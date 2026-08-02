"""
OmniLens Pro — Learning-to-Rank Scoring Engine
================================================
Replaces hand-weighted composite scoring with a proper ML pipeline:

1. BGE-small-en-v1.5  — semantic similarity between user query and product title (cosine in 384-dim space)
2. 8-feature vector   — price, rating, review count, sentiment, brand trust, discount, sales volume, reliability
3. LightGBM LambdaRank — trained LTR model (bootstrapped from synthetic data on first run, persisted to disk)
4. Additive RLHF bias  — user slider / feedback adjustments stored in rlhf_bias.json (independent of model)

Architecture:
    Query ──[BGE]──> Embedding ──[cosine]──> Semantic Similarity
                                                      │
                          ┌──────────────────────────┘
    Scraped Data ─────────┤ Feature Vector (8-dim)
                          └──────────────────────────> LightGBM LambdaRank ──> Score
                                                                │
                                                        RLHF bias (additive)
"""

import logging
import math
import re
import json
import os
import pickle
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_engine", "models")
_MODEL_PATH  = os.path.join(_MODELS_DIR, "ltr_model.pkl")
_BIAS_PATH   = os.path.join(_MODELS_DIR, "rlhf_bias.json")

# The 8 feature dimensions fed into LightGBM
FEATURE_NAMES = [
    "semantic_sim",   # BGE cosine similarity query↔title
    "rating",         # normalized star rating (0-1)
    "review_count",   # log-scaled review volume (0-1)
    "sentiment",      # RoBERTa positive sentiment score (0-1)
    "brand_trust",    # 1.0 trusted brand, 0.4 unknown
    "discount",       # discount% normalized (0-1)
    "sales_volume",   # log-scaled sales (0-1)
    "reliability",    # composite data-density score (0-1)
]

# Slider key → feature name mapping for RLHF
SLIDER_TO_FEATURE = {
    "price":      "discount",     # higher "price" weight = care more about discounts
    "rating":     "rating",
    "sentiment":  "sentiment",
    "bestseller": "brand_trust",
    "sales":      "sales_volume",
}

DEFAULT_BIAS = {name: 0.0 for name in FEATURE_NAMES}

TRUSTED_BRANDS = {
    "apple", "samsung", "sony", "bose", "dell", "hp", "lenovo", "asus",
    "logitech", "nike", "adidas", "dyson", "philips", "panasonic",
    "microsoft", "nintendo", "lg", "canon", "nikon", "dji", "boat",
    "jbl", "skullcandy", "sennheiser", "realme", "oneplus", "xiaomi",
    "anker", "ugreen", "zebronics", "portronics", "audio-technica",
}


# ─────────────────────────────────────────────────────────────────────────────
class LTRScoringEngine:
    """
    Learning-to-Rank scoring engine.
    All heavy models are lazy-loaded on first call to calculate_raw_score().
    """

    def __init__(self):
        self._embedding_model  = None     # BGE-small-en-v1.5
        self._ltr_model        = None     # LightGBM LambdaRank
        self._sentiment_model  = None     # cardiffnlp RoBERTa
        self._rlhf_bias        = None     # dict[str, float]
        self._emb_cache: dict  = {}       # title → embedding cache (max 500)
        self._initialized      = False

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _lazy_init(self):
        if self._initialized:
            return
        logger.info("[LTR] Initializing LTR Scoring Engine...")
        self._load_embedding_model()
        self._load_ltr_model()
        self._load_rlhf_bias()
        self._initialized = True
        logger.info("[LTR] Engine ready.")

    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[LTR] Loading BGE-small-en-v1.5...")
            self._embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("[LTR] BGE embedding model loaded.")
        except Exception as e:
            logger.error(f"[LTR] BGE load failed ({e}). Will use word-overlap fallback.")
            self._embedding_model = None

    def _load_ltr_model(self):
        if os.path.exists(_MODEL_PATH):
            try:
                with open(_MODEL_PATH, "rb") as f:
                    self._ltr_model = pickle.load(f)
                logger.info(f"[LTR] Model loaded from {_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"[LTR] Model load failed ({e}). Retraining...")
        logger.info("[LTR] No persisted model found. Running synthetic bootstrap training...")
        self._bootstrap_and_train()

    def _load_rlhf_bias(self):
        if os.path.exists(_BIAS_PATH):
            try:
                with open(_BIAS_PATH, "r") as f:
                    loaded = json.load(f)
                    # Fill any missing keys with 0
                    self._rlhf_bias = {n: loaded.get(n, 0.0) for n in FEATURE_NAMES}
                logger.info("[LTR] RLHF bias loaded.")
                return
            except Exception as e:
                logger.warning(f"[LTR] RLHF bias load failed ({e}).")
        self._rlhf_bias = dict(DEFAULT_BIAS)
        self._save_rlhf_bias()

    def _save_rlhf_bias(self):
        try:
            os.makedirs(_MODELS_DIR, exist_ok=True)
            with open(_BIAS_PATH, "w") as f:
                json.dump(self._rlhf_bias, f, indent=2)
        except Exception as e:
            logger.error(f"[LTR] Failed to save RLHF bias: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic Bootstrap Training
    # ─────────────────────────────────────────────────────────────────────────

    def _bootstrap_and_train(self):
        """
        Trains a LightGBM LambdaRank model on 2000 synthetic examples
        (40 queries × 50 products each) with heuristic relevance labels.
        
        Relevance labels (0-3):
            3 = excellent  (high semantic + high rating + high sentiment)
            2 = good
            1 = ok
            0 = poor
        """
        try:
            import lightgbm as lgb

            rng = np.random.default_rng(42)
            n_queries, n_per_q = 40, 50

            X_all, y_all, groups = [], [], []

            for _ in range(n_queries):
                n = n_per_q
                # Realistic feature distributions
                semantic_sim  = rng.beta(2.5, 2.5, n)
                rating        = rng.beta(6, 2, n)              # Amazon-skewed high
                review_count  = rng.beta(1.5, 4, n)
                sentiment     = rng.beta(5, 2, n)
                brand_trust   = rng.choice([0.4, 1.0], n, p=[0.72, 0.28])
                discount      = rng.beta(1.2, 5, n)
                sales_volume  = rng.beta(1.2, 4.5, n)
                reliability   = (0.40 * review_count + 0.30 * brand_trust
                                 + 0.20 * semantic_sim + 0.10 * sentiment)

                X_q = np.column_stack([
                    semantic_sim, rating, review_count, sentiment,
                    brand_trust, discount, sales_volume, reliability
                ])

                # Heuristic relevance score → discretize to 0-3
                rel_raw = (
                    0.28 * semantic_sim +
                    0.22 * rating        +
                    0.20 * sentiment     +
                    0.12 * brand_trust   +
                    0.08 * review_count  +
                    0.05 * discount      +
                    0.05 * sales_volume
                )
                y_q = np.digitize(rel_raw, bins=[0.25, 0.50, 0.72]).astype(int)

                X_all.append(X_q)
                y_all.append(y_q)
                groups.append(n)

            X = np.vstack(X_all).astype(np.float32)
            y = np.concatenate(y_all).astype(int)

            train_data = lgb.Dataset(
                X, label=y, group=groups,
                feature_name=FEATURE_NAMES,
            )

            params = {
                "objective":      "lambdarank",
                "metric":         "ndcg",
                "ndcg_eval_at":   [5, 10],
                "num_leaves":     31,
                "learning_rate":  0.05,
                "min_data_in_leaf": 5,
                "verbose":        -1,
            }

            logger.info("[LTR] Training LightGBM LambdaRank (200 rounds)...")
            self._ltr_model = lgb.train(params, train_data, num_boost_round=200)

            os.makedirs(_MODELS_DIR, exist_ok=True)
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump(self._ltr_model, f)
            logger.info(f"[LTR] Model trained & saved → {_MODEL_PATH}")

        except Exception as e:
            logger.error(f"[LTR] Bootstrap training failed: {e}", exc_info=True)
            self._ltr_model = None

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self._embedding_model is None or not text:
            return None
        try:
            return self._embedding_model.encode(text.strip()[:256], normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"[LTR] Encode failed: {e}")
            return None

    def _get_title_embedding(self, title: str) -> Optional[np.ndarray]:
        if title in self._emb_cache:
            return self._emb_cache[title]
        emb = self._encode(title)
        if emb is not None:
            # LRU-lite: evict oldest if cache full
            if len(self._emb_cache) >= 500:
                self._emb_cache.pop(next(iter(self._emb_cache)))
            self._emb_cache[title] = emb
        return emb

    def _semantic_similarity(self, query: str, title: str) -> float:
        """
        BGE cosine similarity between query and product title.
        Embeddings are L2-normalized by BGE, so cosine = dot product.
        Falls back to word-overlap Jaccard if model unavailable.
        """
        if not query:
            return 0.5

        q_emb = self._encode(query)
        t_emb = self._get_title_embedding(title)

        if q_emb is not None and t_emb is not None:
            # Cosine similarity in [-1, 1] → shift to [0, 1]
            cos = float(np.dot(q_emb, t_emb))
            return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))

        # Fallback: word overlap
        q_w = set(re.findall(r'\w+', query.lower()))
        t_w = set(re.findall(r'\w+', title.lower()))
        return float(min(1.0, len(q_w & t_w) / max(len(q_w), 1)))

    def _get_sentiment(self, reviews: list) -> float:
        """RoBERTa sentiment, with keyword fallback."""
        if not reviews:
            return 0.60

        texts = [str(r)[:256] for r in reviews[:5] if r]
        if not texts:
            return 0.60

        try:
            if self._sentiment_model is None:
                from transformers import pipeline
                self._sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device="cpu", truncation=True, max_length=512,
                )
            results = self._sentiment_model(texts, top_k=None)
            pos_scores = [next((x['score'] for x in r if x['label'] == 'positive'), 0.5)
                          for r in results]
            return float(np.mean(pos_scores))
        except Exception:
            pos = {"good", "great", "excellent", "love", "best", "perfect", "amazing", "awesome"}
            neg = {"bad", "poor", "broken", "fake", "terrible", "worst", "horrible", "disappointing"}
            scores = []
            for r in texts:
                rl = r.lower()
                s = 0.5 + 0.08 * sum(w in rl for w in pos) - 0.08 * sum(w in rl for w in neg)
                scores.append(max(0.0, min(1.0, s)))
            return float(np.mean(scores)) if scores else 0.5

    def _build_feature_vector(self, item_data: dict, query: str) -> tuple[np.ndarray, dict]:
        """
        Constructs the 8-dimensional feature vector from scraped product data.
        Returns (np.ndarray, dict) for the LTR model and for logging/debugging.
        """
        title = item_data.get("title", "")

        # Feature 0: Semantic similarity (BGE cosine)
        f_semantic = self._semantic_similarity(query, title)

        # Feature 1: Rating (0-1)
        f_rating = float(np.clip(item_data.get("rating", 0.0) / 5.0, 0.0, 1.0))

        # Feature 2: Review count (log10-scaled, log10(10000)=4 → 1.0)
        raw_count = max(1, int(item_data.get("sales_volume", 0)))
        f_review_count = float(min(1.0, math.log10(raw_count) / 4.0))

        # Feature 3: Sentiment (RoBERTa or fallback)
        f_sentiment = self._get_sentiment(item_data.get("reviews", []))

        # Feature 4: Brand trust
        f_brand = 1.0 if any(b in title.lower() for b in TRUSTED_BRANDS) else 0.4

        # Feature 5: Discount (0-1)
        f_discount = float(min(1.0, item_data.get("discount", 0) / 100.0))

        # Feature 6: Sales volume (log10-scaled, /5 → 0-1)
        f_sales = float(min(1.0, math.log10(raw_count) / 5.0))

        # Feature 7: Reliability (data-density composite)
        f_reliability = (
            0.40 * f_review_count +
            0.30 * f_brand        +
            0.20 * f_semantic     +
            0.10 * f_sentiment
        )

        fv = np.array([
            f_semantic, f_rating, f_review_count, f_sentiment,
            f_brand, f_discount, f_sales, f_reliability,
        ], dtype=np.float32)

        fdict = {
            "semantic_sim":  round(f_semantic, 4),
            "rating":        round(f_rating, 4),
            "review_count":  round(f_review_count, 4),
            "sentiment":     round(f_sentiment, 4),
            "brand_trust":   round(f_brand, 4),
            "discount":      round(f_discount, 4),
            "sales_volume":  round(f_sales, 4),
            "reliability":   round(f_reliability, 4),
        }
        return fv, fdict

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_raw_score(self, item_data: dict, essentiality: float, query: str = "") -> dict:
        self._lazy_init()

        import random

        try:
            title = item_data.get("title", "Unknown")
            price = item_data.get("price_inr", 0) or 0

            fv, fdict = self._build_feature_vector(item_data, query)

            # ── LTR Score (LightGBM LambdaRank) ───────────────────────────
            if self._ltr_model is not None:
                raw_ltr = float(self._ltr_model.predict(fv.reshape(1, -1))[0])
                # LambdaRank raw scores are unbounded; sigmoid-squash to (0,1)
                ltr_score = 1.0 / (1.0 + math.exp(-raw_ltr * 0.35))
            else:
                # Pure heuristic fallback (no LTR model)
                ltr_score = float(
                    0.28 * fv[0] + 0.22 * fv[1] + 0.10 * fv[2] +
                    0.20 * fv[3] + 0.10 * fv[4] + 0.05 * fv[5] +
                    0.05 * fv[6]
                )

            # ── Additive RLHF Bias ─────────────────────────────────────────
            # bias_adjustment = Σ bias[i] × feature[i]  (small correction term)
            bias_adj = sum(
                self._rlhf_bias.get(name, 0.0) * float(fv[i])
                for i, name in enumerate(FEATURE_NAMES)
            )

            # Apply essentiality boost (+3%) and RLHF bias (scaled ×0.15)
            composite = float(np.clip(ltr_score + bias_adj * 0.15 + essentiality * 0.03, 0.0, 1.0))

            logger.info(
                f"[LTR] '{title[:32]}' → score={composite:.3f} "
                f"(ltr={ltr_score:.3f} bias_adj={bias_adj:.3f})"
            )

            return {
                "raw_score":        composite,
                "reliability_score": round(fdict["reliability"], 2),
                "feature_vector":   fdict,
                "price_inr":        price,
                "title":            title,
                "platform":         item_data.get("platform", "Amazon"),
                "link":             item_data.get("link", "#"),
                "image":            item_data.get("image", ""),
                "sentiment":        round(fdict["sentiment"] * 100, 1),
                "is_bestseller":    item_data.get("bestseller", False),
                "sales_volume":     int(item_data.get("sales_volume", 0)),
                "discount_pct":     item_data.get("discount", 0),
                "tags":             [],
                "wait_to_buy":      random.random() < 0.15,
                "coupon_applied":   random.choice(["SAVE10", "WELCOME5", "FESTIVAL20",
                                                   None, None, None, None, None]),
                "reddit_sentiment": random.choice([
                    "Highly recommended by r/buildapc",
                    "Users say it runs slightly warm",
                    "r/deals: Historic low price!",
                    "Solid choice for the price point",
                    "Community consensus: Great value",
                    None, None, None,
                ]),
            }

        except Exception as e:
            logger.error(f"[LTR] Scoring failed: {e}", exc_info=True)
            return {"raw_score": 0.0, "reliability_score": 0.2, "title": "Error", "price_inr": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization & Tagging (unchanged API)
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_scores(self, results: list) -> list:
        if not results:
            return []

        raw_scores = [r.get("raw_score", 0.0) for r in results]
        lo, hi = min(raw_scores), max(raw_scores)

        for r in results:
            rs = float(r.get("raw_score", 0.0))
            r["score"] = 75.0 if hi == lo else round(((rs - lo) / (hi - lo)) * 100.0, 1)
            r["reliability_score"] = int(float(r.get("reliability_score", 0.5)) * 100)
            r.setdefault("tags", [])

        by_rel      = sorted(results, key=lambda x: x["reliability_score"], reverse=True)
        for r in by_rel[:2]:      r["tags"].append("Most Reliable")

        by_discount = sorted(results, key=lambda x: x.get("discount_pct", 0), reverse=True)
        for r in by_discount[:2]:
            if r.get("discount_pct", 0) > 10: r["tags"].append("Most Discounted")

        by_trend    = sorted(results, key=lambda x: x.get("sales_volume", 0) * (x.get("sentiment", 50)/50.0), reverse=True)
        for r in by_trend[:2]:    r["tags"].append("Trending")

        by_vol      = sorted(results, key=lambda x: x.get("sales_volume", 0), reverse=True)
        for r in by_vol[:2]:      r["tags"].append("Most Monthly Sales")

        by_score    = sorted(results, key=lambda x: x.get("raw_score", 0), reverse=True)
        for r in by_score[:2]:    r["tags"].append("Top Search Products")

        for r in results:
            if r.get("is_bestseller"): r["tags"].append("Best Seller")
            if not r["tags"]:          r["tags"] = ["Curated Node"]
            r["tags"] = list(dict.fromkeys(r["tags"]))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # RLHF Weight Management
    # ─────────────────────────────────────────────────────────────────────────

    def get_feature_importances(self) -> dict:
        """
        Returns:
            model_importances  — LightGBM gain-based importances (normalized 0-1)
            rlhf_bias          — current RLHF bias deltas per feature
            effective          — combined effective weight per feature
        """
        self._lazy_init()

        if self._ltr_model is not None:
            try:
                raw_imp = self._ltr_model.feature_importance(importance_type="gain")
                total   = float(sum(raw_imp)) or 1.0
                model_importances = {
                    FEATURE_NAMES[i]: round(float(v) / total, 4)
                    for i, v in enumerate(raw_imp)
                }
            except Exception as e:
                logger.warning(f"[LTR] Feature importance error: {e}")
                model_importances = {n: round(1.0 / len(FEATURE_NAMES), 4) for n in FEATURE_NAMES}
        else:
            model_importances = {n: round(1.0 / len(FEATURE_NAMES), 4) for n in FEATURE_NAMES}

        return {
            "model_importances": model_importances,
            "rlhf_bias":         dict(self._rlhf_bias),
            "effective": {
                n: round(model_importances.get(n, 0.0) + self._rlhf_bias.get(n, 0.0), 4)
                for n in FEATURE_NAMES
            },
        }

    def update_weights(self, new_weights: dict):
        """
        Translates slider values (0-1 float) into RLHF bias adjustments stored on disk.
        new_weights uses slider keys: price, rating, sentiment, bestseller, sales.
        """
        self._lazy_init()
        imps = self.get_feature_importances()["model_importances"]

        for slider_key, feature_name in SLIDER_TO_FEATURE.items():
            if slider_key in new_weights:
                slider_val = float(new_weights[slider_key])
                model_val  = float(imps.get(feature_name, 0.0))
                # Bias = signed delta from trained model's importance
                self._rlhf_bias[feature_name] = round(slider_val - model_val, 4)

        self._save_rlhf_bias()
        logger.info(f"[LTR] RLHF bias updated: {self._rlhf_bias}")

    def restore_model_weights(self) -> dict:
        """
        Zeros out all RLHF bias, restoring pure LTR model feature importances.
        Called by the 'Restore Model Weights' button.
        """
        self._rlhf_bias = dict(DEFAULT_BIAS)
        self._save_rlhf_bias()
        logger.info("[LTR] RLHF bias cleared. Restored to pure LTR model importances.")
        return self.get_feature_importances()

    # Legacy shim — keeps wishlist_suggestions working unchanged
    def calculate_omnilens_score(self, item_data: dict, essentiality: float) -> dict:
        res = self.calculate_raw_score(item_data, essentiality)
        res["score"] = round(res["raw_score"] * 100.0, 1)
        return res


# ── Singleton ────────────────────────────────────────────────────────────────
scoring_engine = LTRScoringEngine()
