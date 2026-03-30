import logging
import math
import re

logger = logging.getLogger(__name__)

class ScoringEngine:
    def __init__(self):
        self._sentiment_analyzer = None
        self._hiraspark_skb = None   # Lazy-loaded HieraSpark SpectralKernelBank
        # Complex Weights - Multi-signal calibration
        self.weights = {
            "semantic_match": 0.18,
            "brand_trust":    0.14,
            "rating":         0.18,
            "sentiment":      0.23,
            "volume":         0.09,
            "price_value":    0.10,
            "hiraspark":      0.08,  # HieraSpark spectral novelty score
        }
        self.TRUSTED_BRANDS = {
            "apple", "samsung", "sony", "bose", "dell", "hp", "lenovo", "asus",
            "logitech", "nike", "adidas", "dyson", "philips", "panasonic",
            "microsoft", "nintendo", "lg", "canon", "nikon", "dji"
        }

    def _get_analyzer(self):
        if self._sentiment_analyzer is None:
            logger.info("Loading Sentiment Classifier pipeline (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
            try:
                from transformers import pipeline
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                    device="cpu", 
                    truncation=True, 
                    max_length=512
                )
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
        return self._sentiment_analyzer

    def _get_hiraspark_skb(self):
        """Lazy-init a fixed-seed SpectralKernelBank for deterministic scoring."""
        if self._hiraspark_skb is None:
            try:
                import torch
                from ml_engine.models.hiraspark_adapter import SpectralKernelBank
                import torch
                torch.manual_seed(42)
                self._hiraspark_skb = SpectralKernelBank(hidden_size=64, n_kernels=8)
                self._hiraspark_skb.eval()
                logger.info("HieraSpark SKB loaded for spectral scoring.")
            except Exception as e:
                logger.warning(f"HieraSpark unavailable (CPU mode OK): {e}")
                self._hiraspark_skb = False  # sentinel: don't try again
        return self._hiraspark_skb if self._hiraspark_skb else None

    def _compute_hiraspark_novelty(self, title: str) -> float:
        """Spectral novelty score: measures information density of product title.
        
        Primary path: HieraSpark SpectralKernelBank (if torch available).
        Fallback: Character bigram entropy — deterministic and meaningful.
        """
        try:
            import torch
            skb = self._get_hiraspark_skb()
            if skb is None:
                return self._bigram_entropy(title)
            raw = [b / 255.0 for b in title.encode('utf-8')[:64]]
            raw += [0.0] * (64 - len(raw))
            x = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(1, 64, 64)
            with torch.no_grad():
                energy = skb(x)          # (1, 64, 1), bounded by tanh
                score = (energy.mean().item() + 1.0) / 2.0  # → [0, 1]
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return self._bigram_entropy(title)

    def _bigram_entropy(self, title: str) -> float:
        """Character bigram entropy — information-theoretic title novelty proxy."""
        import math
        n = len(title)
        if n < 2:
            return 0.5
        bigrams = [title[i:i+2] for i in range(n - 1)]
        freq: dict = {}
        for b in bigrams:
            freq[b] = freq.get(b, 0) + 1
        total = len(bigrams)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        return min(entropy / 5.0, 1.0)   # max_entropy ≈ 5 bits for 2-grams

    def _calculate_semantic_match(self, title: str, query: str) -> float:
        """Calculates fuzzy overlap between product title and search query."""
        t_words = set(re.findall(r'\w+', title.lower()))
        q_words = set(re.findall(r'\w+', query.lower()))
        if not q_words: return 0.5
        intersection = t_words.intersection(q_words)
        return min(1.0, len(intersection) / len(q_words))

    def _get_brand_authority(self, title: str) -> float:
        """Checks if a trusted brand name is present in the title."""
        title_low = title.lower()
        if any(brand in title_low for brand in self.TRUSTED_BRANDS):
            return 1.0
        return 0.4 # Baseline for unverified/generic brands

    def calculate_raw_score(self, item_data: dict, essentiality: float, query: str = "") -> dict:
        """
        Calculates a multi-dimensional composite score (0.0 – 1.0).
        reliability_score is now a transparent metric based on data density.
        """
        try:
            # 1. Semantic Match (Relevance)
            title = item_data.get('title', 'Unknown')
            semantic_score = self._calculate_semantic_match(title, query)

            # 2. Sentiment Analysis
            reviews = item_data.get('reviews', [])
            if not reviews:
                avg_sentiment = 0.6
                sentiment_confidence = 0.3 # Low confidence without reviews
            else:
                try:
                    analyzer = self._get_analyzer()
                    if analyzer:
                        sentiments = analyzer(reviews, top_k=None)
                        pos_scores = [next((x['score'] for x in res if x['label'] == 'positive'), 0.5) for res in sentiments]
                        avg_sentiment = float(sum(pos_scores)) / len(pos_scores) if pos_scores else 0.5
                        sentiment_confidence = 1.0
                    else: raise ValueError
                except:
                    # Fallback keyword sentiment
                    pos = {"good", "great", "excellent", "love", "best", "perfect"}
                    neg = {"bad", "poor", "broken", "cheap", "fake", "terrible"}
                    scores = [0.5 + (sum(1 for w in pos if w in r.lower())*0.1) - (sum(1 for w in neg if w in r.lower())*0.1) for r in reviews]
                    avg_sentiment = float(sum(scores)) / len(scores) if scores else 0.5
                    sentiment_confidence = 0.7

            # 3. Rating & Volume (Non-linear)
            raw_rating = item_data.get('rating', 0.0)
            norm_rating = min(max(raw_rating / 5.0, 0.0), 1.0)
            
            review_count = item_data.get('sales_volume', 0) # Mapping sales_volume to review volume for consistency
            volume_score = min(1.0, math.log10(max(1, review_count)) / 4.0) # log10(10000) = 4

            # 4. Brand Authority
            brand_score = self._get_brand_authority(title)

            # 5. Price Value
            price = item_data.get('price_inr', 0) or 0
            price_score = 0.7 # Default
            if price > 0:
                # Award points for presence of significant discount
                discount = item_data.get('discount', 0)
                price_score = min(1.0, 0.5 + (discount / 100.0))

            # ==== Reliability Score (Transparency Metric) ====
            # Reliability is based on data density: Review Volume, Brand Trust, and Semantic Alignment
            reliability = (
                (volume_score * 0.4) + 
                (brand_score * 0.3) + 
                (semantic_score * 0.2) + 
                (sentiment_confidence * 0.1)
            )

            # HieraSpark Spectral Novelty Score
            hiraspark_score = self._compute_hiraspark_novelty(title)

            # ==== Weighted Composite Score (Final Ranking) ====
            w = self.weights
            composite = (
                (semantic_score   * w["semantic_match"]) +
                (brand_score      * w["brand_trust"]) +
                (norm_rating      * w["rating"]) +
                (avg_sentiment    * w["sentiment"]) +
                (volume_score     * w["volume"]) +
                (price_score      * w["price_value"]) +
                (hiraspark_score  * w.get("hiraspark", 0.08))
            )

            # Essentiality boost
            composite = min(max(composite + (essentiality * 0.05), 0.0), 1.0)

            logger.info(f"Scored '{title[:30]}' -> Rank:{composite:.3f} Rel:{reliability:.3f}")

            import random
            
            # Mock Novelty Upgrades
            wait_to_buy = random.random() < 0.15  # 15% chance to flag Wait-to-Buy
            
            coupons = ["SAVE10", "WELCOME5", "FESTIVAL20", None, None, None, None, None]
            coupon_applied = random.choice(coupons)
            
            reddit_mock_sentiments = [
                "Highly recommended by r/buildapc",
                "Users say it runs slightly warm",
                "r/deals: Historic low price!",
                "Solid choice for the price point",
                "Community consensus: Great value",
                None, None, None
            ]
            reddit_sentiment = random.choice(reddit_mock_sentiments)

            return {
                "raw_score":        composite,
                "reliability_score": round(reliability, 2),
                "hiraspark_novelty": round(hiraspark_score, 3),
                "price_inr":        price,
                "title":            title,
                "platform":         item_data.get('platform', 'Amazon'),
                "link":             item_data.get('link', '#'),
                "image":            item_data.get('image', ''),
                "sentiment":        round(avg_sentiment * 100, 1),
                "is_bestseller":    item_data.get('bestseller', False),
                "sales_volume":     review_count,
                "discount_pct":     item_data.get('discount', 0),
                "tags":             [],
                "wait_to_buy":      wait_to_buy,
                "coupon_applied":   coupon_applied,
                "reddit_sentiment": reddit_sentiment
            }

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return {"raw_score": 0.0, "reliability_score": 0.2, "title": "Error", "price_inr": 0}

    def normalize_scores(self, results: list[dict]) -> list[dict]:
        """Normalizes scores and assigns high-fidelity tags."""
        if not results: return []
        
        raw_scores = [r.get("raw_score", 0.0) for r in results]
        lo, hi = min(raw_scores), max(raw_scores)

        for r in results:
            rs = float(r.get("raw_score", 0.0))
            normalized = 75.0 if hi == lo else ((rs - lo) / (hi - lo)) * 100.0
            r["score"] = round(normalized, 1)
            
            # Map reliability internal (0.1) to UI percent (0-100)
            r["reliability_score"] = int(float(r.get("reliability_score", 0.5)) * 100)

        # ── Tagging Logic ──
        # 1. Most Reliable (Highest Reliability Score)
        by_rel = sorted(results, key=lambda x: x["reliability_score"], reverse=True)
        for r in by_rel[:2]: r["tags"].append("Most Reliable")
        
        # 2. Most Discounted (Highest Discount Percentage)
        by_discount = sorted(results, key=lambda x: x.get("discount_pct", 0), reverse=True)
        for r in by_discount[:2]:
            if r.get("discount_pct", 0) > 10: r["tags"].append("Most Discounted")
            
        # 3. Trending (High volume + Recent/High Sentiment)
        by_trending = sorted(results, key=lambda x: (x.get("sales_volume", 0) * (x.get("sentiment", 50)/50.0)), reverse=True)
        for r in by_trending[:2]: r["tags"].append("Trending")
        
        # 4. Most Monthly Sales (Highest Volume)
        by_volume = sorted(results, key=lambda x: x.get("sales_volume", 0), reverse=True)
        for r in by_volume[:2]: r["tags"].append("Most Monthly Sales")

        # 5. Top Search Products (Best Semantic Match)
        # Note: semantic_score isn't in final dict, but we can reuse title similarity if needed
        # For now, use the raw_score as proxy for "Top Search"
        by_score = sorted(results, key=lambda x: x.get("raw_score", 0), reverse=True)
        for r in by_score[:2]: r["tags"].append("Top Search Products")

        for r in results:
            if r.get("is_bestseller"): r["tags"].append("Best Seller")
            if not r["tags"]: r["tags"] = ["Curated Node"]
            # Deduplicate tags
            r["tags"] = list(dict.fromkeys(r["tags"]))

        return results

    def calculate_omnilens_score(self, item_data: dict, essentiality: float) -> dict:
        """Legacy shim."""
        res = self.calculate_raw_score(item_data, essentiality)
        res["score"] = round(res["raw_score"] * 100.0, 1)
        return res

    def update_weights(self, new_weights: dict):
        """Map generic global weights to specific scoring engine weights."""
        if "price" in new_weights:
            self.weights["price_value"] = new_weights["price"]
        if "rating" in new_weights:
            self.weights["rating"] = new_weights["rating"]
        if "sentiment" in new_weights:
            self.weights["sentiment"] = new_weights["sentiment"]
        if "bestseller" in new_weights or "sales" in new_weights:
            # Combine bestseller + sales into volume and brand_trust
            combined = (new_weights.get("bestseller", 0) + new_weights.get("sales", 0)) / 2
            self.weights["volume"] = combined
            self.weights["brand_trust"] = combined

scoring_engine = ScoringEngine()
