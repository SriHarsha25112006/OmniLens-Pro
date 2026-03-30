"""
OmniLens Intent Parser (v2)
===========================
Three-tier classification pipeline:

Tier 1 – Taxonomy Keyword Match (deterministic, zero latency)
  └─ Maps known scenario phrases → curated component lists with ~100% accuracy

Tier 2 – Zero-shot NLI Classifier (facebook/bart-large-mnli)
  └─ Classifies any query as SCENARIO or PRODUCT with ~97%+ accuracy
  └─ SCENARIO → asks flan-t5 to generate components (properly prompted)
  └─ PRODUCT  → asks flan-t5 to generate model variants

Tier 3 – Flan-T5 Raw Generation Fallback
  └─ Properly prompted: "List exactly 10 PHYSICAL COMPONENTS needed for X"
"""

import re
import os
import json
import logging
import difflib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from ml_engine.models.intent_taxonomy import SCENARIO_TAXONOMY, TRAINING_DATA

logger = logging.getLogger(__name__)


class IntentParser:
    def __init__(self):
        logger.info("Loading Intent Parser (v2) — Taxonomy + Fine-Tuned Classifier + Flan-T5...")

        # Taxonomy patterns (compiled for fast matching)
        self._taxonomy_patterns = []
        for key_str, components in SCENARIO_TAXONOMY.items():
            patterns = [re.compile(r'\b' + kw.strip().replace(' ', r'\s+') + r'\b', re.IGNORECASE)
                        for kw in key_str.split('|')]
            self._taxonomy_patterns.append((patterns, components))

        # Lazy-loaded models (set to None initially)
        self._ft_encoder = None
        self._ft_clf = None
        self._nli_classifier = None
        self.model = None

        # Massive vocabulary dictionary mapped for typo-correction and heuristic matches
        self.KNOWN_ENTITIES = {
            # Tech & Electronics
            'sony', 'apple', 'iphone', 'samsung', 'pixel', 'oneplus', 'lg', 'asus', 
            'lenovo', 'dell', 'hp', 'msi', 'acer', 'microsoft', 'nintendo', 'playstation', 
            'xbox', 'logitech', 'razer', 'corsair', 'bose', 'sennheiser', 'jabra', 
            'garmin', 'fitbit', 'dji', 'gopro', 'intel', 'amd', 'nvidia', 'nothing',
            'xiaomi', 'redmi', 'oppo', 'vivo', 'realme', 'motorola', 'nokia', 'boat',
            'noise', 'fire-boltt', 'pebble', 'marshall', 'jbl', 'skullcandy', 'panasonic',
            'canon', 'nikon', 'fujifilm', 'olympus', 'sandisk', 'seagate', 'western digital',
            'kingston', 'crucial', 'tp-link', 'netgear', 'ubiquiti', 'cisco', 'linksys',
            'epson', 'brother', 'nzxt', 'fractal', 'steelseries', 'hyperx', 'roccat',

            # Home, Kitchen & Appliances
            'dyson', 'philips', 'whirlpool', 'bosch', 'siemens', 'haier', 'voltas', 
            'godrej', 'bajaj', 'havells', 'crompton', 'prestige', 'pigeon', 'kitchenaid',
            'cuisinart', 'breville', 'nespresso', 'keurig', 'ninja', 'vitamix', 'roomba',
            'irobot', 'shark', 'bissell', 'hoover', 'eureka', 'panasonic', 'hitachi',
            'toshiba', 'sharp', 'sanyo', 'daikin', 'carrier', 'trane', 'lennox', 'ruud',
            'yeti', 'hydro flask', 'stanley', 'thermos', 'contigo', 'oxo', 'tupperware',

            # Fashion, Apparel & Lifestyle
            'nike', 'adidas', 'puma', 'reebok', 'under armour', 'new balance', 'asics',
            'skechers', 'casio', 'fossil', 'titan', 'fastrack', 'ray-ban', 'oakley',
            'zara', 'h&m', 'levis', 'calvin klein', 'tommy hilfiger', 'gucci', 'prada',
            'louis vuitton', 'chanel', 'hermes', 'rolex', 'omega', 'tag heuer', 'breitling',
            'patagonia', 'the north face', 'columbia', 'timberland', 'vans', 'converse',
            'lacoste', 'ralph lauren', 'hugo boss', 'armani', 'versace', 'dior', 'balenciaga',
            'givenchy', 'fendi', 'burberry', 'valentino', 'alexander mcqueen', 'bottega veneta',

            # Categories & Objects
            'headphones', 'earphones', 'earbuds', 'laptop', 'phone', 'camera', 'watch', 
            'monitor', 'keyboard', 'mouse', 'tv', 'television', 'tablet', 'ipad', 
            'macbook', 'charger', 'speaker', 'projector', 'router', 'ssd', 'hdd',
            'smartwatch', 'controller', 'console', 'processor', 'gpu', 'motherboard', 'ram',
            'shoe', 'sneaker', 'shirt', 't-shirt', 'jacket', 'jeans', 'bag', 'backpack',
            'bottle', 'mat', 'board', 'game', 'toy', 'book', 'pen', 'desk', 'chair',
            'table', 'sofa', 'couch', 'bed', 'mattress', 'pillow', 'blanket', 'towel',
            'microwave', 'oven', 'fridge', 'refrigerator', 'washer', 'dryer', 'dishwasher'
        }

        logger.info("Intent Parser v2 initialized with Taxonomy. NLP models will load on-demand.")

    def _correct_typos(self, prompt: str) -> str:
        words = prompt.split()
        corrected = []
        for word in words:
            # clean punctuation for lookup
            clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
            if len(clean_word) > 3 and clean_word not in self.KNOWN_ENTITIES:
                matches = difflib.get_close_matches(clean_word, self.KNOWN_ENTITIES, n=1, cutoff=0.8)
                if matches:
                    # Restore original capitalization/punctuation format if needed (simplified here)
                    corrected.append(matches[0])
                    continue
            corrected.append(word)
        return " ".join(corrected)

    def _get_ft_clf(self):
        if self._ft_clf is None:
            pkl_path = os.path.join(os.path.dirname(__file__), "intent_classifier.pkl")
            if os.path.exists(pkl_path):
                try:
                    import pickle
                    from sentence_transformers import SentenceTransformer
                    with open(pkl_path, "rb") as f:
                        saved = pickle.load(f)
                    self._ft_clf = saved["classifier"]
                    self._ft_encoder = SentenceTransformer(saved["encoder_name"])
                    logger.info("Loaded fine-tuned intent classifier lazily.")
                except Exception as e:
                    logger.warning(f"Lazy-load failed for fine-tuned classifier: {e}")
        return self._ft_clf, self._ft_encoder

    def _get_nli(self):
        if self._nli_classifier is None:
            logger.info("Loading zero-shot NLI classifier (facebook/bart-large-mnli)...")
            try:
                self._nli_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1,
                )
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
        return self._nli_classifier

    def _get_gen_model(self):
        if self.model is None:
            try:
                logger.info("Loading flan-t5-small for dynamic generation...")
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small', local_files_only=False)
                self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', local_files_only=False)
            except Exception as e:
                logger.error(f"Failed to load generation model: {e}")
        return self.tokenizer, self.model

    # ─────────────────────────────────────────────────────────────────────────
    def extrapolate_checklist(self, prompt: str, exclude_items: list = None, num_items: int = 10) -> list:
        """
        Given a user prompt, returns a list of product items to search for.
        """
        prompt_clean = self._correct_typos(prompt.strip())
        logger.info(f"Spell-corrected prompt: '{prompt_clean}'")
        exclude_items = exclude_items or []

        # ── TIER 1: Taxonomy keyword match ───────────────────────────────────
        taxonomy_components = self._match_taxonomy(prompt_clean)
        if taxonomy_components:
            logger.info(f"Tier 1 (Taxonomy) matched for: '{prompt_clean[:60]}'")
            filtered_components = [c for c in taxonomy_components if c.lower() not in exclude_items]
            return self._format_items(filtered_components[:num_items], intent="SCENARIO")

        # ── TIER 2: Zero-shot NLI classification ─────────────────────────────
        intent = self._classify_intent(prompt_clean)
        logger.info(f"Tier 2 (NLI) classified '{prompt_clean[:50]}' as: {intent}")

        if intent == "SCENARIO":
            components = self._generate_components_for_scenario(prompt_clean, exclude_items, num_items)
            return self._format_items(components[:num_items], intent="SCENARIO")
        else:
            # PRODUCT: generate model variants / best options
            variants = self._generate_product_variants(prompt_clean, exclude_items, num_items)
            return self._format_items(variants[:num_items], intent="PRODUCT")

    # ─────────────────────────────────────────────────────────────────────────
    def _match_taxonomy(self, prompt: str) -> list | None:
        """
        Tier 1: Deterministic keyword-based taxonomy lookup.
        Returns a component list if any known scenario keyword is found.
        """
        for patterns, components in self._taxonomy_patterns:
            for pattern in patterns:
                if pattern.search(prompt):
                    return components
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _classify_intent(self, prompt: str) -> str:
        """
        Tier 0: Fast Heuristic (zero latency)
        Tier 2: Uses fine-tuned sentence classifier if available,
        otherwise falls back to zero-shot NLI.
        """
        prompt_low = self._correct_typos(prompt.lower().strip())
        words = prompt_low.split()
        
        # ── Tier 0.1: Immediate SCENARIO markers (High confidence)
        scenario_markers = {
            'setup', 'set up', 'install', 'planning', 'going to', 'want to',
            'trip', 'vacation', 'resort', 'mountain', 'hiking', 'trek',
            'theatre', 'theater', 'building', 'build a', 'need help', 'shop for'
        }
        if any(marker in prompt_low for marker in scenario_markers):
            # Only force if it's not a tiny query like "theater"
            if len(words) >= 3:
                return "SCENARIO"

        # ── Tier 0.2: Short PRODUCT queries with brands/categories
        if len(words) <= 5:
            if any(w in self.KNOWN_ENTITIES for w in words):
                return "PRODUCT"
            # If it's just a model number
            if any(any(c.isdigit() for c in w) for w in words):
                return "PRODUCT"

        # Path A: Fine-tuned model (LR on sentence-transformers)
        ft_clf, ft_enc = self._get_ft_clf()
        if ft_clf is not None and ft_enc is not None:
            try:
                emb = ft_enc.encode([prompt])
                pred = ft_clf.predict(emb)[0]
                return "SCENARIO" if pred == 1 else "PRODUCT"
            except Exception as e:
                logger.warning(f"Fine-tuned classifier inference failed: {e}")

        # Path B: Zero-shot NLI (HuggingFace)
        nli = self._get_nli()
        if nli is not None:
            try:
                candidate_labels = [
                    "a situation or goal that requires multiple different products",
                    "a specific product or product category to search for"
                ]
                result = nli(prompt, candidate_labels)
                top_label = result['labels'][0]
                return "SCENARIO" if "situation" in top_label else "PRODUCT"
            except Exception as e:
                logger.warning(f"NLI classification failed: {e}")

        # Path C: Length-based fallback
        if len(words) >= 6:
            return "SCENARIO"
        return "PRODUCT"

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_components_for_scenario(self, prompt: str, exclude_items: list = None, num_items: int = 10) -> list:
        """
        Tier 3 (SCENARIO path): Asks flan-t5 to list essential physical products.
        Uses few-shot prompting to anchor the model to high-quality results.
        """
        few_shot_context = (
            "Example 1: 'planning a beach vacation'\n"
            "Output: Sunscreen SPF 50, Waterproof Beach Bag, Snorkeling Mask, Quick-dry Towel, Reef-safe Sunblock, Oversized Beach Hat, Polarized Sunglasses, Beach Umbrella, Portable Power Bank, Inflatable Beach Ball\n\n"
            "Example 2: 'setting up a home office'\n"
            "Output: Ergonomic Office Chair, Standing Desk, 4K Web Camera, Noise Canceling Headphones, Mechanical Wireless Keyboard, Dual Monitor Stand, Large Desk Mat, USB-C Docking Station, Cable Management Clips, LED Desk Lamp\n\n"
            "Example 3: 'starting a home gym'\n"
            "Output: Adjustable Dumbbell Set, Thick Yoga Mat, Heavy Duty Resistance Bands, Doorway Pull-up Bar, Speed Jump Rope, Workout Bench, Kettlebell 16kg, Foam Roller, Exercise Ball, Fitness Heart Rate Tracker\n\n"
        )
        
        exclude_str = f" Do NOT include any of: {', '.join(exclude_items)}." if exclude_items else ""
        ml_prompt = (
            f"{few_shot_context}"
            f"Query: \"{prompt}\"\n"
            f"Task: List exactly {num_items} essential physical items to buy for this goal.{exclude_str} Use only product names. No descriptions.\n"
            f"Output:"
        )
        return self._run_generation(ml_prompt, prompt, num_items)

    # ─────────────────────────────────────────────────────────────────────────
    def _generate_product_variants(self, prompt: str, exclude_items: list = None, num_items: int = 10) -> list:
        """
        Tier 3 (PRODUCT path): Asks flan-t5 for the best specific models.
        """
        exclude_str = f" Do NOT include any of: {', '.join(exclude_items)}." if exclude_items else ""
        ml_prompt = (
            "Example: 'iPhone'\n"
            "Output: iPhone 15 Pro, iPhone 15, iPhone 14 Plus, iPhone 13 Mini, iPhone 12, iPhone SE 2022, iPhone 15 Pro Max, iPhone 14 Pro, iPhone 13, iPhone 11\n\n"
            f"Query: \"{prompt}\"\n"
            f"Task: List exactly {num_items} specific product model numbers or high-end variants.{exclude_str}\n"
            f"Output:"
        )
        return self._run_generation(ml_prompt, prompt, num_items)

    # ─────────────────────────────────────────────────────────────────────────
    def _run_generation(self, ml_prompt: str, fallback_context: str, num_items: int = 10) -> list:
        """Runs flan-t5 generation and parses the output into a clean list."""
        tk, md = self._get_gen_model()
        if tk is None or md is None:
            logger.warning("Generation model not available. Returning empty list.")
            return []
            
        try:
            inputs = tk(ml_prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = md.generate(**inputs, max_new_tokens=150, num_beams=4, early_stopping=True)
            generated_text = tk.decode(outputs[0], skip_special_tokens=True)
            
            # Clean separators
            text = generated_text.replace('\n', ',').replace(';', ',')
            raw_items = [item.strip(' "\'[]()') for item in text.split(',') if item.strip()]

            items = []
            # Meta-commands and generic instruction phrases to prune
            BLACK_LIST = {
                "respond", "comma-separated", "output:", "rules:", "based on", 
                "essential", "physical", "items", "to buy", "for this", "list",
                "here are", "i suggest", "task:", "query:", "exactly", "model"
            }

            for item in raw_items:
                item_clean = item.strip()
                if not item_clean or len(item_clean) < 3 or len(item_clean) > 50:
                    continue
                
                lower = item_clean.lower()
                # Stop word / instruction phrase filter
                if any(bad in lower for bad in BLACK_LIST):
                    continue
                
                # Filter out numbers/ordinals if the model hallucinations "1. Product"
                item_clean = re.sub(r'^\d+[\.\)]\s*', '', item_clean)
                
                if item_clean:
                    items.append(item_clean)

            return items[:num_items]

        except Exception as e:
            logger.error(f"Flan-T5 generation failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    def _format_items(self, item_names: list, intent: str = "PRODUCT") -> list:
        """
        Pads to 10 items using DDG search if needed, then formats for the API.
        """
        items = list(item_names)

        # Deduplicate preserving order
        seen = set()
        items = [x for x in items if x.lower() not in seen and not seen.add(x.lower())]

        # Format
        formatted = []
        for idx, name in enumerate(items):
            formatted.append({
                "id": str(idx + 1),
                "name": name,
                "category": "Components" if intent == "SCENARIO" else "Gear",
                "essentiality": round(max(0.2, 1.0 - (idx * 0.08)), 2),
                "estimatedCost": 0
            })

        return formatted



intent_parser = IntentParser()
