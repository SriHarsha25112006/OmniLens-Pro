"""
OmniLens Unified Neural Engine (flan-t5-small Core)
===================================================
Combines Query Clarification, Typo Rewriting, Intent Parsing,
and Checklist Extrapolation into a single unified Neural Pipeline powered by google/flan-t5-small.

Alignment with LTR Scoring Engine:
  - _format_items() emits: id, name, search_query, essentiality, category
  - name        → human-readable label (shown in UI)
  - search_query→ the exact string sent to Amazon scraper
  - essentiality→ [0.0, 1.0] priority weight fed into calculate_raw_score()
  - _classify_intent() → called by main.py L184 for fast-path vs scenario routing
"""

import re
import logging
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ml_engine.models.intent_taxonomy import SCENARIO_TAXONOMY, TRAINING_DATA

logger = logging.getLogger(__name__)


class UnifiedIntentEngine:
    """
    Unified Single-Model Neural Engine combining Query Clarification and Intent Parsing using flan-t5-small.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._is_initialized = False

        # Fast taxonomy lookup patterns (compiled once at startup)
        self._taxonomy_patterns = []
        for key_str, components in SCENARIO_TAXONOMY.items():
            patterns = [re.compile(r'\b' + kw.strip().replace(' ', r'\s+') + r'\b', re.IGNORECASE)
                        for kw in key_str.split('|')]
            self._taxonomy_patterns.append((patterns, components))

        # Intent keywords for fast SCENARIO detection (no model needed)
        self._scenario_signals = {
            "want to go", "planning", "going on", "help me shop", "need gear",
            "preparing for", "setting up", "setting up a", "build a", "help me",
            "what do i need", "what should i buy", "need things for", "upgrade my",
            "trip", "vacation", "journey", "tour", "excursion", "adventure",
            "setup", "rig", "build", "renovation", "camping", "hiking", "skiing",
            "snowboarding", "backpacking", "travel", "trekking",
        }

        # PRODUCT fast-path keywords (strongly suggest a specific item)
        self._product_signals = {
            "iphone", "samsung", "galaxy", "sony", "pixel", "oneplus", "realme",
            "nvidia", "rtx", "gtx", "amd", "radeon", "intel", "amd ryzen",
            "macbook", "lenovo", "dell xps", "hp spectre",
            "best", "recommend", "under", "budget", "compare",
        }

    def _lazy_init(self):
        if self._is_initialized:
            return
        try:
            logger.info("Loading unified Flan-T5 neural engine (google/flan-t5-small)...")
            self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
            self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
            self.model.eval()
            self._is_initialized = True
            logger.info("Unified Flan-T5 neural engine loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Flan-T5 model: {e}")

    def _generate_text(self, prompt: str, max_length: int = 128) -> str:
        self._lazy_init()
        if not self.model or not self.tokenizer:
            return ""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=96,
                    num_beams=2,
                    early_stopping=True,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Flan-T5 generation error: {e}")
            return ""

    # ── 1. INTENT CLASSIFICATION ────────────────────────────────────────────────

    def _classify_intent(self, query: str) -> str:
        """
        Called by main.py to determine if query is SCENARIO or PRODUCT.
        Uses fast keyword heuristics before falling back to word-length check.

        Returns:
            "SCENARIO" → user described a situation/goal (checklist needed)
            "PRODUCT"  → user asked for a specific item (fast-path)
        """
        q = query.lower().strip()
        words = q.split()

        # 1. Check strong scenario signals
        for sig in self._scenario_signals:
            if sig in q:
                return "SCENARIO"

        # 2. Check taxonomy keyword match → always SCENARIO
        for patterns, _ in self._taxonomy_patterns:
            for p in patterns:
                if p.search(q):
                    return "SCENARIO"

        # 3. Check strong product signals
        for sig in self._product_signals:
            if sig in q:
                return "PRODUCT"

        # 4. Word count heuristic: long = SCENARIO, short = PRODUCT
        # e.g. "wireless headphones" (2 words) = PRODUCT
        #      "I want to go for a ski trip" (8 words) = SCENARIO
        if len(words) >= 5:
            return "SCENARIO"

        return "PRODUCT"

    # ── 2. UNIFIED QUERY CLARIFIER ──────────────────────────────────────────────

    def clarify(self, raw_input: str) -> Dict[str, Any]:
        """
        Runs neural text rewriting and intent classification via flan-t5-small.
        """
        if not raw_input or not raw_input.strip():
            return {
                "corrected_input": "",
                "understood_as": "Empty query",
                "formatted_prompt": "",
                "confidence": "low",
                "query_type": "AMBIGUOUS",
                "changes_made": [],
                "needs_confirmation": False,
            }

        raw = raw_input.strip()

        # Neural paraphrase & correction via Flan-T5
        rewrite_prompt = f"Fix spelling errors, normalize slang, and rewrite into a clear shopping prompt: {raw}"
        rewritten = self._generate_text(rewrite_prompt)
        if not rewritten or len(rewritten) < 3:
            rewritten = raw

        # Detect intent via the shared classifier
        intent = self._classify_intent(raw)
        query_type = intent  # "SCENARIO" or "PRODUCT"

        if query_type == "SCENARIO":
            formatted_prompt = (
                f"{rewritten[0].upper() + rewritten[1:]}. "
                "Please help me shop for all essential gear and items."
            ) if not rewritten.lower().startswith("i ") else rewritten
            understood_as = f'Shopping scenario: "{raw}"'
        else:
            formatted_prompt = f"Find me the best {rewritten} with good ratings and value for money."
            understood_as = f'Product search: "{raw}"'

        changes_made = []
        if raw.lower() != rewritten.lower():
            changes_made.append(f'"{raw}" → "{rewritten}" (neural rewrite)')

        return {
            "corrected_input": rewritten,
            "understood_as": understood_as,
            "formatted_prompt": formatted_prompt,
            "confidence": "high",
            "query_type": query_type,
            "changes_made": changes_made,
            "needs_confirmation": bool(changes_made) or query_type == "SCENARIO",
        }

    # ── 3. CHECKLIST EXTRAPOLATION ──────────────────────────────────────────────

    def extrapolate_checklist(self, prompt: str, exclude_items: list = None, num_items: int = 10) -> list:
        """
        Extrapolates product checklist from a scenario prompt.
        Uses taxonomy lookup first, falls back to Flan-T5 neural generation.

        Returns a list of dicts with:
          id, name, search_query, essentiality, category
        These keys are consumed by main.py's process_item() and passed to
        scoring_engine.calculate_raw_score(..., query=original_query).
        """
        exclude_set = {e.lower().strip() for e in (exclude_items or [])}
        clean_prompt = prompt.strip()

        # ── Fast taxonomy lookup ─────────────────────────────────────────────
        for patterns, components in self._taxonomy_patterns:
            for p in patterns:
                if p.search(clean_prompt):
                    filtered = [
                        c for c in components
                        if c.lower().strip() not in exclude_set
                    ]
                    return self._format_items(filtered[:num_items], intent_type="SCENARIO")

        # ── Flan-T5 neural generation ────────────────────────────────────────
        gen_prompt = (
            f"List {num_items} essential physical product items to buy for: {clean_prompt}. "
            "Give specific product names separated by commas."
        )
        generated = self._generate_text(gen_prompt, max_length=256)

        items: list[str] = []
        if generated:
            raw_splits = re.split(r'[,;\n]+|\d+\.', generated)
            for part in raw_splits:
                cleaned = part.strip().strip("•-").strip()
                if len(cleaned) > 2 and cleaned.lower() not in exclude_set:
                    items.append(cleaned)

        # Fallback if Flan-T5 generates fewer than 3 items
        if len(items) < 3:
            items = [
                f"{clean_prompt} Primary Unit",
                f"High-Grade {clean_prompt} Accessories",
                f"Essential {clean_prompt} Gear",
                f"Protective Carrying Case for {clean_prompt}",
                f"Premium Maintenance Kit for {clean_prompt}",
            ]

        return self._format_items(items[:num_items], intent_type="SCENARIO")

    def _format_items(self, components: list, intent_type: str) -> list:
        """
        Formats a list of product name strings into the dict schema consumed by:
          - process_item()          → uses 'id', 'name', 'search_query', 'essentiality'
          - calculate_raw_score()   → uses 'essentiality', passed 'query' from original prompt
          - UI stream events        → uses 'name', 'id', 'category'

        Essentiality is assigned in descending priority order:
          Item 1 → 1.0 (most essential), Item N → ~0.4 (least essential)
        This feeds into the LTR scoring engine as an additive boost (+3% per unit).
        """
        n = len(components)
        formatted = []
        for idx, comp in enumerate(components, 1):
            name = comp.strip()
            if not name:
                continue

            # Essentiality: linear decay from 1.0 → 0.4 across the list
            essentiality = round(1.0 - (idx - 1) / max(n, 1) * 0.6, 2)

            formatted.append({
                "id":           str(idx),
                "name":         name,           # human-readable label shown in UI
                "search_query": name,           # string sent to Amazon scraper
                "essentiality": essentiality,   # [0.4, 1.0] → fed into LTR score
                "category":     "Equipment" if intent_type == "SCENARIO" else "Products",
                "status":       "pending",
            })
        return formatted


# ── Singletons ────────────────────────────────────────────────────────────────
intent_parser = UnifiedIntentEngine()
# Legacy alias — some older imports still use query_clarifier
query_clarifier = intent_parser
