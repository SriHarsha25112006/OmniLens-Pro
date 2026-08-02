"""
OmniLens Unified Neural Engine (flan-t5-small Core)
===================================================
Combines Query Clarification, Typo Rewriting, Intent Parsing,
and Checklist Extrapolation into a single unified Neural Pipeline powered by google/flan-t5-small.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ml_engine.models.intent_taxonomy import SCENARIO_TAXONOMY

logger = logging.getLogger(__name__)


class UnifiedIntentEngine:
    """
    Unified Single-Model Neural Engine combining Query Clarification and Intent Parsing using flan-t5-small.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._is_initialized = False

        # Fast taxonomy lookup patterns
        self._taxonomy_patterns = []
        for key_str, components in SCENARIO_TAXONOMY.items():
            patterns = [re.compile(r'\b' + kw.strip().replace(' ', r'\s+') + r'\b', re.IGNORECASE)
                        for kw in key_str.split('|')]
            self._taxonomy_patterns.append((patterns, components))

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
                outputs = self.model.generate(inputs.input_ids, max_new_tokens=64, num_beams=2, early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Flan-T5 generation error: {e}")
            return ""

    # ── 1. UNIFIED QUERY CLARIFIER ──────────────────────────────────────────

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

        # Generate neural paraphrase & correction via Flan-T5
        rewrite_prompt = f"Fix spelling errors, normalize slang, and rewrite into a clear shopping prompt: {raw}"
        rewritten = self._generate_text(rewrite_prompt)
        if not rewritten or len(rewritten) < 3:
            rewritten = raw

        # Detect intent type (SCENARIO vs PRODUCT)
        words = raw.lower().split()
        is_scenario = len(words) >= 4 or any(w in raw.lower() for w in ["want", "need", "trip", "build", "setup", "vacation", "hiking", "skiing", "gaming"])
        query_type = "SCENARIO" if is_scenario else "PRODUCT"

        if query_type == "SCENARIO":
            formatted_prompt = f"{rewritten[0].upper() + rewritten[1:]}. Please help me shop for all essential gear and items." if not rewritten.lower().startswith("i ") else rewritten
            understood_as = f'Shopping scenario: "{raw}"'
        else:
            formatted_prompt = f"Find me the best {rewritten} with good ratings and value for money."
            understood_as = f'Product search: "{raw}"'

        changes_made = []
        if raw.lower() != rewritten.lower():
            changes_made.append(f'"{raw}" → "{rewritten}" (neural inference)')

        return {
            "corrected_input": rewritten,
            "understood_as": understood_as,
            "formatted_prompt": formatted_prompt,
            "confidence": "high",
            "query_type": query_type,
            "changes_made": changes_made,
            "needs_confirmation": bool(changes_made) or query_type == "SCENARIO",
        }

    # ── 2. UNIFIED CHECKLIST EXTRAPOLATION ─────────────────────────────────

    def extrapolate_checklist(self, prompt: str, exclude_items: list = None, num_items: int = 10) -> list:
        """
        Extrapolates 10 product items using taxonomy lookup or flan-t5 neural generation.
        """
        exclude_items = set(exclude_items or [])
        clean_prompt = prompt.strip()

        # Fast taxonomy check
        for patterns, components in self._taxonomy_patterns:
            for p in patterns:
                if p.search(clean_prompt):
                    filtered = [c for c in components if c.lower() not in exclude_items]
                    return self._format_items(filtered[:num_items], intent_type="SCENARIO")

        # Dynamic neural generation using Flan-T5
        gen_prompt = f"List 10 essential physical product items to buy for: {clean_prompt}"
        generated = self._generate_text(gen_prompt, max_length=128)

        # Parse generated comma/newline separated items
        items = []
        if generated:
            raw_splits = re.split(r'[,;\n\d+\.]+', generated)
            for part in raw_splits:
                cleaned = part.strip()
                if len(cleaned) > 2 and cleaned.lower() not in exclude_items:
                    items.append(cleaned)

        # Fallback if generation produces fewer items
        if len(items) < 3:
            items = [
                f"{clean_prompt} Primary Unit",
                f"High-Grade {clean_prompt} Accessories",
                f"Essential {clean_prompt} Gear",
                f"Protective Carrying Case for {clean_prompt}",
                f"Premium Maintenance Kit for {clean_prompt}"
            ]

        return self._format_items(items[:num_items], intent_type="SCENARIO")

    def _format_items(self, components: list, intent_type: str) -> list:
        formatted = []
        for idx, comp in enumerate(components, 1):
            name = comp.strip()
            if not name:
                continue
            formatted.append({
                "id": str(idx),
                "query": name,
                "category": "Equipment" if intent_type == "SCENARIO" else "Products",
                "status": "pending"
            })
        return formatted


# Unified Singleton Instance
intent_parser = UnifiedIntentEngine()
# Expose query_clarifier alias pointing to the unified engine
query_clarifier = intent_parser
