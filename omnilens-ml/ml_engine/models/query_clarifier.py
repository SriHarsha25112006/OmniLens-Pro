"""
QueryClarifier (OmniLens Pro v2)
================================
A lightweight pre-processing model that sits between the user's raw input
and the main shopping pipeline. It handles:

1. Spell correction (difflib fuzzy match against a curated entity/word vocab)
2. Intent reformulation (scenario → structured prompt template)
3. Ambiguity detection (flags vague/incomplete queries)
4. Formatting (standardises the output into a clean, structured prompt
   ready for the OmniLens pipeline)

The clarified prompt is returned to the frontend for YES/NO confirmation
before the main search begins. If the user rejects, they retype and the
clarifier runs again.

Architecture Notes (HieraSpark context):
- This is a deterministic NLP pre-processor (no neural inference cost).
- It runs in <10ms, adding zero perceptible latency.
- The structured output it produces significantly reduces ambiguity noise
  passed into intent_parser + scraper, improving pipeline accuracy.
- Can be extended with a lightweight seq2seq rewriter (flan-t5-small)
  for full paraphrase correction if needed in future.
"""

import re
import difflib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Correction Lexicon ────────────────────────────────────────────────────────
# Comprehensive general English + shopping-domain vocabulary for fuzzy matching
CORRECTION_LEXICON = {
    # Activities / Scenarios
    "hiking", "trekking", "camping", "skiing", "snowboarding", "surfing", "swimming",
    "cycling", "running", "jogging", "yoga", "meditation", "gaming", "streaming",
    "cooking", "baking", "gardening", "painting", "photography", "fishing", "hunting",
    "traveling", "backpacking", "road trip", "vacation", "festival", "concert",
    "wedding", "party", "birthday", "office", "studying", "reading", "journaling",
    "workout", "gym", "pilates", "crossfit", "boxing", "dance", "marathon",

    # Tech & Electronics
    "laptop", "headphones", "earphones", "earbuds", "keyboard", "mouse", "monitor",
    "tablet", "charger", "cable", "router", "speaker", "camera", "drone",
    "smartwatch", "projector", "microphone", "webcam", "storage", "battery",
    "graphics", "processor", "motherboard", "memory", "cooling", "gaming",

    # Home & Lifestyle
    "furniture", "mattress", "pillow", "blanket", "kitchen", "cooking", "appliance",
    "vacuum", "blender", "toaster", "coffee", "refrigerator", "microwave",
    "curtains", "lighting", "decoration", "cleaning", "organizer", "storage",

    # Fashion & Sports
    "shoes", "sneakers", "boots", "sandals", "jacket", "hoodie", "shirt",
    "pants", "jeans", "dress", "socks", "gloves", "helmet", "goggles",
    "backpack", "bag", "wallet", "sunglasses", "watch", "jewelry",

    # Common misspelling targets
    "necessary", "essential", "equipment", "accessories", "supplies", "gear",
    "professional", "budget", "affordable", "premium", "wireless", "portable",
    "waterproof", "lightweight", "durable", "comfortable", "stylish",
    "beginner", "advanced", "outdoor", "indoor", "travel", "everyday",
    "recommend", "suggest", "find", "help", "want", "need", "looking",
    "something", "anything", "everything", "everything", "nothing",

    # Brands (common typo targets)
    "apple", "samsung", "sony", "google", "microsoft", "amazon", "nike",
    "adidas", "logitech", "bose", "jbl", "philips", "corsair", "razer",
}

# ── Scenario Signals → Formatted Intent Template ──────────────────────────────
SCENARIO_TEMPLATES = [
    # (trigger_patterns, formatted_template)
    (
        [r'\bski(ing)?\b', r'\bsnowboard(ing)?\b', r'\bski\s+trip\b'],
        "I want to go for a skiing / snowboarding trip, help me shop for all the necessary gear and clothing."
    ),
    (
        [r'\bhik(e|ing)\b', r'\btrek(king)?\b'],
        "I am planning a hiking / trekking trip. Help me find all the essential gear, clothing, and accessories."
    ),
    (
        [r'\bcamping\b'],
        "I want to go camping. Help me shop for essential camping gear including shelter, cooking, and safety items."
    ),
    (
        [r'\bhome\s+office\b', r'\bwork\s+from\s+home\b', r'\bwfh\b'],
        "I am setting up a home office. Help me find all the essential equipment: desk, chair, monitors, peripherals."
    ),
    (
        [r'\bhome\s+gym\b', r'\bgym\s+setup\b'],
        "I want to set up a home gym. Help me find all the essential workout equipment and accessories."
    ),
    (
        [r'\bbeach\b.*\bvacation\b', r'\bbeach\s+trip\b'],
        "I am going on a beach vacation. Help me shop for all the necessary items: swimwear, sunscreen, gear, and accessories."
    ),
    (
        [r'\bwedding\b'],
        "I am shopping for a wedding. Help me find all the essential items for the event."
    ),
    (
        [r'\bbaby\b.*\b(shower|room|nursery|essentials)\b'],
        "I am preparing for a new baby. Help me shop for all the essential baby items and accessories."
    ),
    (
        [r'\bgaming\s+setup\b', r'\bpc\s+build\b', r'\bgaming\s+room\b'],
        "I am building a gaming setup / PC. Help me find all the essential components and peripherals."
    ),
    (
        [r'\b(road\s+trip|travel|traveling|backpacking)\b'],
        "I am going on a trip / traveling. Help me shop for all the necessary travel gear and accessories."
    ),
    (
        [r'\b(yoga|pilates|meditation)\b'],
        "I want to start yoga / pilates / meditation. Help me find all the essential equipment and accessories."
    ),
    (
        [r'\b(photography|photographer)\b'],
        "I am getting into photography. Help me find the best cameras, lenses, and accessories."
    ),
    (
        [r'\b(cycling|biking|bicycle)\b'],
        "I want to go cycling. Help me find the best bikes, helmets, and cycling accessories."
    ),
    (
        [r'\b(marathon|running|jogging)\b'],
        "I am training for a marathon / running. Help me find the best running shoes, gear, and accessories."
    ),
    (
        [r'\b(cooking|baking|kitchen)\b.*\b(setup|tools|equipment|accessories)\b'],
        "I want to set up a cooking / baking kitchen. Help me find the best kitchen tools and equipment."
    ),
    (
        [r'\b(surfing|surf)\b'],
        "I want to go surfing. Help me find surfboards, wetsuits, and essential surfing accessories."
    ),
    (
        [r'\b(rock\s+climbing|bouldering|climbing)\b'],
        "I want to go rock climbing / bouldering. Help me find harnesses, shoes, chalk, and climbing accessories."
    ),
]

# ── Common Spelling Patterns for Non-Vocabulary Words ─────────────────────────
MANUAL_CORRECTIONS = {
    # Common misspellings
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "need to",
    "plz": "please",
    "pls": "please",
    "lmk": "let me know",
    "asap": "as soon as possible",
    "w/": "with",
    "w/o": "without",
    "bc": "because",
    "b/c": "because",
    "tbh": "to be honest",
    "idk": "I don't know",
    "idc": "I don't care",
    "imo": "in my opinion",
    "ngl": "not gonna lie",
    "smth": "something",
    "sth": "something",
    "nd": "and",
    "n": "and",
    "r": "are",
    "ur": "your",
    "u": "you",
    "hv": "have",
    "hav": "have",
    "thx": "thanks",
    "ty": "thank you",
    "btw": "by the way",
    "fyi": "for your information",
    "irl": "in real life",
    "imo": "in my opinion",
    "lol": "",  # remove
    "lmao": "",  # remove
    "omg": "",  # remove
    "omfg": "",  # remove
}


class QueryClarifier:
    """
    Lightweight query understanding and reformulation engine.
    
    Pipeline:
    1. apply_manual_corrections() → fix slang/abbreviations
    2. correct_spelling() → difflib fuzzy match against CORRECTION_LEXICON
    3. detect_scenario_template() → map to structured intent if recognizable
    4. format_clarified_prompt() → produce clean, formatted output

    Returns a ClarificationResult with:
    - corrected_input: spell-corrected version of raw input
    - understood_as: a human-readable summary of what was understood
    - formatted_prompt: the clean prompt to pass downstream
    - confidence: 'high' | 'medium' | 'low'
    - query_type: 'SCENARIO' | 'PRODUCT' | 'AMBIGUOUS'
    - changes_made: list of specific corrections/changes for transparency
    """

    def __init__(self):
        self._scenario_patterns = []
        for trigger_list, template in SCENARIO_TEMPLATES:
            compiled = [re.compile(p, re.IGNORECASE) for p in trigger_list]
            self._scenario_patterns.append((compiled, template))
        logger.info("QueryClarifier initialized.")

    def clarify(self, raw_input: str) -> dict:
        """
        Main entry point. Returns a full clarification result.
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
        changes_made = []

        # Step 1: Manual corrections (slang/abbreviations)
        corrected = self._apply_manual_corrections(raw, changes_made)

        # Step 2: Spell correction (difflib fuzzy match)
        corrected = self._correct_spelling(corrected, changes_made)

        # Step 3: Scenario template mapping
        template_match = self._detect_scenario_template(corrected)

        # Step 4: Build formatted prompt
        formatted_prompt, understood_as, confidence, query_type = self._build_formatted_prompt(
            corrected, raw, template_match, changes_made
        )

        needs_confirmation = bool(changes_made) or confidence in ("medium", "low") or template_match is not None

        return {
            "corrected_input": corrected,
            "understood_as": understood_as,
            "formatted_prompt": formatted_prompt,
            "confidence": confidence,
            "query_type": query_type,
            "changes_made": changes_made,
            "needs_confirmation": needs_confirmation,
        }

    # ── Step 1 ────────────────────────────────────────────────────────────────
    def _apply_manual_corrections(self, text: str, changes: list) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = word.lower().strip(".,!?;:")
            if clean in MANUAL_CORRECTIONS:
                replacement = MANUAL_CORRECTIONS[clean]
                if replacement:
                    result.append(replacement)
                    changes.append(f'"{word}" → "{replacement}"')
                else:
                    changes.append(f'Removed "{word}" (informal expression)')
                # skip appending word
            else:
                result.append(word)
        return " ".join(result).strip()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    def _correct_spelling(self, text: str, changes: list) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = re.sub(r"[^a-zA-Z]", "", word.lower())
            if len(clean) > 3 and clean not in CORRECTION_LEXICON:
                close = difflib.get_close_matches(clean, CORRECTION_LEXICON, n=1, cutoff=0.82)
                if close and close[0] != clean:
                    changes.append(f'"{word}" → "{close[0]}" (spelling)')
                    result.append(close[0])
                    continue
            result.append(word)
        return " ".join(result)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    def _detect_scenario_template(self, text: str) -> Optional[str]:
        for patterns, template in self._scenario_patterns:
            if any(p.search(text) for p in patterns):
                return template
        return None

    # ── Step 4 ────────────────────────────────────────────────────────────────
    def _build_formatted_prompt(
        self, corrected: str, raw: str, template: Optional[str], changes: list
    ) -> tuple[str, str, str, str]:
        """
        Returns: (formatted_prompt, understood_as, confidence, query_type)
        """
        # Case A: Matched a well-known scenario template → high confidence 
        if template:
            return (
                template,
                f'Shopping scenario: "{corrected}"',
                "high",
                "SCENARIO",
            )

        # Case B: Looks like a specific product query
        #   Heuristic: short (≤6 words), contains brand/product keyword, no action verbs
        words = corrected.lower().split()
        has_action = any(w in {"want", "need", "help", "going", "planning", "setting"} for w in words)
        is_short = len(words) <= 6
        has_product_word = any(w in CORRECTION_LEXICON for w in words)

        if is_short and not has_action and has_product_word:
            formatted = f"Find me the best {corrected} available in India with good reviews and value for money."
            return (
                formatted,
                f'Product search: "{corrected}"',
                "high" if not changes else "medium",
                "PRODUCT",
            )

        # Case C: Multi-word action sentence — treat as scenario
        if has_action and len(words) >= 4:
            # Standardize punctuation + capitalize
            clean = corrected.strip()
            if not clean.endswith("."):
                clean += "."
            # Ensure first letter capitalized
            clean = clean[0].upper() + clean[1:]
            formatted = (
                f"{clean} Please help me build a complete shopping list with all the "
                f"essential products, gear, and accessories needed."
            )
            return (
                formatted,
                f'Shopping scenario: "{corrected}"',
                "medium" if changes else "high",
                "SCENARIO",
            )

        # Case D: Ambiguous / unclear → low confidence
        clean = corrected.strip()
        if not clean.endswith("?") and not clean.endswith("."):
            clean += "."
        formatted = (
            f"Help me shop for: {clean} "
            f"Find the most relevant products with good ratings and value for money."
        )
        return (
            formatted,
            f'Shopping for: "{corrected}" (interpreted broadly)',
            "low",
            "AMBIGUOUS",
        )


# Singleton
query_clarifier = QueryClarifier()
