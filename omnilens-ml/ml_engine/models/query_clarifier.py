"""
QueryClarifier (OmniLens Pro v3 - Deep Learning Edition)
=========================================================
Neural Pre-Processing Engine powered by PyTorch, HuggingFace Transformers,
Vector Embedding Cosine Similarity, and Seq2Seq Generative Rewriting.

Replaces static regex lists and pre-defined dictionaries with:
1. Neural Text Normalization & Paraphrase Formatting via Flan-T5 Seq2Seq model.
2. Vector Embedding Cosine Similarity Classification for intent & archetype mapping.
3. Dynamic Character Sequence Alignment for transparent correction tracking.
"""

import re
import difflib
import logging
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Core Archetype Vectors for Cosine Similarity Matching
ARCHETYPE_DESCRIPTIONS = {
    "SCENARIO_TRIP": "outdoor activity adventure trip travel sports skiing hiking camping beach vacation",
    "SCENARIO_SETUP": "tech build workstation gaming PC office setup home gym room setup computer",
    "SCENARIO_HOME": "home furniture kitchen appliances interior decoration living space cleaning lifestyle",
    "SCENARIO_FASHION": "fashion apparel footwear clothing outfit accessories dress shoes jacket style",
    "PRODUCT_DIRECT": "specific single electronic product device gadget item purchase brand name"
}


class NeuralQueryClarifier:
    """
    Deep Learning powered query understanding and reformulation engine.
    Uses vector embedding cosine similarity and neural sequence-to-sequence generation.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._archetype_embeddings = {}
        self._is_initialized = False

    def _lazy_init(self):
        """Lazy load PyTorch transformer model and compute archetype embeddings."""
        if self._is_initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_name = "google/flan-t5-small"
            logger.info(f"Loading DL QueryClarifier model: {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._model.eval()

            # Pre-compute vector embeddings for archetype descriptions using model encoder
            with torch.no_grad():
                for key, desc in ARCHETYPE_DESCRIPTIONS.items():
                    inputs = self._tokenizer(desc, return_tensors="pt", truncation=True, max_length=64)
                    encoder_outputs = self._model.encoder(**inputs)
                    # Mean pooling over hidden states to get 512-dim embedding vector
                    embedding = encoder_outputs.last_hidden_state.mean(dim=1)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    self._archetype_embeddings[key] = embedding

            self._is_initialized = True
            logger.info("DL QueryClarifier model & archetype vector embeddings loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DL QueryClarifier model: {e}")

    def _get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Compute 512-dim normalized vector embedding for an input string."""
        if not self._is_initialized or not self._model:
            return None
        try:
            with torch.no_grad():
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                encoder_outputs = self._model.encoder(**inputs)
                embedding = encoder_outputs.last_hidden_state.mean(dim=1)
                return F.normalize(embedding, p=2, dim=1)
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None

    def _compute_cosine_similarities(self, text_embedding: torch.Tensor) -> Dict[str, float]:
        """Compute Cosine Similarity scores between query embedding and core archetypes."""
        similarities = {}
        if text_embedding is None:
            return similarities

        with torch.no_grad():
            for key, arch_emb in self._archetype_embeddings.items():
                # Cosine Similarity = (u · v) / (||u|| ||v||)
                cos_sim = F.cosine_similarity(text_embedding, arch_emb).item()
                similarities[key] = max(0.0, cos_sim)
        return similarities

    def _generate_neural_rewrite(self, raw_input: str) -> str:
        """Use Flan-T5 Seq2Seq model to fix spelling, normalize slang, and reformat text."""
        if not self._is_initialized or not self._model:
            return raw_input

        try:
            # Neural prompt instruction for end-to-end text correction and presentable formatting
            prompt = (
                f"Fix spelling errors, normalize slang, and rewrite into a clear presentable shopping prompt: {raw_input}"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs.input_ids,
                    max_new_tokens=64,
                    num_beams=2,
                    early_stopping=True
                )
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if decoded and len(decoded) > 5:
                decoded = decoded[0].upper() + decoded[1:]
                return decoded
        except Exception as e:
            logger.error(f"Neural text rewrite error: {e}")
        return raw_input

    def _extract_changes(self, raw: str, rewritten: str) -> List[str]:
        """Detect modified words between raw input and neural output for user transparency."""
        raw_words = re.findall(r'\w+', raw.lower())
        rewritten_words = re.findall(r'\w+', rewritten.lower())
        changes = []

        if not raw_words or not rewritten_words:
            return changes

        for rw in raw_words:
            if len(rw) >= 3 and rw not in rewritten_words:
                best_match = None
                best_sim = 0.5
                for n_w in rewritten_words:
                    if len(n_w) >= 3:
                        sim = difflib.SequenceMatcher(None, rw, n_w).ratio()
                        if sim > best_sim:
                            best_sim = sim
                            best_match = n_w
                if best_match:
                    changes.append(f'"{rw}" → "{best_match}" (contextual inference)')
        return changes

    def clarify(self, raw_input: str) -> Dict[str, Any]:
        """
        Main entry point called by FastAPI /api/clarify_query.
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
        self._lazy_init()

        # Step 1: Neural Seq2Seq Rewriting & Spelling Normalization
        neural_rewritten = self._generate_neural_rewrite(raw)
        changes_made = self._extract_changes(raw, neural_rewritten)

        # Step 2: Vector Embedding Extraction
        query_emb = self._get_embedding(neural_rewritten)

        # Step 3: Cosine Similarity Vector Matching
        sim_scores = self._compute_cosine_similarities(query_emb)

        # Determine highest scoring archetype via Cosine Similarity
        best_archetype = "PRODUCT_DIRECT"
        max_sim = 0.0
        if sim_scores:
            best_archetype, max_sim = max(sim_scores.items(), key=lambda x: x[1])

        # Step 4: Map Cosine Similarity & Embedding to Structured Output
        if max_sim >= 0.65:
            confidence = "high"
        elif max_sim >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        is_scenario = "SCENARIO" in best_archetype or len(raw.split()) >= 4 or "want" in raw.lower() or "trip" in raw.lower() or "build" in raw.lower()
        query_type = "SCENARIO" if is_scenario else "PRODUCT"

        # Format final presentable prompt
        if query_type == "SCENARIO":
            if not neural_rewritten.lower().startswith("i "):
                formatted_prompt = f"{neural_rewritten} Please help me shop for all essential gear, items, and accessories."
            else:
                formatted_prompt = neural_rewritten
            understood_as = f'Shopping scenario: "{raw}" (contextual inference)'
        else:
            formatted_prompt = f"Find me the best {neural_rewritten} with good ratings and value for money."
            understood_as = f'Product search: "{raw}" (contextual inference)'

        needs_confirmation = bool(changes_made) or confidence in ("medium", "low") or query_type == "SCENARIO"

        return {
            "corrected_input": neural_rewritten,
            "understood_as": understood_as,
            "formatted_prompt": formatted_prompt,
            "confidence": confidence,
            "query_type": query_type,
            "changes_made": changes_made,
            "needs_confirmation": needs_confirmation,
        }


# Singleton Instance
query_clarifier = NeuralQueryClarifier()
