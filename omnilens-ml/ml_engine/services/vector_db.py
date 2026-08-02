import os
import faiss
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_FAISS_IDX_PATH = os.path.join(_DB_DIR, "products.index")
_METADATA_PATH = os.path.join(_DB_DIR, "products_meta.pkl")

class VectorDBService:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = None
        self.metadata = [] # List of dicts, parallel to FAISS index
        self._url_set = set() # For quick deduplication
        self._initialized = False
        self._model = None

    def _lazy_init(self):
        if self._initialized:
            return
        os.makedirs(_DB_DIR, exist_ok=True)
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        logger.info("[FAISS] Loading BGE-small-en-v1.5 for Vector DB...")
        self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Load FAISS index and metadata
        if os.path.exists(_FAISS_IDX_PATH) and os.path.exists(_METADATA_PATH):
            try:
                self.index = faiss.read_index(_FAISS_IDX_PATH)
                with open(_METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                self._url_set = {m.get("link") for m in self.metadata if m.get("link")}
                logger.info(f"[FAISS] Loaded {self.index.ntotal} products from disk.")
            except Exception as e:
                logger.error(f"[FAISS] Failed to load index: {e}")
                self._init_empty()
        else:
            self._init_empty()
            
        self._initialized = True

    def _init_empty(self):
        # IndexFlatIP = Inner Product (equivalent to Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._url_set = set()

    def _save(self):
        try:
            faiss.write_index(self.index, _FAISS_IDX_PATH)
            with open(_METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"[FAISS] Failed to save index: {e}")

    def upsert_products(self, products: List[Dict[str, Any]]):
        """Add new products to the FAISS index if they don't already exist."""
        self._lazy_init()
        if not products:
            return

        new_prods = []
        for p in products:
            if not p.get("title") or not p.get("link"):
                continue
            if p["link"] not in self._url_set:
                new_prods.append(p)
                self._url_set.add(p["link"])

        if not new_prods:
            return

        titles = [p["title"][:256] for p in new_prods]
        try:
            embeddings = self._model.encode(titles, normalize_embeddings=True)
            self.index.add(embeddings)
            self.metadata.extend(new_prods)
            self._save()
            logger.info(f"[FAISS] Upserted {len(new_prods)} new products. Total: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"[FAISS] Upsert failed: {e}")

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Retrieve top_k products matching the query."""
        self._lazy_init()
        if self.index.ntotal == 0 or not query:
            return []

        try:
            q_emb = self._model.encode([query], normalize_embeddings=True)
            distances, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    item = dict(self.metadata[idx])
                    # Store semantic similarity retrieved from FAISS
                    item["_faiss_score"] = float(dist)
                    results.append(item)
            return results
        except Exception as e:
            logger.error(f"[FAISS] Search failed: {e}")
            return []

vector_db = VectorDBService()
