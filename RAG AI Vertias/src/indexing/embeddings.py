import hashlib
import numpy as np
from typing import List, Dict
from ..config import settings


class EmbeddingService:
    """Computes dense vector representations and sparse lexical token frequencies."""

    def __init__(self):
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.api_key = settings.OPENAI_API_KEY
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of document texts."""
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=texts,
                    dimensions=self.dimensions
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                # Log or fall back
                pass

        # Deterministic offline vector fallback for testing / local execution
        return [self._generate_deterministic_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query."""
        vectors = self.embed_documents([query])
        return vectors[0]

    def compute_sparse_vector(self, text: str) -> Dict[int, float]:
        """Computes basic sparse term frequencies mapped to integer indices for BM25 hybrid matching."""
        tokens = text.lower().split()
        sparse_vec: Dict[int, float] = {}
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:7], 16) % 100000
            sparse_vec[idx] = sparse_vec.get(idx, 0.0) + 1.0

        # Term frequency normalization
        total_tokens = len(tokens) or 1
        for k in sparse_vec:
            sparse_vec[k] = float(sparse_vec[k] / total_tokens)

        return sparse_vec

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """Generates a stable, normalized float vector from text hash for offline testing."""
        hasher = hashlib.sha256(text.encode("utf-8"))
        seed = int(hasher.hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimensions)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
