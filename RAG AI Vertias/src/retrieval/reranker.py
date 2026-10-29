from typing import List, Tuple
from ..config import settings
from ..models.schemas import ChunkMetadata


class RerankerService:
    """Neural cross-encoder reranking service with confidence threshold filtering."""

    def __init__(self):
        self.api_key = settings.COHERE_API_KEY
        self.model = settings.COHERE_RERANK_MODEL
        self.threshold = settings.RERANKER_CONFIDENCE_THRESHOLD
        self.client = None

        if self.api_key:
            try:
                import cohere
                self.client = cohere.ClientV2(api_key=self.api_key)
            except Exception:
                self.client = None

    def rerank(
        self, query: str, candidates: List[ChunkMetadata], top_n: int = 7
    ) -> Tuple[List[ChunkMetadata], bool]:
        """
        Reranks candidate chunks against the user query.
        Returns (reranked_chunks, has_sufficient_confidence).
        If all candidates fail the threshold (< 0.45), has_sufficient_confidence is False.
        """
        if not candidates:
            return [], False

        # If Cohere client is active
        if self.client:
            try:
                docs = [c.text for c in candidates]
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=min(top_n, len(candidates)),
                )

                reranked_chunks = []
                for result in response.results:
                    chunk = candidates[result.index]
                    chunk.rerank_score = float(result.relevance_score)
                    if chunk.rerank_score >= self.threshold:
                        reranked_chunks.append(chunk)

                if not reranked_chunks:
                    return [], False

                return reranked_chunks[:top_n], True
            except Exception:
                pass

        # Fallback local lexical-semantic scoring when running offline or without API key
        scored_candidates = []
        query_words = set(query.lower().split())

        for chunk in candidates:
            chunk_words = set(chunk.text.lower().split())
            intersection = query_words.intersection(chunk_words)
            # Simple Jaccard + existing vector score approximation
            lexical_sim = len(intersection) / (len(query_words) or 1)
            vector_sim = chunk.rerank_score if chunk.rerank_score is not None else 0.5
            combined_score = float(0.4 * lexical_sim + 0.6 * vector_sim)

            chunk.rerank_score = combined_score
            if combined_score >= self.threshold:
                scored_candidates.append(chunk)

        scored_candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        if not scored_candidates:
            return [], False

        return scored_candidates[:top_n], True
