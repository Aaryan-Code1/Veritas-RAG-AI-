import time
from typing import List, Tuple, Dict, Any, Optional
from ..config import settings
from ..models.schemas import ChunkMetadata, QueryRequest
from ..indexing.embeddings import EmbeddingService
from ..indexing.qdrant_client import QdrantStore
from .reranker import RerankerService


class QueryEngine:
    """End-to-end retrieval coordinator handling query rewriting, search, reranking, and assembly."""

    def __init__(self, qdrant_store: Optional[QdrantStore] = None):
        self.embedding_service = EmbeddingService()
        self.store = qdrant_store or QdrantStore()
        self.reranker = RerankerService()

    def rewrite_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Condenses conversational context or cleans punctuation for search.
        In Phase 1, strips filler words; extensible to multi-turn agent rewrite.
        """
        clean_query = query.strip()
        # Remove trailing question marks or pleasantries if needed
        return clean_query

    def retrieve_context(
        self, request: QueryRequest
    ) -> Tuple[List[ChunkMetadata], bool, Dict[str, int]]:
        """
        Executes candidate retrieval, reranking, and diversity filtering.
        Returns (selected_chunks, is_confident, latency_metrics).
        """
        metrics = {
            "retrieval_latency_ms": 0,
            "rerank_latency_ms": 0,
            "candidate_count": 0,
        }

        # Step 1: Query embedding
        t0 = time.time()
        rewritten_q = self.rewrite_query(request.query, request.session_id)
        query_vec = self.embedding_service.embed_query(rewritten_q)

        # Step 2: Dense + RBAC candidate search
        candidates = self.store.search_candidates(
            query_vector=query_vec,
            user_groups=request.user_groups,
            limit=settings.TOP_K_RETRIEVAL,
        )
        t1 = time.time()
        metrics["retrieval_latency_ms"] = int((t1 - t0) * 1000)
        metrics["candidate_count"] = len(candidates)

        if not candidates:
            return [], False, metrics

        # Step 3: Neural Cross-Encoder Reranking & Threshold Gate
        t2 = time.time()
        reranked_chunks, has_confidence = self.reranker.rerank(
            query=rewritten_q,
            candidates=candidates,
            top_n=settings.FINAL_CONTEXT_CHUNKS,
        )
        t3 = time.time()
        metrics["rerank_latency_ms"] = int((t3 - t2) * 1000)

        if not has_confidence or not reranked_chunks:
            return [], False, metrics

        # Step 4: Apply MMR / Deduplication
        final_chunks = self._apply_mmr_diversity(reranked_chunks, target_k=settings.FINAL_CONTEXT_CHUNKS)

        return final_chunks, True, metrics

    def format_context_for_prompt(self, chunks: List[ChunkMetadata]) -> str:
        """Assembles chunks into structured XML context blocks with 1-based indexing."""
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            block = (
                f'<context_source id="{i}" '
                f'doc_title="{chunk.document_title}" '
                f'page="{chunk.page_number or 1}" '
                f'section="{chunk.section_name or "General"}" '
                f'url="{chunk.source_url}">\n'
                f"{chunk.text}\n"
                f"</context_source>"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _apply_mmr_diversity(
        self, chunks: List[ChunkMetadata], target_k: int
    ) -> List[ChunkMetadata]:
        """Simple text diversity filter to prevent identical chunks from dominating context."""
        selected: List[ChunkMetadata] = []
        seen_docs = set()

        # Prioritize top chunk
        if chunks:
            selected.append(chunks[0])
            seen_docs.add((chunks[0].document_id, chunks[0].section_name))

        for chunk in chunks[1:]:
            key = (chunk.document_id, chunk.section_name)
            # Allow multiple chunks from same doc only if we have budget
            if key not in seen_docs or len(selected) < 3:
                selected.append(chunk)
                seen_docs.add(key)

            if len(selected) >= target_k:
                break

        return selected
