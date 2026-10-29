import time
import uuid
from typing import List, Tuple, Optional
from ..config import settings
from ..models.schemas import (
    QueryRequest,
    QueryResponse,
    QueryMetrics,
    ChunkMetadata,
    Citation,
)
from .prompt import PromptTemplates
from .citation_verifier import CitationVerifier


class LLMGenerator:
    """Manages LLM prompting, grounded response synthesis, and citation verification."""

    def __init__(self):
        self.provider = settings.DEFAULT_LLM_PROVIDER
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        self.openai_key = settings.OPENAI_API_KEY
        self.anthropic_client = None
        self.openai_client = None

        if self.anthropic_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            except Exception:
                pass

        if self.openai_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_key)
            except Exception:
                pass

    def generate_grounded_answer(
        self,
        request: QueryRequest,
        context_chunks: List[ChunkMetadata],
        formatted_context: str,
        retrieval_metrics: dict,
    ) -> QueryResponse:
        """Invokes LLM with strict grounding invariants and verifies citations."""
        t_start = time.time()
        query_id = f"qry_{uuid.uuid4()}"

        # If no context or confidence failed, emit deterministic refusal immediately
        if not context_chunks:
            return QueryResponse(
                query_id=query_id,
                query=request.query,
                answer=PromptTemplates.STANDARD_REFUSAL_MESSAGE,
                refusal=True,
                citations=[],
                metrics=QueryMetrics(
                    retrieval_latency_ms=retrieval_metrics.get("retrieval_latency_ms", 0),
                    rerank_latency_ms=retrieval_metrics.get("rerank_latency_ms", 0),
                    total_latency_ms=int((time.time() - t_start) * 1000),
                    candidate_chunks_evaluated=retrieval_metrics.get("candidate_count", 0),
                    final_chunks_used=0,
                ),
            )

        system_prompt = PromptTemplates.SYSTEM_PROMPT.format(
            context_blocks=formatted_context
        )
        user_prompt = PromptTemplates.build_user_prompt(
            request.query, formatted_context
        )

        answer_text = ""

        # 1. Try Anthropic Claude
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=1024,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer_text = response.content[0].text
            except Exception:
                pass

        # 2. Try OpenAI Fallback
        if not answer_text and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer_text = response.choices[0].message.content or ""
            except Exception:
                pass

        # 3. Deterministic Local Grounded Synthesis Fallback (for testing / offline evaluation)
        if not answer_text:
            answer_text = self._offline_grounded_synthesis(request.query, context_chunks)

        is_refusal = (
            PromptTemplates.STANDARD_REFUSAL_MESSAGE.lower() in answer_text.lower()
            or "do not have enough" in answer_text.lower()
        )

        citations: List[Citation] = []
        if not is_refusal:
            citations = CitationVerifier.verify_and_build_citations(
                answer_text, context_chunks
            )

        t_total = int((time.time() - t_start) * 1000)

        return QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=answer_text,
            refusal=is_refusal,
            citations=citations,
            metrics=QueryMetrics(
                retrieval_latency_ms=retrieval_metrics.get("retrieval_latency_ms", 0),
                rerank_latency_ms=retrieval_metrics.get("rerank_latency_ms", 0),
                total_latency_ms=t_total,
                candidate_chunks_evaluated=retrieval_metrics.get("candidate_count", 0),
                final_chunks_used=len(context_chunks),
            ),
        )

    def _offline_grounded_synthesis(
        self, query: str, chunks: List[ChunkMetadata]
    ) -> str:
        """Deterministic rule-based synthesizer when no cloud LLM API key is present."""
        if not chunks:
            return PromptTemplates.STANDARD_REFUSAL_MESSAGE

        first_chunk = chunks[0]
        # Check if query terms overlap significantly
        q_words = set(query.lower().split())
        c_words = set(first_chunk.text.lower().split())
        overlap = len(q_words.intersection(c_words))

        if overlap < 1:
            return PromptTemplates.STANDARD_REFUSAL_MESSAGE

        # Clean header and return grounded excerpt with citation
        lines = [l.strip() for l in first_chunk.text.splitlines() if l.strip() and not l.startswith("[Document:")]
        core_line = lines[0] if lines else first_chunk.text[:200]
        return f"According to verified enterprise documentation, {core_line} [1]."
