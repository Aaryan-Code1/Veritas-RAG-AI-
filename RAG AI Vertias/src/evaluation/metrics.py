from typing import List, Dict, Any
from ..models.schemas import Citation, ChunkMetadata


class EvaluationMetrics:
    """Calculates retrieval, generation, citation, and refusal metrics."""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], ground_truth_docs: List[str]) -> float:
        """Fraction of relevant documents retrieved in top-k."""
        if not ground_truth_docs:
            return 1.0
        matched = set(retrieved_docs).intersection(set(ground_truth_docs))
        return float(len(matched) / len(ground_truth_docs))

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], ground_truth_docs: List[str]) -> float:
        """Fraction of retrieved items that are relevant."""
        if not retrieved_docs:
            return 0.0
        matched = set(retrieved_docs).intersection(set(ground_truth_docs))
        return float(len(matched) / len(retrieved_docs))

    @staticmethod
    def mrr(retrieved_docs: List[str], ground_truth_docs: List[str]) -> float:
        """Mean Reciprocal Rank of the first relevant document."""
        if not ground_truth_docs:
            return 1.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in ground_truth_docs:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def citation_precision(citations: List[Citation]) -> float:
        """Ratio of citations where entailment was verified."""
        if not citations:
            return 1.0
        verified = sum(1 for c in citations if c.entailment_verified)
        return float(verified / len(citations))

    @staticmethod
    def refusal_accuracy(actual_refusal: bool, expected_refusal: bool) -> float:
        """Binary check whether the system correctly refused an unanswerable query."""
        return 1.0 if actual_refusal == expected_refusal else 0.0

    @staticmethod
    def faithfulness(answer: str, context_chunks: List[ChunkMetadata]) -> float:
        """
        Calculates lexical groundedness ratio of answer sentences
        supported by any of the context chunks.
        """
        if not answer or not context_chunks:
            return 1.0

        all_context = " ".join([c.text.lower() for c in context_chunks])
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        supported = 0
        for sent in sentences:
            words = set(sent.lower().split()) - {"according", "to", "verified", "enterprise", "documentation"}
            if not words:
                supported += 1
                continue
            overlap = len(words.intersection(set(all_context.split()))) / len(words)
            if overlap >= 0.25:
                supported += 1

        return float(supported / len(sentences))
