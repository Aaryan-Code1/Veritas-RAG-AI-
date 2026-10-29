import json
import os
import uuid
from typing import List, Dict, Any, Optional
from ..models.schemas import EvalItem, EvalRunResponse, QueryRequest
from ..retrieval.query_engine import QueryEngine
from ..generation.generator import LLMGenerator
from .metrics import EvaluationMetrics


class EvaluationRunner:
    """Executes automated benchmark evaluation sweeps against golden datasets."""

    def __init__(self, query_engine: Optional[QueryEngine] = None, generator: Optional[LLMGenerator] = None):
        self.query_engine = query_engine or QueryEngine()
        self.generator = generator or LLMGenerator()

    def load_dataset(self, filepath: Optional[str] = None) -> List[EvalItem]:
        """Loads golden evaluation items from JSON file."""
        if not filepath:
            filepath = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [EvalItem(**item) for item in data]

    def run_evaluations(
        self, items: Optional[List[EvalItem]] = None, sample_size: Optional[int] = None
    ) -> EvalRunResponse:
        """Executes full benchmark evaluation across the golden dataset."""
        if items is None:
            items = self.load_dataset()

        if sample_size and sample_size < len(items):
            items = items[:sample_size]

        recall_scores = []
        precision_scores = []
        mrr_scores = []
        faithfulness_scores = []
        citation_precision_scores = []
        refusal_scores = []

        for item in items:
            req = QueryRequest(
                query=item.query_text,
                user_groups=item.required_acl_groups,
            )

            # Retrieve
            chunks, is_confident, r_metrics = self.query_engine.retrieve_context(req)
            retrieved_titles = [c.document_title for c in chunks]

            # Retrieval metrics
            recall_scores.append(
                EvaluationMetrics.recall_at_k(retrieved_titles, item.ground_truth_doc_titles)
            )
            precision_scores.append(
                EvaluationMetrics.precision_at_k(retrieved_titles, item.ground_truth_doc_titles)
            )
            mrr_scores.append(
                EvaluationMetrics.mrr(retrieved_titles, item.ground_truth_doc_titles)
            )

            # Generate
            formatted_ctx = self.query_engine.format_context_for_prompt(chunks)
            res = self.generator.generate_grounded_answer(req, chunks, formatted_ctx, r_metrics)

            # Generation & Refusal metrics
            refusal_scores.append(
                EvaluationMetrics.refusal_accuracy(res.refusal, item.expected_refusal)
            )
            faithfulness_scores.append(
                EvaluationMetrics.faithfulness(res.answer, chunks)
            )
            citation_precision_scores.append(
                EvaluationMetrics.citation_precision(res.citations)
            )

        n = len(items) or 1
        summary_metrics = {
            "mean_recall_at_k": round(sum(recall_scores) / n, 4),
            "mean_precision_at_k": round(sum(precision_scores) / n, 4),
            "mean_mrr": round(sum(mrr_scores) / n, 4),
            "mean_faithfulness": round(sum(faithfulness_scores) / n, 4),
            "mean_citation_precision": round(sum(citation_precision_scores) / n, 4),
            "mean_refusal_accuracy": round(sum(refusal_scores) / n, 4),
        }

        # Quality Gates Check
        passed = (
            summary_metrics["mean_faithfulness"] >= 0.90
            and summary_metrics["mean_refusal_accuracy"] >= 0.90
            and summary_metrics["mean_citation_precision"] >= 0.90
        )

        return EvalRunResponse(
            eval_run_id=f"eval_{uuid.uuid4()}",
            total_queries=len(items),
            metrics=summary_metrics,
            pass_fail_status="PASSED" if passed else "FAILED",
        )
