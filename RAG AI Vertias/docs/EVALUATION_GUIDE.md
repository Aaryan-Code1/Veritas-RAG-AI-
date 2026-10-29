# VeritasAI Evaluation & Benchmarking Guide

## 1. Overview
In VeritasAI, evaluation is treated as a continuous software engineering discipline. Every change to chunking, embeddings, prompt templates, or rerankers must pass regression gates before merging into the main branch.

## 2. Evaluation Dataset Format
Golden datasets are stored in `src/evaluation/golden_dataset.json` following this schema:

```json
{
  "eval_id": "eval_001",
  "query_text": "What is our customer data retention period under GDPR?",
  "ground_truth_answer": "Customer PII must be permanently deleted within 30 days of a verified request.",
  "ground_truth_chunk_ids": ["chk_sec_01", "chk_sec_02"],
  "ground_truth_doc_titles": ["Enterprise Global Privacy Policy 2025"],
  "query_type": "single_hop",
  "expected_refusal": false,
  "required_acl_groups": ["group:everyone"]
}
```

## 3. Metrics Computed

| Metric | Threshold | Method | Description |
| :--- | :--- | :--- | :--- |
| **Recall@7** | $\ge 92.0\%$ | Exact ID / Title Match | Percentage of ground truth chunks retrieved in top 7 |
| **Precision@7** | $\ge 88.0\%$ | Match Ratio | Ratio of retrieved chunks that are relevant |
| **Faithfulness** | $\ge 98.0\%$ | LLM Judge / NLI | Ratio of answer claims supported by retrieved context |
| **Citation Precision** | $\ge 95.0\%$ | NLI Entailment | Ratio of inline citations that entail the asserted claim |
| **Refusal Accuracy** | $\ge 96.0\%$ | Exact Classifier | Correct refusal rate on unanswerable and out-of-domain queries |

## 4. Running Evaluations Locally

```bash
# Run automated evaluations against golden dataset
python scripts/run_evals.py --dataset src/evaluation/golden_dataset.json --sample-size 50
```
