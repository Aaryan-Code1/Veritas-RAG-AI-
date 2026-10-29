<div align="center">

# 🛡️ VeritasAI: Enterprise RAG & Compliance Intelligence Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%201.12-red.svg?logo=qdrant)](https://qdrant.tech)
[![CI/CD Eval Gates](https://img.shields.io/badge/Evals-Ragas%20%7C%20DeepEval-orange.svg)](docs/EVALUATION_GUIDE.md)

**A high-precision, hallucination-resistant Retrieval-Augmented Generation (RAG) platform with auditable inline citations, zero-leakage RBAC filtering, and an automated continuous evaluation framework.**

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [Key Features](#-key-features) • [API Reference](#-api-reference) • [Evaluation Suite](#-evaluation-framework) • [Documentation](docs/PRD.md)

</div>

---

## 📖 Overview

Enterprise teams (Security, Legal, Engineering, Compliance) face severe bottlenecks searching through fragmented, contradictory documentation. Generic conversational LLMs pose major business risks: they fabricate plausible-sounding falsehoods, leak restricted payroll or executive data across team boundaries, and cannot supply verifiable page-level audit trails.

**VeritasAI** solves this through a zero-hallucination operational contract:
1. **Hybrid Retrieval with RBAC:** Merges dense vector embeddings (`text-embedding-3-large`) with sparse BM25 lexical search via Reciprocal Rank Fusion (RRF), executing strict security pre-filtering inside the vector index.
2. **Neural Cross-Encoder Reranking:** Filters candidate chunks using Cohere Rerank v3.5, triggering an immediate deterministic refusal if relevance falls below calibrated thresholds ($\text{Score} < 0.45$).
3. **Auditable Claim-Level Citations:** Answers include clickable inline markers (`[1]`, `[2]`) linked to verified document snippets, page numbers, and deep URLs.
4. **Post-Generation NLI Verification:** Validates generated claims against retrieved context using Natural Language Inference (NLI) before final delivery.
5. **Continuous Evaluation Suite:** Rigorous CI/CD regression gates benchmark Faithfulness, Context Recall, Citation Precision, and Refusal Accuracy.

---

## 🏛️ Architecture

```
                          [ User Query / API Client ]
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   FastAPI Gateway Router  │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │ Ingestion Pipeline    │             │ Query Pipeline        │
        │ - Docling / Parser    │             │ - Query Rewriter      │
        │ - Structure Chunker   │             │ - Hybrid Search (RRF) │
        │ - PII Sanitizer       │             │ - Cohere Reranker     │
        │ - Vector Embeddings   │             │ - MMR Context Window  │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    ▼                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 Qdrant Vector Database                      │
        │   - Dense HNSW (1536d) + Sparse BM25 + RBAC Payload Filter  │
        └─────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │ Grounded Generator    │
                                              │ - Claude 3.5 Sonnet   │
                                              │ - Inline Citations    │
                                              └───────────┬───────────┘
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │ NLI Citation Verifier │
                                              │ - Entailment Check    │
                                              └───────────────────────┘
```

---

## ✨ Key Features

- **🛡️ Strict Groundedness & Refusal:** Instructed to synthesize answers *only* from context. Declines with a standard refusal when context is missing, preventing guesswork.
- **📑 Verifiable Inline Citations:** Every factual assertion is tagged with `[n]` referencing page number, document version, and verbatim text.
- **🔐 Zero-Leakage Access Control (RBAC):** In-index filtering guarantees users never retrieve content from repositories their credentials cannot view.
- **⚡ High-Throughput Hybrid Search:** Dense semantic search + BM25 keyword matching fused using Reciprocal Rank Fusion ($k=60$).
- **🎯 Neural Reranking:** Filters noise and eliminates irrelevant chunks via cross-attention scoring.
- **📊 Continuous Evaluation Framework:** Integrated benchmarking scripts measuring Faithfulness, Recall, Precision, and Latency against golden datasets.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- API Keys: Anthropic / OpenAI, Cohere

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/veritas-ai-rag.git
cd veritas-ai-rag

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY, COHERE_API_KEY, and OPENAI_API_KEY
```

### 3. Start Infrastructure (Qdrant & Redis)

```bash
docker-compose up -d qdrant redis
```

### 4. Seed Sample Enterprise Documents

```bash
python scripts/seed_sample_docs.py
```

### 5. Launch the FastAPI Application

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at: **http://localhost:8000/docs**

---

## 📡 API Reference

### 1. Submit Query (`POST /api/v1/query`)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is our data retention period under GDPR and how does S3 deletion execute?",
       "user_groups": ["group:everyone", "group:sec-ops"],
       "stream": false
     }'
```

#### Sample Response with Grounded Citations
```json
{
  "query_id": "qry_88e3c1b2-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "answer": "Under our Global Privacy Policy, customer PII must be permanently deleted within 30 days of a verified GDPR erasure request [1]. In AWS S3, deletion is executed via automated lifecycle rules that transition flagged objects to Glacier Flexible Recovery before permanent expiration [2].",
  "refusal": false,
  "citations": [
    {
      "citation_id": 1,
      "document_title": "Enterprise Global Privacy Policy 2025",
      "page_number": 14,
      "section_name": "Section 4.1: GDPR Data Subject Rights",
      "verbatim_snippet": "All verified requests for customer erasure under GDPR Article 17 must execute full data purge across active datastores within 30 calendar days.",
      "source_url": "https://wiki.enterprise.internal/pages/viewpage.action?pageId=9843210",
      "entailment_verified": true
    },
    {
      "citation_id": 2,
      "document_title": "S3 Storage Lifecycle Architecture Spec",
      "page_number": 6,
      "section_name": "S3 Lifecycle Configuration",
      "verbatim_snippet": "Lifecycle policies tag customer-deleted objects for transition and trigger permanent deletion upon expiration of the compliance retention buffer.",
      "source_url": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h/view",
      "entailment_verified": true
    }
  ],
  "metrics": {
    "retrieval_latency_ms": 340,
    "rerank_latency_ms": 180,
    "total_latency_ms": 2850
  }
}
```

### 2. Ingest Document (`POST /api/v1/ingest`)

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Cloud Security Baseline Specification",
       "source_system": "confluence",
       "source_url": "https://wiki.enterprise.internal/pages/viewpage.action?pageId=554433",
       "content": "# Cloud Security Baseline\n## Data Encryption\nAll database volumes enforce AES-256 encryption at rest...",
       "access_control_list": ["group:everyone", "group:engineering"]
     }'
```

---

## 🧪 Evaluation Framework

Run automated regression benchmarks against the curated golden dataset:

```bash
python scripts/run_evals.py --sample-size 50
```

### Target Quality Gates
| Metric | Target | Failure Gate |
| :--- | :--- | :--- |
| **Faithfulness** | $\ge 98.0\%$ | Build fails if $< 97.5\%$ |
| **Citation Precision** | $\ge 95.0\%$ | Build fails if $< 94.0\%$ |
| **Refusal Accuracy** | $\ge 96.0\%$ | Build fails if $< 95.0\%$ |
| **Context Recall@7** | $\ge 92.0\%$ | Build fails if $< 90.0\%$ |

---

## 📂 Project Structure

```
.
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore patterns
├── Dockerfile                # API container image definition
├── docker-compose.yml        # Multi-container local stack (Qdrant, Redis, API)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Build configuration & tooling settings
├── docs/
│   ├── PRD.md                # Comprehensive Product Requirements Document
│   ├── ARCHITECTURE.md       # Technical architecture specification
│   └── EVALUATION_GUIDE.md   # Benchmarking and metrics guide
├── src/
│   ├── config.py             # Pydantic Settings configuration
│   ├── api/                  # FastAPI routes, schemas, and app lifecycle
│   ├── ingestion/            # Document parsing, chunking, and PII masking
│   ├── indexing/             # Dense & sparse vectorization, Qdrant client
│   ├── retrieval/            # Hybrid search, Cohere reranking, MMR query engine
│   ├── generation/           # System prompt, LLM generator, NLI citation verifier
│   └── evaluation/           # Golden benchmark dataset, metrics, and eval runner
├── tests/                    # Unit and integration test suite
└── scripts/
    ├── seed_sample_docs.py   # Populates local Qdrant with sample data
    └── run_evals.py          # Automated evaluation benchmark script
```

---

## 🔒 Security & Privacy

- **Zero Data Retention:** Foundation model calls run under enterprise zero-data-retention terms.
- **RBAC Pre-filtering:** Document permissions are checked inside the database index before any chunk text is returned to the reranker or LLM.
- **PII Scrubbing:** Automated sanitization redacts sensitive personally identifiable information prior to indexing.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
