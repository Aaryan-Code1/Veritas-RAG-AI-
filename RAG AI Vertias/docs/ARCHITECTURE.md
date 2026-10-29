# VeritasAI Technical Architecture

## 1. System Overview

VeritasAI implements a modular, high-throughput, enterprise-grade RAG architecture designed to eliminate hallucinations, enforce zero data leakage across Role-Based Access Control (RBAC) boundaries, and provide cryptographic citation traceability.

```
                           +---------------------------+
                           |  Client Web App / API     |
                           +-------------+-------------+
                                         |
                                         v
                         +---------------+---------------+
                         |   FastAPI Application Gateway |
                         +---------------+---------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      [ Write / Ingest Path ]                             [ Read / Query Path ]
               |                                                   |
               v                                                   v
+--------------+--------------+                   +----------------+---------------+
| Ingestion & Layout Analysis |                   | Query Understanding & Rewrite |
+--------------+--------------+                   +----------------+---------------+
               |                                                   |
               v                                                   v
+--------------+--------------+                   +----------------+---------------+
| Structure-Aware Chunker     |                   | Qdrant Hybrid Search (Dense+BM25)
+--------------+--------------+                   | + Strict RBAC Pre-Filter      |
               |                                  +----------------+---------------+
               v                                                   |
+--------------+--------------+                                    v
| PII & Secret Masking Engine |                   +----------------+---------------+
+--------------+--------------+                   | Cross-Encoder Reranker         |
               |                                  +----------------+---------------+
               v                                                   |
+--------------+--------------+                                    v
| Vector & BM25 Embedder      |                   +----------------+---------------+
+--------------+--------------+                   | Context Assembly & MMR         |
               |                                  +----------------+---------------+
               v                                                   |
+--------------+--------------+                                    v
| Qdrant Vector & Payload DB  | <=================+ Grounded LLM Stream (Claude)   |
+-----------------------------+                   +----------------+---------------+
                                                                   |
                                                                   v
                                                  +----------------+---------------+
                                                  | Post-Gen NLI Citation Verifier |
                                                  +--------------------------------+
```

## 2. Ingestion Details
- **Docling / Marker:** High-fidelity document parsing capturing headers, lists, code, and table boundaries.
- **Hierarchical Chunking:** 512-token chunks with 64-token overlap, snapping to punctuation boundaries. Contextual breadcrumbs (`[Document: ... | Section: ...]`) prepended to every chunk before embedding.
- **PII Scrubbing:** Redacts SSNs, credit cards, emails, and AWS access tokens.

## 3. Query Details
- **RRF Fusion:**
  $$RRF\_Score(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{60 + r_m(d)}$$
- **Reranker Confidence Gate:** If top rerank score $< 0.45$, the query triggers deterministic refusal immediately without making an LLM API call.
- **Verification:** Sentences mapped to `[n]` citations undergo Natural Language Inference entailment checks against retrieved source text.
