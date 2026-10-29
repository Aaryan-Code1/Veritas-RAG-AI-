# Product Requirements Document (PRD)

**Product Name:** VeritasAI — Enterprise Knowledge & Compliance RAG System  
**Document Version:** 1.0.0-FINAL  
**Owner:** Senior AI Product Manager & Lead ML Engineer  
**Status:** Approved for Engineering Implementation  

---

## 1. Product Context

| Attribute | Specification | Operational Details & Labelled Assumptions |
| :--- | :--- | :--- |
| **Product Name** | **VeritasAI** | Enterprise Retrieval-Augmented Generation & Auditable Compliance Intelligence Platform |
| **Domain / Use Case** | Enterprise Knowledge Search, Security & Compliance Auditing, Legal/HR Policy Discovery | Unified semantic search and synthetic question answering across internal engineering docs, security architecture, SOC 2/ISO-27001 audit binders, HR policies, and customer contract addendums. |
| **Target Users** | Solutions Architects, Security & Compliance Auditors, Legal Counsel, Technical Support Tier 3 | ~2,500 monthly active internal enterprise employees, scaling to 15,000 cross-functional knowledge workers in Phase 2. |
| **Data Sources** | Unstructured & Semi-structured Enterprise Repositories | Confluence, Google Drive (Docs, Sheets, Slides), Notion workspaces, Jira, scanned and digital PDF contracts/whitepapers, DOCX SOPs, and internal developer portals (Markdown/HTML). |
| **Corpus Size** | ~120,000 Documents (~650,000 pages) | Raw data volume: ~450 GB (text, structured tables, and embedded raster figures). Growth rate: ~3,500 new/updated documents per month. |
| **Languages** | Primary: English (92%); Secondary: Spanish, German, French, Japanese (8%) | Cross-lingual retrieval allows English queries against non-English documents via aligned multilingual vector embeddings. |
| **Update Frequency** | Event-Driven & Scheduled Batch | Webhook-driven near-real-time ingestion (<15 min sync latency) for Notion/Confluence/Google Drive; daily batch delta sync (02:00 UTC) for legacy file shares and scanned audit binders. |
| **Constraints** | Strict Latency, Data Sovereignty, PII/RBAC Enforcement | - **Latency Target:** TTFT $\le 1,200\text{ ms}$; Total Response Completion $\le 4,500\text{ ms}$ (p95).<br>- **Deployment:** Single-tenant AWS GovCloud / Commercial VPC (EKS).<br>- **Privacy/Compliance:** Zero Data Retention (ZDR) guarantee; automated PII redaction; strict Document-Level Access Control (RBAC/ABAC).<br>- **Compute Budget:** $\le \$0.015$ average inference cost per query. |
| **Recommended Stack** | Modern Hybrid RAG Engine | - **LLM Provider:** Anthropic Claude 3.5 Sonnet (primary) / GPT-4o (secondary fallback).<br>- **Vector DB:** Qdrant Enterprise (distributed HNSW with on-disk payload quantization).<br>- **Reranker:** Cohere Rerank v3.5 (API) with local fallback to `BAAI/bge-reranker-large`.<br>- **Framework:** Custom LlamaIndex / LangGraph orchestration.<br>- **Backend:** Python 3.11+, FastAPI (asynchronous I/O), Celery + Redis. |

---

## 2. Core Product Requirement

Every time an authenticated user submits a natural-language query, VeritasAI **must**:
1. **Retrieve:** Extract the top-$k$ most semantically and lexically relevant chunks from the verified enterprise corpus matching the user’s exact role-based access control (RBAC) permissions.
2. **Ground:** Synthesize an accurate, coherent answer using **exclusively** the retrieved, validated context chunks.
3. **Attribute (Citations):** Provide precise, interactive inline citation markers (`[1]`, `[2]`) anchored to specific claims, mapped to an inspectable source panel showing document title, version, page/section, hyperlink, and highlighted verbatim supporting quote.
4. **Deterministic Refusal:** When retrieved context is missing, incomplete, or ambiguous, output: *"I do not have enough verified information in the provided documentation to answer this question."*

---

## 3. System Architecture & Components

VeritasAI separates the write path (**Ingestion Pipeline**) and the read path (**Query & Verification Pipeline**).

### Ingestion Pipeline
1. **Document Connectors:** Extract files via webhooks and scheduled sync.
2. **Document Parsing Engine:** Handles PDF, DOCX, HTML, and Markdown, preserving table integrity and heading hierarchies.
3. **Structural Chunker:** Chunks text into 512-token segments (64-token overlap) snapping to sentence boundaries, prepending contextual breadcrumbs.
4. **PII Masking:** Sanitizes credit cards, SSNs, credentials before indexing.
5. **Embedding Generator:** Computes 1536-d dense vectors (`text-embedding-3-large`) and sparse BM25 representations.
6. **Vector Storage:** Persists vectors and metadata payloads into Qdrant.

### Query Pipeline
1. **Query Condensation & Rewriting:** Resolves chat history and generates standalone search terms.
2. **Hybrid Search:** Runs dense k-NN search and sparse BM25 in Qdrant with pre-filtering on user ACLs.
3. **Score Fusion:** Combines dense and sparse ranks via Reciprocal Rank Fusion ($k=60$).
4. **Cross-Encoder Reranking:** Re-scores candidates using Cohere Rerank v3.5; drops chunks below threshold $0.45$.
5. **Diversity (MMR):** Selects top-7 diverse context chunks ($\lambda = 0.75$).
6. **Generation:** Claude 3.5 Sonnet generates grounded answers with inline `[n]` citation markers.
7. **Post-Generation NLI Verification:** Validates claims against source snippets, flagging any unsupported claims.

---

## 4. Evaluation Framework

- **Golden Dataset:** 1,200 curated and verified queries (70% answerable, 20% refusal/out-of-domain, 10% adversarial/ambiguous).
- **Retrieval Metrics:** Recall@7 ($\ge 92\%$), Precision@7 ($\ge 88\%$), MRR ($\ge 0.85$), Hit Rate ($\ge 96\%$).
- **Generation Metrics:** Faithfulness ($\ge 98\%$), Answer Relevance ($\ge 90\%$), Refusal Accuracy ($\ge 96\%$).
- **Citation Metrics:** Citation Precision ($\ge 95\%$), Citation Recall ($\ge 98\%$).
- **CI/CD Quality Gates:** PRs fail build if Faithfulness drops below $97.5\%$ or Citation Precision drops below $94\%$.
