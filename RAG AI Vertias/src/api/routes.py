import uuid
import hashlib
from fastapi import APIRouter, HTTPException, Depends
from ..models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentIngestRequest,
    DocumentResponse,
    EvalRunRequest,
    EvalRunResponse,
)
from ..ingestion.chunker import HierarchicalChunker
from ..indexing.embeddings import EmbeddingService
from ..indexing.qdrant_client import QdrantStore
from ..retrieval.query_engine import QueryEngine
from ..generation.generator import LLMGenerator
from ..evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/api/v1")

# Singletons / Dependency injections
_store = None
_embedding_service = None
_chunker = None
_query_engine = None
_generator = None


def get_store():
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = HierarchicalChunker()
    return _chunker


def get_query_engine(store: QdrantStore = Depends(get_store)):
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(qdrant_store=store)
    return _query_engine


def get_generator():
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    engine: QueryEngine = Depends(get_query_engine),
    generator: LLMGenerator = Depends(get_generator),
):
    """
    Submits a query for RBAC-filtered retrieval, cross-encoder reranking,
    grounded synthesis, and citation verification.
    """
    chunks, is_confident, r_metrics = engine.retrieve_context(request)
    formatted_ctx = engine.format_context_for_prompt(chunks)
    response = generator.generate_grounded_answer(request, chunks, formatted_ctx, r_metrics)
    return response


@router.post("/ingest", response_model=DocumentResponse)
async def ingest_document(
    request: DocumentIngestRequest,
    chunker: HierarchicalChunker = Depends(get_chunker),
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: QdrantStore = Depends(get_store),
):
    """
    Ingests and parses a document, performs PII sanitization, computes
    hierarchical chunks, generates embeddings, and indexes into Qdrant.
    """
    doc_id = f"doc_{uuid.uuid4()}"
    content_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()

    chunks = chunker.chunk_document(doc_id=doc_id, doc_req=request)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content resulted in zero valid chunks.")

    texts = [c.text for c in chunks]
    dense_vectors = embedder.embed_documents(texts)
    indexed_count = store.upsert_chunks(chunks=chunks, dense_vectors=dense_vectors)

    return DocumentResponse(
        document_id=doc_id,
        status="INDEXED",
        chunks_indexed=indexed_count,
        content_hash=f"sha256_{content_hash}",
        message=f"Successfully indexed {indexed_count} chunks with RBAC permissions.",
    )


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_evaluation_benchmark(
    request: EvalRunRequest,
    engine: QueryEngine = Depends(get_query_engine),
    generator: LLMGenerator = Depends(get_generator),
):
    """
    Executes automated benchmark evaluation sweep across the golden dataset.
    """
    runner = EvaluationRunner(query_engine=engine, generator=generator)
    results = runner.run_evaluations(sample_size=request.sample_size)
    return results
