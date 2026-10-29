from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., description="Document title")
    source_system: str = Field(..., description="Origin system: confluence, gdrive, notion, pdf")
    source_url: str = Field(..., description="Canonical source URL")
    content: str = Field(..., description="Raw text, markdown, or HTML content")
    file_type: str = Field(default="text/markdown", description="MIME type or format")
    version: str = Field(default="1.0", description="Document version string")
    access_control_list: List[str] = Field(
        default=["group:everyone"], description="List of user emails and groups authorized to access"
    )
    is_confidential: bool = Field(default=False, description="Flag for sensitive documents")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary custom tags")


class DocumentResponse(BaseModel):
    document_id: str
    status: str
    chunks_indexed: int
    content_hash: str
    message: str


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source_system: str
    source_url: str
    page_number: Optional[int] = 1
    section_name: Optional[str] = "General"
    chunk_index: int
    text: str
    access_control_list: List[str]
    is_confidential: bool = False
    rerank_score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question from the user")
    session_id: Optional[str] = Field(default=None, description="Session ID for multi-turn history")
    user_groups: List[str] = Field(
        default=["group:everyone"], description="Active authenticated user groups and email"
    )
    stream: bool = Field(default=False, description="Whether to stream response via SSE")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")


class Citation(BaseModel):
    citation_id: int
    document_id: str
    document_title: str
    source_url: str
    source_system: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    verbatim_snippet: str
    entailment_verified: bool = True
    match_confidence: float = 1.0


class QueryMetrics(BaseModel):
    retrieval_latency_ms: int
    rerank_latency_ms: int
    total_latency_ms: int
    candidate_chunks_evaluated: int
    final_chunks_used: int


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    refusal: bool
    citations: List[Citation]
    metrics: QueryMetrics


class EvalItem(BaseModel):
    eval_id: str
    query_text: str
    ground_truth_answer: str
    ground_truth_chunk_ids: List[str]
    ground_truth_doc_titles: List[str]
    query_type: str = "single_hop"  # single_hop, multi_hop, refusal, adversarial
    expected_refusal: bool = False
    required_acl_groups: List[str] = ["group:everyone"]


class EvalRunRequest(BaseModel):
    golden_dataset_id: Optional[str] = "default"
    sample_size: Optional[int] = None
    metrics: List[str] = [
        "faithfulness",
        "citation_precision",
        "refusal_accuracy",
        "recall_at_k",
    ]


class EvalRunResponse(BaseModel):
    eval_run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_queries: int
    metrics: Dict[str, float]
    pass_fail_status: str
