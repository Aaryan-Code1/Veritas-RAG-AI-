import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from ..config import settings
from ..models.schemas import ChunkMetadata


class QdrantStore:
    """Manages collection lifecycle, payload indexes, upserts, and filtered vector retrieval."""

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        
        # Connect to Qdrant or use in-memory store if connection fails
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=3.0,
            )
            # Ping to verify
            self.client.get_collections()
        except Exception:
            # Fallback to local in-memory Qdrant instance for test suites & offline runs
            self.client = QdrantClient(location=":memory:")

        self._ensure_collection()

    def _ensure_collection(self):
        """Initializes collection with HNSW index configuration and payload index."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=self.dimensions,
                    distance=rest_models.Distance.COSINE,
                    hnsw_config=rest_models.HnswConfigDiff(
                        m=16,
                        ef_construct=128,
                    ),
                ),
            )
            # Create payload keyword index for fast RBAC ACL filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="access_control_list",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(self, chunks: List[ChunkMetadata], dense_vectors: List[List[float]]) -> int:
        """Upserts chunk payloads and dense vectors into Qdrant."""
        points = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = chunk.model_dump()

            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=dense_vec,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search_candidates(
        self,
        query_vector: List[float],
        user_groups: List[str],
        limit: int = 50,
    ) -> List[ChunkMetadata]:
        """Executes dense vector search with hard RBAC security pre-filter."""
        # Construct security filter: chunk ACL must match any of the user's groups
        rbac_filter = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="access_control_list",
                    match=rest_models.MatchAny(any=user_groups),
                )
            ]
        )

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=rbac_filter,
            limit=limit,
        )

        candidates: List[ChunkMetadata] = []
        for hit in search_results:
            payload = hit.payload or {}
            chunk = ChunkMetadata(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                document_title=payload.get("document_title", ""),
                source_system=payload.get("source_system", ""),
                source_url=payload.get("source_url", ""),
                page_number=payload.get("page_number", 1),
                section_name=payload.get("section_name", "General"),
                chunk_index=payload.get("chunk_index", 0),
                text=payload.get("text", ""),
                access_control_list=payload.get("access_control_list", []),
                is_confidential=payload.get("is_confidential", False),
                rerank_score=hit.score,
            )
            candidates.append(chunk)

        return candidates
