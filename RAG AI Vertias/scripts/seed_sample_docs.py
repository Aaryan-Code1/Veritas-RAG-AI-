"""
Seed script to populate sample enterprise compliance and infrastructure
documents into the local vector database.
"""
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.schemas import DocumentIngestRequest
from src.ingestion.chunker import HierarchicalChunker
from src.indexing.embeddings import EmbeddingService
from src.indexing.qdrant_client import QdrantStore

SAMPLE_DOCUMENTS = [
    DocumentIngestRequest(
        title="Enterprise Global Privacy Policy 2025",
        source_system="confluence",
        source_url="https://wiki.enterprise.internal/pages/privacy-policy-2025",
        version="2025.1",
        access_control_list=["group:everyone"],
        is_confidential=False,
        content="""# Global Data Privacy Policy

## Section 4.1: GDPR Data Subject Rights
All verified requests for customer data erasure under GDPR Article 17 must execute full data purge across active datastores within 30 calendar days. The privacy engineering team is notified automatically via webhook.

## Section 4.2: Data Retention
Customer audit logs and access histories are retained for exactly 365 calendar days before permanent cryptographic erasure.
""",
    ),
    DocumentIngestRequest(
        title="S3 Storage Lifecycle Architecture Spec",
        source_system="gdrive",
        source_url="https://drive.google.com/file/d/s3-lifecycle-spec-2025/view",
        version="3.0",
        access_control_list=["group:everyone", "group:engineering"],
        is_confidential=False,
        content="""# S3 Storage Lifecycle Architecture

## S3 Lifecycle Configuration
Production database backups and customer deletion flags trigger S3 lifecycle management rules. Lifecycle policies tag customer-deleted objects for transition and trigger permanent deletion upon expiration of the compliance retention buffer. Objects transition to Glacier Flexible Recovery before permanent expiration.

## Disaster Recovery Backups
Primary PostgreSQL databases take automated hourly snapshots retained for 35 days across two distinct AWS availability zones.
""",
    ),
    DocumentIngestRequest(
        title="Enterprise Cloud Security Baseline Architecture",
        source_system="confluence",
        source_url="https://wiki.enterprise.internal/pages/cloud-sec-baseline",
        version="2.2",
        access_control_list=["group:sec-ops", "group:engineering"],
        is_confidential=True,
        content="""# Cloud Security Baseline Architecture

## Data Protection and Encryption
Production databases enforce AES-256 encryption at rest utilizing AWS KMS Customer Managed Keys (CMKs) rotated automatically every 365 days. Unencrypted volumes are automatically terminated by AWS Config rules.

## Access Control Lists
Direct SSH access to production EC2 instances is strictly forbidden. All administrative access must traverse Teleport Bastion hosts with multi-factor authentication.
""",
    ),
]


def main():
    print("Initializing VeritasAI Ingestion Pipeline...")
    chunker = HierarchicalChunker()
    embedder = EmbeddingService()
    store = QdrantStore()

    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        print(f"Ingesting: '{doc.title}' ({doc.source_system})...")
        chunks = chunker.chunk_document(doc_id=f"seed_{doc.title[:8]}", doc_req=doc)
        dense_vecs = embedder.embed_documents([c.text for c in chunks])
        count = store.upsert_chunks(chunks=chunks, dense_vectors=dense_vecs)
        total_chunks += count
        print(f"  -> Indexed {count} chunks.")

    print(f"\nSeeding complete! Successfully indexed {total_chunks} chunks into collection '{store.collection_name}'.")


if __name__ == "__main__":
    main()
