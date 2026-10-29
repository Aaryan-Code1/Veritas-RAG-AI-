import uuid
import re
from typing import List, Dict, Any
from ..models.schemas import ChunkMetadata, DocumentIngestRequest
from .sanitizer import PIISanitizer
from .parser import DocumentParser


class HierarchicalChunker:
    """Structure-aware chunker with sentence snapping and contextual header prepending."""

    def __init__(self, target_chars: int = 1800, overlap_chars: int = 200):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk_document(
        self, doc_id: str, doc_req: DocumentIngestRequest
    ) -> List[ChunkMetadata]:
        # Clean & sanitize content
        raw_text = doc_req.content
        if doc_req.file_type == "text/html":
            raw_text = DocumentParser.parse_html(raw_text)

        sanitized_text = PIISanitizer.sanitize(raw_text)
        sections = DocumentParser.extract_sections(sanitized_text)

        chunks: List[ChunkMetadata] = []
        chunk_idx = 0

        for sec in sections:
            sec_name = sec["section_name"]
            sec_content = sec["content"]

            if not sec_content:
                continue

            sec_chunks = self._split_text(sec_content)

            for text_segment in sec_chunks:
                # Prepend contextual header breadcrumbs
                contextual_header = (
                    f"[Document: {doc_req.title} | Section: {sec_name} | Version: {doc_req.version}]\n"
                )
                full_chunk_text = contextual_header + text_segment

                chunk_obj = ChunkMetadata(
                    chunk_id=f"chk_{uuid.uuid4()}",
                    document_id=doc_id,
                    document_title=doc_req.title,
                    source_system=doc_req.source_system,
                    source_url=doc_req.source_url,
                    page_number=1,
                    section_name=sec_name,
                    chunk_index=chunk_idx,
                    text=full_chunk_text,
                    access_control_list=doc_req.access_control_list,
                    is_confidential=doc_req.is_confidential,
                )
                chunks.append(chunk_obj)
                chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Splits text into chunks respecting sentence boundaries."""
        if len(text) <= self.target_chars:
            return [text.strip()]

        # Split on sentence terminals
        sentences = re.split(r"(?<=[.?!])\s+", text)
        result_chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self.target_chars and current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                result_chunks.append(chunk_str)

                # Overlap calculation
                overlap_tokens = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    if overlap_len + len(prev_sent) <= self.overlap_chars:
                        overlap_tokens.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break

                current_chunk = overlap_tokens + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        if current_chunk:
            result_chunks.append(" ".join(current_chunk).strip())

        return result_chunks
