import re
from typing import List, Tuple, Dict, Any
from ..models.schemas import ChunkMetadata, Citation


class CitationVerifier:
    """Verifies that generated inline citations [n] correspond to actual supporting source text."""

    CITATION_REGEX = re.compile(r"\[(\d+)\]")

    @classmethod
    def extract_citations_from_text(cls, text: str) -> List[int]:
        """Finds all unique integer citation markers in text, sorted."""
        matches = cls.CITATION_REGEX.findall(text)
        ids = sorted(list(set(int(m) for m in matches)))
        return ids

    @classmethod
    def verify_and_build_citations(
        cls, answer_text: str, context_chunks: List[ChunkMetadata]
    ) -> List[Citation]:
        """
        Parses citations [n] from answer_text, maps them to context_chunks[n-1],
        and verifies that the cited text actually supports the claim.
        """
        cited_indices = cls.extract_citations_from_text(answer_text)
        citations: List[Citation] = []

        chunk_map: Dict[int, ChunkMetadata] = {
            i + 1: chunk for i, chunk in enumerate(context_chunks)
        }

        # Sentence-by-sentence extraction
        sentences = re.split(r"(?<=[.?!])\s+", answer_text)

        for cid in cited_indices:
            if cid not in chunk_map:
                continue

            chunk = chunk_map[cid]

            # Find the sentence(s) that cited this marker
            citing_sentences = [s for s in sentences if f"[{cid}]" in s]

            # Simple lexical overlap verification between citing sentence and chunk text
            entailment_passed = True
            if citing_sentences:
                overlap = cls._calculate_lexical_entailment(citing_sentences[0], chunk.text)
                entailment_passed = overlap >= 0.20  # Minimum 20% semantic token overlap

            # Extract most relevant snippet from chunk
            verbatim_snippet = cls._extract_best_snippet(citing_sentences, chunk.text)

            citation = Citation(
                citation_id=cid,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                source_url=chunk.source_url,
                source_system=chunk.source_system,
                page_number=chunk.page_number,
                section_name=chunk.section_name,
                verbatim_snippet=verbatim_snippet,
                entailment_verified=entailment_passed,
                match_confidence=round(chunk.rerank_score or 0.95, 2),
            )
            citations.append(citation)

        return citations

    @staticmethod
    def _calculate_lexical_entailment(sentence: str, chunk_text: str) -> float:
        """Calculates token overlap ratio between claim sentence and source chunk."""
        clean_s = set(re.sub(r"[^a-zA-Z0-9\s]", "", sentence.lower()).split())
        clean_c = set(re.sub(r"[^a-zA-Z0-9\s]", "", chunk_text.lower()).split())
        # Remove common stop words
        stopwords = {"the", "a", "an", "and", "or", "is", "are", "in", "to", "for", "of", "with", "that", "this"}
        s_words = clean_s - stopwords
        if not s_words:
            return 1.0
        overlap = len(s_words.intersection(clean_c)) / len(s_words)
        return float(overlap)

    @staticmethod
    def _extract_best_snippet(citing_sentences: List[str], chunk_text: str) -> str:
        """Finds the most representative 1-2 sentences in the chunk text."""
        chunk_lines = [line.strip() for line in chunk_text.splitlines() if line.strip() and not line.startswith("[Document:")]
        if chunk_lines:
            # Return first substantial sentence/line up to 250 characters
            snippet = chunk_lines[0]
            if len(snippet) > 250:
                snippet = snippet[:247] + "..."
            return snippet
        return chunk_text[:200]
