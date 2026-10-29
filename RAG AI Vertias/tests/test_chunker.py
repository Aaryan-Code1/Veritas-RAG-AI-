import unittest
from src.ingestion.sanitizer import PIISanitizer
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import HierarchicalChunker
from src.models.schemas import DocumentIngestRequest


class TestChunkerAndParser(unittest.TestCase):

    def test_pii_sanitizer(self):
        dirty_text = "Employee SSN is 123-45-6789 and Card is 4532-1234-5678-9010."
        sanitized = PIISanitizer.sanitize(dirty_text)
        self.assertNotIn("123-45-6789", sanitized)
        self.assertIn("[REDACTED_SSN]", sanitized)
        self.assertIn("[REDACTED_CARD]", sanitized)

    def test_document_parser_markdown(self):
        md = """# Section 1
This is line one.

## Section 2
This is section two.
"""
        sections = DocumentParser.extract_sections(md)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["section_name"], "Section 1")
        self.assertEqual(sections[1]["section_name"], "Section 2")

    def test_hierarchical_chunker(self):
        chunker = HierarchicalChunker(target_chars=500, overlap_chars=50)
        req = DocumentIngestRequest(
            title="Test Policy",
            source_system="confluence",
            source_url="https://test.internal",
            content="""# Privacy
Customer records are deleted in 30 days. This is a mandatory standard.

# Security
All keys must be rotated annually.
""",
        )
        chunks = chunker.chunk_document("doc_test_123", req)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("[Document: Test Policy | Section:", chunks[0].text)
        self.assertEqual(chunks[0].document_id, "doc_test_123")


if __name__ == "__main__":
    unittest.main()
