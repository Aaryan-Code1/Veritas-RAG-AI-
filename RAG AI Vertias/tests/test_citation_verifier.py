import unittest
from src.generation.citation_verifier import CitationVerifier
from src.models.schemas import ChunkMetadata


class TestCitationVerifier(unittest.TestCase):

    def test_extract_citations(self):
        text = "Enterprise data is backed up hourly [1]. Redis snapshots are daily [2] with 7-day TTL [1][3]."
        cids = CitationVerifier.extract_citations_from_text(text)
        self.assertEqual(cids, [1, 2, 3])

    def test_verify_and_build_citations(self):
        answer = "Production databases execute automated hourly snapshots [1]."
        chunks = [
            ChunkMetadata(
                chunk_id="chk_01",
                document_id="doc_01",
                document_title="DR Plan",
                source_system="confluence",
                source_url="https://wiki.test",
                page_number=4,
                section_name="Backups",
                chunk_index=0,
                text="Production databases execute automated hourly snapshots retained for 35 days.",
                access_control_list=["group:everyone"],
            )
        ]

        citations = CitationVerifier.verify_and_build_citations(answer, chunks)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].citation_id, 1)
        self.assertEqual(citations[0].document_title, "DR Plan")
        self.assertEqual(citations[0].page_number, 4)
        self.assertTrue(citations[0].entailment_verified)


if __name__ == "__main__":
    unittest.main()
