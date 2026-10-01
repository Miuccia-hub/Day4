import math
import tempfile
import unittest
from pathlib import Path

from rag_core import EvidenceStore, change_control_review, chunk_document, embed_text


ROOT = Path(__file__).resolve().parents[1]


class RagCoreTests(unittest.TestCase):
    def test_embedding_is_normalized_and_deterministic(self):
        first = embed_text("change in control termination")
        second = embed_text("change in control termination")
        self.assertEqual(first, second)
        self.assertTrue(math.isclose(sum(value * value for value in first), 1.0, abs_tol=1e-9))

    def test_chunk_metadata_and_line_numbers(self):
        path = ROOT / "contracts" / "Software_Development_and_Services_Agreement_Canvassian_Bravocat_2023_8.txt"
        chunks = chunk_document(path, ROOT, max_chars=600)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(item.source.startswith("contracts/") for item in chunks))
        self.assertTrue(all(item.line_start <= item.line_end for item in chunks))
        self.assertIn("Bravocat", chunks[0].clients)

    def test_change_control_review_finds_adverse_clients(self):
        rows = {row["client"]: row for row in change_control_review(ROOT)}
        self.assertEqual(rows["Bravocat"]["status"], "ADVERSE")
        self.assertEqual(rows["Charlemont"]["status"], "ADVERSE")
        self.assertEqual(rows["Echona"]["status"], "ADVERSE")
        self.assertEqual(rows["PayWise"]["control_clauses"], 0)

    def test_local_search_retrieves_jane_departure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(ROOT, Path(directory) / "index")
            store._collection = None
            store.backend = "local"
            store.ensure_index(force=True)
            results = store.search("Jane Wu potential departure stepping down within six months", top_k=5)
            self.assertTrue(any("Leadership_Transition" in result.chunk.source for result in results))


if __name__ == "__main__":
    unittest.main()
