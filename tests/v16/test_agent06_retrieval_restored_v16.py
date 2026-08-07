from __future__ import annotations
import unittest
import pandas as pd

from src.tools.draft_writing.retrieval import (
    dedupe_evidence,
    query_chroma_restricted,
    query_csv_restricted,
    retrieve_section_evidence,
)


class RecordingCollection:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs["where"]["source_filename"]
        return self.responses.get(source, {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        })


class TestAgent06RetrievalRestoredV16(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"source_filename": "a.pdf", "chunk_id": "a1", "text": "alpha beta gamma"},
            {"source_filename": "a.pdf", "chunk_id": "a2", "text": "unrelated content"},
            {"source_filename": "a.pdf", "chunk_id": "a3", "text": "alpha"},
            {"source_filename": "b.pdf", "chunk_id": "b1", "text": "alpha beta"},
            {"source_filename": "b.pdf", "chunk_id": "b2", "text": "zero overlap"},
        ])
        self.valid = {(r.source_filename, r.chunk_id) for r in self.df.itertuples()}

    def test_per_source_k_and_real_chunk_count_limit(self):
        collection = RecordingCollection()
        query_chroma_restricted(
            collection, self.df, "alpha beta", ["a.pdf", "b.pdf"], 8,
            valid_source_chunk_pairs=self.valid,
        )
        self.assertEqual(8 // 2 + 1, 5)
        self.assertEqual(collection.calls[0]["n_results"], 3)
        self.assertEqual(collection.calls[1]["n_results"], 2)

    def test_returned_source_must_match_requested_source(self):
        collection = RecordingCollection({
            "a.pdf": {
                "documents": [["alpha", "alpha"]],
                "metadatas": [[
                    {"source_filename": "wrong.pdf", "chunk_id": "a1"},
                    {"source_filename": "a.pdf", "chunk_id": "a2"},
                ]],
                "distances": [[0.0, 0.2]],
            }
        })
        rows = query_chroma_restricted(
            collection, self.df, "alpha", ["a.pdf"], 3,
            valid_source_chunk_pairs=self.valid,
        )
        self.assertEqual([row["chunk_id"] for row in rows], ["a2"])

    def test_csv_score_is_normalized_and_zero_scores_are_kept(self):
        rows = query_csv_restricted(
            self.df, "alpha beta delta", ["a.pdf"], 10,
            valid_source_chunk_pairs=self.valid,
        )
        by_id = {row["chunk_id"]: row for row in rows}
        self.assertAlmostEqual(by_id["a1"]["score"], 2 / 3)
        self.assertAlmostEqual(by_id["a3"]["score"], 1 / 3)
        self.assertEqual(by_id["a2"]["score"], 0.0)

    def test_max_evidence_chars_applies_to_chroma_and_csv(self):
        long_text = "x" * 100
        df = pd.DataFrame([
            {"source_filename": "a.pdf", "chunk_id": "a1", "text": long_text},
        ])
        valid = {("a.pdf", "a1")}
        collection = RecordingCollection({
            "a.pdf": {
                "documents": [[long_text]],
                "metadatas": [[{"source_filename": "a.pdf", "chunk_id": "a1"}]],
                "distances": [[0.1]],
            }
        })
        chroma_rows = query_chroma_restricted(
            collection, df, "x", ["a.pdf"], 1, 12, valid,
        )
        csv_rows = query_csv_restricted(df, "x", ["a.pdf"], 1, 12, valid)
        self.assertEqual(len(chroma_rows[0]["text"]), 12)
        self.assertEqual(len(csv_rows[0]["text"]), 12)

    def test_dedupe_rejects_invalid_pairs_keeps_best_and_stable_order(self):
        rows = [
            {"source_filename": "b.pdf", "chunk_id": "b1", "text": "alpha", "score": 0.5},
            {"source_filename": "a.pdf", "chunk_id": "a2", "text": "alpha", "score": 0.5},
            {"source_filename": "a.pdf", "chunk_id": "a1", "text": "alpha", "score": 0.5},
            {"source_filename": "a.pdf", "chunk_id": "a1", "text": "better", "score": 0.9},
            {"source_filename": "x.pdf", "chunk_id": "x1", "text": "invalid", "score": 1.0},
        ]
        result = dedupe_evidence(rows, self.valid)
        self.assertEqual(
            [(r["source_filename"], r["chunk_id"], r["score"]) for r in result],
            [("a.pdf", "a1", 0.9), ("a.pdf", "a2", 0.5), ("b.pdf", "b1", 0.5)],
        )

    def test_retrieve_sequence_chroma_then_csv_dedupe_then_top_k(self):
        collection = RecordingCollection({
            "a.pdf": {
                "documents": [["alpha beta gamma"]],
                "metadatas": [[{"source_filename": "a.pdf", "chunk_id": "a1"}]],
                "distances": [[0.4]],
            }
        })
        section = {
            "section_title": "alpha",
            "purpose": "beta",
            "papers_to_use": [{"source_filename": "a.pdf"}],
        }
        rows = retrieve_section_evidence(section, collection, self.df, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(collection.calls[0]["where"], {"source_filename": "a.pdf"})
        self.assertEqual(rows[0]["chunk_id"], "a1")
        self.assertEqual(rows[0]["retrieval_method"], "csv_lexical_restricted")
        self.assertEqual(rows[1]["retrieval_method"], "csv_lexical_restricted")


if __name__ == "__main__":
    unittest.main()
