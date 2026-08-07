from __future__ import annotations

import copy
import unittest

import pandas as pd

from src.tools.draft_writing.hybrid_retrieval import (
    balanced_hybrid_selection,
    deduplicate_candidates,
    query_chroma_candidates,
    query_csv_ranked_candidates,
    reciprocal_rank_fusion,
    retrieve_section_evidence_hybrid,
)


class FakeCollection:
    def __init__(self, by_source):
        self.by_source = by_source
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        source = kwargs["where"]["source_filename"]
        rows = self.by_source.get(source, [])[: kwargs["n_results"]]
        return {
            "documents": [[row["text"] for row in rows]],
            "metadatas": [[
                {
                    "source_filename": row.get("returned_source", source),
                    "chunk_id": row.get("chunk_id"),
                }
                for row in rows
            ]],
            "distances": [[row.get("distance", 0.1) for row in rows]],
        }


def row(source, chunk, score=1.0, text="substantive evidence", retrieval_source="chroma"):
    return {
        "source_filename": source,
        "chunk_id": chunk,
        "text": text,
        "score": score,
        "retrieval_source": retrieval_source,
        "retrieval_sources": [retrieval_source],
    }


class DeduplicationTests(unittest.TestCase):
    def test_duplicate_chroma_chroma(self):
        result = deduplicate_candidates([row("a", "1", 0.2), row("a", "1", 0.8)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 0.8)

    def test_duplicate_csv_csv(self):
        result = deduplicate_candidates([
            row("a", "1", 0.3, retrieval_source="csv"),
            row("a", "1", 0.4, retrieval_source="csv"),
        ])
        self.assertEqual(len(result), 1)

    def test_duplicate_chroma_csv_merges_sources(self):
        result = deduplicate_candidates([
            row("a", "1", 0.8, retrieval_source="chroma"),
            row("a", "1", 0.7, retrieval_source="csv"),
        ])
        self.assertEqual(result[0]["retrieval_sources"], ["chroma", "csv"])

    def test_incomplete_pairs_rejected(self):
        result = deduplicate_candidates([row("", "1"), row("a", ""), row("a", "1", text="")])
        self.assertEqual(result, [])

    def test_same_chunk_id_different_papers_is_distinct(self):
        result = deduplicate_candidates([row("a", "1"), row("b", "1")])
        self.assertEqual(len(result), 2)

    def test_same_paper_different_chunks_is_distinct(self):
        result = deduplicate_candidates([row("a", "1"), row("a", "2")])
        self.assertEqual(len(result), 2)

    def test_unauthorized_and_invalid_pairs_rejected(self):
        result = deduplicate_candidates(
            [row("a", "1"), row("b", "2")],
            allowed_papers=["a"],
            valid_source_chunk_pairs={("a", "9"), ("b", "2")},
        )
        self.assertEqual(result, [])


class RRFTests(unittest.TestCase):
    def test_only_chroma(self):
        fused = reciprocal_rank_fusion([row("a", "1")], [], rrf_k=60)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)
        self.assertEqual(fused[0]["retrieval_sources"], ["chroma"])

    def test_only_csv(self):
        fused = reciprocal_rank_fusion([], [row("a", "1", retrieval_source="csv")], rrf_k=60)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)
        self.assertEqual(fused[0]["retrieval_sources"], ["csv"])

    def test_present_in_both_exact_formula(self):
        c = row("a", "1"); c["chroma_rank"] = 2
        s = row("a", "1", retrieval_source="csv"); s["csv_rank"] = 3
        fused = reciprocal_rank_fusion([c], [s], rrf_k=60)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 62 + 1 / 63)
        self.assertEqual(fused[0]["retrieval_sources"], ["chroma", "csv"])

    def test_rrf_k_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_tie_break_is_deterministic(self):
        a = row("b", "2"); a["chroma_rank"] = 1
        b = row("a", "1"); b["chroma_rank"] = 1
        first = reciprocal_rank_fusion([a, b], [], rrf_k=60)
        second = reciprocal_rank_fusion([a, b], [], rrf_k=60)
        self.assertEqual(first, second)
        self.assertEqual((first[0]["source_filename"], first[0]["chunk_id"]), ("a", "1"))


class CSVQueryTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"source_filename": "a.pdf", "chunk_id": "1", "text": "Neural forecasting accuracy metric."},
            {"source_filename": "a.pdf", "chunk_id": "2", "text": "Forecasting only."},
            {"source_filename": "b.pdf", "chunk_id": "3", "text": "MÉTRICA de precisión clínica."},
            {"source_filename": "x.pdf", "chunk_id": "9", "text": "neural forecasting accuracy"},
            {"source_filename": "a.pdf", "chunk_id": "4", "text": ""},
        ])
        self.valid = {(r.source_filename, r.chunk_id) for r in self.df.itertuples()}

    def query(self, text):
        return query_csv_ranked_candidates(
            self.df, text, ["a.pdf", "b.pdf"],
            candidate_multiplier=3,
            top_k_evidence_per_section=8,
            max_candidates_per_source=24,
            valid_source_chunk_pairs=self.valid,
        )

    def test_exact_and_partial_lexical_matches_rank(self):
        result = self.query("neural forecasting accuracy")
        self.assertEqual(result[0]["chunk_id"], "1")
        self.assertGreater(result[0]["score"], result[1]["score"])

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.query(""), [])

    def test_empty_content_excluded(self):
        self.assertNotIn("4", [r["chunk_id"] for r in self.query("forecasting")])

    def test_accents_case_and_punctuation_normalized(self):
        result = self.query("metrica, PRECISION!")
        self.assertEqual(result[0]["chunk_id"], "3")

    def test_unauthorized_paper_excluded_even_with_best_match(self):
        self.assertNotIn("9", [r["chunk_id"] for r in self.query("neural forecasting accuracy")])

    def test_lexical_tie_is_stable(self):
        df = pd.DataFrame([
            {"source_filename": "b.pdf", "chunk_id": "2", "text": "clinical model"},
            {"source_filename": "a.pdf", "chunk_id": "1", "text": "clinical model"},
        ])
        result = query_csv_ranked_candidates(
            df, "clinical", ["a.pdf", "b.pdf"], candidate_multiplier=2,
            top_k_evidence_per_section=2, max_candidates_per_source=2,
            valid_source_chunk_pairs={("a.pdf", "1"), ("b.pdf", "2")},
        )
        self.assertEqual([(r["source_filename"], r["chunk_id"]) for r in result], [("a.pdf", "1"), ("b.pdf", "2")])

    def test_other_domain_vocabulary(self):
        result = self.query("clinica precision metrica")
        self.assertEqual(result[0]["chunk_id"], "3")


class ChromaQueryTests(unittest.TestCase):
    def test_restriction_dedup_and_per_source_limit(self):
        df = pd.DataFrame([
            {"source_filename": "a", "chunk_id": "1", "text": "one"},
            {"source_filename": "a", "chunk_id": "2", "text": "two"},
            {"source_filename": "b", "chunk_id": "3", "text": "three"},
        ])
        collection = FakeCollection({
            "a": [
                {"chunk_id": "1", "text": "one", "distance": 0.1},
                {"chunk_id": "1", "text": "one duplicate", "distance": 0.2},
                {"chunk_id": "2", "text": "two", "distance": 0.3},
                {"chunk_id": "x", "text": "wrong source", "returned_source": "x"},
            ],
            "b": [{"chunk_id": "3", "text": "three", "distance": 0.1}],
        })
        result = query_chroma_candidates(
            collection, df, "query", ["a", "b"], candidate_multiplier=3,
            top_k_evidence_per_section=2, max_candidates_per_source=1,
            valid_source_chunk_pairs={("a", "1"), ("a", "2"), ("b", "3")},
        )
        self.assertEqual(len(result), 2)
        self.assertEqual({r["source_filename"] for r in result}, {"a", "b"})


class BalancedSelectionTests(unittest.TestCase):
    def make_rankings(self):
        chroma = [row("a", str(i), 1 - i / 100) for i in range(1, 6)]
        csv = [row("b", str(i), 1 - i / 100, retrieval_source="csv") for i in range(1, 6)]
        for i, item in enumerate(chroma, 1): item["chroma_rank"] = i
        for i, item in enumerate(csv, 1): item["csv_rank"] = i
        return chroma, csv, reciprocal_rank_fusion(chroma, csv, rrf_k=60)

    def select(self, chroma, csv, fused, **overrides):
        params = dict(
            chroma_quota=3, csv_quota=3, rrf_quota=2,
            top_k_evidence_per_section=8, max_candidates_per_source=24,
            max_evidence_chars=18000, allowed_papers=["a", "b"],
            valid_source_chunk_pairs={(r["source_filename"], r["chunk_id"]) for r in chroma + csv},
        )
        params.update(overrides)
        return balanced_hybrid_selection(chroma, csv, fused, **params)

    def test_both_sources_fill_quotas(self):
        c, s, f = self.make_rankings(); result = self.select(c, s, f)
        self.assertEqual(len(result), 8)
        self.assertEqual(sum(r["selection_bucket"] == "chroma_quota" for r in result), 3)
        self.assertEqual(sum(r["selection_bucket"] == "csv_quota" for r in result), 3)

    def test_chroma_deficit_moves_to_rrf(self):
        c, s, _ = self.make_rankings(); c = c[:1]; f = reciprocal_rank_fusion(c, s, rrf_k=60)
        result = self.select(c, s, f)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]["selection_bucket"], "chroma_quota")

    def test_csv_deficit_moves_to_rrf(self):
        c, s, _ = self.make_rankings(); s = s[:1]; f = reciprocal_rank_fusion(c, s, rrf_k=60)
        result = self.select(c, s, f)
        self.assertEqual(len(result), 6)

    def test_none_fill_and_no_candidates(self):
        self.assertEqual(self.select([], [], []), [])

    def test_less_candidates_than_top_k_and_no_duplicates(self):
        c = [row("a", "1")]; c[0]["chroma_rank"] = 1
        s = [row("a", "1", retrieval_source="csv")]; s[0]["csv_rank"] = 1
        f = reciprocal_rank_fusion(c, s, rrf_k=60)
        result = self.select(c, s, f)
        self.assertEqual(len(result), 1)

    def test_quota_sum_must_equal_top_k(self):
        c, s, f = self.make_rankings()
        with self.assertRaisesRegex(ValueError, "retrieval_quotas"):
            self.select(c, s, f, rrf_quota=1)

    def test_unauthorized_high_score_never_enters(self):
        c, s, f = self.make_rankings()
        bad = row("x", "99", 99); bad["chroma_rank"] = 1
        result = self.select([bad] + c, s, reciprocal_rank_fusion([bad] + c, s, rrf_k=60))
        self.assertNotIn("x", [r["source_filename"] for r in result])

    def test_final_per_source_limit(self):
        c, s, f = self.make_rankings()
        result = self.select(c, s, f, max_candidates_per_source=2)
        self.assertLessEqual(sum(r["source_filename"] == "a" for r in result), 2)
        self.assertLessEqual(sum(r["source_filename"] == "b" for r in result), 2)

    def test_character_limit_skips_oversized_first_and_continues(self):
        c = [row("a", "1", text="x" * 20), row("a", "2", text="short")]
        for i, item in enumerate(c, 1): item["chroma_rank"] = i
        result = self.select(c, [], reciprocal_rank_fusion(c, [], rrf_k=60), max_evidence_chars=10)
        self.assertEqual([r["chunk_id"] for r in result], ["2"])

    def test_character_limit_excludes_next_that_would_overflow(self):
        c = [row("a", "1", text="12345"), row("a", "2", text="67890"), row("a", "3", text="x")]
        for i, item in enumerate(c, 1): item["chroma_rank"] = i
        result = self.select(c, [], reciprocal_rank_fusion(c, [], rrf_k=60), max_evidence_chars=6)
        self.assertEqual([r["chunk_id"] for r in result], ["1", "3"])

    def test_deterministic_repeated_execution(self):
        c, s, f = self.make_rankings()
        self.assertEqual(self.select(c, s, f), self.select(c, s, f))


class EndToEndHybridTests(unittest.TestCase):
    def test_retrieve_section_evidence_hybrid_trace_fields(self):
        df = pd.DataFrame([
            {"source_filename": "a.pdf", "chunk_id": "1", "text": "neural model accuracy"},
            {"source_filename": "a.pdf", "chunk_id": "2", "text": "forecast metric comparison"},
            {"source_filename": "b.pdf", "chunk_id": "3", "text": "clinical metric evidence"},
        ])
        collection = FakeCollection({
            "a.pdf": [{"chunk_id": "1", "text": "neural model accuracy", "distance": 0.1}],
            "b.pdf": [{"chunk_id": "3", "text": "clinical metric evidence", "distance": 0.2}],
        })
        section = {
            "section_title": "Model accuracy comparison",
            "purpose": "Compare forecast metrics",
            "papers_to_use": ["a.pdf", "b.pdf"],
        }
        result = retrieve_section_evidence_hybrid(
            section, collection, df, candidate_multiplier=3,
            chroma_quota=1, csv_quota=1, rrf_quota=1, rrf_k=60,
            top_k_evidence_per_section=3, max_evidence_chars=18000,
            max_candidates_per_source=24,
        )
        self.assertTrue(result)
        required = {
            "source_filename", "chunk_id", "text", "retrieval_source",
            "retrieval_sources", "chroma_rank", "csv_rank", "rrf_score",
            "selection_bucket", "selection_order",
        }
        for item in result:
            self.assertTrue(required.issubset(item))
        self.assertEqual(result, retrieve_section_evidence_hybrid(
            section, collection, df, candidate_multiplier=3,
            chroma_quota=1, csv_quota=1, rrf_quota=1, rrf_k=60,
            top_k_evidence_per_section=3, max_evidence_chars=18000,
            max_candidates_per_source=24,
        ))


if __name__ == "__main__":
    unittest.main()
