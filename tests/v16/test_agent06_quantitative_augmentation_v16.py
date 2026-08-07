from __future__ import annotations

import json
import unittest

import pandas as pd

from src.tools.draft_writing.quantitative_augmentation import (
    DEFAULT_CONFIRMED_STATUSES,
    augment_evidence_with_quantitative_chunks_greedy,
    build_quantitative_chunk_candidates,
    normalize_chunk_ids,
    normalize_confirmed_quantitative_rows,
)


def chunks(*rows):
    return pd.DataFrame(rows, columns=["source_filename", "chunk_id", "text"])


def base(source, chunk_id, text, score=0.0, **extra):
    row = {
        "source_filename": source,
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "rrf_score": score,
        "retrieval_sources": ["chroma", "csv"],
    }
    row.update(extra)
    return row


def qrow(source="a.pdf", chunk="c1", value="10%", status="confirmed_in_source_chunk", **extra):
    row = {
        "source_filename": source,
        "chunk_id": chunk,
        "value": value,
        "verification_status": status,
    }
    row.update(extra)
    return row


class ChunkIdNormalizationTests(unittest.TestCase):
    def test_list_tuple_and_set(self):
        self.assertEqual(normalize_chunk_ids(["c1", "c2", "c1", ""]), ["c1", "c2"])
        self.assertEqual(normalize_chunk_ids(("c1", "c2")), ["c1", "c2"])
        self.assertEqual(normalize_chunk_ids({"c2", "c1"}), ["c1", "c2"])

    def test_json_serialized_list(self):
        self.assertEqual(normalize_chunk_ids('["c1", "c2", "c1"]'), ["c1", "c2"])

    def test_delimited_and_single(self):
        self.assertEqual(normalize_chunk_ids("c1,c2;c3|c2\nc4"), ["c1", "c2", "c3", "c4"])
        self.assertEqual(normalize_chunk_ids("c1"), ["c1"])

    def test_empty(self):
        self.assertEqual(normalize_chunk_ids(None), [])
        self.assertEqual(normalize_chunk_ids(""), [])


class ConfirmedRowTests(unittest.TestCase):
    def test_primary_and_equivalent_statuses(self):
        rows = normalize_confirmed_quantitative_rows([
            qrow(status="confirmed_in_source_chunk"),
            qrow(chunk="c2", status="confirmed_literal_in_source_chunk"),
        ])
        self.assertEqual(len(rows), 2)
        self.assertIn("confirmed_literal_in_source_chunk", DEFAULT_CONFIRMED_STATUSES)

    def test_unknown_empty_and_partial_status_rejected(self):
        rows = normalize_confirmed_quantitative_rows([
            qrow(status=""),
            qrow(status="not_confirmed"),
            qrow(status="confirmed_in_source_chunk_extra"),
            qrow(status="contains confirmed_in_source_chunk text"),
        ])
        self.assertEqual(rows, [])

    def test_missing_required_fields_rejected(self):
        rows = normalize_confirmed_quantitative_rows([
            qrow(source=""),
            qrow(chunk=""),
            qrow(value=""),
        ])
        self.assertEqual(rows, [])

    def test_value_field_priority(self):
        row = qrow(value="", numeric_value="1.2", reported_value="9")
        normalized = normalize_confirmed_quantitative_rows([row])
        self.assertEqual(normalized[0]["_value"], "1.2")

    def test_direct_and_checked_ids_are_merged_stably(self):
        row = qrow(chunk="c1", source_chunk_ids_checked='["c2", "c1", "c3"]')
        normalized = normalize_confirmed_quantitative_rows([row])
        self.assertEqual(normalized[0]["_chunk_ids"], ["c1", "c2", "c3"])


class CandidateConstructionTests(unittest.TestCase):
    def setUp(self):
        self.df = chunks(
            ("a.pdf", "c1", "Accuracy was 95% on dataset A."),
            ("a.pdf", "c2", "Error values were 1.3 and 10 % respectively."),
            ("b.pdf", "c1", "Accuracy was 95% on dataset B."),
        )
        self.valid = {(r.source_filename, r.chunk_id) for r in self.df.itertuples()}

    def build(self, rows, allowed=("a.pdf", "b.pdf")):
        return build_quantitative_chunk_candidates(
            self.df,
            rows,
            allowed_papers=list(allowed),
            valid_source_chunk_pairs=self.valid,
        )

    def test_authorized_real_literal_chunk_accepted(self):
        result = self.build([qrow(value="95%", dataset="dataset A")])
        self.assertEqual([(r["source_filename"], r["chunk_id"]) for r in result], [("a.pdf", "c1")])

    def test_unauthorized_source_rejected(self):
        result = self.build([qrow(source="b.pdf", chunk="c1", value="95%")], allowed=("a.pdf",))
        self.assertEqual(result, [])

    def test_nonexistent_chunk_and_invalid_pair_rejected(self):
        self.assertEqual(self.build([qrow(chunk="missing")]), [])
        result = build_quantitative_chunk_candidates(
            self.df,
            [qrow(value="95%")],
            allowed_papers=["a.pdf"],
            valid_source_chunk_pairs={("a.pdf", "c2")},
        )
        self.assertEqual(result, [])

    def test_same_chunk_id_in_different_papers_is_documentally_distinct(self):
        result = self.build([
            qrow(source="a.pdf", chunk="c1", value="95%", dataset="dataset A"),
            qrow(source="b.pdf", chunk="c1", value="95%", dataset="dataset B"),
        ])
        self.assertEqual([(r["source_filename"], r["chunk_id"]) for r in result], [("a.pdf", "c1"), ("b.pdf", "c1")])

    def test_multiple_rows_group_into_one_chunk(self):
        result = self.build([
            qrow(chunk="c2", value="1.3", metric="rmse", row_id="r1"),
            qrow(chunk="c2", value="10%", metric="mape", row_id="r2"),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["quantitative_values"], ["1.3", "10%"])
        self.assertEqual(result[0]["quantitative_row_ids"], ["r1", "r2"])

    def test_literal_false_positives_rejected(self):
        local = chunks(("a.pdf", "x", "Values were 11.3, 100 and 70."))
        valid = {("a.pdf", "x")}
        for value in ("1.3", "10", "7"):
            result = build_quantitative_chunk_candidates(
                local, [qrow(chunk="x", value=value)], allowed_papers=["a.pdf"], valid_source_chunk_pairs=valid
            )
            self.assertEqual(result, [])

    def test_percentage_spacing_and_decimal_comma(self):
        result = self.build([
            qrow(chunk="c2", value="10%"),
            qrow(chunk="c2", value="1,3", metric="rmse"),
        ])
        self.assertEqual(result[0]["quantitative_values"], ["1,3", "10%"])

    def test_percentage_without_symbol_rejected(self):
        local = chunks(("a.pdf", "x", "The reported score was 10."))
        result = build_quantitative_chunk_candidates(
            local, [qrow(chunk="x", value="10%")], allowed_papers=["a.pdf"], valid_source_chunk_pairs={("a.pdf", "x")}
        )
        self.assertEqual(result, [])

    def test_adjacent_unit_is_allowed_for_non_percentage(self):
        local = chunks(("a.pdf", "x", "The delay was 10 ms."))
        result = build_quantitative_chunk_candidates(
            local, [qrow(chunk="x", value="10", unit="ms")], allowed_papers=["a.pdf"], valid_source_chunk_pairs={("a.pdf", "x")}
        )
        self.assertEqual(len(result), 1)

    def test_max_rows_is_deterministic(self):
        rows = [qrow(chunk="c1", value="95%"), qrow(chunk="c2", value="1.3")]
        result = build_quantitative_chunk_candidates(
            self.df, rows, allowed_papers=["a.pdf"], valid_source_chunk_pairs=self.valid, max_quantitative_rows_per_section=1
        )
        self.assertEqual([(r["chunk_id"]) for r in result], ["c1"])


class GreedySelectionTests(unittest.TestCase):
    def setUp(self):
        self.df = chunks(
            ("a.pdf", "a1", "RMSE 1.0 and MAE 2.0 on dataset A."),
            ("a.pdf", "a2", "RMSE 1.0 on dataset A."),
            ("b.pdf", "b1", "Accuracy 95% on dataset B."),
            ("c.pdf", "c1", "Accuracy 95% on dataset C."),
            ("a.pdf", "a3", "Background evidence without numbers."),
            ("b.pdf", "b2", "Second background evidence."),
            ("c.pdf", "c2", "Third background evidence."),
        )
        self.valid = {(r.source_filename, r.chunk_id) for r in self.df.itertuples()}
        self.allowed = ["a.pdf", "b.pdf", "c.pdf"]

    def execute_case(self, base_rows, qrows, **overrides):
        params = dict(
            allowed_papers=self.allowed,
            top_k_evidence_per_section=4,
            quantitative_evidence_quota=2,
            max_evidence_chars=10000,
            max_candidates_per_source=4,
            valid_source_chunk_pairs=self.valid,
        )
        params.update(overrides)
        return augment_evidence_with_quantitative_chunks_greedy(base_rows, self.df, qrows, **params)

    def test_multi_value_chunk_wins_and_redundant_chunk_is_not_added(self):
        rows = [
            qrow(chunk="a1", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(chunk="a1", value="2.0", metric="mae", dataset="dataset A"),
            qrow(chunk="a2", value="1.0", metric="rmse", dataset="dataset A"),
        ]
        result = self.execute_case([], rows)
        self.assertEqual(result[0]["chunk_id"], "a1")
        self.assertEqual(result[0]["quantitative_marginal_gain"], 2)
        self.assertNotIn("a2", [r["chunk_id"] for r in result])

    def test_second_chunk_with_new_coverage_is_added(self):
        rows = [
            qrow(chunk="a1", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        result = self.execute_case([], rows)
        self.assertEqual({r["chunk_id"] for r in result}, {"a1", "b1"})

    def test_metric_diversity_breaks_equal_marginal_tie(self):
        rows = [
            qrow(chunk="a1", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(chunk="a1", value="2.0", metric="rmse", dataset="dataset A"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        # a1 gain 2 but one metric; b1 gain 1, so gain remains primary.
        result = self.execute_case([], rows, quantitative_evidence_quota=1)
        self.assertEqual(result[0]["chunk_id"], "a1")

    def test_source_diversity_breaks_equal_tie(self):
        base_rows = [base("a.pdf", "a3", "Background evidence without numbers.")]
        rows = [
            qrow(chunk="a2", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        self.assertEqual(result[1]["source_filename"], "b.pdf")

    def test_stable_source_chunk_tiebreak(self):
        rows = [
            qrow(source="c.pdf", chunk="c1", value="95%", metric="accuracy", dataset="dataset C"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        result = self.execute_case([], rows, quantitative_evidence_quota=1)
        self.assertEqual((result[0]["source_filename"], result[0]["chunk_id"]), ("b.pdf", "b1"))

    def test_same_value_different_datasets_have_distinct_coverage_keys(self):
        rows = [
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
            qrow(source="c.pdf", chunk="c1", value="95%", metric="accuracy", dataset="dataset C"),
        ]
        result = self.execute_case([], rows)
        keys = {key for row in result for key in row["quantitative_coverage_keys"]}
        self.assertEqual(len(keys), 2)

    def test_same_value_different_metrics_have_distinct_keys(self):
        local = chunks(("a.pdf", "x", "Accuracy 95% and recall 95% on dataset A."))
        valid = {("a.pdf", "x")}
        result = augment_evidence_with_quantitative_chunks_greedy(
            [], local,
            [qrow(chunk="x", value="95%", metric="accuracy", dataset="dataset A"), qrow(chunk="x", value="95%", metric="recall", dataset="dataset A")],
            allowed_papers=["a.pdf"], top_k_evidence_per_section=2, quantitative_evidence_quota=1,
            max_evidence_chars=1000, max_candidates_per_source=2, valid_source_chunk_pairs=valid,
        )
        self.assertEqual(len(result[0]["quantitative_coverage_keys"]), 2)

    def test_all_values_already_covered_by_same_base_chunk(self):
        base_rows = [base("a.pdf", "a1", "RMSE 1.0 and MAE 2.0 on dataset A.")]
        rows = [qrow(chunk="a1", value="1.0", metric="rmse", dataset="dataset A")]
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        self.assertEqual(len(result), 1)
        self.assertNotEqual(result[0].get("selection_bucket"), "quantitative_greedy")

    def test_partial_base_coverage_adds_only_new_chunk(self):
        base_rows = [base("a.pdf", "a2", "RMSE 1.0 on dataset A.")]
        rows = [
            qrow(chunk="a2", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        self.assertEqual([r["chunk_id"] for r in result], ["a2", "b1"])

    def test_same_value_in_other_paper_does_not_cover(self):
        base_rows = [base("c.pdf", "c1", "Accuracy 95% on dataset C.")]
        rows = [qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B")]
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        self.assertEqual([r["source_filename"] for r in result], ["c.pdf", "b.pdf"])

    def test_contradictory_context_is_not_counted_as_covered(self):
        base_rows = [base("b.pdf", "b1", "Accuracy 95% on dataset B.")]
        rows = [qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset C")]
        # Candidate row itself cannot be built because dataset context is metadata only; literal is present,
        # but base coverage requires dataset C to appear, so no false coverage key is introduced.
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        self.assertEqual(len(result), 1)

    def test_quantitative_chunk_already_in_base_is_not_duplicated(self):
        base_rows = [base("a.pdf", "a1", "RMSE 1.0 and MAE 2.0 on dataset A.")]
        result = self.execute_case(base_rows, [qrow(chunk="a1", value="1.0")])
        self.assertEqual(len({(r["source_filename"], r["chunk_id"]) for r in result}), len(result))

    def test_quota_zero_returns_hybrid_up_to_top_k(self):
        base_rows = [base("a.pdf", "a3", "Background evidence without numbers."), base("b.pdf", "b2", "Second background evidence.")]
        result = self.execute_case(base_rows, [qrow(chunk="a1", value="1.0")], quantitative_evidence_quota=0)
        self.assertEqual([r["chunk_id"] for r in result], ["a3", "b2"])

    def test_unused_quantitative_slots_return_to_hybrid(self):
        base_rows = [
            base("a.pdf", "a3", "Background evidence without numbers."),
            base("b.pdf", "b2", "Second background evidence."),
            base("c.pdf", "c2", "Third background evidence."),
        ]
        result = self.execute_case(base_rows, [], top_k_evidence_per_section=3, quantitative_evidence_quota=2)
        self.assertEqual([r["chunk_id"] for r in result], ["a3", "b2", "c2"])

    def test_less_hybrid_than_base_slots(self):
        result = self.execute_case([base("a.pdf", "a3", "Background evidence without numbers.")], [qrow(source="b.pdf", chunk="b1", value="95%")])
        self.assertEqual(len(result), 2)

    def test_source_limit_applies_to_final_output(self):
        base_rows = [base("a.pdf", "a3", "Background evidence without numbers.")]
        rows = [qrow(chunk="a1", value="1.0"), qrow(chunk="a2", value="1.0")]
        result = self.execute_case(base_rows, rows, max_candidates_per_source=1)
        self.assertEqual(sum(r["source_filename"] == "a.pdf" for r in result), 1)

    def test_character_limit_skips_oversized_quantitative_and_tries_next(self):
        local = chunks(
            ("a.pdf", "big", "1.0 " + "x" * 100),
            ("b.pdf", "small", "2.0"),
        )
        valid = {("a.pdf", "big"), ("b.pdf", "small")}
        result = augment_evidence_with_quantitative_chunks_greedy(
            [], local,
            [qrow(chunk="big", value="1.0"), qrow(source="b.pdf", chunk="small", value="2.0")],
            allowed_papers=["a.pdf", "b.pdf"], top_k_evidence_per_section=2, quantitative_evidence_quota=2,
            max_evidence_chars=10, max_candidates_per_source=2, valid_source_chunk_pairs=valid,
        )
        self.assertEqual([(r["source_filename"], r["chunk_id"]) for r in result], [("b.pdf", "small")])

    def test_output_traceability_and_hybrid_metadata_preserved(self):
        base_rows = [base("a.pdf", "a3", "Background evidence without numbers.", chroma_rank=1, csv_rank=2)]
        rows = [qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B", row_id="r9")]
        result = self.execute_case(base_rows, rows, top_k_evidence_per_section=2, quantitative_evidence_quota=1)
        q = result[1]
        for field in ("source_filename", "chunk_id", "text", "selection_bucket", "selection_order", "quantitative_values", "quantitative_coverage_keys", "quantitative_marginal_gain", "quantitative_row_ids", "verification_statuses"):
            self.assertIn(field, q)
        self.assertEqual(q["selection_bucket"], "quantitative_greedy")

    def test_deterministic_repeated_execution(self):
        rows = [
            qrow(chunk="a1", value="1.0", metric="rmse", dataset="dataset A"),
            qrow(source="b.pdf", chunk="b1", value="95%", metric="accuracy", dataset="dataset B"),
        ]
        first = self.execute_case([], rows)
        second = self.execute_case([], rows)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_invalid_limits_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "quantitative_evidence_quota"):
            self.execute_case([], [], top_k_evidence_per_section=1, quantitative_evidence_quota=2)
        with self.assertRaisesRegex(ValueError, "max_evidence_chars"):
            self.execute_case([], [], max_evidence_chars=0)
        with self.assertRaisesRegex(ValueError, "max_candidates_per_source"):
            self.execute_case([], [], max_candidates_per_source=0)


if __name__ == "__main__":
    unittest.main()
