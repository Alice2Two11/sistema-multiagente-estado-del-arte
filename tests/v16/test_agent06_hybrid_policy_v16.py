from __future__ import annotations

import re
import unittest

from src.config.draft_writing_policy_config import (
    DEFAULT_DRAFT_WRITING_POLICY,
    LEGACY_RETRIEVAL_STRATEGY,
    PLANNED_HYBRID_RETRIEVAL_STRATEGY,
    get_draft_writing_policy,
    validate_draft_writing_policy,
)


class DraftWritingHybridPolicyTests(unittest.TestCase):
    def test_defaults_keep_contractual_legacy_strategy_until_integration(self):
        policy = get_draft_writing_policy()
        expected = {
            "retrieval_strategy": LEGACY_RETRIEVAL_STRATEGY,
            "candidate_multiplier": 3,
            "top_k_evidence_per_section": 8,
            "chroma_quota": 3,
            "csv_quota": 3,
            "rrf_quota": 2,
            "rrf_k": 60,
            "quantitative_evidence_quota": 2,
            "organizational_target_words": 40,
            "max_evidence_chars": 18000,
            "max_quantitative_rows_per_section": 12,
        }
        for key, value in expected.items():
            self.assertEqual(policy[key], value)
        self.assertEqual(policy["stage_version"], "06_AGENTIC_V16_BEHAVIOR_PRESERVING")
        self.assertEqual(policy["rag_version"], "legacy_chroma_then_csv_restricted_v1")

    def test_planned_hybrid_strategy_can_be_declared_explicitly(self):
        policy = get_draft_writing_policy(
            {"retrieval_strategy": PLANNED_HYBRID_RETRIEVAL_STRATEGY}
        )
        self.assertEqual(
            policy["retrieval_strategy"],
            PLANNED_HYBRID_RETRIEVAL_STRATEGY,
        )

    def test_unsupported_strategy_is_rejected(self):
        self.assert_policy_error(
            {"retrieval_strategy": "unknown_strategy"},
            "DRAFT_POLICY_INVALID:retrieval_strategy:unsupported_strategy",
        )

    def test_all_required_policy_fields_are_present(self):
        required = {
            "retrieval_strategy",
            "candidate_multiplier",
            "chroma_quota",
            "csv_quota",
            "rrf_quota",
            "rrf_k",
            "top_k_evidence_per_section",
            "max_evidence_chars",
            "max_candidates_per_source",
            "quantitative_evidence_quota",
            "max_quantitative_rows_per_section",
            "organizational_target_words",
            "organizational_minimum_words",
            "organizational_maximum_words",
            "substantive_minimum_ratio",
            "substantive_maximum_ratio",
        }
        self.assertTrue(required.issubset(DEFAULT_DRAFT_WRITING_POLICY))

    def test_valid_override_is_applied(self):
        policy = get_draft_writing_policy(
            {
                "top_k_evidence_per_section": 6,
                "chroma_quota": 2,
                "csv_quota": 2,
                "rrf_quota": 2,
                "quantitative_evidence_quota": 1,
            }
        )
        self.assertEqual(policy["top_k_evidence_per_section"], 6)
        self.assertEqual(policy["quantitative_evidence_quota"], 1)

    def test_legacy_top_k_override_derives_compatible_quotas(self):
        policy = get_draft_writing_policy({"top_k_evidence_per_section": 3})
        self.assertEqual(
            policy["chroma_quota"] + policy["csv_quota"] + policy["rrf_quota"],
            3,
        )
        self.assertEqual(
            (policy["chroma_quota"], policy["csv_quota"], policy["rrf_quota"]),
            (1, 1, 1),
        )

    def assert_policy_error(self, overrides, expected_message):
        with self.assertRaisesRegex(ValueError, f"^{re.escape(expected_message)}$"):
            get_draft_writing_policy(overrides)

    def test_candidate_multiplier_must_be_positive(self):
        self.assert_policy_error(
            {"candidate_multiplier": 0},
            "DRAFT_POLICY_INVALID:candidate_multiplier:must_be_greater_than_or_equal_to_1",
        )

    def test_top_k_must_be_positive(self):
        self.assert_policy_error(
            {"top_k_evidence_per_section": 0},
            "DRAFT_POLICY_INVALID:top_k_evidence_per_section:must_be_greater_than_0",
        )

    def test_top_k_numeric_string_is_rejected_without_coercion(self):
        self.assert_policy_error(
            {"top_k_evidence_per_section": "8"},
            "DRAFT_POLICY_INVALID_TYPE:top_k_evidence_per_section:expected_integer",
        )

    def test_top_k_non_numeric_string_is_rejected_without_native_error(self):
        self.assert_policy_error(
            {"top_k_evidence_per_section": "abc"},
            "DRAFT_POLICY_INVALID_TYPE:top_k_evidence_per_section:expected_integer",
        )

    def test_top_k_float_is_rejected_without_coercion(self):
        self.assert_policy_error(
            {"top_k_evidence_per_section": 3.5},
            "DRAFT_POLICY_INVALID_TYPE:top_k_evidence_per_section:expected_integer",
        )

    def test_top_k_boolean_is_rejected(self):
        self.assert_policy_error(
            {"top_k_evidence_per_section": True},
            "DRAFT_POLICY_INVALID_TYPE:top_k_evidence_per_section:expected_integer",
        )

    def test_rrf_k_must_be_positive(self):
        self.assert_policy_error(
            {"rrf_k": 0},
            "DRAFT_POLICY_INVALID:rrf_k:must_be_greater_than_0",
        )

    def test_quantitative_quota_must_not_be_negative(self):
        self.assert_policy_error(
            {"quantitative_evidence_quota": -1},
            "DRAFT_POLICY_INVALID:quantitative_evidence_quota:must_be_between_0_and_top_k_evidence_per_section",
        )

    def test_quantitative_quota_must_not_exceed_top_k(self):
        self.assert_policy_error(
            {"quantitative_evidence_quota": 9},
            "DRAFT_POLICY_INVALID:quantitative_evidence_quota:must_be_between_0_and_top_k_evidence_per_section",
        )

    def test_each_retrieval_quota_must_be_nonnegative(self):
        self.assert_policy_error(
            {"chroma_quota": -1, "csv_quota": 7, "rrf_quota": 2},
            "DRAFT_POLICY_INVALID:chroma_quota:must_be_greater_than_or_equal_to_0",
        )

    def test_retrieval_quotas_must_sum_to_top_k(self):
        self.assert_policy_error(
            {"chroma_quota": 3, "csv_quota": 3, "rrf_quota": 1},
            "DRAFT_POLICY_INVALID:retrieval_quotas:chroma_quota_plus_csv_quota_plus_rrf_quota_must_equal_top_k_evidence_per_section",
        )

    def test_partial_quota_override_is_rejected(self):
        self.assert_policy_error(
            {"chroma_quota": 4},
            "DRAFT_POLICY_INVALID:retrieval_quotas:all_quota_overrides_must_be_provided_together",
        )

    def test_errors_are_reproducible(self):
        messages = []
        for _ in range(2):
            try:
                get_draft_writing_policy({"rrf_k": 0})
            except ValueError as error:
                messages.append(str(error))
        self.assertEqual(messages[0], messages[1])

    def test_policy_must_be_mapping(self):
        with self.assertRaisesRegex(
            ValueError,
            "^DRAFT_POLICY_INVALID_TYPE:policy:expected_mapping$",
        ):
            validate_draft_writing_policy([])


if __name__ == "__main__":
    unittest.main()
