from __future__ import annotations

import copy
import unittest

from src.config.verification_policy_config import (
    REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES,
    REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT,
    get_verification_input_policy,
)
from src.tools.verification.validation import validate_independent_reverification_response


class Phase63SVocabularyAndSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = get_verification_input_policy()
        self.context = {
            "correction_id": "CORR-1",
            "claim_id": "CLAIM-1",
            "correction_action_type": "REPLACE_NUMERIC_VALUE",
            "allowed_evidence_ids": ("E1",),
            "target_issue_codes": ("UNSUPPORTED_NUMERIC_VALUE",),
            "policy": self.policy,
        }

    def response(self, *, verdict="SUPPORTED", support="STRONG", observed=None, evidence=None):
        return {
            "correction_id": "CORR-1",
            "claim_id": "CLAIM-1",
            "proposed_verdict": verdict,
            "support_level": support,
            "evidence_ids_used": ["E1"] if evidence is None else evidence,
            "observed_issue_codes": [] if observed is None else observed,
            "target_issues_resolved": ["UNSUPPORTED_NUMERIC_VALUE"],
            "supported_meaning_preserved": True,
            "intended_semantic_change_valid": True,
            "unintended_semantic_change_absent": True,
            "scope_assessment": "NOT_APPLICABLE",
            "numeric_assessment": "VALID",
            "attribution_assessment": "NOT_APPLICABLE",
            "citation_assessment": "NOT_APPLICABLE",
            "manual_review_recommended": False,
            "reason_codes": ["TARGET_ISSUE_APPEARS_RESOLVED"],
            "rationale": "La evidencia autorizada respalda el claim virtual.",
            "confidence": 0.9,
        }

    def assert_code(self, code, response):
        with self.assertRaisesRegex(ValueError, code):
            validate_independent_reverification_response(response, context=self.context)

    def test_technical_code_rejected_as_observed_issue(self):
        self.assert_code("REVERIFICATION_UNKNOWN_ISSUE_CODE", self.response(observed=["LLM_INVOCATION_FAILED"]))

    def test_retrieval_code_rejected_as_observed_issue(self):
        self.assert_code("REVERIFICATION_UNKNOWN_ISSUE_CODE", self.response(observed=["RETRIEVAL_TECHNICAL_BLOCKER"]))

    def test_scientific_issue_allowed(self):
        r = self.response(verdict="PARTIALLY_SUPPORTED", support="PARTIAL", observed=["PARTIAL_SUPPORT"])
        r["target_issues_resolved"] = []
        out = validate_independent_reverification_response(r, context=self.context)
        self.assertEqual(out["observed_issue_codes"], ("PARTIAL_SUPPORT",))

    def test_supported_strong_allowed(self):
        self.assertEqual(validate_independent_reverification_response(self.response(), context=self.context)["support_level"], "STRONG")

    def test_supported_partial_rejected(self):
        self.assert_code("REVERIFICATION_VERDICT_SUPPORT_LEVEL_MISMATCH", self.response(support="PARTIAL"))

    def test_partially_supported_partial_allowed(self):
        r = self.response(verdict="PARTIALLY_SUPPORTED", support="PARTIAL", observed=["PARTIAL_SUPPORT"])
        r["target_issues_resolved"] = []
        self.assertEqual(validate_independent_reverification_response(r, context=self.context)["proposed_verdict"], "PARTIALLY_SUPPORTED")

    def test_not_evaluated_strong_rejected(self):
        r = self.response(verdict="NOT_EVALUATED", support="STRONG", evidence=[])
        r["target_issues_resolved"] = []
        self.assert_code("REVERIFICATION_VERDICT_SUPPORT_LEVEL_MISMATCH", r)

    def test_not_evaluated_none_allowed(self):
        r = self.response(verdict="NOT_EVALUATED", support="NONE", evidence=[])
        r["target_issues_resolved"] = []
        self.assertEqual(validate_independent_reverification_response(r, context=self.context)["support_level"], "NONE")

    def test_not_applicable_none_allowed(self):
        r = self.response(verdict="NOT_APPLICABLE", support="NONE", evidence=[])
        r["target_issues_resolved"] = []
        self.assertEqual(validate_independent_reverification_response(r, context=self.context)["support_level"], "NONE")

    def test_matrix_is_closed_and_deterministic(self):
        self.assertEqual(set(REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT), {
            "NOT_APPLICABLE", "NOT_EVALUATED", "SUPPORTED", "PARTIALLY_SUPPORTED",
            "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE",
        })
        self.assertEqual(REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT, get_verification_input_policy()["reverification_support_level_by_verdict"])
        self.assertNotIn("RETRIEVAL_TECHNICAL_BLOCKER", REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES)


if __name__ == "__main__":
    unittest.main()
