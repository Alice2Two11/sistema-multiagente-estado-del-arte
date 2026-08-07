from __future__ import annotations

import unittest

from src.config.verification_policy_config import (
    REVERIFICATION_APPLICATION_ORDER_FIELDS,
    REVERIFICATION_PROCESS_NAME,
    get_verification_input_policy,
)
from src.tools.verification.traceability import (
    TRACEABILITY_PROVISIONAL_UNITS,
    CorrectionReverificationInputContract,
    CorrectionReverificationResultContract,
)
from src.tools.verification.validation import (
    validate_correction_reverification_input_contract,
    validate_correction_reverification_result_contract,
    validate_reverification_block_matrix,
)


class Phase61ContractsTests(unittest.TestCase):
    def valid_input(self):
        policy = get_verification_input_policy()
        return {
            "correction_id": "CORR-1",
            "claim_id": "C1",
            "section_id": "S1",
            "original_claim_text": "El modelo obtuvo 94 %.",
            "proposed_claim_text": "El modelo obtuvo 95 %.",
            "source_verdict": "PARTIALLY_SUPPORTED",
            "source_issue_codes": ["UNSUPPORTED_NUMERIC_VALUE"],
            "target_issue_codes": ["UNSUPPORTED_NUMERIC_VALUE"],
            "correction_action_type": "REPLACE_NUMERIC_VALUE",
            "claim_span_in_section": {
                "coordinate_base": "SECTION_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
                "base_text_fingerprint": "section-fp", "start": 0, "end": 21,
                "text": "El modelo obtuvo 94 %.",
            },
            "target_span_in_claim": {
                "coordinate_base": "CLAIM_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
                "base_text_fingerprint": "original-fp", "start": 17, "end": 21, "text": "94 %",
            },
            "replacement_text": "95 %",
            "evidence_ids": ["E1"],
            "authorized_evidence": [{"evidence_id": "E1", "authorized_for_section": True}],
            "correction_validation_result": {"proposal_status": "ACCEPTED_FOR_REVERIFICATION", "correction_applied": False},
            "proposal_fingerprint": "proposal-fp",
            "proposed_claim_text_fingerprint": "proposed-text-fp",
            "original_claim_fingerprint": "original-fp",
            "original_section_fingerprint": "section-fp",
            "base_claim_fingerprint": "original-fp",
            "base_section_fingerprint": "section-fp",
            "application_order_key": ["S1", 0, 17, "CORR-1"],
            "attempt_context": {"attempt_number": 1},
            "policy": policy,
        }

    def valid_result(self):
        return {
            "correction_id": "CORR-1",
            "claim_id": "C1",
            "section_id": "S1",
            "reverification_execution_status": "COMPLETED",
            "scientific_outcome": "SUPPORTED",
            "acceptance_decision": "ACCEPT_FOR_07C",
            "original_verdict": "PARTIALLY_SUPPORTED",
            "proposed_verdict": "SUPPORTED",
            "original_issue_codes": ["UNSUPPORTED_NUMERIC_VALUE"],
            "remaining_issue_codes": [],
            "resolved_issue_codes": ["UNSUPPORTED_NUMERIC_VALUE"],
            "new_issue_codes": [],
            "evidence_used": ["E1"],
            "supported_meaning_preserved": True,
            "intended_semantic_change_valid": True,
            "unintended_semantic_change_absent": True,
            "scope_change_valid": True,
            "numeric_change_valid": True,
            "attribution_change_valid": True,
            "citation_change_valid": True,
            "hallucination_risk_before": "HIGH",
            "hallucination_risk_after": "LOW",
            "hallucination_risk_delta": "REDUCED",
            "risk_policy_version": "AGENT07_HALLUCINATION_RISK_V1",
            "risk_before_recomputed": True,
            "risk_after_computed": True,
            "manual_review_required": False,
            "reason_codes": ["TARGET_ISSUE_RESOLVED", "RISK_REDUCED"],
            "technical_issue_codes": [],
            "tool_usage": {"reverification_retrieval_rounds": 0},
            "decision_trace": [],
            "raw_attempts": [],
            "result_contract_valid": True,
            "correction_applied": False,
        }

    def test_default_policy_freezes_virtual_process(self):
        policy = get_verification_input_policy()
        self.assertEqual(policy["reverification_process_name"], REVERIFICATION_PROCESS_NAME)
        self.assertEqual(policy["reverification_retrieval_rounds"], 0)
        self.assertTrue(policy["require_frozen_reverification_evidence"])

    def test_policy_rejects_nonzero_retrieval_rounds(self):
        with self.assertRaisesRegex(ValueError, "must_be_zero"):
            get_verification_input_policy({"reverification_retrieval_rounds": 1})

    def test_policy_acceptance_limit_is_configurable(self):
        policy = get_verification_input_policy({"max_accepted_proposals_per_claim": 2})
        self.assertEqual(policy["max_accepted_proposals_per_claim"], 2)

    def test_policy_rejects_zero_acceptance_limit(self):
        with self.assertRaises(ValueError):
            get_verification_input_policy({"max_accepted_proposals_per_claim": 0})

    def test_application_order_is_frozen(self):
        self.assertEqual(
            get_verification_input_policy()["reverification_application_order_fields"],
            REVERIFICATION_APPLICATION_ORDER_FIELDS,
        )

    def test_valid_input_contract(self):
        value = validate_correction_reverification_input_contract(self.valid_input())
        self.assertEqual(value["target_issue_codes"], ("UNSUPPORTED_NUMERIC_VALUE",))

    def test_proposal_and_proposed_text_fingerprints_remain_separate(self):
        value = validate_correction_reverification_input_contract(self.valid_input())
        self.assertEqual(value["proposal_fingerprint"], "proposal-fp")
        self.assertEqual(value["proposed_claim_text_fingerprint"], "proposed-text-fp")

    def test_target_issue_must_exist_in_original_issues(self):
        value = self.valid_input()
        value["target_issue_codes"] = ["ATTRIBUTION_ERROR"]
        with self.assertRaisesRegex(ValueError, "TARGET_ISSUE_CODE_NOT_PRESENT"):
            validate_correction_reverification_input_contract(value)

    def test_target_issue_cannot_be_empty(self):
        value = self.valid_input()
        value["target_issue_codes"] = []
        with self.assertRaisesRegex(ValueError, "target_issue_codes:empty"):
            validate_correction_reverification_input_contract(value)

    def test_evidence_must_be_frozen_and_authorized_context(self):
        value = self.valid_input()
        value["evidence_ids"] = ["E2"]
        with self.assertRaisesRegex(ValueError, "REVERIFICATION_EVIDENCE_NOT_FROZEN"):
            validate_correction_reverification_input_contract(value)

    def test_application_order_key_is_normalized(self):
        value = validate_correction_reverification_input_contract(self.valid_input())
        self.assertEqual(value["application_order_key"], ("S1", 0, 17, "CORR-1"))

    def test_non_authorized_frozen_evidence_is_rejected(self):
        value = self.valid_input()
        value["authorized_evidence"][0]["authorized_for_section"] = False
        with self.assertRaisesRegex(ValueError, "REVERIFICATION_EVIDENCE_NOT_AUTHORIZED"):
            validate_correction_reverification_input_contract(value)

    def test_proposal_status_must_be_accepted_for_reverification(self):
        value = self.valid_input()
        value["correction_validation_result"]["proposal_status"] = "REJECTED"
        with self.assertRaisesRegex(ValueError, "PROPOSAL_STATUS_NOT_ALLOWED"):
            validate_correction_reverification_input_contract(value)

    def test_result_risk_policy_version_must_match(self):
        value = self.valid_result()
        value["risk_policy_version"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "RISK_POLICY_VERSION_MISMATCH"):
            validate_correction_reverification_result_contract(value)

    def test_valid_result_contract(self):
        value = validate_correction_reverification_result_contract(self.valid_result())
        self.assertEqual(value["acceptance_decision"], "ACCEPT_FOR_07C")

    def test_result_rejects_physical_application(self):
        value = self.valid_result()
        value["correction_applied"] = True
        with self.assertRaisesRegex(ValueError, "PHYSICAL_APPLICATION_FORBIDDEN"):
            validate_correction_reverification_result_contract(value)

    def test_execution_scientific_and_acceptance_are_independent_fields(self):
        value = self.valid_result()
        value.update({
            "reverification_execution_status": "COMPLETED",
            "scientific_outcome": "AMBIGUOUS",
            "acceptance_decision": "DEFER_TO_MANUAL_REVIEW",
            "manual_review_required": True,
            "reason_codes": ["SCIENTIFIC_AMBIGUITY"],
        })
        result = validate_correction_reverification_result_contract(value)
        self.assertEqual(result["scientific_outcome"], "AMBIGUOUS")

    def test_unknown_acceptance_decision_rejected(self):
        value = self.valid_result()
        value["acceptance_decision"] = "ACCEPT_CORRECTION"
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_DECISION_UNKNOWN"):
            validate_correction_reverification_result_contract(value)

    def test_unknown_reason_code_rejected(self):
        value = self.valid_result()
        value["reason_codes"] = ["INVENTED"]
        with self.assertRaisesRegex(ValueError, "REASON_CODE_UNKNOWN"):
            validate_correction_reverification_result_contract(value)

    def test_contractual_block_matrix(self):
        validate_reverification_block_matrix(
            category="CONTRACTUAL_INCOMPATIBILITY",
            execution_status="BLOCKED",
            acceptance_decision="REJECT_PROPOSAL",
        )

    def test_temporary_dependency_block_matrix(self):
        validate_reverification_block_matrix(
            category="TEMPORARY_TECHNICAL_DEPENDENCY",
            execution_status="BLOCKED",
            acceptance_decision="DEFER_TO_MANUAL_REVIEW",
        )

    def test_negative_scientific_result_matrix(self):
        validate_reverification_block_matrix(
            category="NEGATIVE_SCIENTIFIC_RESULT",
            execution_status="COMPLETED",
            acceptance_decision="REJECT_PROPOSAL",
        )

    def test_ambiguity_matrix(self):
        validate_reverification_block_matrix(
            category="SCIENTIFIC_AMBIGUITY",
            execution_status="COMPLETED",
            acceptance_decision="DEFER_TO_MANUAL_REVIEW",
        )

    def test_block_matrix_rejects_wrong_pair(self):
        with self.assertRaisesRegex(ValueError, "BLOCK_MATRIX_VIOLATION"):
            validate_reverification_block_matrix(
                category="CONTRACTUAL_INCOMPATIBILITY",
                execution_status="BLOCKED",
                acceptance_decision="DEFER_TO_MANUAL_REVIEW",
            )

    def test_traceability_units_do_not_double_count(self):
        self.assertEqual(TRACEABILITY_PROVISIONAL_UNITS["claims"], ("claim_id",))
        self.assertEqual(TRACEABILITY_PROVISIONAL_UNITS["issues"], ("claim_id", "issue_code"))
        self.assertEqual(
            TRACEABILITY_PROVISIONAL_UNITS["evidence"],
            ("claim_id", "correction_id", "evidence_id"),
        )

    def test_input_dataclass_round_trip(self):
        data = self.valid_input()
        obj = CorrectionReverificationInputContract(**data)
        validated = validate_correction_reverification_input_contract(obj.to_dict())
        self.assertEqual(validated["claim_id"], "C1")

    def test_result_dataclass_round_trip(self):
        data = self.valid_result()
        obj = CorrectionReverificationResultContract(**data)
        validated = validate_correction_reverification_result_contract(obj.to_dict())
        self.assertEqual(validated["technical_issue_codes"], ())

    def test_separate_spans_are_required(self):
        value = self.valid_input()
        value.pop("target_span_in_claim")
        with self.assertRaisesRegex(ValueError, "target_span_in_claim"):
            validate_correction_reverification_input_contract(value)

    def test_wrong_span_base_rejected(self):
        value = self.valid_input()
        value["target_span_in_claim"]["coordinate_base"] = "SECTION_TEXT"
        with self.assertRaisesRegex(ValueError, "coordinate_base"):
            validate_correction_reverification_input_contract(value)

    def test_base_fingerprint_mismatch_rejected(self):
        value = self.valid_input()
        value["base_claim_fingerprint"] = "other"
        value["target_span_in_claim"]["base_text_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "BASE_CLAIM_FINGERPRINT_MISMATCH"):
            validate_correction_reverification_input_contract(value)

    def test_application_order_key_mismatch_rejected(self):
        value = self.valid_input()
        value["application_order_key"] = ["S1", 1, 17, "CORR-1"]
        with self.assertRaisesRegex(ValueError, "APPLICATION_ORDER_KEY_MISMATCH"):
            validate_correction_reverification_input_contract(value)

    def test_correction_applied_must_be_present(self):
        value = self.valid_input()
        value["correction_validation_result"].pop("correction_applied")
        with self.assertRaisesRegex(ValueError, "CORRECTION_APPLIED_REQUIRED"):
            validate_correction_reverification_input_contract(value)

    def test_empty_evidence_rejected(self):
        value = self.valid_input()
        value["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "EVIDENCE_REQUIRED"):
            validate_correction_reverification_input_contract(value)

    def test_duplicate_authorized_evidence_id_rejected(self):
        value = self.valid_input()
        value["authorized_evidence"].append({"evidence_id": "E1", "authorized_for_section": True})
        with self.assertRaisesRegex(ValueError, "AUTHORIZED_EVIDENCE_ID_DUPLICATE"):
            validate_correction_reverification_input_contract(value)

    def test_blocked_acceptance_rejected_in_main_validator(self):
        value = self.valid_result()
        value["reverification_execution_status"] = "BLOCKED"
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_REQUIRES_COMPLETED|BLOCKED_CANNOT_ACCEPT"):
            validate_correction_reverification_result_contract(value)

    def test_not_evaluated_acceptance_rejected(self):
        value = self.valid_result()
        value["scientific_outcome"] = "NOT_EVALUATED"
        with self.assertRaisesRegex(ValueError, "EVALUATED_OUTCOME"):
            validate_correction_reverification_result_contract(value)

    def test_manual_review_acceptance_rejected(self):
        value = self.valid_result()
        value["manual_review_required"] = True
        with self.assertRaisesRegex(ValueError, "MANUAL_REVIEW"):
            validate_correction_reverification_result_contract(value)

    def test_increased_risk_acceptance_rejected(self):
        value = self.valid_result()
        value["hallucination_risk_delta"] = "INCREASED"
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_RISK_INVALID"):
            validate_correction_reverification_result_contract(value)

    def test_not_comparable_risk_acceptance_rejected(self):
        value = self.valid_result()
        value["hallucination_risk_delta"] = "NOT_COMPARABLE"
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_RISK_INVALID"):
            validate_correction_reverification_result_contract(value)

    def test_result_contract_valid_false_rejected(self):
        value = self.valid_result()
        value["result_contract_valid"] = False
        with self.assertRaisesRegex(ValueError, "RESULT_CONTRACT_NOT_VALID"):
            validate_correction_reverification_result_contract(value)


if __name__ == "__main__":
    unittest.main()
