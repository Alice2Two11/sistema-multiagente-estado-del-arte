from copy import deepcopy

from src.tools.verification.resolution import resolve_multiple_correction_proposals
from src.tools.verification.validation import (
    build_provisional_verification_traceability_bundle,
    validate_claim_verification_result_contract,
)


def _tool_usage():
    return {
        "tool_names_considered": ("VERIFICATION_LLM",),
        "tool_names_selected": ("VERIFICATION_LLM",),
        "tools_considered": 1,
        "tools_selected": 1,
        "retrieval_requested": 0,
        "retrieval_rounds": 0,
        "evidence_selected": 0,
        "llm_calls": 3,
        "format_attempts": 3,
        "schema_validation_attempts": 3,
        "scientific_judgment_attempts": 3,
        "format_retries": 2,
        "schema_retries": 2,
    }


def _terminal_unresolved_claim(*, judgment_status="BLOCKED"):
    return {
        "claim_id": "S2_C6",
        "claim_uid": "uid-S2_C6",
        "claim_type": "SUBSTANTIVE_FACTUAL",
        "scientific_judgment_required": True,
        "execution_status": "COMPLETED",
        "technical_status": "LLM_VALIDATION_ATTEMPTS_EXHAUSTED",
        "technical_issue_codes": ("LLM_VALIDATION_ATTEMPTS_EXHAUSTED",),
        "scientific_judgment_status": judgment_status,
        "scientific_verdict": "NOT_EVALUATED",
        "support_level": "NONE",
        "deterministic_issue_codes": (),
        "semantic_issue_codes": (),
        "eligible_evidence": (),
        "deterministically_discarded_evidence": (),
        "evidence_used": (),
        "evidence_rejected": (),
        "contradiction_assessment": {"type": "NONE", "evidence_ids": ()},
        "numeric_assessment": "NOT_APPLICABLE",
        "attribution_assessment": "NOT_APPLICABLE",
        "extrapolation_assessment": "NOT_APPLICABLE",
        "hallucination_risk": "MEDIUM",
        "llm_correction_recommendation": False,
        "final_correction_eligibility": "MANUAL_REVIEW_REQUIRED",
        "manual_review_required": True,
        "reason_codes": (),
        "tool_usage": _tool_usage(),
        "decision_trace": ("SCIENTIFIC_JUDGMENT_BLOCKED",),
        "raw_attempts": (),
        "result_contract_valid": True,
        "scientific_validation_ok": False,
        "validation_ok": True,
    }


def _aggregation_input(claim):
    return {
        "claim_verification_records": ({"section_id": "S2", "claim_verification_result": claim},),
        "correction_proposals": (),
        "correction_reverification_inputs": (),
        "correction_precheck_results": (),
        "independent_reverification_results": (),
        "before_after_comparison_results": (),
        "policy_versions": {"verification": "v1"},
        "schema_versions": {"bundle": "v1"},
        "additional_llm_calls": 0,
        "additional_retrieval_rounds": 0,
        "correction_applied": False,
        "official_artifacts_created": False,
    }


def test_terminal_llm_exhaustion_is_contract_valid_unresolved_science():
    blocked = validate_claim_verification_result_contract(_terminal_unresolved_claim())
    assert blocked["scientific_verdict"] == "NOT_EVALUATED"
    assert blocked["manual_review_required"] is True

    # Some historical terminal serializers used COMPLETED to mean execution ended.
    completed_terminal = validate_claim_verification_result_contract(
        _terminal_unresolved_claim(judgment_status="COMPLETED")
    )
    assert completed_terminal["technical_status"] == "LLM_VALIDATION_ATTEMPTS_EXHAUSTED"


def test_valid_bundle_with_manual_unresolved_claim_is_partial_not_invalid():
    bundle = build_provisional_verification_traceability_bundle(
        _aggregation_input(_terminal_unresolved_claim())
    )
    assert bundle.aggregation_status == "PARTIAL"
    assert bundle.metrics_status == "COMPUTED"
    assert bundle.normalized_bundle_status == "COMPUTED"
    assert bundle.normalized_bundle_fingerprint is not None
    assert "PARTIAL_MANUAL_REVIEW_REQUIRED" in bundle.partial_reason_codes
    assert "AGGREGATION_ROW_MANUAL_REVIEW_REQUIRED" in bundle.aggregation_warnings
    assert not any(code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID") for code in bundle.aggregation_issue_codes)

    resolution = resolve_multiple_correction_proposals(bundle)
    assert resolution.aggregation_status == "PARTIAL"
    assert resolution.resolution_status == "PARTIAL"
    assert resolution.eligible_for_07c is False


def test_really_invalid_claim_record_stays_invalid_and_resolution_blocked():
    broken = deepcopy(_terminal_unresolved_claim())
    broken["manual_review_required"] = False
    bundle = build_provisional_verification_traceability_bundle(_aggregation_input(broken))
    assert bundle.aggregation_status == "INVALID"
    assert bundle.metrics_status == "NOT_COMPUTED"
    assert bundle.normalized_bundle_status == "NOT_COMPUTABLE"
    assert bundle.normalized_bundle_fingerprint is None
    assert any(code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID") for code in bundle.aggregation_issue_codes)

    resolution = resolve_multiple_correction_proposals(bundle)
    assert resolution.resolution_status == "BLOCKED"
    assert resolution.resolution_issue_codes == ("MULTI_PROPOSAL_BLOCKED_INVALID_BUNDLE",)

import pytest


@pytest.fixture(params=("S2_C6", "S3_C2"))
def sanitized_real_terminal_fixture(request):
    """Sanitized shape observed for the two rejected runtime records."""
    row = _terminal_unresolved_claim(judgment_status="COMPLETED")
    row["claim_id"] = request.param
    return row


def test_sanitized_s2_c6_and_s3_c2_match_only_admitted_terminal_combination(
    sanitized_real_terminal_fixture,
):
    row = sanitized_real_terminal_fixture
    assert row["technical_status"] == "LLM_VALIDATION_ATTEMPTS_EXHAUSTED"
    assert row["technical_issue_codes"] == ("LLM_VALIDATION_ATTEMPTS_EXHAUSTED",)
    assert row["scientific_judgment_status"] == "COMPLETED"
    assert row["scientific_verdict"] == "NOT_EVALUATED"
    assert row["support_level"] == "NONE"
    assert row["manual_review_required"] is True
    assert row["final_correction_eligibility"] == "MANUAL_REVIEW_REQUIRED"
    assert validate_claim_verification_result_contract(row)["claim_id"] == row["claim_id"]


def test_unknown_technical_status_with_manual_fields_is_invalid_and_bundle_invalid():
    unknown = _terminal_unresolved_claim(judgment_status="COMPLETED")
    unknown["technical_status"] = "UNKNOWN_TERMINAL_STATE"
    unknown["technical_issue_codes"] = ("LLM_VALIDATION_ATTEMPTS_EXHAUSTED",)
    with pytest.raises(ValueError, match="CLAIM_VERIFICATION_TECHNICAL_STATUS_UNKNOWN"):
        validate_claim_verification_result_contract(unknown)

    bundle = build_provisional_verification_traceability_bundle(_aggregation_input(unknown))
    assert bundle.aggregation_status == "INVALID"
    assert any(code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID") for code in bundle.aggregation_issue_codes)


def test_registered_but_not_policy_admitted_completed_status_stays_invalid():
    row = _terminal_unresolved_claim(judgment_status="COMPLETED")
    row["technical_status"] = "LLM_UNAVAILABLE"
    row["technical_issue_codes"] = ("LLM_UNAVAILABLE",)
    with pytest.raises(ValueError, match="CLAIM_VERIFICATION_TECHNICAL_JUDGMENT_INCOHERENT"):
        validate_claim_verification_result_contract(row)
