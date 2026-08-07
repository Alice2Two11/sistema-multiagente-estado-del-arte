from __future__ import annotations

import copy

from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.validation import (
    compare_virtual_reverification_before_after,
    validate_before_after_comparison_result_contract,
)


def valid_triplet():
    c, p, r, _ = setup_result()
    return c, p, r


def test_input_contractually_invalid_is_rejected_without_exception():
    c, p, r = valid_triplet(); c = dict(c); c.pop("claim_id")
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["reason_codes"] == ("COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_precheck_contractually_invalid_is_rejected():
    c, p, r = valid_triplet(); p = dict(p); p.pop("proposal_fingerprint")
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["reason_codes"] == ("COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",)


def test_reverification_result_contractually_invalid_is_rejected():
    c, p, r = valid_triplet(); r = dict(r); r["numeric_assessment"] = "UNKNOWN"
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


def test_missing_identity_field_rejected():
    c, p, r = valid_triplet(); r = dict(r); r["proposal_fingerprint"] = ""
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_source_issues_modified_after_snapshot_rejected():
    c, p, r = valid_triplet(); c = copy.deepcopy(c); c["source_issue_codes"].append("INVALID_CITATION")
    out = compare_virtual_reverification_before_after(c, p, r)
    assert "COMPARISON_CONTEXT_SNAPSHOT_MISMATCH" in out["reason_codes"]


def test_target_issues_modified_after_snapshot_rejected():
    c, p, r = valid_triplet(); c = copy.deepcopy(c); c["target_issue_codes"] = ["INVALID_CITATION"]
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_policy_modified_after_snapshot_rejected():
    c, p, r = valid_triplet(); c = copy.deepcopy(c); c["policy"]["max_reverification_llm_attempts"] = 9
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] in {"REJECT_PROPOSAL", "DEFER_TO_MANUAL_REVIEW"}


def test_evidence_modified_after_snapshot_rejected():
    c, p, r = valid_triplet(); c = copy.deepcopy(c); c["authorized_evidence"][0]["canonical_text"] += " alterado"
    out = compare_virtual_reverification_before_after(c, p, r)
    assert "COMPARISON_CONTEXT_SNAPSHOT_MISMATCH" in out["reason_codes"]


def test_old_context_fingerprint_with_same_identity_strings_rejected():
    c, p, r = valid_triplet(); p = dict(p); p["reverification_context_fingerprint"] = "0" * 64
    r = dict(r); r["reverification_context_fingerprint"] = "0" * 64
    out = compare_virtual_reverification_before_after(c, p, r)
    assert "COMPARISON_CONTEXT_SNAPSHOT_MISMATCH" in out["reason_codes"]


def test_reported_resolution_mismatch_never_accepts():
    c, p, r = valid_triplet(); r = dict(r); r["target_issues_resolved_reported"] = ()
    out = compare_virtual_reverification_before_after(c, p, r)
    assert "REPORTED_RESOLUTION_MISMATCH" in out["reason_codes"]
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"


def test_unknown_reason_code_rejected_by_result_contract():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    out = dict(out); out["reason_codes"] = ("UNKNOWN",)
    try:
        validate_before_after_comparison_result_contract(out)
    except ValueError as e:
        assert str(e) == "COMPARISON_UNKNOWN_REASON_CODE"
    else: raise AssertionError("expected failure")


def test_unknown_technical_code_rejected_by_result_contract():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    out = dict(out); out["technical_issue_codes"] = ("UNKNOWN",)
    try:
        validate_before_after_comparison_result_contract(out)
    except ValueError as e:
        assert str(e) == "COMPARISON_UNKNOWN_TECHNICAL_ISSUE_CODE"
    else: raise AssertionError("expected failure")


def test_false_issue_partition_rejected():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    out = dict(out); out["resolved_issue_codes"] = ()
    try:
        validate_before_after_comparison_result_contract(out)
    except ValueError as e:
        assert str(e) == "COMPARISON_ISSUE_PARTITION_INVALID"
    else: raise AssertionError("expected failure")


def test_unknown_assessment_rejected():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    out = dict(out); out["numeric_assessment"] = "UNKNOWN"
    try: validate_before_after_comparison_result_contract(out)
    except ValueError: pass
    else: raise AssertionError("expected failure")


def test_unknown_risk_rejected():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    out = dict(out); out["hallucination_risk_before"] = "UNKNOWN"
    try: validate_before_after_comparison_result_contract(out)
    except ValueError: pass
    else: raise AssertionError("expected failure")


def test_invalid_policy_returns_auditable_defer():
    c, p, r = valid_triplet(); c = copy.deepcopy(c); c["policy"]["reverification_comparison_risk_policy_version"] = "BAD"
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert out["additional_llm_calls"] == 0


def test_missing_result_is_deferred_as_technical_dependency():
    c, p, _ = valid_triplet()
    out = compare_virtual_reverification_before_after(c, p, {})
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert "COMPARISON_RESULT_ABSENT" in out["technical_issue_codes"]


def test_mixed_identity_is_rejected():
    c, p, r = valid_triplet(); r = dict(r); r["claim_id"] = "OTHER"
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_no_calls_retrieval_or_application():
    c, p, r = valid_triplet(); out = compare_virtual_reverification_before_after(c, p, r)
    assert out["additional_llm_calls"] == 0
    assert out["retrieval_rounds"] == 0
    assert out["correction_applied"] is False
