from __future__ import annotations

from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.validation import compare_virtual_reverification_before_after


def triplet():
    c,p,r,_ = setup_result()
    return dict(c), dict(p), dict(r)


def assert_safe(out):
    assert out["additional_llm_calls"] == 0
    assert out["retrieval_rounds"] == 0
    assert out["correction_applied"] is False
    assert out["acceptance_decision"] in {"REJECT_PROPOSAL", "DEFER_TO_MANUAL_REVIEW"}


def test_result63_invented_verdict_returns_auditable_failure():
    c,p,r=triplet(); r["proposed_verdict"]="INVENTED"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["proposed_verdict"] == "NOT_EVALUATED"
    assert_safe(out)


def test_input_invented_source_verdict_returns_auditable_failure():
    c,p,r=triplet(); c["source_verdict"]="INVENTED"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["original_verdict"] == "NOT_EVALUATED"
    assert_safe(out)


def test_invented_action_rejects_with_closed_safe_action():
    c,p,r=triplet(); c["correction_action_type"]="INVENTED"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_CORRECTION_ACTION_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["correction_action_type"] == "NOT_AVAILABLE"
    assert_safe(out)


def test_failure_result_with_invalid_mappings_never_raises():
    out=compare_virtual_reverification_before_after({"correction_id":"x"},{"precheck_status":"PRECHECK_PASSED"},{"claim_id":"y"})
    assert_safe(out)


def test_precheck_blocked_does_not_continue_to_snapshots_and_preserves_reason():
    c,p,r=triplet(); p["precheck_status"]="PRECHECK_BLOCKED"; p["reason_codes"]=("REVERIFICATION_POLICY_UNAVAILABLE",)
    p["frozen_evidence_snapshot_fingerprint"]="stale"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_BLOCKED",)
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert out["decision_trace"][0]["precheck_reason_codes"] == ("REVERIFICATION_POLICY_UNAVAILABLE",)
    assert_safe(out)


def test_precheck_rejected_does_not_continue_to_snapshots_and_preserves_reason():
    c,p,r=triplet(); p["precheck_status"]="PRECHECK_REJECTED"; p["reason_codes"]=("SCOPE_EXPANSION_DETECTED",)
    p["frozen_evidence_snapshot_fingerprint"]="stale"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_REJECTED",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["decision_trace"][0]["precheck_reason_codes"] == ("SCOPE_EXPANSION_DETECTED",)
    assert_safe(out)


def test_incomplete_result_mapping_returns_auditable_failure():
    c,p,r=triplet(); r={"correction_id":r["correction_id"]}
    out=compare_virtual_reverification_before_after(c,p,r)
    assert_safe(out)


def test_invalid_policy_returns_auditable_failure():
    c,p,r=triplet(); c["policy"]={"require_frozen_reverification_evidence":False}
    out=compare_virtual_reverification_before_after(c,p,r)
    assert_safe(out)


def test_valid_acceptance_unchanged():
    c,p,r=triplet(); out=compare_virtual_reverification_before_after(c,p,r)
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"
    assert out["additional_llm_calls"] == 0
    assert out["retrieval_rounds"] == 0
    assert out["correction_applied"] is False

def test_invented_action_failure_defer_uses_safe_contractual_action():
    from src.tools.verification.validation import _comparison_failure_result
    out=_comparison_failure_result(
        reverification_input={"correction_action_type":"INVENTED"},
        precheck_result={}, reverification_result={},
        reason="COMPARISON_PRECHECK_BLOCKED", decision="DEFER_TO_MANUAL_REVIEW",
    )
    assert out["correction_action_type"] == "NOT_AVAILABLE"
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert_safe(out)
