from __future__ import annotations

from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.validation import (
    compare_virtual_reverification_before_after,
    validate_before_after_comparison_result_contract,
)


def triplet():
    c, p, r, _ = setup_result()
    return dict(c), dict(p), dict(r)


def test_missing_action_uses_not_available():
    c,p,r=triplet(); c.pop("correction_action_type")
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["correction_action_type"] == "NOT_AVAILABLE"
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_invented_action_is_not_transformed_to_real_action():
    c,p,r=triplet(); c["correction_action_type"]="INVENTED"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["correction_action_type"] == "NOT_AVAILABLE"
    assert out["correction_action_type"] != "SPLIT_CLAIM"


def test_not_available_never_accepts():
    c,p,r=triplet(); out=compare_virtual_reverification_before_after(c,p,r)
    out["correction_action_type"]="NOT_AVAILABLE"
    try:
        validate_before_after_comparison_result_contract(out)
    except ValueError:
        pass
    else:
        raise AssertionError("NOT_AVAILABLE must never allow ACCEPT_FOR_07C")


def test_valid_real_action_is_preserved():
    c,p,r=triplet(); out=compare_virtual_reverification_before_after(c,p,r)
    assert out["correction_action_type"] == c["correction_action_type"]
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"


def _blocked(reason, technical=()):
    c,p,r=triplet(); p["precheck_status"]="PRECHECK_BLOCKED"
    p["reason_codes"]=(reason,); p["technical_issue_codes"]=tuple(technical)
    return compare_virtual_reverification_before_after(c,p,r)


def test_known_temporary_reason_defers():
    out=_blocked("REVERIFICATION_POLICY_UNAVAILABLE")
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert out["decision_trace"][0]["gate_classification"] == "TEMPORARY_TECHNICAL"


def test_known_permanent_reason_rejects():
    out=_blocked("PROPOSAL_FINGERPRINT_MISMATCH")
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["decision_trace"][0]["gate_classification"] == "PERMANENT_CONTRACTUAL"


def test_unknown_reason_is_invalid_and_rejected():
    out=_blocked("SOMETHING_UNKNOWN")
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_word_technical_does_not_imply_temporary():
    out=_blocked("FAKE_TECHNICAL_REASON")
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_unknown_technical_issue_invalidates_gate():
    out=_blocked("REVERIFICATION_POLICY_UNAVAILABLE", ("UNKNOWN_TECH",))
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_gate_keeps_reason_and_technical_codes_separate():
    out=_blocked("REVERIFICATION_DEPENDENCY_UNAVAILABLE", ("REVERIFICATION_DEPENDENCY_UNAVAILABLE",))
    trace=out["decision_trace"][0]
    assert trace["precheck_reason_codes"] == ("REVERIFICATION_DEPENDENCY_UNAVAILABLE",)
    assert trace["precheck_technical_issue_codes"] == ("REVERIFICATION_DEPENDENCY_UNAVAILABLE",)
    assert trace["gate_classification"] == "TEMPORARY_TECHNICAL"
    assert out["observed_issue_codes"] == ()


def test_previous_valid_acceptance_unchanged():
    c,p,r=triplet(); out=compare_virtual_reverification_before_after(c,p,r)
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"
    assert out["correction_action_type"] == "REPLACE_NUMERIC_VALUE"
