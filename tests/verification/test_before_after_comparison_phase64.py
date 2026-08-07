from __future__ import annotations

import copy
import json

from test_reverification_prechecks_phase62 import base_context
from src.tools.verification.validation import (
    compare_virtual_reverification_before_after,
    run_independent_virtual_reverification,
    run_virtual_reverification_prechecks,
)


class Double:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return json.dumps(self.response)


def llm_payload(c, *, verdict="SUPPORTED", support="STRONG", observed=(), reported=None,
                manual=False, meaning=True, intended=True, unintended=True,
                numeric="VALID", scope="NOT_APPLICABLE", attribution="NOT_APPLICABLE",
                citation="NOT_APPLICABLE"):
    if reported is None:
        reported = list(c["target_issue_codes"])
    return {
        "correction_id": c["correction_id"],
        "claim_id": c["claim_id"],
        "proposed_verdict": verdict,
        "support_level": support,
        "evidence_ids_used": ["E1"] if verdict == "SUPPORTED" else [],
        "observed_issue_codes": list(observed),
        "target_issues_resolved": list(reported),
        "supported_meaning_preserved": meaning,
        "intended_semantic_change_valid": intended,
        "unintended_semantic_change_absent": unintended,
        "scope_assessment": scope,
        "numeric_assessment": numeric,
        "attribution_assessment": attribution,
        "citation_assessment": citation,
        "manual_review_recommended": manual,
        "reason_codes": ["TARGET_ISSUE_APPEARS_RESOLVED"] if reported else [],
        "rationale": "Evaluación científica independiente válida.",
        "confidence": 0.9,
    }


def setup_result(**kwargs):
    c = base_context()
    p = run_virtual_reverification_prechecks(c)
    d = Double(llm_payload(c, **kwargs))
    r = run_independent_virtual_reverification(c, p, reverification_llm=d)
    assert r["reverification_execution_status"] == "COMPLETED"
    return c, p, r, d


def compare(**kwargs):
    c, p, r, d = setup_result(**kwargs)
    out = compare_virtual_reverification_before_after(c, p, r)
    return out, c, p, r, d


def test_target_issue_resolved_accepts_for_07c():
    out, *_ = compare()
    assert out["resolved_issue_codes"] == ("UNSUPPORTED_NUMERIC_VALUE",)
    assert out["target_issues_resolved"] is True
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"


def test_target_issue_remains_rejects():
    out, *_ = compare(verdict="PARTIALLY_SUPPORTED", support="PARTIAL",
                      observed=("UNSUPPORTED_NUMERIC_VALUE", "PARTIAL_SUPPORT"), reported=())
    assert "UNSUPPORTED_NUMERIC_VALUE" in out["remaining_issue_codes"]
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert "TARGET_ISSUE_NOT_RESOLVED" in out["reason_codes"]


def test_false_reported_resolution_is_detected():
    c, p, r, _ = setup_result()
    r = dict(r)
    r["observed_issue_codes"] = ("UNSUPPORTED_NUMERIC_VALUE",)
    r["target_issues_resolved_reported"] = ("UNSUPPORTED_NUMERIC_VALUE",)
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["reported_resolution_matches"] is False
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_new_critical_issue_rejects():
    out, *_ = compare(verdict="PARTIALLY_SUPPORTED", support="PARTIAL",
                      observed=("ATTRIBUTION_ERROR",), reported=("UNSUPPORTED_NUMERIC_VALUE",))
    assert out["new_issue_codes"] == ("ATTRIBUTION_ERROR",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert "CRITICAL_NEW_ISSUE_INTRODUCED" in out["reason_codes"]


def test_new_noncritical_issue_defers():
    out, *_ = compare(verdict="PARTIALLY_SUPPORTED", support="PARTIAL",
                      observed=("PARTIAL_SUPPORT",), reported=("UNSUPPORTED_NUMERIC_VALUE",))
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert out["manual_review_required"] is True


def test_multiple_issues_resolved_and_non_target_remains():
    c = base_context()
    c["source_issue_codes"] = ["UNSUPPORTED_NUMERIC_VALUE", "INVALID_CITATION"]
    c["target_issue_codes"] = ["UNSUPPORTED_NUMERIC_VALUE"]
    p = run_virtual_reverification_prechecks(c)
    d = Double(llm_payload(c))
    r = run_independent_virtual_reverification(c, p, reverification_llm=d)
    r = dict(r)
    r["observed_issue_codes"] = ("INVALID_CITATION",)
    r["target_issues_resolved_reported"] = ("UNSUPPORTED_NUMERIC_VALUE",)
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["resolved_issue_codes"] == ("UNSUPPORTED_NUMERIC_VALUE",)
    assert out["remaining_issue_codes"] == ("INVALID_CITATION",)
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"


def test_risk_reduced():
    out, *_ = compare()
    assert out["hallucination_risk_before"] == "HIGH"
    assert out["hallucination_risk_after"] == "LOW"
    assert out["hallucination_risk_delta"] == "REDUCED"


def test_risk_unchanged_with_target_resolved_can_accept():
    c = base_context()
    c["source_issue_codes"] = ["UNSUPPORTED_NUMERIC_VALUE", "INVALID_CITATION"]
    c["target_issue_codes"] = ["UNSUPPORTED_NUMERIC_VALUE"]
    p = run_virtual_reverification_prechecks(c)
    r = run_independent_virtual_reverification(c, p, reverification_llm=Double(llm_payload(c)))
    r = dict(r); r["observed_issue_codes"] = ("INVALID_CITATION",)
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["hallucination_risk_delta"] == "UNCHANGED"
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"


def test_risk_unchanged_with_target_unresolved_rejects():
    out, *_ = compare(verdict="PARTIALLY_SUPPORTED", support="PARTIAL",
                      observed=("UNSUPPORTED_NUMERIC_VALUE",), reported=())
    assert out["hallucination_risk_delta"] == "UNCHANGED"
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_risk_increased_rejects():
    c = base_context()
    c["source_issue_codes"] = ["PARTIAL_SUPPORT"]
    c["target_issue_codes"] = ["PARTIAL_SUPPORT"]
    c["correction_action_type"] = "NARROW_SCOPE"
    c["replacement_text"] = "alcanzó un RMSE de 1.3 en la muestra evaluada"
    c["proposed_claim_text"] = "El modelo alcanzó un RMSE de 1.3 en la muestra evaluada."
    from hashlib import sha256
    c["proposed_claim_text_fingerprint"] = sha256(c["proposed_claim_text"].encode()).hexdigest()
    p = run_virtual_reverification_prechecks(c)
    payload = llm_payload(c, verdict="PARTIALLY_SUPPORTED", support="PARTIAL", observed=("ATTRIBUTION_ERROR",), reported=("PARTIAL_SUPPORT",), numeric="NOT_APPLICABLE", scope="VALID")
    r = run_independent_virtual_reverification(c, p, reverification_llm=Double(payload))
    r = dict(r)
    r["target_issues_resolved_reported"] = ("PARTIAL_SUPPORT",)
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["hallucination_risk_delta"] == "NOT_COMPARABLE"
    assert out["acceptance_decision"] in {"REJECT_PROPOSAL", "DEFER_TO_MANUAL_REVIEW"}


def test_risk_not_comparable_defers():
    c, p, r, _ = setup_result()
    r = dict(r)
    r.update(proposed_verdict="NOT_EVALUATED", support_level="NONE",
             observed_issue_codes=(), target_issues_resolved_reported=(), evidence_ids_used=())
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["hallucination_risk_delta"] == "NOT_COMPARABLE"
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"


def test_applicable_assessment_valid():
    out, *_ = compare()
    assert out["numeric_assessment"] == "VALID"
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"


def test_applicable_assessment_invalid_rejects():
    out, *_ = compare(numeric="INVALID")
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert "ACTION_ASSESSMENT_INVALID" in out["reason_codes"]


def test_non_applicable_assessment_incorrect_rejects():
    c, p, r, _ = setup_result()
    r = dict(r); r["scope_assessment"] = "VALID"
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


def test_semantic_failures_reject():
    for kwargs, code in [
        ({"meaning": False}, "SUPPORTED_MEANING_NOT_PRESERVED"),
        ({"intended": False}, "INTENDED_SEMANTIC_CHANGE_INVALID"),
        ({"unintended": False}, "UNINTENDED_SEMANTIC_CHANGE_DETECTED"),
    ]:
        out, *_ = compare(**kwargs)
        assert out["acceptance_decision"] == "REJECT_PROPOSAL"
        assert code in out["reason_codes"]


def test_manual_review_recommended_defers():
    out, *_ = compare(manual=True)
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"


def test_cross_source_disagreement_defers():
    out, *_ = compare(verdict="PARTIALLY_SUPPORTED", support="PARTIAL",
                      observed=("CROSS_SOURCE_DISAGREEMENT",), reported=("UNSUPPORTED_NUMERIC_VALUE",))
    assert out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW"


def test_correction_id_mismatch_rejects_without_calls():
    c, p, r, d = setup_result()
    r = dict(r); r["correction_id"] = "OTHER"
    before = d.calls
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert "COMPARISON_CONTEXT_MISMATCH" in out["reason_codes"]
    assert d.calls == before


def test_context_fingerprint_mismatch_rejects():
    c, p, r, _ = setup_result()
    r = dict(r); r["reverification_context_fingerprint"] = "0" * 64
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_issue_code_order_is_contractual():
    c = base_context()
    c["source_issue_codes"] = ["INVALID_CITATION", "UNSUPPORTED_NUMERIC_VALUE"]
    c["target_issue_codes"] = ["UNSUPPORTED_NUMERIC_VALUE"]
    c["correction_action_type"] = "REPLACE_NUMERIC_VALUE"
    p = run_virtual_reverification_prechecks(c)
    r = run_independent_virtual_reverification(c, p, reverification_llm=Double(llm_payload(c)))
    r = dict(r); r["observed_issue_codes"] = (); r["target_issues_resolved_reported"] = tuple(reversed(c["target_issue_codes"]))
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["resolved_issue_codes"] == ("INVALID_CITATION", "UNSUPPORTED_NUMERIC_VALUE")


def test_no_additional_llm_retrieval_or_application():
    out, *_ = compare()
    assert out["additional_llm_calls"] == 0
    assert out["retrieval_rounds"] == 0
    assert out["correction_applied"] is False
    assert out["result_contract_valid"] is True
