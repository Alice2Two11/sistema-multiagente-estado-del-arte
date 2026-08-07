from __future__ import annotations

import pytest

from test_before_after_comparison_phase64 import setup_result
from src.config.verification_policy_config import (
    PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES,
    PRECHECK_GATE_TECHNICAL_ISSUE_CODES,
    PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES,
    PRECHECK_RUNTIME_EMITTED_REASON_CODES,
    PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES,
)
from src.tools.verification.validation import (
    audit_precheck_gate_reason_code_coverage,
    compare_virtual_reverification_before_after,
)


def triplet():
    c, p, r, _ = setup_result()
    return dict(c), dict(p), dict(r)


def gate(status: str, reason: str, technical=()):
    c, p, r = triplet()
    p["precheck_status"] = status
    p["reason_codes"] = (reason,)
    p["technical_issue_codes"] = tuple(technical)
    return compare_virtual_reverification_before_after(c, p, r)


@pytest.mark.parametrize(
    "reason",
    (
        "REVERIFICATION_EVIDENCE_NOT_AUTHORIZED",
        "REVERIFICATION_CORPUS_WIDE_EVIDENCE_FORBIDDEN",
        "DOCUMENT_IDENTITY_INVALID",
        "REVERIFICATION_EVIDENCE_ORDER_MISMATCH",
    ),
)
def test_known_evidence_contract_failures_preserve_original_reason(reason):
    out = gate("PRECHECK_BLOCKED", reason)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_BLOCKED",)
    assert out["decision_trace"][0]["precheck_reason_codes"] == (reason,)
    assert "COMPARISON_PRECHECK_INVALID" not in out["reason_codes"]


@pytest.mark.parametrize(
    "reason",
    (
        "UNSUPPORTED_NEW_NUMERIC_VALUE",
        "NUMERIC_CONTEXT_MISMATCH",
        "UNSUPPORTED_NEW_ATTRIBUTION",
        "ATTRIBUTION_RELATION_NOT_SUPPORTED",
        "NEW_CITATION_MARKER_MISSING",
        "NEW_CITATION_DOES_NOT_SUPPORT_PROPOSED_CLAIM",
        "SCOPE_EXPANSION_DETECTED",
        "QUALIFICATION_DIFFERENTIAL_INVALID",
        "REMOVAL_ALTERS_SUPPORTED_MEANING",
    ),
)
def test_known_action_rejections_preserve_original_reason(reason):
    out = gate("PRECHECK_REJECTED", reason)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_REJECTED",)
    trace = out["decision_trace"][0]
    assert trace["precheck_reason_codes"] == (reason,)
    assert trace["gate_classification"] == "DETERMINISTIC_SCIENTIFIC_REJECTION"
    assert "COMPARISON_PRECHECK_INVALID" not in out["reason_codes"]


def test_unknown_reason_still_produces_invalid_gate():
    out = gate("PRECHECK_BLOCKED", "TRULY_UNKNOWN_PRECHECK_REASON")
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_parameterized_contract_family_is_classified_and_original_preserved():
    reason = "TARGET_ISSUE_CODE_NOT_PRESENT:INVALID_CITATION"
    out = gate("PRECHECK_BLOCKED", reason)
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_BLOCKED",)
    assert out["decision_trace"][0]["precheck_reason_codes"] == (reason,)


def test_categories_are_pairwise_disjoint():
    temporary = set(PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES)
    contractual = set(PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES)
    scientific = set(PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES)
    assert temporary.isdisjoint(contractual)
    assert temporary.isdisjoint(scientific)
    assert contractual.isdisjoint(scientific)


def test_automatic_runtime_code_coverage_is_complete():
    audit = audit_precheck_gate_reason_code_coverage()
    assert audit["uncovered"] == ()
    assert audit["category_overlaps"] == ()
    assert audit["uncategorized_extras"] == ()
    assert audit["duplicated_emittable"] == ()
    assert set(PRECHECK_RUNTIME_EMITTED_REASON_CODES).issubset(
        set(PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES)
        | set(PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES)
        | set(PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES)
    )


def test_phase62_gate_catalog_excludes_llm_invocation_failure():
    assert "REVERIFICATION_LLM_INVOCATION_FAILED" not in PRECHECK_RUNTIME_EMITTED_REASON_CODES
    assert "REVERIFICATION_LLM_INVOCATION_FAILED" not in PRECHECK_GATE_TECHNICAL_ISSUE_CODES


def test_valid_acceptance_is_unchanged():
    c, p, r = triplet()
    out = compare_virtual_reverification_before_after(c, p, r)
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"
    assert out["additional_llm_calls"] == 0
    assert out["retrieval_rounds"] == 0
    assert out["correction_applied"] is False
