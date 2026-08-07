from __future__ import annotations

import copy
import pytest

from test_before_after_comparison_phase64 import setup_result
from test_reverification_prechecks_phase62 import base_context
from src.config.verification_policy_config import (
    PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES,
    PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES,
    PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES,
    PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES,
)
from src.tools.verification.validation import (
    audit_precheck_gate_reason_code_coverage,
    compare_virtual_reverification_before_after,
    discover_precheck_parameterized_reason_code_families,
    run_virtual_reverification_prechecks,
)


def _gate(reason: str):
    c, p, r, _ = setup_result()
    p = dict(p)
    p["precheck_status"] = "PRECHECK_BLOCKED"
    p["reason_codes"] = (reason,)
    p["technical_issue_codes"] = ()
    return compare_virtual_reverification_before_after(dict(c), p, dict(r))


@pytest.mark.parametrize(
    ("mutator", "expected_prefix"),
    (
        (lambda c: c.__setitem__("replacement_text", 7), "REVERIFICATION_CONTRACT_INVALID:replacement_text"),
        (lambda c: c.__setitem__("application_order_key", ["S1"]), "REVERIFICATION_CONTRACT_INVALID:application_order_key"),
        (lambda c: c.__setitem__("target_issue_codes", []), "REVERIFICATION_CONTRACT_INVALID:target_issue_codes:empty"),
        (lambda c: c.__setitem__("claim_span_in_section", "bad"), "REVERIFICATION_CONTRACT_INVALID:claim_span_in_section"),
        (lambda c: c.__setitem__("target_span_in_claim", "bad"), "REVERIFICATION_CONTRACT_INVALID:target_span_in_claim"),
        (lambda c: c.__setitem__("authorized_evidence", "bad"), "REVERIFICATION_CONTRACT_INVALID:authorized_evidence"),
        (lambda c: c.__setitem__("attempt_context", "bad"), "REVERIFICATION_CONTRACT_INVALID:attempt_context"),
    ),
)
def test_real_parameterized_contract_branches_are_permanent_and_preserved(mutator, expected_prefix):
    context = base_context()
    mutator(context)
    precheck = run_virtual_reverification_prechecks(context)
    assert precheck["precheck_status"] == "PRECHECK_BLOCKED"
    reason = precheck["reason_codes"][0]
    assert reason.startswith(expected_prefix)

    out = _gate(reason)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_BLOCKED",)
    assert out["decision_trace"][0]["gate_classification"] == "PERMANENT_CONTRACTUAL"
    assert out["decision_trace"][0]["precheck_reason_codes"] == (reason,)
    assert "COMPARISON_PRECHECK_INVALID" not in out["reason_codes"]


def test_contract_invalid_family_is_declared_discovered_and_classified_once():
    assert "REVERIFICATION_CONTRACT_INVALID" in PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES
    assert "REVERIFICATION_CONTRACT_INVALID" in PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES
    assert "REVERIFICATION_CONTRACT_INVALID" in discover_precheck_parameterized_reason_code_families()
    assert "REVERIFICATION_CONTRACT_INVALID" not in PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES
    assert "REVERIFICATION_CONTRACT_INVALID" not in PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES


def test_unknown_parameterized_family_remains_invalid():
    out = _gate("NEW_UNCLASSIFIED_FAMILY:field")
    assert out["reason_codes"] == ("COMPARISON_PRECHECK_INVALID",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def test_non_circular_audit_detects_artificial_unclassified_emittable_family():
    audit = audit_precheck_gate_reason_code_coverage(
        extra_emittable_families=("ARTIFICIAL_PRODUCTIVE_FAMILY",)
    )
    assert "ARTIFICIAL_PRODUCTIVE_FAMILY" in audit["uncovered"]
    assert "ARTIFICIAL_PRODUCTIVE_FAMILY" in audit["uncovered_parameterized_families"]


def test_real_source_audit_is_complete_and_categories_remain_disjoint():
    audit = audit_precheck_gate_reason_code_coverage()
    assert audit["uncovered"] == ()
    assert audit["uncovered_parameterized_families"] == ()
    assert audit["category_overlaps"] == ()
    assert audit["uncategorized_extras"] == ()
    assert audit["duplicated_emittable"] == ()
