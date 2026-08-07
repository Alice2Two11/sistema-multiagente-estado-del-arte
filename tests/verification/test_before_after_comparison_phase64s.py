from __future__ import annotations

import copy
import pytest

from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.validation import (
    compare_virtual_reverification_before_after,
    validate_before_after_comparison_result_contract,
)


def triplet():
    c, p, r, _ = setup_result()
    return c, p, r


def test_result63_unknown_reason_code_is_rejected_by_official_validator():
    c,p,r=triplet(); r=dict(r); r["reason_codes"]=("UNKNOWN",)
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


def test_result63_unknown_technical_code_is_rejected_by_official_validator():
    c,p,r=triplet(); r=dict(r); r["technical_issue_codes"]=("UNKNOWN",)
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


def test_result63_supported_without_evidence_is_rejected():
    c,p,r=triplet(); r=dict(r); r["evidence_ids_used"]=()
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


def test_result63_incompatible_assessment_is_rejected():
    c,p,r=triplet(); r=dict(r); r["scope_assessment"]="VALID"
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_REVERIFICATION_RESULT_INVALID",)


@pytest.mark.parametrize("where", ["input", "precheck", "result"])
def test_identity_missing_in_each_structure(where):
    c,p,r=triplet()
    obj={"input":c,"precheck":p,"result":r}[where]
    obj=dict(obj); obj.pop("correction_id")
    if where=="input": c=obj
    elif where=="precheck": p=obj
    else: r=obj
    out=compare_virtual_reverification_before_after(c,p,r)
    assert out["reason_codes"] == ("COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",)
    assert out["acceptance_decision"] == "REJECT_PROPOSAL"


def accepted_result():
    c,p,r=triplet()
    return compare_virtual_reverification_before_after(c,p,r)


def rejected_by_contract(mutator):
    out=copy.deepcopy(accepted_result())
    mutator(out)
    with pytest.raises(ValueError, match="COMPARISON_RESULT_SCHEMA_INVALID"):
        validate_before_after_comparison_result_contract(out)


def test_accept_with_invalid_applicable_assessment_rejected():
    rejected_by_contract(lambda x: x.__setitem__("numeric_assessment", "INVALID"))


def test_accept_with_meaning_not_preserved_rejected():
    rejected_by_contract(lambda x: x.__setitem__("supported_meaning_preserved", False))


def test_accept_with_invalid_intended_change_rejected():
    rejected_by_contract(lambda x: x.__setitem__("intended_semantic_change_valid", False))


def test_accept_with_unintended_change_rejected():
    rejected_by_contract(lambda x: x.__setitem__("unintended_semantic_change_absent", False))


def test_accept_with_critical_new_issue_rejected():
    def mutate(x):
        x["observed_issue_codes"]=("INVALID_CITATION",)
        x["resolved_issue_codes"]=("UNSUPPORTED_NUMERIC_VALUE",)
        x["remaining_issue_codes"]=()
        x["new_issue_codes"]=("INVALID_CITATION",)
    rejected_by_contract(mutate)


def test_defer_requires_manual_review_true():
    out=copy.deepcopy(accepted_result())
    out["acceptance_decision"]="DEFER_TO_MANUAL_REVIEW"
    out["manual_review_required"]=False
    with pytest.raises(ValueError, match="COMPARISON_RESULT_SCHEMA_INVALID"):
        validate_before_after_comparison_result_contract(out)


def test_complete_valid_acceptance_contract_passes():
    out=accepted_result()
    assert out["acceptance_decision"] == "ACCEPT_FOR_07C"
    assert out["manual_review_required"] is False
    assert validate_before_after_comparison_result_contract(out) == out
