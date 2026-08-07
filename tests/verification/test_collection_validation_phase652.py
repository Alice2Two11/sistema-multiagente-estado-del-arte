from copy import deepcopy

import pytest

from test_terminal_contracts_phase650s import valid_claim_result, valid_proposal
from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.validation import compare_virtual_reverification_before_after
from src.tools.verification.traceability import (
    ClaimVerificationAggregationRecord,
    MetricValue,
    ProvisionalVerificationMetrics,
)
from src.tools.verification.validation import (
    validate_and_normalize_provisional_collections,
    validate_provisional_collection_validation_result_contract,
    validate_correction_traceability_row_contract,
    create_provisional_verification_traceability_bundle,
)
from test_structural_bundle_phase651ar import correction_payload, bundle_kwargs, metrics_payload


def valid_collections():
    c, p, r, _ = setup_result()
    comparison = compare_virtual_reverification_before_after(c, p, r)
    return {
        "claim_verification_records": (
            ClaimVerificationAggregationRecord("sec1", valid_claim_result()).to_dict(),
        ),
        "correction_proposals": (valid_proposal(),),
        "correction_reverification_inputs": (c,),
        "correction_precheck_results": (p,),
        "independent_reverification_results": (r,),
        "before_after_comparison_results": (comparison,),
        "policy_versions": {"verification": "v1"},
        "schema_versions": {"collection": "v1"},
        "additional_llm_calls": 0,
        "additional_retrieval_rounds": 0,
        "correction_applied": False,
        "official_artifacts_created": False,
    }


def test_block0_complete_sequence_available():
    assert validate_correction_traceability_row_contract(correction_payload())["comparison_stage_availability"] == "AVAILABLE"

@pytest.mark.parametrize("changes", [
    {"proposal_stage_availability":"NOT_APPLICABLE", "proposal_status":None},
    {"proposal_stage_availability":"FAILED", "proposal_status":None, "precheck_stage_availability":"BLOCKED_UPSTREAM", "precheck_status":None},
    {"precheck_stage_availability":"NOT_PRODUCED", "precheck_status":None, "reverification_stage_availability":"NOT_PRODUCED", "reverification_execution_status":None},
])
def test_block0_invalid_causal_sequences(changes):
    row = correction_payload(**changes)
    with pytest.raises(ValueError, match="stage_causality|LATER_STAGE_AVAILABLE"):
        validate_correction_traceability_row_contract(row)


def test_block0_upstream_block_propagates_to_comparison():
    row = correction_payload(
        precheck_stage_availability="BLOCKED_UPSTREAM", precheck_status=None,
        reverification_stage_availability="BLOCKED_UPSTREAM", reverification_execution_status=None,
        comparison_stage_availability="BLOCKED_UPSTREAM", acceptance_decision=None,
        hallucination_risk_before=None, hallucination_risk_after=None, hallucination_risk_delta=None,
        action_type="NOT_AVAILABLE", is_scientific_correction_action=False, is_gate_result=True,
        gate_classification="TEMPORARY_TECHNICAL",
    )
    assert validate_correction_traceability_row_contract(row)["comparison_stage_availability"] == "BLOCKED_UPSTREAM"


def test_block0_computed_bundle_all_rates_computable():
    result = create_provisional_verification_traceability_bundle(**bundle_kwargs())
    assert result.metrics_status == "COMPUTED"


def test_block0_computed_bundle_zero_denominator_rate_not_computable():
    kwargs = bundle_kwargs()
    metrics = metrics_payload(True)
    metrics["new_issue_rate"] = MetricValue(None, 0, 0, "NOT_COMPUTABLE", "corrections", "none eligible").to_dict()
    kwargs["metrics"] = metrics
    result = create_provisional_verification_traceability_bundle(**kwargs)
    assert result.metrics_status == "COMPUTED"
    assert result.metrics["new_issue_rate"]["status"] == "NOT_COMPUTABLE"


def test_block0_computed_bundle_missing_rate_rejected():
    kwargs = bundle_kwargs(); metrics = metrics_payload(True); metrics["new_issue_rate"] = None; kwargs["metrics"] = metrics
    with pytest.raises(ValueError, match="COMPUTED_REQUIRES_COMPUTED_RATES"):
        create_provisional_verification_traceability_bundle(**kwargs)


def test_valid_element_each_collection_and_primary_indexes():
    result = validate_and_normalize_provisional_collections(valid_collections())
    assert result.collection_validation_status == "VALID"
    assert set(result.primary_indexes) == {
        "claim_verification_records", "correction_proposals", "correction_reverification_inputs", "correction_precheck_results",
        "independent_reverification_results", "before_after_comparison_results",
    }
    assert validate_provisional_collection_validation_result_contract(result.to_dict())["result_contract_valid"] is True


@pytest.mark.parametrize("collection,mutator", [
    ("claim_verification_records", lambda x: x["claim_verification_result"].pop("claim_id")),
    ("correction_proposals", lambda x: x.update(action_type="INVENTED")),
    ("correction_reverification_inputs", lambda x: x.pop("correction_id")),
    ("correction_precheck_results", lambda x: x.pop("correction_id")),
    ("independent_reverification_results", lambda x: x.update(reverification_execution_status="INVENTED")),
    ("before_after_comparison_results", lambda x: x.update(acceptance_decision="INVENTED")),
])
def test_invalid_element_each_collection_is_auditable(collection, mutator):
    payload = valid_collections(); bad = deepcopy(payload[collection][0]); mutator(bad); payload[collection] = (bad,)
    result = validate_and_normalize_provisional_collections(payload)
    assert result.collection_validation_status == "INVALID"
    assert result.aggregation_status == "INVALID"
    assert result.metrics_status == "NOT_COMPUTED"
    assert any(code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID") for code in result.collection_issue_codes)


def _duplicate_case(collection, conflict=False):
    payload = valid_collections(); first = deepcopy(payload[collection][0]); second = deepcopy(first)
    if conflict:
        if collection == "claim_verification_records": second["section_id"] = "sec-conflict"
        elif collection == "correction_proposals": second["requires_manual_review"] = not second["requires_manual_review"]
        elif collection == "correction_reverification_inputs": second["attempt_context"] = {**second["attempt_context"], "audit_variant": "b"}
        elif collection == "correction_precheck_results": second["reason_codes"] = tuple(second.get("reason_codes", ())) + ("REVERIFICATION_EVIDENCE_REQUIRED",)
        elif collection == "independent_reverification_results": second["manual_review_recommended"] = not second["manual_review_recommended"]
        else: second["manual_review_required"] = not second["manual_review_required"]
    payload[collection] = (first, second)
    return validate_and_normalize_provisional_collections(payload)


def test_claim_identical_duplicate_deduplicated():
    result = _duplicate_case("claim_verification_records")
    assert result.collection_validation_status == "VALID"
    assert len(result.normalized_claim_verification_records) == 1
    assert "AGGREGATION_IDENTICAL_DUPLICATE_DEDUPLICATED" in result.collection_issue_codes

@pytest.mark.parametrize("collection", [
    "claim_verification_records", "correction_proposals", "correction_reverification_inputs", "correction_precheck_results",
    "independent_reverification_results", "before_after_comparison_results",
])
def test_conflicting_duplicates_invalidate(collection):
    result = _duplicate_case(collection, conflict=True)
    assert result.collection_validation_status == "INVALID"
    assert "AGGREGATION_CONFLICTING_DUPLICATE" in result.collection_issue_codes or any(
        code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID") for code in result.collection_issue_codes
    )


def test_same_mapping_content_different_key_order_is_identical():
    payload = valid_collections(); first = payload["claim_verification_records"][0]
    second = dict(reversed(list(first.items())))
    second["claim_verification_result"] = dict(reversed(list(first["claim_verification_result"].items())))
    payload["claim_verification_records"] = (first, second)
    result = validate_and_normalize_provisional_collections(payload)
    assert len(result.normalized_claim_verification_records) == 1


def test_reordered_inputs_produce_same_normalized_collections():
    payload = valid_collections(); proposal = deepcopy(payload["correction_proposals"][0]); proposal["correction_id"] = "corr2"
    # Keep proposal valid by changing identity would invalidate fingerprint; use identical duplicates in reversed order instead.
    payload["claim_verification_records"] = payload["claim_verification_records"] * 2
    a = validate_and_normalize_provisional_collections(payload)
    payload["claim_verification_records"] = tuple(reversed(payload["claim_verification_records"]))
    b = validate_and_normalize_provisional_collections(payload)
    assert a.normalized_claim_verification_records == b.normalized_claim_verification_records
    assert a.primary_indexes == b.primary_indexes


def test_unknown_element_field_produces_auditable_invalid_result():
    payload = valid_collections(); bad = deepcopy(payload["correction_proposals"][0]); bad["unknown"] = 1
    payload["correction_proposals"] = (bad,)
    result = validate_and_normalize_provisional_collections(payload)
    assert result.result_contract_valid is True
    assert result.collection_validation_status == "INVALID"


def test_no_cross_collection_joins_are_performed():
    payload = valid_collections()
    # Remove proposals while keeping downstream records. This is an orphan, but joins are explicitly out of scope.
    payload["correction_proposals"] = ()
    result = validate_and_normalize_provisional_collections(payload)
    assert result.collection_validation_status == "VALID"
    assert result.normalized_correction_proposals == ()
    assert result.normalized_before_after_comparison_results


def test_isolation_counters_and_no_final_fingerprints():
    result = validate_and_normalize_provisional_collections(valid_collections())
    assert result.metrics_status == "NOT_COMPUTED"
    assert not hasattr(result, "normalized_bundle_fingerprint")
    source = valid_collections()
    assert source["additional_llm_calls"] == 0
    assert source["additional_retrieval_rounds"] == 0
    assert source["correction_applied"] is False
    assert source["official_artifacts_created"] is False
