import math
import pytest

from src.tools.verification.traceability import (
    MetricValue, ClaimTraceabilityRow, CorrectionTraceabilityRow,
    ReverificationTraceabilityRow, ProvisionalVerificationMetrics,
)
from src.tools.verification.validation import (
    validate_metric_value_contract, validate_claim_traceability_row_contract,
    validate_correction_traceability_row_contract,
    validate_reverification_traceability_row_contract,
    create_provisional_verification_traceability_bundle,
    validate_provisional_verification_traceability_bundle_contract,
    validate_provisional_verification_aggregation_input_contract,
)


def rate(value=0.0, numerator=0, denominator=1, status="COMPUTED"):
    return MetricValue(value, numerator, denominator, status, "unique entities", "eligible population").to_dict()


def metrics_payload(computed=True):
    rate_value = rate() if computed else None
    return ProvisionalVerificationMetrics(
        candidate_issue_resolution_rate=rate_value,
        accepted_issue_resolution_rate=rate_value,
        correction_acceptance_rate=rate_value,
        new_issue_rate=rate_value,
        hallucination_risk_reduction_rate=rate_value,
        recommendations_generated=MetricValue(None, 0, 0, "NOT_COMPUTABLE", "recommendations", "no identity"),
    ).to_dict()


def bundle_kwargs(status="VALID", metrics_status=None):
    if metrics_status is None:
        metrics_status = {"VALID":"COMPUTED", "PARTIAL":"PARTIALLY_COMPUTED", "INVALID":"NOT_COMPUTED"}[status]
    return dict(
        claim_traceability_rows=(), correction_traceability_rows=(),
        claim_evidence_traceability_rows=(), correction_evidence_traceability_rows=(),
        reverification_traceability_rows=(), metrics=metrics_payload(metrics_status == "COMPUTED"),
        aggregation_status=status, metrics_status=metrics_status,
        partial_reason_codes=("PARTIAL_EXPECTED",) if status == "PARTIAL" else (),
        aggregation_issue_codes=(), aggregation_warnings=(("AGGREGATION_CLAIM_WITHOUT_PROPOSAL",) if status == "PARTIAL" else ()),
        normalized_bundle_status="NOT_COMPUTABLE", normalized_bundle_fingerprint=None,
        aggregation_audit_fingerprint=None, input_collection_fingerprints={"claims": None},
        policy_versions={"verification":"v1"}, schema_versions={"bundle":"v1"},
        correction_applied=False, official_artifacts_created=False,
        additional_llm_calls=0, additional_retrieval_rounds=0,
    )


def claim_payload(**changes):
    value = ClaimTraceabilityRow(
        "c1", "s1", "SUBSTANTIVE_FACTUAL", "Claim", "SUPPORTED", (), "LOW", False,
        True, ("x1",), ("ACCEPT_FOR_07C",), ("x1",), (), (), (), False,
    ).to_dict()
    value.update(changes)
    return value


def correction_payload(**changes):
    value = CorrectionTraceabilityRow(
        "x1", "c1", "s1", "REPLACE_NUMERIC_VALUE", True, False,
        "AVAILABLE", "AVAILABLE", "AVAILABLE", "AVAILABLE",
        "ACCEPTED_FOR_REVERIFICATION", "PRECHECK_PASSED", "COMPLETED", "ACCEPT_FOR_07C",
        ("UNSUPPORTED_NUMERIC_VALUE",), ("UNSUPPORTED_NUMERIC_VALUE",), (), (),
        "HIGH", "LOW", "REDUCED", "a"*64, "b"*64, "c"*64, "d"*64,
        (), (), (), (), None, False,
    ).to_dict()
    value.update(changes)
    return value


def reverify_payload(**changes):
    value = ReverificationTraceabilityRow(
        "x1", "c1", "s1", "v1", "COMPLETED", 1, 1, 0, 1, 0,
        ("e1",), (), (), True, False, "ACCEPT_FOR_07C",
        "a"*64, "b"*64, "c"*64, "d"*64,
    ).to_dict()
    value.update(changes)
    return value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_metric_rejects_nonfinite(bad):
    with pytest.raises(ValueError, match="FINITE_REQUIRED"):
        validate_metric_value_contract(rate(bad, 1, 1))


def test_metric_rejects_numerator_above_denominator():
    with pytest.raises(ValueError, match="EXCEEDS_DENOMINATOR"):
        validate_metric_value_contract(rate(2.0, 2, 1))


def test_claim_closed_catalogs_and_proposal_coherence():
    with pytest.raises(ValueError, match="source_verdict:UNKNOWN"):
        validate_claim_traceability_row_contract(claim_payload(source_verdict="INVENTED"))
    with pytest.raises(ValueError, match="RELATED_FIELDS_MUST_BE_EMPTY"):
        validate_claim_traceability_row_contract(claim_payload(has_correction_proposal=False))
    with pytest.raises(ValueError, match="NOT_SUBSET"):
        validate_claim_traceability_row_contract(claim_payload(individual_accepted_correction_ids=("x2",)))
    with pytest.raises(ValueError, match="MUST_BE_DISJOINT"):
        validate_claim_traceability_row_contract(claim_payload(individual_rejected_correction_ids=("x1",)))


@pytest.mark.parametrize("field,value", [
    ("proposal_status", "INVENTED"), ("precheck_status", "INVENTED"),
    ("acceptance_decision", "INVENTED"), ("hallucination_risk_before", "INVENTED"),
])
def test_correction_closed_enums(field, value):
    with pytest.raises(ValueError, match="UNKNOWN"):
        validate_correction_traceability_row_contract(correction_payload(**{field:value}))


def test_stage_availability_matrix():
    with pytest.raises(ValueError, match="MUST_BE_NULL_WHEN_UNAVAILABLE"):
        validate_correction_traceability_row_contract(correction_payload(precheck_stage_availability="NOT_PRODUCED"))
    with pytest.raises(ValueError, match="REQUIRED_WHEN_AVAILABLE"):
        validate_correction_traceability_row_contract(correction_payload(precheck_status=None))
    bad = correction_payload(precheck_stage_availability="BLOCKED_UPSTREAM", precheck_status=None)
    with pytest.raises(ValueError, match="LATER_STAGE_AVAILABLE"):
        validate_correction_traceability_row_contract(bad)


def test_action_gate_coherence():
    with pytest.raises(ValueError, match="FLAGS_INCOHERENT"):
        validate_correction_traceability_row_contract(correction_payload(is_gate_result=True))
    gate = correction_payload(
        action_type="NOT_AVAILABLE", is_scientific_correction_action=False, is_gate_result=True,
        gate_classification=None, acceptance_decision="REJECT_PROPOSAL",
    )
    with pytest.raises(ValueError, match="gate_classification"):
        validate_correction_traceability_row_contract(gate)


def test_reverification_closed_and_attempt_coherence():
    with pytest.raises(ValueError, match="CANNOT_ACCEPT"):
        validate_reverification_traceability_row_contract(reverify_payload(reverification_execution_status="FAILED"))
    with pytest.raises(ValueError, match="EXCEEDS_ATTEMPTS"):
        validate_reverification_traceability_row_contract(reverify_payload(format_retries=2))


def test_bundle_fingerprint_state_rules():
    kwargs = bundle_kwargs(); kwargs["normalized_bundle_status"]="COMPUTED"
    with pytest.raises(ValueError, match="SHA256_REQUIRED"):
        create_provisional_verification_traceability_bundle(**kwargs)
    kwargs = bundle_kwargs(); kwargs["normalized_bundle_fingerprint"]="a"*64
    with pytest.raises(ValueError, match="MUST_BE_NULL"):
        create_provisional_verification_traceability_bundle(**kwargs)
    kwargs = bundle_kwargs(); kwargs["aggregation_audit_fingerprint"]="bad"
    with pytest.raises(ValueError, match="SHA256_REQUIRED"):
        create_provisional_verification_traceability_bundle(**kwargs)


def test_computed_metrics_requires_rates_and_versions_nonempty():
    kwargs = bundle_kwargs(); kwargs["metrics"] = metrics_payload(False)
    with pytest.raises(ValueError, match="COMPUTED_REQUIRES_COMPUTED_RATES"):
        create_provisional_verification_traceability_bundle(**kwargs)
    kwargs = bundle_kwargs(); kwargs["policy_versions"]={"verification":""}
    with pytest.raises(ValueError, match="NONEMPTY_STRING_MAPPING_REQUIRED"):
        create_provisional_verification_traceability_bundle(**kwargs)


def test_constructor_creates_new_frozen_instance_without_mutation(monkeypatch):
    import src.tools.verification.traceability as traceability
    original = traceability.ProvisionalVerificationTraceabilityBundle
    seen = []
    class Spy(original):
        def __new__(cls, *args, **kwargs):
            seen.append(dict(kwargs))
            return super().__new__(cls)
    monkeypatch.setattr(traceability, "ProvisionalVerificationTraceabilityBundle", Spy)
    result = create_provisional_verification_traceability_bundle(**bundle_kwargs())
    assert len(seen) == 1
    assert seen[0]["result_contract_valid"] is True
    assert result.result_contract_valid is True


@pytest.mark.parametrize("status", ["VALID", "PARTIAL", "INVALID"])
def test_valid_bundles_three_states(status):
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs(status))
    assert validate_provisional_verification_traceability_bundle_contract(bundle.to_dict())["aggregation_status"] == status


def test_input_versions_must_be_nonempty():
    value = {
        "claim_verification_records": (), "correction_proposals": (), "correction_precheck_results": (),
        "independent_reverification_results": (), "before_after_comparison_results": (),
        "policy_versions": {"verification":""}, "schema_versions": {"input":"v1"},
        "additional_llm_calls": 0, "additional_retrieval_rounds": 0,
        "correction_applied": False, "official_artifacts_created": False,
    }
    with pytest.raises(ValueError, match="NONEMPTY_STRING_MAPPING_REQUIRED"):
        validate_provisional_verification_aggregation_input_contract(value)
