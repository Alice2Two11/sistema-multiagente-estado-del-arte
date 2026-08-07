from dataclasses import asdict
import json
import pytest

from src.config.verification_policy_config import (
    AGGREGATION_STATUSES, METRICS_STATUSES, TRACE_STAGE_AVAILABILITIES,
    METRIC_COMPUTATION_STATUSES, NORMALIZED_BUNDLE_STATUSES,
)
from src.tools.verification.traceability import (
    ProvisionalVerificationAggregationInput, MetricValue, ClaimTraceabilityRow,
    CorrectionTraceabilityRow, ClaimEvidenceTraceabilityRow,
    CorrectionEvidenceTraceabilityRow, ReverificationTraceabilityRow,
    ProvisionalVerificationMetrics,
)
from src.tools.verification.validation import (
    validate_provisional_verification_aggregation_input_contract,
    validate_metric_value_contract, validate_claim_traceability_row_contract,
    validate_correction_traceability_row_contract,
    validate_claim_evidence_traceability_row_contract,
    validate_correction_evidence_traceability_row_contract,
    validate_reverification_traceability_row_contract,
    validate_provisional_verification_traceability_bundle_contract,
    create_provisional_verification_traceability_bundle,
)


def metric_not_computable():
    return MetricValue(None, 0, 0, "NOT_COMPUTABLE", "recommendation identity", "no stable identity")


def metrics():
    return ProvisionalVerificationMetrics(
        candidate_issue_resolution_rate=MetricValue(0.0, 0, 1, "COMPUTED", "claim issues", "all"),
        accepted_issue_resolution_rate=MetricValue(0.0, 0, 1, "COMPUTED", "claim issues", "accepted"),
        correction_acceptance_rate=MetricValue(0.0, 0, 1, "COMPUTED", "corrections", "reverified"),
        new_issue_rate=MetricValue(0.0, 0, 1, "COMPUTED", "corrections", "reverified"),
        hallucination_risk_reduction_rate=MetricValue(0.0, 0, 1, "COMPUTED", "corrections", "comparable"),
        recommendations_generated=metric_not_computable(),
    )


def claim_row():
    return ClaimTraceabilityRow(
        "c1", "s1", "SUBSTANTIVE_FACTUAL", "Claim", "SUPPORTED", (), "LOW", False,
        False, (), (), (), (), (), (), False,
    )


def correction_row(action="REPLACE_NUMERIC_VALUE", scientific=True, gate=False):
    return CorrectionTraceabilityRow(
        "x1", "c1", "s1", action, scientific, gate,
        "AVAILABLE", "AVAILABLE", "AVAILABLE", "AVAILABLE",
        "ACCEPTED_FOR_REVERIFICATION", "PRECHECK_PASSED", "COMPLETED", "ACCEPT_FOR_07C",
        ("UNSUPPORTED_NUMERIC_VALUE",), ("UNSUPPORTED_NUMERIC_VALUE",), (), (),
        "HIGH", "LOW", "REDUCED", "a"*64, "b"*64, "c"*64, "d"*64,
        (), (), (), (), None, False,
    )


def bundle_kwargs(status="VALID"):
    if status == "VALID":
        metrics_status, partial, normalized_status, normalized = "COMPUTED", (), "NOT_COMPUTABLE", None
    elif status == "PARTIAL":
        metrics_status, partial, normalized_status, normalized = "PARTIALLY_COMPUTED", ("PARTIAL_EXPECTED",), "NOT_COMPUTABLE", None
    else:
        metrics_status, partial, normalized_status, normalized = "NOT_COMPUTED", (), "NOT_COMPUTABLE", None
    return dict(
        claim_traceability_rows=(), correction_traceability_rows=(),
        claim_evidence_traceability_rows=(), correction_evidence_traceability_rows=(),
        reverification_traceability_rows=(), metrics=(metrics().to_dict() if status != "INVALID" else ProvisionalVerificationMetrics(recommendations_generated=metric_not_computable()).to_dict()),
        aggregation_status=status, metrics_status=metrics_status,
        partial_reason_codes=partial, aggregation_issue_codes=(), aggregation_warnings=(("AGGREGATION_CLAIM_WITHOUT_PROPOSAL",) if status == "PARTIAL" else ()),
        normalized_bundle_status=normalized_status, normalized_bundle_fingerprint=normalized,
        aggregation_audit_fingerprint=None, input_collection_fingerprints={},
        policy_versions={"verification":"v1"}, schema_versions={"bundle":"v1"}, correction_applied=False,
        official_artifacts_created=False, additional_llm_calls=0,
        additional_retrieval_rounds=0,
    )


def test_all_closed_enum_values_are_present():
    assert AGGREGATION_STATUSES == ("VALID", "PARTIAL", "INVALID")
    assert METRICS_STATUSES == ("COMPUTED", "PARTIALLY_COMPUTED", "NOT_COMPUTED")
    assert set(TRACE_STAGE_AVAILABILITIES) == {"AVAILABLE", "NOT_PRODUCED", "NOT_APPLICABLE", "BLOCKED_UPSTREAM", "FAILED"}
    assert METRIC_COMPUTATION_STATUSES == ("COMPUTED", "NOT_COMPUTABLE")
    assert NORMALIZED_BUNDLE_STATUSES == ("COMPUTED", "NOT_COMPUTABLE")


def test_unknown_enum_rejected():
    row = correction_row().to_dict(); row["proposal_stage_availability"] = "MAGIC"
    with pytest.raises(ValueError, match="UNKNOWN"):
        validate_correction_traceability_row_contract(row)


def test_empty_batch_input_valid_and_zero_invariants():
    value = ProvisionalVerificationAggregationInput(policy_versions={"verification":"v1"}, schema_versions={"input":"v1"}).to_dict()
    assert validate_provisional_verification_aggregation_input_contract(value)["claim_verification_records"] == ()
    value["additional_llm_calls"] = 1
    with pytest.raises(ValueError, match="MUST_BE_ZERO"):
        validate_provisional_verification_aggregation_input_contract(value)


def test_claim_row_valid_and_final_field_unknown():
    value = claim_row().to_dict()
    assert validate_claim_traceability_row_contract(value)["claim_id"] == "c1"
    value["final_claim_decision"] = "SUPPORTED"
    with pytest.raises(ValueError, match="UNKNOWN_FIELDS"):
        validate_claim_traceability_row_contract(value)


def test_scientific_correction_row_valid():
    assert validate_correction_traceability_row_contract(correction_row().to_dict())["is_scientific_correction_action"]


def test_gate_not_available_valid_but_never_scientific():
    row = correction_row("NOT_AVAILABLE", False, True).to_dict()
    row["acceptance_decision"] = "REJECT_PROPOSAL"
    row["gate_classification"] = "PERMANENT_CONTRACTUAL"
    assert validate_correction_traceability_row_contract(row)["is_gate_result"]
    row["is_scientific_correction_action"] = True
    with pytest.raises(ValueError, match="GATE_ONLY"):
        validate_correction_traceability_row_contract(row)


def test_not_available_cannot_accept():
    row = correction_row("NOT_AVAILABLE", False, True).to_dict()
    with pytest.raises(ValueError, match="CANNOT_ACCEPT"):
        validate_correction_traceability_row_contract(row)


def test_claim_evidence_has_no_correction_id_and_not_evaluated_allowed():
    row = ClaimEvidenceTraceabilityRow("c1", "s1", "e1", "p.pdf", "ch1", "0"*64, "SUPPORT", True, True, "NOT_EVALUATED")
    value = row.to_dict()
    assert "correction_id" not in value
    assert validate_claim_evidence_traceability_row_contract(value)["supports_original_claim"] == "NOT_EVALUATED"


def test_correction_evidence_requires_correction_id():
    row = CorrectionEvidenceTraceabilityRow("c1", "x1", "s1", "e1", "p.pdf", "ch1", "SUPPORT", True, True, True, "SUPPORTED", "f"*64)
    assert validate_correction_evidence_traceability_row_contract(row.to_dict())["correction_id"] == "x1"


def test_reverification_row_valid():
    row = ReverificationTraceabilityRow("x1", "c1", "s1", "v1", "COMPLETED", 1, 1, 0, 1, 0, ("e1",), (), (), True, False, "ACCEPT_FOR_07C", "a"*64, "b"*64, "c"*64, "d"*64)
    assert validate_reverification_traceability_row_contract(row.to_dict())["reverification_llm_calls"] == 1


def test_metric_computed_and_not_computable():
    computed = MetricValue(0.5, 1, 2, "COMPUTED", "claims", "accepted")
    assert validate_metric_value_contract(computed.to_dict())["value"] == 0.5
    nc = metric_not_computable()
    assert validate_metric_value_contract(nc.to_dict())["value"] is None


def test_denominator_zero_cannot_return_zero_float():
    bad = MetricValue(0.0, 0, 0, "NOT_COMPUTABLE", "claims", "none")
    with pytest.raises(ValueError, match="MUST_BE_NULL"):
        validate_metric_value_contract(bad.to_dict())


@pytest.mark.parametrize("status", ["VALID", "PARTIAL", "INVALID"])
def test_bundle_statuses(status):
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs(status))
    assert bundle.result_contract_valid is True
    assert validate_provisional_verification_traceability_bundle_contract(bundle.to_dict())["aggregation_status"] == status


def test_invalid_bundle_requires_null_normalized_fingerprint():
    kwargs = bundle_kwargs("INVALID"); kwargs["normalized_bundle_fingerprint"] = "a"*64
    with pytest.raises(ValueError, match="MUST_BE_NULL"):
        create_provisional_verification_traceability_bundle(**kwargs)


def test_result_contract_valid_is_derived():
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs())
    assert bundle.result_contract_valid is True
    value = bundle.to_dict(); value["result_contract_valid"] = False
    with pytest.raises(ValueError, match="MUST_BE_DERIVED_TRUE"):
        validate_provisional_verification_traceability_bundle_contract(value)


def test_unknown_bundle_field_rejected():
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs()).to_dict()
    bundle["surprise"] = 1
    with pytest.raises(ValueError, match="UNKNOWN_FIELDS"):
        validate_provisional_verification_traceability_bundle_contract(bundle)


def test_serialization_deterministic():
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs())
    first = json.dumps(bundle.to_dict(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(bundle.to_dict(), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_isolation_flags_remain_zero_false():
    bundle = create_provisional_verification_traceability_bundle(**bundle_kwargs())
    assert bundle.additional_llm_calls == 0
    assert bundle.additional_retrieval_rounds == 0
    assert bundle.correction_applied is False
    assert bundle.official_artifacts_created is False
