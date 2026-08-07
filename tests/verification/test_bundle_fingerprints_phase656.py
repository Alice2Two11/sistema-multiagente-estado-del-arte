from copy import deepcopy
import math
import pytest

from test_referential_integrity_phase653 import aligned_payload
from src.tools.verification.validation import (
    validate_and_normalize_provisional_collections,
    validate_provisional_referential_integrity,
    build_provisional_traceability_rows,
    aggregate_provisional_verification_metrics,
    validate_provisional_metrics_aggregation_result_contract,
    compute_provisional_collection_fingerprints,
    build_provisional_verification_traceability_bundle,
    _phase656_audit_payload,
    _phase656_sha256_payload,
)


def pipeline(payload=None):
    p = payload or aligned_payload()
    cv = validate_and_normalize_provisional_collections(p)
    rr = validate_provisional_referential_integrity(cv)
    rows = build_provisional_traceability_rows(rr)
    metrics = aggregate_provisional_verification_metrics(rows, cv)
    return p, cv, rr, rows, metrics


def test_collection_fingerprints_are_sha256_and_order_independent():
    p = aligned_payload()
    cv1 = validate_and_normalize_provisional_collections(p)
    q = {k: (tuple(reversed(v)) if isinstance(v, tuple) else v) for k, v in p.items()}
    cv2 = validate_and_normalize_provisional_collections(q)
    a = compute_provisional_collection_fingerprints(cv1)
    b = compute_provisional_collection_fingerprints(cv2)
    assert a == b and len(a) == 6
    assert all(len(value) == 64 for value in a.values())


def test_bundle_normalized_fingerprint_ignores_input_order_and_mapping_order():
    p = aligned_payload()
    a = build_provisional_verification_traceability_bundle(p)
    q = {k: (tuple(reversed(v)) if isinstance(v, tuple) else v) for k, v in p.items()}
    q["policy_versions"] = dict(reversed(tuple(q["policy_versions"].items())))
    b = build_provisional_verification_traceability_bundle(q)
    assert a.normalized_bundle_fingerprint == b.normalized_bundle_fingerprint


def test_identical_duplicate_changes_audit_not_normalized():
    p = aligned_payload()
    base = build_provisional_verification_traceability_bundle(p)
    q = deepcopy(p)
    q["correction_proposals"] = q["correction_proposals"] + (deepcopy(q["correction_proposals"][0]),)
    duplicate = build_provisional_verification_traceability_bundle(q)
    assert base.normalized_bundle_fingerprint == duplicate.normalized_bundle_fingerprint
    assert base.aggregation_audit_fingerprint != duplicate.aggregation_audit_fingerprint


def test_warning_changes_audit_payload_not_normalized_payload():
    _, cv, rr, rows, metrics = pipeline()
    fingerprints = compute_provisional_collection_fingerprints(cv)
    common = dict(
        collection=cv.to_dict(), referential=rr.to_dict(), rows=rows.to_dict(),
        metric_result=metrics.to_dict(), input_fingerprints=fingerprints,
        aggregation_status="VALID", metrics_status="COMPUTED", partial_reason_codes=(),
    )
    first = _phase656_audit_payload(**common)
    changed_rows = rows.to_dict(); changed_rows["row_warnings"] = ("AGGREGATION_ROW_SOURCE_CLAIM_TEXT_UNAVAILABLE",)
    second = _phase656_audit_payload(**{**common, "rows": changed_rows})
    assert _phase656_sha256_payload(first) != _phase656_sha256_payload(second)


def test_invalid_batch_has_only_audit_fingerprint():
    p = aligned_payload(); q = deepcopy(p["correction_proposals"][0]); q["reason_codes"] = ("different",)
    p["correction_proposals"] = p["correction_proposals"] + (q,)
    bundle = build_provisional_verification_traceability_bundle(p)
    assert bundle.aggregation_status == "INVALID"
    assert bundle.metrics_status == "NOT_COMPUTED"
    assert bundle.normalized_bundle_fingerprint is None
    assert bundle.normalized_bundle_status == "NOT_COMPUTABLE"
    assert len(bundle.aggregation_audit_fingerprint) == 64
    assert bundle.result_contract_valid is True


def test_row_metric_and_policy_changes_change_normalized_fingerprint():
    p = aligned_payload(); base = build_provisional_verification_traceability_bundle(p)
    q = deepcopy(p); q["policy_versions"] = {**q["policy_versions"], "verification": "changed"}
    changed_policy = build_provisional_verification_traceability_bundle(q)
    assert changed_policy.normalized_bundle_fingerprint != base.normalized_bundle_fingerprint
    # A semantically valid counter change changes metrics and therefore the normalized hash.
    r = deepcopy(p); claim = deepcopy(r["claim_verification_records"][0]); result = dict(claim["claim_verification_result"])
    result["tool_usage"] = {**result["tool_usage"], "llm_calls": result["tool_usage"]["llm_calls"] + 1}
    claim["claim_verification_result"] = result; r["claim_verification_records"] = (claim,)
    changed_metric = build_provisional_verification_traceability_bundle(r)
    assert changed_metric.normalized_bundle_fingerprint != base.normalized_bundle_fingerprint


def test_formula_contract_rejects_total_and_rate_tampering():
    _, _, _, _, result = pipeline(); raw = result.to_dict()
    raw["metrics"] = dict(raw["metrics"]); raw["metrics"]["total_llm_calls"] += 1
    with pytest.raises(ValueError, match="TOTAL_LLM_CALLS_MISMATCH"):
        validate_provisional_metrics_aggregation_result_contract(raw)
    raw = result.to_dict(); raw["metrics"] = dict(raw["metrics"])
    rate = dict(raw["metrics"]["accepted_issue_resolution_rate"]); rate["numerator"] = 0; rate["value"] = 0.0
    raw["metrics"]["accepted_issue_resolution_rate"] = rate
    with pytest.raises(ValueError, match="NUMERATOR_MISMATCH"):
        validate_provisional_metrics_aggregation_result_contract(raw)
    raw = result.to_dict(); raw["metrics"] = dict(raw["metrics"])
    rate = dict(raw["metrics"]["accepted_issue_resolution_rate"]); rate["denominator"] = 2; rate["value"] = .5
    raw["metrics"]["accepted_issue_resolution_rate"] = rate
    with pytest.raises(ValueError, match="DENOMINATOR_MISMATCH"):
        validate_provisional_metrics_aggregation_result_contract(raw)


def test_formula_contract_rejects_count_incoherence():
    _, _, _, _, result = pipeline(); raw = result.to_dict(); raw["metrics"] = dict(raw["metrics"])
    raw["metrics"]["issues_before"] = 3
    raw["metrics"]["candidate_claim_issues_resolved"] = 1
    raw["metrics"]["accepted_claim_issues_resolved"] = 2
    raw["metrics"]["candidate_issue_resolution_rate"] = {**raw["metrics"]["candidate_issue_resolution_rate"], "numerator": 1, "denominator": 3, "value": 1/3}
    raw["metrics"]["accepted_issue_resolution_rate"] = {**raw["metrics"]["accepted_issue_resolution_rate"], "numerator": 2, "denominator": 3, "value": 2/3}
    with pytest.raises(ValueError, match="ISSUE_COUNT_COHERENCE"):
        validate_provisional_metrics_aggregation_result_contract(raw)
    raw = result.to_dict(); raw["metrics"] = dict(raw["metrics"])
    raw["metrics"]["corrections_accepted_for_07c"] = 2
    raw["metrics"]["correction_acceptance_rate"] = {**raw["metrics"]["correction_acceptance_rate"], "numerator": 2, "denominator": 2, "value": 1.0}
    raw["metrics"]["corrections_reverified"] = 1
    with pytest.raises(ValueError):
        validate_provisional_metrics_aggregation_result_contract(raw)


def test_one_correction_with_two_new_issues_counts_once_in_rate():
    _, cv, _, rows, _ = pipeline(); raw = rows.to_dict(); correction = dict(raw["correction_traceability_rows"][0])
    correction["new_issue_codes"] = ("PARTIAL_SUPPORT", "INSUFFICIENT_EVIDENCE")
    raw["correction_traceability_rows"] = (correction,)
    from src.tools.verification.validation import validate_provisional_traceability_rows_result_contract
    normalized = validate_provisional_traceability_rows_result_contract(raw)
    result = aggregate_provisional_verification_metrics(normalized, cv)
    assert result.metrics["new_issues_introduced"] == 2
    assert result.metrics["corrections_with_new_issues"] == 1
    assert result.metrics["new_issue_rate"]["numerator"] == 1


def test_invalid_orphan_records_do_not_enter_scientific_population():
    p = aligned_payload()
    orphan = deepcopy(p["independent_reverification_results"][0]); orphan["correction_id"] = "orphan"
    p["independent_reverification_results"] = p["independent_reverification_results"] + (orphan,)
    bundle = build_provisional_verification_traceability_bundle(p)
    assert bundle.aggregation_status == "INVALID"
    assert bundle.metrics_status == "NOT_COMPUTED"


def test_bundle_is_derived_and_has_zero_side_effect_invariants():
    bundle = build_provisional_verification_traceability_bundle(aligned_payload())
    assert bundle.result_contract_valid is True
    assert bundle.correction_applied is False
    assert bundle.official_artifacts_created is False
    assert bundle.additional_llm_calls == 0
    assert bundle.additional_retrieval_rounds == 0
