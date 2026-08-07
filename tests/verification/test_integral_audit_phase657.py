from copy import deepcopy
import json
import re
import pytest

from test_referential_integrity_phase653 import aligned_payload
from src.tools.verification.validation import (
    build_provisional_verification_traceability_bundle,
    validate_and_normalize_provisional_collections,
    validate_provisional_collection_validation_result_contract,
    validate_provisional_verification_traceability_bundle_contract,
    _phase656_audit_payload,
    _phase656_sha256_payload,
)


def invalid_payload(marker, *, reverse_mapping=False):
    payload = aligned_payload()
    proposal = deepcopy(payload["correction_proposals"][0])
    proposal["unexpected_invalid_field"] = marker
    if reverse_mapping:
        proposal = dict(reversed(tuple(proposal.items())))
    payload["correction_proposals"] = (proposal,)
    return payload


def test_different_invalid_elements_change_audit_fingerprint_without_raw_content():
    a = build_provisional_verification_traceability_bundle(invalid_payload("SECRET-A"))
    b = build_provisional_verification_traceability_bundle(invalid_payload("SECRET-B"))
    assert a.aggregation_status == b.aggregation_status == "INVALID"
    assert a.normalized_bundle_fingerprint is b.normalized_bundle_fingerprint is None
    assert a.aggregation_audit_fingerprint != b.aggregation_audit_fingerprint
    serialized_a = json.dumps(a.to_dict(), sort_keys=True)
    serialized_b = json.dumps(b.to_dict(), sort_keys=True)
    assert "SECRET-A" not in serialized_a
    assert "SECRET-B" not in serialized_b


def test_same_invalid_element_is_stable_and_mapping_order_independent():
    first = build_provisional_verification_traceability_bundle(invalid_payload("SAME"))
    second = build_provisional_verification_traceability_bundle(invalid_payload("SAME"))
    reordered = build_provisional_verification_traceability_bundle(invalid_payload("SAME", reverse_mapping=True))
    assert first.aggregation_audit_fingerprint == second.aggregation_audit_fingerprint
    assert first.aggregation_audit_fingerprint == reordered.aggregation_audit_fingerprint


def test_invalid_element_record_is_closed_and_sha256_validated():
    collection = validate_and_normalize_provisional_collections(invalid_payload("A"))
    assert len(collection.invalid_element_records) == 1
    record = collection.invalid_element_records[0]
    assert set(record) == {"collection", "position", "reason_code", "raw_element_fingerprint"}
    assert re.fullmatch(r"[0-9a-f]{64}", record["raw_element_fingerprint"])
    raw = collection.to_dict()
    bad = dict(raw["invalid_element_records"][0])
    bad["raw_element_fingerprint"] = "bad"
    raw["invalid_element_records"] = (bad,)
    with pytest.raises(ValueError, match="INVALID_SHA256"):
        validate_provisional_collection_validation_result_contract(raw)


def test_invalid_element_requires_matching_positional_issue():
    collection = validate_and_normalize_provisional_collections(invalid_payload("A"))
    raw = collection.to_dict()
    raw["collection_issue_codes"] = tuple(
        code for code in raw["collection_issue_codes"]
        if not code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID:")
    )
    with pytest.raises(ValueError, match="ISSUE_REQUIRED|POSITIONAL_ISSUE_MISMATCH"):
        validate_provisional_collection_validation_result_contract(raw)


def test_policy_and_schema_versions_change_audit_even_for_invalid_batch():
    payload = invalid_payload("A")
    base = build_provisional_verification_traceability_bundle(payload)
    policy_changed = deepcopy(payload)
    policy_changed["policy_versions"] = {**policy_changed["policy_versions"], "verification": "changed-policy"}
    schema_changed = deepcopy(payload)
    schema_changed["schema_versions"] = {**schema_changed["schema_versions"], "bundle": "changed-schema"}
    by_policy = build_provisional_verification_traceability_bundle(policy_changed)
    by_schema = build_provisional_verification_traceability_bundle(schema_changed)
    assert base.normalized_bundle_fingerprint is None
    assert by_policy.normalized_bundle_fingerprint is None
    assert by_schema.normalized_bundle_fingerprint is None
    assert base.aggregation_audit_fingerprint != by_policy.aggregation_audit_fingerprint
    assert base.aggregation_audit_fingerprint != by_schema.aggregation_audit_fingerprint


def test_audit_payload_versions_are_canonically_ordered():
    empty = {
        "invalid_element_records": (), "duplicate_records": (), "collection_issue_codes": (), "collection_warnings": (),
    }
    common = dict(
        collection=empty, referential={}, rows={}, metric_result={}, input_fingerprints={},
        aggregation_status="INVALID", metrics_status="NOT_COMPUTED", partial_reason_codes=(),
    )
    a = _phase656_audit_payload(**common, policy_versions={"b":"2","a":"1"}, schema_versions={"z":"9","x":"7"})
    b = _phase656_audit_payload(**common, policy_versions={"a":"1","b":"2"}, schema_versions={"x":"7","z":"9"})
    assert _phase656_sha256_payload(a) == _phase656_sha256_payload(b)


def test_unknown_aggregation_codes_and_warnings_are_rejected():
    bundle = build_provisional_verification_traceability_bundle(aligned_payload())
    raw = bundle.to_dict(); raw["aggregation_issue_codes"] = ("INVENTED_ISSUE",)
    with pytest.raises(ValueError, match="aggregation_issue_codes:UNKNOWN"):
        validate_provisional_verification_traceability_bundle_contract(raw)
    raw = bundle.to_dict(); raw["aggregation_warnings"] = ("INVENTED_WARNING",)
    with pytest.raises(ValueError, match="aggregation_warnings:UNKNOWN"):
        validate_provisional_verification_traceability_bundle_contract(raw)


def test_invalid_state_matrix_is_closed():
    bundle = build_provisional_verification_traceability_bundle(invalid_payload("A"))
    raw = bundle.to_dict(); raw["normalized_bundle_status"] = "COMPUTED"; raw["normalized_bundle_fingerprint"] = "a" * 64
    with pytest.raises(ValueError, match="INVALID|NORMALIZED"):
        validate_provisional_verification_traceability_bundle_contract(raw)
    raw = bundle.to_dict(); raw["metrics_status"] = "COMPUTED"
    with pytest.raises(ValueError, match="INVALID|METRICS"):
        validate_provisional_verification_traceability_bundle_contract(raw)


def test_valid_pipeline_is_deterministic_and_side_effect_free():
    payload = aligned_payload()
    original = deepcopy(payload)
    first = build_provisional_verification_traceability_bundle(payload)
    reordered = {key: (tuple(reversed(value)) if isinstance(value, tuple) else value) for key, value in payload.items()}
    second = build_provisional_verification_traceability_bundle(reordered)
    assert payload == original
    assert first.normalized_bundle_fingerprint == second.normalized_bundle_fingerprint
    assert first.aggregation_audit_fingerprint == second.aggregation_audit_fingerprint
    assert first.result_contract_valid is True
    assert first.correction_applied is False
    assert first.official_artifacts_created is False
    assert first.additional_llm_calls == 0
    assert first.additional_retrieval_rounds == 0
