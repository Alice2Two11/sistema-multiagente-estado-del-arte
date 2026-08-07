from copy import deepcopy
import pytest

from src.tools.verification.traceability import (
    ClaimTraceabilityRow, CorrectionTraceabilityRow, MetricValue,
    ProvisionalVerificationMetrics,
)
from src.tools.verification.validation import create_provisional_verification_traceability_bundle
from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.resolution import (
    resolve_multiple_correction_proposals,
    validate_provisional_multi_proposal_resolution_result,
)


def mv():
    return MetricValue(None,0,0,"NOT_COMPUTABLE","entities","eligible").to_dict()


def metrics(computed=True):
    rate = mv() if computed else None
    return ProvisionalVerificationMetrics(
        candidate_issue_resolution_rate=rate,
        accepted_issue_resolution_rate=rate,
        correction_acceptance_rate=rate,
        new_issue_rate=rate,
        hallucination_risk_reduction_rate=rate,
        recommendations_generated=mv(),
    ).to_dict()


def claim(cid="c1", text="Alpha beta gamma.", corrections=(), decisions=(), accepted=(), rejected=(), deferred=()):
    return ClaimTraceabilityRow(
        cid,"s1","SUBSTANTIVE_FACTUAL",text,"PARTIALLY_SUPPORTED",
        ("PARTIAL_SUPPORT",),"MEDIUM",False,bool(corrections),tuple(corrections),tuple(decisions),
        tuple(accepted),tuple(rejected),tuple(deferred),("PARTIAL_SUPPORT",),bool(deferred),False,
    ).to_dict()


def corr(correction_id, start, end, target, replacement, decision="ACCEPT_FOR_07C", *, resolved=("PARTIAL_SUPPORT",), new=()):
    text="Alpha beta gamma."
    proposed=text[:start]+replacement+text[end:]
    return CorrectionTraceabilityRow(
        correction_id,"c1","s1","NARROW_SCOPE",True,False,
        "AVAILABLE","AVAILABLE","AVAILABLE","AVAILABLE",
        "ACCEPTED_FOR_REVERIFICATION","PRECHECK_PASSED","COMPLETED",decision,
        ("PARTIAL_SUPPORT",),tuple(resolved),(),tuple(new),"MEDIUM","LOW","REDUCED",
        "a"*64,"b"*64,"c"*64,"d"*64,(),(),(),(),None,decision=="DEFER_TO_MANUAL_REVIEW",False,
        fingerprint_text(text),{"start":start,"end":end,"text":target,"coordinate_system":"PYTHON_CODEPOINT_OFFSETS"},replacement,proposed,
    ).to_dict()


def bundle(corrections=(), claim_row=None, status="VALID"):
    if status=="INVALID":
        return create_provisional_verification_traceability_bundle(
            claim_traceability_rows=(),correction_traceability_rows=(),claim_evidence_traceability_rows=(),
            correction_evidence_traceability_rows=(),reverification_traceability_rows=(),metrics=metrics(False),
            aggregation_status="INVALID",metrics_status="NOT_COMPUTED",partial_reason_codes=(),
            aggregation_issue_codes=("AGGREGATION_METRICS_INPUT_INVALID",),aggregation_warnings=(),
            normalized_bundle_status="NOT_COMPUTABLE",normalized_bundle_fingerprint=None,
            aggregation_audit_fingerprint="f"*64,input_collection_fingerprints={},
            policy_versions={"verification":"v1"},schema_versions={"bundle":"v1"},
            correction_applied=False,official_artifacts_created=False,additional_llm_calls=0,additional_retrieval_rounds=0,
        )
    if claim_row is None:
        ids=tuple(x["correction_id"] for x in corrections)
        decisions=tuple(x["acceptance_decision"] for x in corrections if x["acceptance_decision"])
        claim_row=claim(corrections=ids,decisions=decisions,
            accepted=tuple(x["correction_id"] for x in corrections if x["acceptance_decision"]=="ACCEPT_FOR_07C"),
            rejected=tuple(x["correction_id"] for x in corrections if x["acceptance_decision"]=="REJECT_PROPOSAL"),
            deferred=tuple(x["correction_id"] for x in corrections if x["acceptance_decision"]=="DEFER_TO_MANUAL_REVIEW"))
    return create_provisional_verification_traceability_bundle(
        claim_traceability_rows=(claim_row,),correction_traceability_rows=tuple(sorted(corrections,key=lambda x:x["correction_id"])),
        claim_evidence_traceability_rows=(),correction_evidence_traceability_rows=(),reverification_traceability_rows=(),
        metrics=metrics(True),aggregation_status=status,metrics_status="COMPUTED",partial_reason_codes=(),
        aggregation_issue_codes=(),aggregation_warnings=(),normalized_bundle_status="COMPUTED",
        normalized_bundle_fingerprint="1"*64,aggregation_audit_fingerprint="2"*64,input_collection_fingerprints={},
        policy_versions={"verification":"v1"},schema_versions={"bundle":"v1"},
        correction_applied=False,official_artifacts_created=False,additional_llm_calls=0,additional_retrieval_rounds=0,
    )


def test_claim_without_corrections():
    r=resolve_multiple_correction_proposals(bundle((),claim(corrections=())))
    assert r.claim_resolution_plans[0]["plan_type"]=="NO_ACCEPTED_CORRECTIONS"
    assert not r.eligible_for_07c


def test_single_accepted():
    r=resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),)))
    p=r.claim_resolution_plans[0]
    assert p["plan_type"]=="SINGLE_ACCEPTED_CORRECTION"
    assert p["virtual_result_text"]=="Alpha B gamma."
    assert p["eligible_for_07c"]


def test_two_disjoint_and_different_lengths_right_to_left():
    rows=(corr("x2",11,16,"gamma","GAMMA-LONG"),corr("x1",6,10,"beta","B"))
    r=resolve_multiple_correction_proposals(bundle(rows))
    assert r.pair_relations[0]["relation_type"]=="INDEPENDENT"
    assert r.claim_resolution_plans[0]["virtual_result_text"]=="Alpha B GAMMA-LONG."
    assert r.claim_resolution_plans[0]["application_order"]==("x2","x1")


def test_redundant_not_applied_twice():
    rows=(corr("x2",6,10,"beta","B"),corr("x1",6,10,"beta","B"))
    p=resolve_multiple_correction_proposals(bundle(rows)).claim_resolution_plans[0]
    assert p["plan_type"]=="MULTIPLE_REDUNDANT_CORRECTIONS"
    assert p["selected_correction_ids"]==("x1",)
    assert p["redundant_correction_ids"]==("x2",)
    assert p["virtual_result_text"]=="Alpha B gamma."


@pytest.mark.parametrize("rows",[
    (corr("x1",6,10,"beta","B"),corr("x2",6,10,"beta","C")),
    (corr("x1",6,12,"beta g","X"),corr("x2",10,16," gamma","Y")),
])
def test_overlapping_conflict_has_no_sequence(rows):
    r=resolve_multiple_correction_proposals(bundle(rows)); p=r.claim_resolution_plans[0]
    assert p["plan_type"]=="MULTIPLE_CONFLICTING_CORRECTIONS"
    assert p["application_order"]==()
    assert not p["eligible_for_07c"]


def test_accepted_and_rejected_keeps_individual_decisions():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G",decision="REJECT_PROPOSAL"))
    p=resolve_multiple_correction_proposals(bundle(rows)).claim_resolution_plans[0]
    assert p["selected_correction_ids"]==("x1",)
    assert p["rejected_correction_ids"]==("x2",)


def test_accepted_and_deferred_requires_manual_review():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G",decision="DEFER_TO_MANUAL_REVIEW"))
    p=resolve_multiple_correction_proposals(bundle(rows)).claim_resolution_plans[0]
    assert p["plan_type"]=="MANUAL_REVIEW_REQUIRED"
    assert not p["eligible_for_07c"]


def test_three_compatible():
    text="Alpha beta gamma."
    rows=(corr("x1",0,5,"Alpha","A"),corr("x2",6,10,"beta","B"),corr("x3",11,16,"gamma","G"))
    p=resolve_multiple_correction_proposals(bundle(rows)).claim_resolution_plans[0]
    assert p["plan_type"]=="MULTIPLE_COMPATIBLE_CORRECTIONS"
    assert p["virtual_result_text"]=="A B G."


def test_input_order_deterministic_and_fingerprints_stable():
    rows=[corr("x2",11,16,"gamma","G"),corr("x1",6,10,"beta","B")]
    a=resolve_multiple_correction_proposals(bundle(tuple(rows)))
    b=resolve_multiple_correction_proposals(bundle(tuple(reversed(rows))))
    assert a.multi_proposal_resolution_fingerprint==b.multi_proposal_resolution_fingerprint
    assert a.to_dict()==b.to_dict()


def test_invalid_bundle_blocked_auditable():
    r=resolve_multiple_correction_proposals(bundle(status="INVALID"))
    assert r.resolution_status=="BLOCKED"
    assert r.multi_proposal_resolution_fingerprint is None
    assert len(r.multi_proposal_audit_fingerprint)==64
    assert not r.eligible_for_07c


def test_audit_fingerprint_sensitive_to_conflict():
    independent=resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G"))))
    conflict=resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),corr("x2",6,10,"beta","C"))))
    assert independent.multi_proposal_audit_fingerprint!=conflict.multi_proposal_audit_fingerprint


def test_result_invariants_zero_and_no_application():
    r=resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),)))
    assert r.additional_llm_calls==r.additional_retrieval_rounds==0
    assert not r.correction_applied and not r.official_artifacts_created
    assert validate_provisional_multi_proposal_resolution_result(r.to_dict())["result_contract_valid"]

# Phase 6.7 adversarial closure tests.
from dataclasses import replace as dc_replace
from src.tools.verification import resolution as resolution_module
from src.config.verification_policy_config import (
    PROVISIONAL_BUNDLE_FINGERPRINT_VERSION,
    PROVISIONAL_BUNDLE_FINGERPRINT_VERSION_V1,
)


def _replan(payload):
    payload = deepcopy(payload)
    payload["claim_resolution_plan_fingerprint"] = resolution_module._sha256(
        resolution_module._plan_payload(payload)
    )
    return payload


def test_phase67_normalized_fingerprint_tamper_rejected():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    data["multi_proposal_resolution_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="MULTI_PROPOSAL_RESOLUTION_FINGERPRINT_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_audit_fingerprint_tamper_rejected():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    data["multi_proposal_audit_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="MULTI_PROPOSAL_AUDIT_FINGERPRINT_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_global_eligibility_is_derived():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    data["eligible_for_07c"] = False
    with pytest.raises(ValueError, match="MULTI_PROPOSAL_GLOBAL_ELIGIBILITY_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_resolution_status_is_derived():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    data["resolution_status"] = "PARTIAL"
    with pytest.raises(ValueError, match="MULTI_PROPOSAL_RESOLUTION_STATUS_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_decision_sets_cannot_overlap():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    plan = dict(data["claim_resolution_plans"][0])
    plan["rejected_correction_ids"] = ("x1",)
    plan["individual_decisions"] = {"x1":"ACCEPT_FOR_07C"}
    data["claim_resolution_plans"] = (_replan(plan),)
    with pytest.raises(ValueError, match="DECISION_SETS_OVERLAP"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_individual_decision_must_match_partition():
    data = resolve_multiple_correction_proposals(bundle((corr("x1",6,10,"beta","B"),))).to_dict()
    plan = dict(data["claim_resolution_plans"][0])
    plan["individual_decisions"] = {"x1":"REJECT_PROPOSAL"}
    data["claim_resolution_plans"] = (_replan(plan),)
    with pytest.raises(ValueError, match="INDIVIDUAL_DECISION_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_single_plan_rejects_two_selected():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G"))
    data=resolve_multiple_correction_proposals(bundle(rows)).to_dict()
    plan=dict(data["claim_resolution_plans"][0]); plan["plan_type"]="SINGLE_ACCEPTED_CORRECTION"
    data["claim_resolution_plans"] = (_replan(plan),)
    with pytest.raises(ValueError, match="SINGLE_INVALID"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_compatible_plan_rejects_blocking_relation():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G"))
    data=resolve_multiple_correction_proposals(bundle(rows)).to_dict()
    pair=dict(data["pair_relations"][0]); pair["relation_type"]="OVERLAPPING_CONFLICTING"; pair["reason_codes"]=("PAIR_OVERLAPPING_INCOMPATIBLE_REPLACEMENTS",)
    data["pair_relations"]=(pair,)
    with pytest.raises(ValueError, match="COMPATIBLE_HAS_BLOCKING_RELATION"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_duplicate_pair_rejected():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G"))
    data=resolve_multiple_correction_proposals(bundle(rows)).to_dict(); pair=data["pair_relations"][0]
    data["pair_relations"]=(pair,pair)
    with pytest.raises(ValueError, match="PAIR_RELATION_DUPLICATE"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_missing_pair_rejected():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G"))
    data=resolve_multiple_correction_proposals(bundle(rows)).to_dict(); data["pair_relations"]=()
    with pytest.raises(ValueError, match="PAIR_RELATION_COVERAGE_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_relation_with_rejected_correction_rejected():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",11,16,"gamma","G",decision="REJECT_PROPOSAL"))
    data=resolve_multiple_correction_proposals(bundle(rows)).to_dict()
    data["pair_relations"] = ({"claim_id":"c1","section_id":"s1","left_correction_id":"x1","right_correction_id":"x2","relation_type":"INDEPENDENT","reason_codes":("PAIR_POSITIONALLY_DISJOINT","PAIR_SEMANTIC_COMPATIBILITY_NOT_ASSERTED")},)
    with pytest.raises(ValueError, match="PAIR_RELATION_CORRECTION_NOT_ACCEPTED"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_phase67_redundancy_audit_survives_defer():
    rows=(corr("x1",6,10,"beta","B"),corr("x2",6,10,"beta","B"),corr("x3",11,16,"gamma","G",decision="DEFER_TO_MANUAL_REVIEW"))
    plan=resolve_multiple_correction_proposals(bundle(rows)).claim_resolution_plans[0]
    assert plan["plan_type"]=="MANUAL_REVIEW_REQUIRED"
    assert plan["selected_correction_ids"]==()
    assert plan["application_order"]==()
    assert plan["redundant_correction_ids"]==("x2",)


def test_phase67_neutral_claim_does_not_block_applicable_claim():
    c1=claim("c1",corrections=("x1",),decisions=("ACCEPT_FOR_07C",),accepted=("x1",))
    c2=claim("c2",text="Neutral claim.",corrections=())
    row=corr("x1",6,10,"beta","B")
    b=create_provisional_verification_traceability_bundle(
        claim_traceability_rows=(c1,c2),correction_traceability_rows=(row,),claim_evidence_traceability_rows=(),
        correction_evidence_traceability_rows=(),reverification_traceability_rows=(),metrics=metrics(True),
        aggregation_status="VALID",metrics_status="COMPUTED",partial_reason_codes=(),aggregation_issue_codes=(),aggregation_warnings=(),
        normalized_bundle_status="COMPUTED",normalized_bundle_fingerprint="1"*64,aggregation_audit_fingerprint="2"*64,
        input_collection_fingerprints={},policy_versions={"verification":"v1"},schema_versions={"bundle":"v2"},
        correction_applied=False,official_artifacts_created=False,additional_llm_calls=0,additional_retrieval_rounds=0,
    )
    result=resolve_multiple_correction_proposals(b)
    plans={p["claim_id"]:p for p in result.claim_resolution_plans}
    assert plans["c2"]["plan_type"]=="NO_ACCEPTED_CORRECTIONS"
    assert not plans["c2"]["requires_07c"] and not plans["c2"]["blocks_07c"]
    assert result.eligible_for_07c and result.resolution_status=="COMPLETED"


def test_phase67_bundle_contract_is_explicitly_v2():
    assert PROVISIONAL_BUNDLE_FINGERPRINT_VERSION_V1=="AGENT07_PROVISIONAL_BUNDLE_V1"
    assert PROVISIONAL_BUNDLE_FINGERPRINT_VERSION=="AGENT07_PROVISIONAL_BUNDLE_V4"
