from copy import deepcopy
from dataclasses import asdict
import pytest

from src.agents.verification_agent import ClaimVerificationResult
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.corrections import (
    CorrectionProposal, _empty_proposal, compute_correction_proposal_fingerprint,
    compute_empty_correction_proposal_fingerprint, fingerprint_text,
)
from src.tools.verification.validation import (
    validate_claim_verification_result_contract,
    validate_correction_proposal_contract,
)


def evidence(eid="ev1"):
    return {"evidence_id":eid,"source_filename":"paper.pdf","chunk_id":"c1","authorized_for_section":True,"usage_role":"SUPPORT"}


def tool_usage(llm=1, req=0, rounds=1):
    return {"tool_names_considered":(),"tool_names_selected":(),"tools_considered":0,"tools_selected":0,
            "retrieval_requested":req,"retrieval_rounds":rounds,"evidence_selected":1,"llm_calls":llm,
            "format_attempts":1,"schema_validation_attempts":1,"scientific_judgment_attempts":1,
            "format_retries":0,"schema_retries":0,"total_response_retries":0}


def claim_result(**overrides):
    data=dict(claim_id="cl1",claim_type="FACTUAL",scientific_judgment_required=True,execution_status="COMPLETED",
      technical_status="OK",technical_issue_codes=(),scientific_judgment_status="COMPLETED",scientific_verdict="SUPPORTED",
      support_level="STRONG",deterministic_issue_codes=(),semantic_issue_codes=(),eligible_evidence=(evidence(),),
      deterministically_discarded_evidence=(),evidence_used=(evidence(),),evidence_rejected=(),
      contradiction_assessment={"type":"NONE","evidence_ids":()},numeric_assessment="NOT_APPLICABLE",
      attribution_assessment="NOT_APPLICABLE",extrapolation_assessment="NOT_APPLICABLE",hallucination_risk="LOW",
      llm_correction_recommendation=False,final_correction_eligibility="NO_CORRECTION_NEEDED",manual_review_required=False,
      reason_codes=(),tool_usage=tool_usage(),decision_trace=("VERDICT_SUPPORTED",),raw_attempts=(),result_contract_valid=True,
      scientific_validation_ok=True,validation_ok=True)
    data.update(overrides)
    return ClaimVerificationResult(**data)


def localized_proposal(status="ACCEPTED_FOR_REVERIFICATION"):
    target={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("10 ms"),"start":0,"end":2,"text":"10"}
    claim_span={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("10 ms"),"start":0,"end":5,"text":"10 ms"}
    kw=dict(correction_id="corr1",claim_id="cl1",section_id="sec1",correction_decision="PROPOSE_CHANGE",action_type="REPLACE_NUMERIC_VALUE",
      proposal_status=status,original_text="10 ms",original_claim_fingerprint=fingerprint_text("10 ms"),original_section_fingerprint=fingerprint_text("10 ms"),
      claim_span_in_section=claim_span,target_span_in_claim=target,localization_method="CONTRACTUAL_SPAN",target_text="10",target_text_fingerprint=fingerprint_text("10"),
      replacement_text="12",proposed_claim_text="12 ms",evidence_ids=("ev1",),reason_codes=("LOCALIZED_NUMERIC_ERROR",),change_scope="TOKEN",semantic_change_level="MINIMAL",
      old_citation_refs=(),new_citation_refs=(),citation_text_span=None,old_numeric_pairs=(("10","ms"),),new_numeric_pairs=(("12","ms"),),metric_context="latency",unit_context="ms",
      old_attribution_elements=(),new_attribution_elements=(),attribution_relation=None,new_entities=(),new_citations=(),new_attributions=(),new_conditions=(),new_technical_terms=(),
      llm_correction_recommendation=True,requires_manual_review=status=="REJECTED",accepted_for_reverification=status=="ACCEPTED_FOR_REVERIFICATION",correction_applied=False,
      final_proposal_status=status,proposal_fingerprint="",prompt_version="AGENT07_CORRECTION_USER_V3_5T",validation_issue_codes=(),decision_path=("LOCALIZATION_CONFIRMED",),raw_attempts=(),
      retry_metrics={"llm_calls":1,"format_attempts":1,"format_retries":0,"schema_validation_attempts":1,"schema_retries":0,"total_response_retries":0})
    kw["proposal_fingerprint"]=compute_correction_proposal_fingerprint(original_claim_fingerprint=kw["original_claim_fingerprint"],original_section_fingerprint=kw["original_section_fingerprint"],target_text_fingerprint=kw["target_text_fingerprint"],claim_id=kw["claim_id"],action_type=kw["action_type"],target_span=kw["target_span_in_claim"],replacement_text=kw["replacement_text"],evidence_ids=kw["evidence_ids"],prompt_version=kw["prompt_version"])
    return CorrectionProposal(**kw)


def empty(decision,status,issues=()):
    return _empty_proposal("cl1","sec1","claim",fingerprint_text("claim"),fingerprint_text("section"),decision,status,("PATH",),get_verification_input_policy(),issues=issues)


def test_real_completed_claim_to_dict():
    out=validate_claim_verification_result_contract(claim_result().to_dict())
    assert out["contradiction_assessment"]["type"]=="NONE"
    assert out["tool_usage"]["llm_calls"]==1


def test_real_deterministic_terminal_reason_overlap():
    r=claim_result(scientific_judgment_required=True,scientific_judgment_status="COMPLETED",scientific_verdict="NOT_EVALUATED",support_level="NONE",
      deterministic_issue_codes=("INVALID_CITATION",),reason_codes=("INVALID_CITATION",),eligible_evidence=(evidence(),),evidence_used=(),
      hallucination_risk="HIGH",final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE",scientific_validation_ok=False,tool_usage=tool_usage(0,0,1))
    assert validate_claim_verification_result_contract(r.to_dict())["reason_codes"]==("INVALID_CITATION",)


def test_real_blocked_claim_to_dict():
    r=claim_result(technical_status="LLM_UNAVAILABLE",technical_issue_codes=("LLM_UNAVAILABLE",),scientific_judgment_status="BLOCKED",
      scientific_verdict="NOT_EVALUATED",support_level="NONE",evidence_used=(),hallucination_risk="MEDIUM",manual_review_required=True,
      final_correction_eligibility="MANUAL_REVIEW_REQUIRED",reason_codes=("LLM_UNAVAILABLE",),scientific_validation_ok=False,tool_usage=tool_usage(0,0,1))
    assert validate_claim_verification_result_contract(r.to_dict())["scientific_judgment_status"]=="BLOCKED"


def test_contradiction_historical_type_shape():
    r=claim_result(scientific_verdict="CONTRADICTED",support_level="NONE",semantic_issue_codes=("CLAIM_EVIDENCE_CONFLICT",),
      contradiction_assessment={"type":"CLAIM_EVIDENCE_CONFLICT","evidence_ids":("ev1",)},hallucination_risk="HIGH",
      final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE",scientific_validation_ok=False)
    assert validate_claim_verification_result_contract(r.to_dict())["contradiction_assessment"]["type"]=="CLAIM_EVIDENCE_CONFLICT"


def test_legacy_contradiction_type_key_rejected():
    d=claim_result().to_dict(); d["contradiction_assessment"]={"contradiction_type":"NONE"}
    with pytest.raises(ValueError): validate_claim_verification_result_contract(d)


def test_localized_accepted_from_dataclass():
    assert validate_correction_proposal_contract(localized_proposal().to_dict())["accepted_for_reverification"] is True


def test_localized_rejected_from_dataclass():
    assert validate_correction_proposal_contract(localized_proposal("REJECTED").to_dict())["proposal_status"]=="REJECTED"


@pytest.mark.parametrize("decision,status",[("NO_CORRECTION","NOT_PROPOSED"),("DEFER_TO_MANUAL_REVIEW","DEFERRED"),("DEFER_TO_MANUAL_REVIEW","REJECTED")])
def test_empty_producer_results(decision,status):
    p=empty(decision,status)
    out=validate_correction_proposal_contract(p.to_dict())
    assert out["action_type"] is None and out["target_span_in_claim"] is None


def test_empty_fingerprint_helper_matches_producer():
    p=empty("NO_CORRECTION","NOT_PROPOSED")
    expected=compute_empty_correction_proposal_fingerprint(claim_id="cl1",decision="NO_CORRECTION",status="NOT_PROPOSED",prompt_version=p.prompt_version)
    assert p.proposal_fingerprint==expected


def test_empty_fingerprint_altered():
    d=empty("NO_CORRECTION","NOT_PROPOSED").to_dict(); d["proposal_fingerprint"]="0"*64
    with pytest.raises(ValueError,match="FINGERPRINT_MISMATCH"): validate_correction_proposal_contract(d)


def test_none_only_for_empty_results():
    d=localized_proposal().to_dict(); d["action_type"]=None
    with pytest.raises(ValueError): validate_correction_proposal_contract(d)


def test_missing_spans_only_for_empty_results():
    d=localized_proposal("REJECTED").to_dict(); d["target_span_in_claim"]=None
    with pytest.raises(ValueError): validate_correction_proposal_contract(d)
    assert validate_correction_proposal_contract(empty("DEFER_TO_MANUAL_REVIEW","REJECTED").to_dict())["target_span_in_claim"] is None


def test_invented_decision_status_combination():
    d=empty("NO_CORRECTION","NOT_PROPOSED").to_dict(); d["proposal_status"]=d["final_proposal_status"]="DEFERRED"
    d["proposal_fingerprint"]=compute_empty_correction_proposal_fingerprint(claim_id=d["claim_id"],decision=d["correction_decision"],status="DEFERRED",prompt_version=d["prompt_version"])
    with pytest.raises(ValueError,match="DECISION_STATUS"): validate_correction_proposal_contract(d)
