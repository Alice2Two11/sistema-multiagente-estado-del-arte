from copy import deepcopy
from dataclasses import asdict
import pytest

from src.tools.verification.corrections import compute_correction_proposal_fingerprint, fingerprint_text
from src.tools.verification.traceability import ClaimVerificationAggregationRecord
from src.tools.verification.validation import (
    validate_claim_verification_result_contract,
    validate_correction_proposal_contract,
    validate_claim_verification_aggregation_record,
)


def valid_claim_result():
    ev={"evidence_id":"ev1","source_filename":"paper.pdf","chunk_id":"c1","authorized_for_section":True,"usage_role":"SUPPORT"}
    return {
      "claim_id":"cl1","claim_type":"FACTUAL","scientific_judgment_required":True,
      "execution_status":"COMPLETED","technical_status":"OK","technical_issue_codes":[],
      "scientific_judgment_status":"COMPLETED","scientific_verdict":"SUPPORTED","support_level":"STRONG",
      "deterministic_issue_codes":[],"semantic_issue_codes":[],"eligible_evidence":[ev],
      "deterministically_discarded_evidence":[],"evidence_used":[ev],"evidence_rejected":[],
      "contradiction_assessment":{"type":"NONE","evidence_ids":[]},"numeric_assessment":"NOT_APPLICABLE",
      "attribution_assessment":"NOT_APPLICABLE","extrapolation_assessment":"NOT_APPLICABLE",
      "hallucination_risk":"LOW","llm_correction_recommendation":False,
      "final_correction_eligibility":"NO_CORRECTION_NEEDED","manual_review_required":False,
      "reason_codes":[],"tool_usage":{"llm_calls":1,"retrieval_requested":0,"retrieval_rounds":1},
      "decision_trace":["VALIDATED"],"raw_attempts":[],"result_contract_valid":True,
      "scientific_validation_ok":True,"validation_ok":True,
    }


def valid_proposal():
    target_span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("10 ms"),"start":0,"end":2,"text":"10"}
    p={
      "correction_id":"corr1","claim_id":"cl1","section_id":"sec1","correction_decision":"PROPOSE_CHANGE",
      "action_type":"REPLACE_NUMERIC_VALUE","proposal_status":"ACCEPTED_FOR_REVERIFICATION",
      "original_text":"10 ms","original_claim_fingerprint":fingerprint_text("10 ms"),"original_section_fingerprint":fingerprint_text("10 ms"),
      "claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("10 ms"),"start":0,"end":5,"text":"10 ms"},
      "target_span_in_claim":target_span,"localization_method":"CONTRACTUAL_SPAN","target_text":"10",
      "target_text_fingerprint":fingerprint_text("10"),"replacement_text":"12","proposed_claim_text":"12 ms","evidence_ids":["ev1"],
      "reason_codes":["LOCALIZED_NUMERIC_ERROR"],"change_scope":"TOKEN","semantic_change_level":"MINIMAL",
      "old_citation_refs":[],"new_citation_refs":[],"citation_text_span":None,
      "old_numeric_pairs":[["10","ms"]],"new_numeric_pairs":[["12","ms"]],"metric_context":"latency","unit_context":"ms",
      "old_attribution_elements":[],"new_attribution_elements":[],"attribution_relation":None,"new_entities":[],
      "new_citations":[],"new_attributions":[],"new_conditions":[],"new_technical_terms":[],
      "llm_correction_recommendation":True,"requires_manual_review":False,"accepted_for_reverification":True,
      "correction_applied":False,"final_proposal_status":"ACCEPTED_FOR_REVERIFICATION","proposal_fingerprint":"",
      "prompt_version":"AGENT07_CORRECTION_USER_V3_5T","validation_issue_codes":[],"decision_path":["PROPOSED"],
      "raw_attempts":[],"retry_metrics":{"llm_calls":1,"format_attempts":1,"schema_attempts":1},
    }
    p["proposal_fingerprint"]=compute_correction_proposal_fingerprint(
      original_claim_fingerprint=p["original_claim_fingerprint"],original_section_fingerprint=p["original_section_fingerprint"],
      target_text_fingerprint=p["target_text_fingerprint"],claim_id=p["claim_id"],action_type=p["action_type"],
      target_span=p["target_span_in_claim"],replacement_text=p["replacement_text"],evidence_ids=p["evidence_ids"],prompt_version=p["prompt_version"])
    return p


def test_claim_valid_complete(): assert validate_claim_verification_result_contract(valid_claim_result())["claim_id"]=="cl1"
def test_claim_missing_field():
    x=valid_claim_result(); x.pop("claim_id")
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_unknown_verdict():
    x=valid_claim_result(); x["scientific_verdict"]="MADE_UP"
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_support_mismatch():
    x=valid_claim_result(); x["support_level"]="PARTIAL"
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_unknown_technical_status():
    x=valid_claim_result(); x["technical_status"]="NOPE"
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_blocked_incoherent():
    x=valid_claim_result(); x["scientific_judgment_status"]="BLOCKED"
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_llm_calls_invalid():
    x=valid_claim_result(); x["tool_usage"]["llm_calls"]=-1
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_retrieval_negative():
    x=valid_claim_result(); x["tool_usage"]["retrieval_rounds"]=-1
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_evidence_identity_conflict():
    x=valid_claim_result(); x["evidence_rejected"]=[{"evidence_id":"ev1","source_filename":"other.pdf","chunk_id":"c2"}]
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_unknown_used_evidence():
    x=valid_claim_result(); x["evidence_used"]=[{"evidence_id":"ev2","source_filename":"paper.pdf","chunk_id":"c2"}]
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_unknown_issue():
    x=valid_claim_result(); x["semantic_issue_codes"]=["UNKNOWN"]
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_unknown_risk():
    x=valid_claim_result(); x["hallucination_risk"]="EXTREME"
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)
def test_claim_recommendation_eligibility_incoherent():
    x=valid_claim_result(); x["llm_correction_recommendation"]=True
    with pytest.raises(ValueError): validate_claim_verification_result_contract(x)

def test_proposal_valid_complete(): assert validate_correction_proposal_contract(valid_proposal())["correction_id"]=="corr1"
def test_proposal_unknown_action():
    x=valid_proposal(); x["action_type"]="NOPE"
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_status_incompatible():
    x=valid_proposal(); x["proposal_status"]="PROPOSED"
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_acceptance_incoherent():
    x=valid_proposal(); x["accepted_for_reverification"]=False
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_applied_true():
    x=valid_proposal(); x["correction_applied"]=True
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_fingerprint_altered():
    x=valid_proposal(); x["replacement_text"]="13"
    with pytest.raises(ValueError, match="PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH"): validate_correction_proposal_contract(x)
def test_proposal_span_invalid():
    x=valid_proposal(); x["target_span_in_claim"]["end"]=-1
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_evidence_duplicate():
    x=valid_proposal(); x["evidence_ids"]=["ev1","ev1"]
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_retry_invalid():
    x=valid_proposal(); x["retry_metrics"]["llm_calls"]=-1
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)
def test_proposal_action_fields_incompatible():
    x=valid_proposal(); x["new_numeric_pairs"]=[]
    with pytest.raises(ValueError): validate_correction_proposal_contract(x)

def test_wrapper_valid_and_deterministic():
    record=ClaimVerificationAggregationRecord("sec1",valid_claim_result())
    first=record.to_dict(); second=record.to_dict(); assert first==second
    assert validate_claim_verification_aggregation_record(first)["section_id"]=="sec1"
def test_wrapper_missing_section():
    with pytest.raises(ValueError): validate_claim_verification_aggregation_record({"section_id":"","claim_verification_result":valid_claim_result()})
def test_wrapper_invalid_result():
    x=valid_claim_result(); x["scientific_verdict"]="NOPE"
    with pytest.raises(ValueError): validate_claim_verification_aggregation_record({"section_id":"sec1","claim_verification_result":x})
def test_wrapper_has_no_invented_ids_or_fingerprints():
    d=ClaimVerificationAggregationRecord("sec1",valid_claim_result()).to_dict()
    assert set(d)=={"section_id","claim_verification_result"}
    assert not any("fingerprint" in k or k.endswith("_id") and k!="section_id" for k in d)
