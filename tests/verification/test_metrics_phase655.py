from copy import deepcopy
import pytest

from test_referential_integrity_phase653 import aligned_payload
from src.tools.verification.traceability import CorrectionTraceabilityRow
from src.tools.verification.validation import (
    validate_and_normalize_provisional_collections,
    validate_provisional_referential_integrity,
    build_provisional_traceability_rows,
    validate_provisional_traceability_rows_result_contract,
    validate_correction_traceability_row_contract,
    aggregate_provisional_verification_metrics,
    validate_provisional_metrics_aggregation_result_contract,
)


def pipeline(payload=None):
    p=payload or aligned_payload()
    cv=validate_and_normalize_provisional_collections(p)
    rr=validate_provisional_referential_integrity(cv)
    rows=build_provisional_traceability_rows(rr)
    return p,cv,rr,rows


def test_resolved_issue_disappears_and_metrics_are_authoritative():
    _,cv,_,rows=pipeline()
    assert rows.claim_traceability_rows[0]['provisional_remaining_issue_codes']==()
    result=aggregate_provisional_verification_metrics(rows,cv)
    m=result.metrics
    assert m['issues_before']==1 and m['candidate_claim_issues_resolved']==1
    assert m['accepted_claim_issues_resolved']==1 and m['issues_remaining']==0
    assert m['verification_llm_calls']==1
    assert m['correction_llm_calls']==1
    assert m['reverification_llm_calls']==1
    assert m['additional_llm_calls']==0 and m['total_llm_calls']==3
    assert m['verification_retrieval_rounds']==1
    assert m['incremental_retrieval_requests']==0
    assert m['recommendations_generated']['status']=='NOT_COMPUTABLE'


def test_claim_without_comparison_keeps_source_issues():
    p=aligned_payload(); p['before_after_comparison_results']=()
    _,_,_,rows=pipeline(p)
    assert rows.claim_traceability_rows[0]['provisional_remaining_issue_codes']==('UNSUPPORTED_NUMERIC_VALUE',)


def test_rejected_resolution_not_accepted(monkeypatch):
    _,cv,_,rows=pipeline()
    raw=rows.to_dict(); c=dict(raw['correction_traceability_rows'][0]); c['acceptance_decision']='REJECT_PROPOSAL'; c['manual_review_required']=False
    raw['correction_traceability_rows']=(c,)
    claim=dict(raw['claim_traceability_rows'][0]); claim['individual_proposal_decisions']=('REJECT_PROPOSAL',); claim['individual_accepted_correction_ids']=(); claim['individual_rejected_correction_ids']=claim['correction_ids']; raw['claim_traceability_rows']=(claim,)
    normalized=validate_provisional_traceability_rows_result_contract(raw)
    result=aggregate_provisional_verification_metrics(normalized,cv)
    assert result.metrics['candidate_claim_issues_resolved']==1
    assert result.metrics['accepted_claim_issues_resolved']==0


def test_metric_rates_and_zero_denominators():
    _,cv,_,rows=pipeline(); m=aggregate_provisional_verification_metrics(rows,cv).metrics
    assert m['accepted_issue_resolution_rate']['value']==1.0
    p=aligned_payload()
    for k in ('correction_proposals','correction_reverification_inputs','correction_precheck_results','independent_reverification_results','before_after_comparison_results'): p[k]=()
    _,cv,_,rows=pipeline(p); m=aggregate_provisional_verification_metrics(rows,cv).metrics
    assert m['correction_acceptance_rate']['status']=='NOT_COMPUTABLE'
    assert m['correction_acceptance_rate']['value'] is None


def test_rows_contract_rejects_duplicate_unsorted_and_internal_orphans():
    _,_,_,rows=pipeline(); raw=rows.to_dict()
    raw['claim_traceability_rows']=raw['claim_traceability_rows']*2
    with pytest.raises(ValueError,match='DUPLICATE'): validate_provisional_traceability_rows_result_contract(raw)
    raw=rows.to_dict(); raw['correction_traceability_rows']=(dict(raw['correction_traceability_rows'][0],claim_id='missing'),)
    with pytest.raises(ValueError,match='CORRECTION_WITHOUT_CLAIM'): validate_provisional_traceability_rows_result_contract(raw)
    raw=rows.to_dict(); e=dict(raw['correction_evidence_traceability_rows'][0],correction_id='missing'); raw['correction_evidence_traceability_rows']=(e,)
    with pytest.raises(ValueError,match='EVIDENCE_WITHOUT_CORRECTION'): validate_provisional_traceability_rows_result_contract(raw)


def test_terminal_empty_action_is_not_gate_or_scientific():
    _,_,_,rows=pipeline(); raw=dict(rows.correction_traceability_rows[0])
    raw.update(action_type=None,is_scientific_correction_action=False,is_gate_result=False,proposal_status='NOT_PROPOSED',precheck_stage_availability='NOT_APPLICABLE',precheck_status=None,reverification_stage_availability='NOT_APPLICABLE',reverification_execution_status=None,comparison_stage_availability='NOT_APPLICABLE',acceptance_decision=None,hallucination_risk_before=None,hallucination_risk_after=None,hallucination_risk_delta=None,gate_classification=None)
    assert validate_correction_traceability_row_contract(raw)['action_type'] is None
    raw['is_gate_result']=True
    with pytest.raises(ValueError): validate_correction_traceability_row_contract(raw)


def test_inputs_reordered_same_metrics_and_no_bundle_fingerprint():
    p=aligned_payload(); _,cv1,_,r1=pipeline(p); a=aggregate_provisional_verification_metrics(r1,cv1).to_dict()
    q={k:(tuple(reversed(v)) if isinstance(v,tuple) else v) for k,v in p.items()}; _,cv2,_,r2=pipeline(q); b=aggregate_provisional_verification_metrics(r2,cv2).to_dict()
    assert a==b
    assert 'normalized_bundle_fingerprint' not in a


def test_metrics_result_contract_and_isolation():
    _,cv,_,rows=pipeline(); result=aggregate_provisional_verification_metrics(rows,cv)
    assert validate_provisional_metrics_aggregation_result_contract(result.to_dict())['result_contract_valid'] is True
    assert result.metrics['additional_retrieval_rounds']==0
    assert not hasattr(result,'normalized_bundle_fingerprint')
