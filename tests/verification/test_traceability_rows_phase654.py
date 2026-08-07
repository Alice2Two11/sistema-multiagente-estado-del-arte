from copy import deepcopy
import pytest

from test_referential_integrity_phase653 import aligned_payload
from src.tools.verification.validation import (
    validate_and_normalize_provisional_collections,
    validate_provisional_referential_integrity,
    validate_provisional_referential_integrity_result_contract,
    build_provisional_traceability_rows,
)


def chain(payload=None):
    return validate_provisional_referential_integrity(
        validate_and_normalize_provisional_collections(payload or aligned_payload())
    )


def test_full_chain_with_reverification_input_and_rows():
    p=aligned_payload(); assert p['correction_reverification_inputs']
    r=chain(p); assert r.referential_validation_status=='VALID'; assert len(r.joined_correction_records)==1
    rows=build_provisional_traceability_rows(r)
    assert rows.row_build_status=='VALID'
    assert len(rows.claim_traceability_rows)==len(rows.correction_traceability_rows)==len(rows.reverification_traceability_rows)==1
    assert rows.metrics_status=='NOT_COMPUTED'


def test_orphan_input_and_precheck_without_input():
    p=aligned_payload(); p['correction_proposals']=()
    r=chain(p); assert 'AGGREGATION_ORPHAN_REVERIFICATION_INPUT' in r.referential_issue_codes
    p=aligned_payload(); p['correction_reverification_inputs']=()
    r=chain(p); assert 'AGGREGATION_PRECHECK_WITHOUT_REVERIFICATION_INPUT' in r.referential_issue_codes

@pytest.mark.parametrize('collection,mutate,code',[
 ('correction_reverification_inputs',lambda x:x.__setitem__('correction_action_type','NARROW_SCOPE'),'AGGREGATION_CORRECTION_ACTION_MISMATCH'),
 ('independent_reverification_results',lambda x:x.__setitem__('evidence_ids_used',('OTHER',)),'AGGREGATION_UNAUTHORIZED_REVERIFICATION_EVIDENCE'),
 ('correction_precheck_results',lambda x:x.__setitem__('frozen_evidence_snapshot_fingerprint','0'*64),'AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH'),
 ('correction_precheck_results',lambda x:x.__setitem__('reverification_context_fingerprint','0'*64),'AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH'),
])
def test_referential_conflicts_rejected(collection,mutate,code):
    p=aligned_payload(); item=deepcopy(p[collection][0]); mutate(item); p[collection]=(item,)
    r=chain(p); assert r.referential_validation_status=='INVALID'; assert code in r.referential_issue_codes
    assert not r.joined_correction_records; assert r.rejected_join_candidates or r.orphan_records



def test_target_issue_input_without_claim_provenance():
    p=aligned_payload(); item=deepcopy(p['correction_reverification_inputs'][0]); item['source_issue_codes']=('INVALID_CITATION',); item['target_issue_codes']=('INVALID_CITATION',); p['correction_reverification_inputs']=(item,)
    r=chain(p); assert r.referential_validation_status=='INVALID'; assert 'AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE' in r.referential_issue_codes

def test_target_issue_comparison_differs_from_input():
    p=aligned_payload(); comp=deepcopy(p['before_after_comparison_results'][0]); comp['target_issue_codes']=(); comp['target_issues_resolved']=False; comp['reported_resolution_matches']=False; comp['acceptance_decision']='DEFER_TO_MANUAL_REVIEW'; comp['manual_review_required']=True; comp['reason_codes']=tuple(sorted(set(comp['reason_codes'])|{'REPORTED_RESOLUTION_MISMATCH'})); p['before_after_comparison_results']=(comp,)
    cv=validate_and_normalize_provisional_collections(p)
    if cv.collection_validation_status=='VALID':
        r=validate_provisional_referential_integrity(cv); assert 'AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE' in r.referential_issue_codes

def test_authorized_evidence_content_and_order_tampering():
    p=aligned_payload(); item=deepcopy(p['correction_reverification_inputs'][0]); item['authorized_evidence'][0]['usage_role']='OTHER'; p['correction_reverification_inputs']=(item,)
    cv=validate_and_normalize_provisional_collections(p)
    assert cv.collection_validation_status=='INVALID' or chain(p).referential_validation_status=='INVALID'


def test_referential_contract_closed_and_tampering_detected():
    raw=chain().to_dict(); raw['orphan_records']=({'collection':'x','primary_id':'1','reason_code':'AGGREGATION_UNKNOWN_CLAIM_ID','invented':1},)
    with pytest.raises(ValueError): validate_provisional_referential_integrity_result_contract(raw)
    raw=chain().to_dict(); raw['identity_conflicts']=({'reason_code':'INVENTED','correction_id':'c','field':'x','observed_values':()},)
    with pytest.raises(ValueError): validate_provisional_referential_integrity_result_contract(raw)


def test_claim_without_proposal_and_legitimate_partial_rows():
    p=aligned_payload()
    for key in ('correction_proposals','correction_reverification_inputs','correction_precheck_results','independent_reverification_results','before_after_comparison_results'): p[key]=()
    r=chain(p); assert r.referential_validation_status=='PARTIAL'
    rows=build_provisional_traceability_rows(r); assert len(rows.claim_traceability_rows)==1; assert not rows.correction_traceability_rows


def test_failed_reverification_without_comparison_and_gate_not_available():
    p=aligned_payload(); p['before_after_comparison_results']=()
    rv=deepcopy(p['independent_reverification_results'][0]); rv['reverification_execution_status']='FAILED'; rv['proposed_verdict']='NOT_EVALUATED'; rv['support_level']='NOT_EVALUATED'; rv['reason_codes']=('REVERIFICATION_LLM_INVOCATION_FAILED',); rv['technical_issue_codes']=('REVERIFICATION_LLM_INVOCATION_FAILED',); rv['evidence_ids_used']=(); p['independent_reverification_results']=(rv,)
    cv=validate_and_normalize_provisional_collections(p)
    if cv.collection_validation_status=='VALID':
        rows=build_provisional_traceability_rows(validate_provisional_referential_integrity(cv)); assert rows.reverification_traceability_rows[0]['acceptance_decision'] is None


def test_evidence_rows_and_support_not_evaluated():
    rows=build_provisional_traceability_rows(chain())
    assert rows.claim_evidence_traceability_rows[0]['supports_original_claim']=='NOT_EVALUATED'
    assert rows.correction_evidence_traceability_rows[0]['supports_proposed_claim']=='NOT_EVALUATED'
    assert rows.correction_evidence_traceability_rows[0]['used_in_reverification'] is True


def test_order_deterministic_and_input_not_mutated():
    p=aligned_payload(); original=deepcopy(p)
    a=build_provisional_traceability_rows(chain(p)).to_dict()
    q={k:(tuple(reversed(v)) if isinstance(v,tuple) else v) for k,v in p.items()}
    b=build_provisional_traceability_rows(chain(q)).to_dict()
    assert a==b and p==original


def test_invalid_joins_produce_no_scientific_rows_and_isolation():
    p=aligned_payload(); x=deepcopy(p['correction_reverification_inputs'][0]); x['claim_id']='other'; p['correction_reverification_inputs']=(x,)
    r=chain(p); rows=build_provisional_traceability_rows(r)
    assert rows.row_build_status=='INVALID'; assert rows.claim_traceability_rows==()
    assert not hasattr(rows,'metrics') and not hasattr(rows,'normalized_bundle_fingerprint')
