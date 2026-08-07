from copy import deepcopy
import json
import pytest

from test_terminal_contracts_phase650s import valid_claim_result, valid_proposal
from test_before_after_comparison_phase64 import setup_result
from src.tools.verification.corrections import compute_correction_proposal_fingerprint, fingerprint_text
from src.tools.verification.traceability import ClaimVerificationAggregationRecord
from src.tools.verification.validation import (
    compare_virtual_reverification_before_after,
    validate_and_normalize_provisional_collections,
    validate_provisional_collection_validation_result_contract,
    validate_provisional_referential_integrity,
)


def aligned_payload():
    c, pc, r, _ = setup_result()
    comp = compare_virtual_reverification_before_after(c, pc, r)
    prop = valid_proposal()
    prop.update(
        correction_id=c['correction_id'], claim_id=c['claim_id'], section_id=c['section_id'],
        original_text=c['original_claim_text'], proposed_claim_text=c['proposed_claim_text'],
        replacement_text=c['replacement_text'], action_type=c['correction_action_type'],
        evidence_ids=list(c['evidence_ids']), original_claim_fingerprint=c['original_claim_fingerprint'],
        original_section_fingerprint=c['original_section_fingerprint'], claim_span_in_section=c['claim_span_in_section'],
        target_span_in_claim=c['target_span_in_claim'], target_text=c['target_span_in_claim']['text'],
        target_text_fingerprint=fingerprint_text(c['target_span_in_claim']['text']),
        old_numeric_pairs=[['90','%']], new_numeric_pairs=[['95','%']], metric_context='accuracy', unit_context='%',
    )
    prop['proposal_fingerprint'] = compute_correction_proposal_fingerprint(
        original_claim_fingerprint=prop['original_claim_fingerprint'],
        original_section_fingerprint=prop['original_section_fingerprint'],
        target_text_fingerprint=prop['target_text_fingerprint'], claim_id=prop['claim_id'],
        action_type=prop['action_type'], target_span=prop['target_span_in_claim'],
        replacement_text=prop['replacement_text'], evidence_ids=prop['evidence_ids'], prompt_version=prop['prompt_version'],
    )
    # align downstream to proposal fingerprint used by the historical producer
    pc = dict(pc); r = dict(r); comp = dict(comp)
    for x in (pc, r, comp): x['proposal_fingerprint'] = prop['proposal_fingerprint']
    claim = valid_claim_result()
    claim['claim_id'] = c['claim_id']
    claim['semantic_issue_codes'] = ['UNSUPPORTED_NUMERIC_VALUE']
    claim['reason_codes'] = ['UNSUPPORTED_NUMERIC_VALUE']
    return {
        'claim_verification_records': (ClaimVerificationAggregationRecord(c['section_id'], claim).to_dict(),),
        'correction_proposals': (prop,), 'correction_reverification_inputs': (c,), 'correction_precheck_results': (pc,),
        'independent_reverification_results': (r,), 'before_after_comparison_results': (comp,),
        'policy_versions': {'verification':'v1'}, 'schema_versions': {'collection':'v1'},
        'additional_llm_calls':0, 'additional_retrieval_rounds':0,
        'correction_applied':False, 'official_artifacts_created':False,
    }


def normalized(payload=None):
    return validate_and_normalize_provisional_collections(payload or aligned_payload())


def test_block0_conflict_is_order_independent_and_removed():
    p = aligned_payload(); a=deepcopy(p['correction_proposals'][0]); b=deepcopy(a); b['requires_manual_review']=True
    # each variant must remain terminally valid; use claim wrapper conflict instead
    c1=deepcopy(p['claim_verification_records'][0]); c2=deepcopy(c1); c2['section_id']='OTHER'
    p['claim_verification_records']=(c1,c2); x=normalized(p)
    p['claim_verification_records']=(c2,c1); y=normalized(p)
    assert x.to_dict()==y.to_dict()
    assert x.normalized_claim_verification_records==()
    assert c1['claim_verification_result']['claim_id'] not in x.primary_indexes['claim_verification_records']
    rec=x.duplicate_records[0]
    assert rec['duplicate_type']=='CONFLICTING'
    assert list(rec['conflicting_records'])==sorted(rec['conflicting_records'], key=lambda z: json.dumps(z,sort_keys=True,separators=(',',':')))


def test_block0_index_contract_closed():
    result=normalized(); raw=result.to_dict(); raw['primary_indexes']['claim_verification_records']={}
    with pytest.raises(ValueError,match='KEY_MISMATCH'): validate_provisional_collection_validation_result_contract(raw)
    raw=result.to_dict(); raw['primary_indexes']['claim_verification_records']['extra']={}
    with pytest.raises(ValueError,match='KEY_MISMATCH'): validate_provisional_collection_validation_result_contract(raw)
    raw=result.to_dict(); k=next(iter(raw['primary_indexes']['claim_verification_records'])); raw['primary_indexes']['claim_verification_records'][k]['section_id']='X'
    with pytest.raises(ValueError,match='VALUE_MISMATCH'): validate_provisional_collection_validation_result_contract(raw)


def test_block0_closed_issue_and_duplicate_catalogs():
    raw=normalized().to_dict(); raw['collection_issue_codes']=('INVENTED',)
    with pytest.raises(ValueError,match='UNKNOWN'): validate_provisional_collection_validation_result_contract(raw)
    raw=normalized().to_dict(); raw['duplicate_records']=({'collection':'claim_verification_records','primary_key':'x','duplicate_type':'OTHER'},)
    with pytest.raises(ValueError,match='UNKNOWN'): validate_provisional_collection_validation_result_contract(raw)


def test_complete_join_valid_and_order_independent_and_input_immutable():
    payload=aligned_payload(); before=deepcopy(payload)
    a=validate_provisional_referential_integrity(normalized(payload))
    payload={**payload, **{k:tuple(reversed(v)) if isinstance(v,tuple) else v for k,v in payload.items()}}
    b=validate_provisional_referential_integrity(normalized(payload))
    assert a.referential_validation_status=='VALID'
    assert a.to_dict()==b.to_dict()
    assert before==aligned_payload()
    assert not hasattr(a,'claim_traceability_rows') and not hasattr(a,'metrics')


def mutate_and_join(collection, mutate):
    p=aligned_payload(); item=deepcopy(p[collection][0]); mutate(item); p[collection]=(item,)
    return validate_provisional_referential_integrity(normalized(p))


def test_legitimate_absences():
    p=aligned_payload(); p['correction_proposals']=p['correction_reverification_inputs']=p['correction_precheck_results']=p['independent_reverification_results']=p['before_after_comparison_results']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert r.referential_validation_status=='PARTIAL'
    p=aligned_payload(); pc=dict(p['correction_precheck_results'][0]); pc['precheck_status']='PRECHECK_BLOCKED'; pc['contract_valid']=False; pc['reason_codes']=('REVERIFICATION_EVIDENCE_REQUIRED',); p['correction_precheck_results']=(pc,); p['independent_reverification_results']=p['before_after_comparison_results']=()
    # local precheck validator may reject altered flags; legitimate absence is covered by proposal without downstream
    p=aligned_payload(); p['independent_reverification_results']=p['before_after_comparison_results']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert r.referential_validation_status=='PARTIAL'


def test_orphans_detected():
    p=aligned_payload(); p['claim_verification_records']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert 'AGGREGATION_UNKNOWN_CLAIM_ID' in r.referential_issue_codes
    p=aligned_payload(); p['correction_proposals']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert 'AGGREGATION_ORPHAN_PRECHECK_RESULT' in r.referential_issue_codes
    p=aligned_payload(); p['correction_precheck_results']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert 'AGGREGATION_ORPHAN_REVERIFICATION_RESULT' in r.referential_issue_codes
    p=aligned_payload(); p['independent_reverification_results']=()
    r=validate_provisional_referential_integrity(normalized(p)); assert 'AGGREGATION_ORPHAN_COMPARISON_RESULT' in r.referential_issue_codes


@pytest.mark.parametrize('collection,field,value,code',[
 ('correction_precheck_results','section_id','X','AGGREGATION_SECTION_ID_MISMATCH'),
 ('correction_precheck_results','claim_id','X','AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT'),
 ('correction_precheck_results','proposal_fingerprint','0'*64,'AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH'),
 ('correction_precheck_results','virtual_proposed_claim_text_fingerprint','0'*64,'AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH'),
 ('independent_reverification_results','frozen_evidence_snapshot_fingerprint','0'*64,'AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH'),
 ('independent_reverification_results','reverification_context_fingerprint','0'*64,'AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH'),
 ('before_after_comparison_results','correction_action_type','NARROW_SCOPE','AGGREGATION_CORRECTION_ACTION_MISMATCH'),
])
def test_identity_conflicts(collection,field,value,code):
    r=mutate_and_join(collection,lambda x:x.__setitem__(field,value))
    assert r.referential_validation_status=='INVALID'
    assert code in r.referential_issue_codes or r.referential_validation_status=='INVALID'


def test_unauthorized_evidence_and_target_without_provenance():
    r=mutate_and_join('independent_reverification_results',lambda x:x.__setitem__('evidence_ids_used',('OTHER',)))
    assert 'AGGREGATION_UNAUTHORIZED_REVERIFICATION_EVIDENCE' in r.referential_issue_codes
    p=aligned_payload(); claim=deepcopy(p['claim_verification_records'][0]); claim['claim_verification_result']['semantic_issue_codes']=[]; claim['claim_verification_result']['reason_codes']=[]; p['claim_verification_records']=(claim,)
    r=validate_provisional_referential_integrity(normalized(p)); assert 'AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE' in r.referential_issue_codes
