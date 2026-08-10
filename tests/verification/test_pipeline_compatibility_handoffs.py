import csv, json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.adapters.agent06_verification_handoff import (
    Agent07RetrieverBinding, build_agent07_input_from_committed_agent06,
    validate_agent07_experiment_compatibility, validate_productive_retriever_binding,
)
from src.adapters.agent07c_handoff import prepare_agent07c_input_from_agent07, validate_original_agent07c_input_artifacts, validate_agent07c_prepared_input_contract, REQUIRED_SAFETY_POLICY
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import AgentResult,DecisionInfo,ExecutionStatus,QualityStatus,RequestedTransition,ToolUsage,TransitionAction
from src.state.pipeline_state import PipelineIdentity,PipelineState,StageState,ArtifactState,DecisionLogEntry
from src.state.fingerprints import sha256_file
from test_multi_proposal_resolution_phase66 import bundle,corr
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from agent07c_test_support import terminal_handoff_args

PATHS={"code_root":"/content/tesis_codigo","project_root":"/content/proyecto_estado_arte","experiment_root":"/content/proyecto_estado_arte/experimento_paper_02"}

def test_config_incompatible_with_00():
    with pytest.raises(ValueError,match="GLOBAL_CONFIG_MISMATCH"):
        validate_agent07_experiment_compatibility(active_config={"verification_policy":{"a":1},"openai_model":"m"},agent07_config={"verification_policy":{"a":2},"verification_model":"m","correction_model":"m"},experiment_paths=PATHS)

def test_retriever_binding_collection_and_embedding(tmp_path):
    cm=tmp_path/'chroma_index_manifest.json'; ch=tmp_path/'chunks_clean_for_rag.csv'
    cm.write_text(json.dumps({"experiment_id":"e","collection_name":"c","embedding_model":"m"})); ch.write_text('chunk_id\n1\n')
    b=Agent07RetrieverBinding('e','c','m',sha256_file(cm),sha256_file(ch))
    assert validate_productive_retriever_binding(binding=b,active_config={"chroma_collection_name":"c","embedding_model":"m"},chroma_manifest_path=cm,chunks_manifest_path=ch,committed_experiment_id='e')["collection_name"]=='c'
    with pytest.raises(ValueError,match="COLLECTION_MISMATCH"):
        validate_productive_retriever_binding(binding={**b.to_dict(),"collection_name":"other"},active_config={"chroma_collection_name":"c","embedding_model":"m"},chroma_manifest_path=cm,chunks_manifest_path=ch,committed_experiment_id='e')
    with pytest.raises(ValueError,match="EMBEDDING_MODEL_MISMATCH"):
        validate_productive_retriever_binding(binding={**b.to_dict(),"embedding_model":"other"},active_config={"chroma_collection_name":"c","embedding_model":"m"},chroma_manifest_path=cm,chunks_manifest_path=ch,committed_experiment_id='e')

def _write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def _store_fixture(tmp_path,repeated=False,foreign_draft=False,conflicting_evidence=False,missing_numeric=False,extra_claim_no_evidence=False,unknown_evidence=False,duplicate_inventory=False):
    sec='Alpha beta gamma. Alpha beta gamma.' if repeated else 'Alpha beta gamma.'
    files={}
    claims=[{"claim_id":"c1","claim":"Alpha beta gamma."}]
    if extra_claim_no_evidence: claims.append({"claim_id":"c2","claim":"beta"})
    if duplicate_inventory: claims.append({"claim_id":"c1","claim":"beta"})
    (tmp_path/'state_of_art_draft.json').write_text(json.dumps({"sections":[{"section_id":"s1","text":sec,"claims":claims}]})); files['state_of_art_draft.json']=tmp_path/'state_of_art_draft.json'
    (tmp_path/'state_of_art_draft.md').write_text(sec); files['state_of_art_draft.md']=tmp_path/'state_of_art_draft.md'
    _write_csv(tmp_path/'draft_sections.csv',[{"section_id":"s1","section_text":sec}]);files['draft_sections.csv']=tmp_path/'draft_sections.csv'
    claim={"claim_id":"c1","section_id":"s1","claim_text":"Alpha beta gamma.","evidence_id":"e1","source_filename":"p.pdf","chunk_id":"ch1","text":"evidence"}
    _write_csv(tmp_path/'draft_claim_evidence.csv',[claim]);files['draft_claim_evidence.csv']=tmp_path/'draft_claim_evidence.csv'
    rag_claim={**claim,'text':'DIFFERENT'} if conflicting_evidence else claim
    if unknown_evidence: rag_claim={**rag_claim,"claim_id":"unknown"}
    _write_csv(tmp_path/'draft_rag_evidence.csv',[rag_claim]);files['draft_rag_evidence.csv']=tmp_path/'draft_rag_evidence.csv'
    numeric_path=tmp_path/'numeric_hallucination_check.csv'
    if missing_numeric:
        numeric_path.write_text('claim_id,section_id,numeric_risk\n')
    else:
        _write_csv(numeric_path,[{"claim_id":"c1","section_id":"s1","numeric_risk":"LOW"}])
    files['numeric_hallucination_check.csv']=numeric_path
    (tmp_path/'draft_validation_report.json').write_text('{}');files['draft_validation_report.json']=tmp_path/'draft_validation_report.json'
    (tmp_path/'draft_generation_manifest.json').write_text(json.dumps({"source_draft_fingerprint":"a"*64,"artifact_identity":"draft06"}));files['draft_generation_manifest.json']=tmp_path/'draft_generation_manifest.json'
    refs={name:ArtifactReference(str(p),sha256_file(p)) for name,p in files.items()}
    result=AgentResult(ExecutionStatus.COMPLETED,QualityStatus.APPROVED,DecisionInfo('OK','ok'),{},(),RequestedTransition(TransitionAction.ADVANCE,'07_agente_verificador','OK',False),refs,ToolUsage(),1,'2026-01-01','',completed_at='2026-01-01')
    log=DecisionLogEntry('d06','2026-01-01','06_agente_redactor','06_agente_redactor',1,{}, {'code':'OK'},(),None,result.to_dict())
    state=PipelineState(PipelineIdentity('exp','run','2026-01-01','2026-01-01','v1'),stages={'06_agente_redactor':StageState(execution_status=ExecutionStatus.COMPLETED)},artifacts={name:ArtifactState(ref,'2026-01-01') for name,ref in refs.items()},decision_log=(log,))
    if foreign_draft:
        p=tmp_path/'DRAFT_foreign.json';p.write_text('{}')
        state=PipelineState(state.identity,stages=state.stages,artifacts={**state.artifacts,'DRAFT_foreign':ArtifactState(ArtifactReference(str(p),sha256_file(p)),'2026-01-01')},decision_log=state.decision_log)
    mapping=tmp_path/'outline_paper_mapping.csv';_write_csv(mapping,[{'section_id':'s1','source_filename':'p.pdf'}])
    return SimpleNamespace(load=lambda:state),mapping

def test_build_contexts_exact_commit_and_no_scientific_defaults(tmp_path):
    out=build_agent07_input_from_committed_agent06(store=_store_fixture(tmp_path,foreign_draft=True)[0],stage_name='06_agente_redactor',agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=_store_fixture(tmp_path,foreign_draft=True)[1])
    c=out['claim_verification_contexts'][0]
    assert (c['claim_id'],c['section_id'])==('c1','s1')
    assert c['claim_span_in_section']['start']==0 and c['eligible_evidence'][0]['source_filename']=='p.pdf'
    assert 'scientific_verdict' not in c and 'final_correction_eligibility' not in c

def test_repeated_claim_without_explicit_span_blocks(tmp_path):
    with pytest.raises(ValueError,match="SPAN_AMBIGUOUS"):
        build_agent07_input_from_committed_agent06(store=_store_fixture(tmp_path,repeated=True)[0],stage_name='06_agente_redactor',agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=_store_fixture(tmp_path,repeated=True)[1])

def _handoff_args(draft,b,r):
    import hashlib
    raw=json.dumps(draft,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    fp=hashlib.sha256(raw).hexdigest()
    section=draft["sections"][0]["text"]
    return dict(
        source_draft_markdown=section,experiment_id="exp",committed_source_draft_fingerprint=fp,
        claim_source_contexts=({"claim_id":"c1","section_id":"s1","original_claim_text":section,
          "claim_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),
          "section_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),
          "source_draft_fingerprint":fp,
          "claim_span_in_section":{"start":0,"end":len(section),"text":section,"base_text_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),"coordinate_system":"PYTHON_CODEPOINT_OFFSETS"}},),
        safety_policy=REQUIRED_SAFETY_POLICY,
        **terminal_handoff_args(b,r,({"claim_id":"c1","section_id":"s1","original_claim_text":section,"claim_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),"section_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),"source_draft_fingerprint":fp,"claim_span_in_section":{"start":0,"end":len(section),"text":section,"base_text_fingerprint":__import__('src.tools.verification.corrections',fromlist=['fingerprint_text']).fingerprint_text(section),"coordinate_system":"PYTHON_CODEPOINT_OFFSETS"}},)),

    )

def test_07c_copy_application_and_conflict_exclusion():
    b=bundle((corr('x1',6,10,'beta','B'),));r=resolve_multiple_correction_proposals(b)
    draft={"sections":[{"section_id":"s1","text":"Alpha beta gamma."}]};original=json.loads(json.dumps(draft))
    out=prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,**_handoff_args(draft,b,r))
    assert out.verified_state_of_art['sections'][0]['text']=='Alpha B gamma.'
    assert draft==original and not out.original_draft_modified and not out.evaluation_ready_emitted
    assert set(out.artifact_payloads)==set(__import__('src.adapters.agent07c_handoff',fromlist=['AGENT07C_REQUIRED_ARTIFACTS']).AGENT07C_REQUIRED_ARTIFACTS)
    assert validate_original_agent07c_input_artifacts(artifact_payloads=out.artifact_payloads,experiment_id='exp')["validation_ok"]
    assert validate_agent07c_prepared_input_contract(out)["result_contract_valid"]
    conflict=resolve_multiple_correction_proposals(bundle((corr('x1',6,10,'beta','B'),corr('x2',6,10,'beta','C'))))
    b2=bundle((corr('x1',6,10,'beta','B'),corr('x2',6,10,'beta','C')))
    out2=prepare_agent07c_input_from_agent07(provisional_bundle=b2.to_dict(),resolution_result=conflict.to_dict(),source_draft=draft,**_handoff_args(draft,b2,conflict))
    assert out2.eligible_claim_ids==() and out2.verified_state_of_art==draft
