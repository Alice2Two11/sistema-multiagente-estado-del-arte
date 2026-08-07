import inspect, json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.adapters.agent06_verification_handoff import (
    build_agent07_input_from_committed_agent06,
    validate_agent06_verification_handoff_contract,
)
from src.adapters.agent07c_handoff import (
    REQUIRED_SAFETY_POLICY,
    prepare_agent07c_input_from_agent07,
    validate_original_agent07c_input_artifacts,
)
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import AgentResult, DecisionInfo, ExecutionStatus, QualityStatus, RequestedTransition, ToolUsage, TransitionAction
from src.state.pipeline_state import PipelineIdentity, PipelineState, StageState, ArtifactState, DecisionLogEntry
from src.state.fingerprints import sha256_file
from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from test_multi_proposal_resolution_phase66 import bundle, claim
from agent07c_test_support import terminal_handoff_args, _fp


def _source_context(text="Alpha beta gamma."):
    draft={"sections":[{"section_id":"s1","text":text}]}
    sfp=fingerprint_text(text)
    ctx={"claim_id":"c1","section_id":"s1","original_claim_text":text,"claim_fingerprint":sfp,"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"eligible_evidence":(),"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":sfp}}
    return draft,(ctx,)


def _prepare_args():
    draft,contexts=_source_context()
    b=bundle((),claim(corrections=())); r=resolve_multiple_correction_proposals(b)
    args=terminal_handoff_args(b,r,contexts)
    return draft,contexts,b,r,args


def test_unbacked_safety_boole_are_not_part_of_public_api():
    assert "safety_evidence" not in inspect.signature(prepare_agent07c_input_from_agent07).parameters


def test_independent_rag_and_evidence_validation_must_be_terminally_backed():
    draft,contexts,b,r,args=_prepare_args()
    args["runtime_result"]["execution_metrics"]["independent_rag_claims"]=0
    with pytest.raises(ValueError,match="INDEPENDENT_RAG_UNPROVEN"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,source_draft_markdown=draft["sections"][0]["text"],experiment_id="exp",committed_source_draft_fingerprint=_fp(draft),claim_source_contexts=contexts,safety_policy=REQUIRED_SAFETY_POLICY,**args)
    args=terminal_handoff_args(b,r,contexts)
    args["runtime_result"]["execution_metrics"]["evidence_candidate_validation_claims"]=0
    with pytest.raises(ValueError,match="EVIDENCE_VALIDATION_UNPROVEN"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,source_draft_markdown=draft["sections"][0]["text"],experiment_id="exp",committed_source_draft_fingerprint=_fp(draft),claim_source_contexts=contexts,safety_policy=REQUIRED_SAFETY_POLICY,**args)


def test_runtime_with_fewer_processed_claims_is_rejected():
    draft,contexts,b,r,args=_prepare_args()
    args["runtime_result"]["execution_metrics"]["claims_processed"]=0
    args["runtime_result"]["execution_metrics"]["independent_rag_claims"]=0
    args["runtime_result"]["execution_metrics"]["evidence_candidate_validation_claims"]=0
    with pytest.raises(ValueError,match="INDEPENDENT_RAG_UNPROVEN"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,source_draft_markdown=draft["sections"][0]["text"],experiment_id="exp",committed_source_draft_fingerprint=_fp(draft),claim_source_contexts=contexts,safety_policy=REQUIRED_SAFETY_POLICY,**args)


def test_scientific_check_false_and_global_contradiction_are_rejected():
    draft,contexts,b,r,args=_prepare_args()
    out=prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,source_draft_markdown=draft["sections"][0]["text"],experiment_id="exp",committed_source_draft_fingerprint=_fp(draft),claim_source_contexts=contexts,safety_policy=REQUIRED_SAFETY_POLICY,**args)
    payloads=dict(out.artifact_payloads)
    report=json.loads(payloads["verification_validation_report.json"])
    report["scientific_handoff_checks"]["claim_coverage_ok"]=False
    # Contradiction: global remains true.
    payloads["verification_validation_report.json"]=json.dumps(report).encode()
    with pytest.raises(ValueError,match="SCIENTIFIC_HANDOFF_GLOBAL_MISMATCH"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")
    report["scientific_handoff_validation_ok"]=False; report["validation_ok"]=False
    payloads["verification_validation_report.json"]=json.dumps(report).encode()
    with pytest.raises(ValueError,match="VALIDATION_REPORT_NOT_OK"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")


def test_global_claim_id_duplicate_across_sections_is_rejected():
    handoff={
      "commit_status":"COMMITTED","run_id":"run","experiment_id":"exp","artifact_identity":"a","schema_version":"v1",
      "source_draft_fingerprint":"a"*64,"agent06_manifest_source_draft_fingerprint":None,
      "claim_verification_contexts":(
        {"claim_id":"dup","section_id":"s1"},{"claim_id":"dup","section_id":"s2"},
      ),
      "expected_claim_ids":("dup","dup"),"claim_inventory_fingerprint":"b"*64,"agent06_decision_id":"d",
      "outline_mapping_fingerprint":"c"*64,"integration_metadata":{},
    }
    with pytest.raises(ValueError,match="GLOBAL_CLAIM_ID_DUPLICATE"):
        validate_agent06_verification_handoff_contract(handoff)


def test_builder_uses_artifacts_captured_from_real_agent06_e2e(tmp_path):
    src=Path(__file__).parents[1]/"fixtures"/"agent06_v17_e2e_snapshot"
    names=("state_of_art_draft.json","state_of_art_draft.md","draft_sections.csv","draft_rag_evidence.csv","draft_claim_evidence.csv","numeric_hallucination_check.csv","draft_validation_report.json","draft_generation_manifest.json")
    refs={}
    for name in names:
        target=tmp_path/name; target.write_bytes((src/name).read_bytes()); refs[name]=ArtifactReference(str(target),sha256_file(target))
    result=AgentResult(ExecutionStatus.COMPLETED,QualityStatus.APPROVED,DecisionInfo("OK","ok"),{},(),RequestedTransition(TransitionAction.ADVANCE,"07","OK",False),refs,ToolUsage(),1,"2026-01-01","",completed_at="2026-01-01")
    log=DecisionLogEntry("d06","2026-01-01","06_agente_redactor","06_agente_redactor",1,{}, {"code":"OK"},(),None,result.to_dict())
    state=PipelineState(PipelineIdentity("exp_synthetic","run_synthetic","2026-01-01","2026-01-01","v1"),stages={"06_agente_redactor":StageState(execution_status=ExecutionStatus.COMPLETED)},artifacts={name:ArtifactState(ref,"2026-01-01") for name,ref in refs.items()},decision_log=(log,))
    mapping=tmp_path/"outline_paper_mapping.csv"; mapping.write_bytes((src/"outline_paper_mapping.csv").read_bytes())
    out=build_agent07_input_from_committed_agent06(store=SimpleNamespace(load=lambda:state),stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    draft=json.loads((src/"state_of_art_draft.json").read_text())
    draft_claim_count=sum(len(section.get("claims",())) for section in draft["sections"])
    assert draft_claim_count==len(out["claim_verification_contexts"])==len(out["expected_claim_ids"])
    assert len(set(out["expected_claim_ids"]))==len(out["expected_claim_ids"])
    assert all(c["claim_id"] and c["section_id"] for c in out["claim_verification_contexts"])
