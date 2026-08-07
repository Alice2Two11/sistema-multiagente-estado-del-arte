from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.claim_verification_context import (
    build_claim_verification_context_from_agent06_handoff,
    classify_claim_from_versioned_policy,
)
from src.adapters.verification_runtime import Agent07RuntimeInput, VerificationRuntimeDependencies, run_agent07_in_memory
from src.agents.verification_agent import VerificationAgent
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import AgentResult, DecisionInfo, ExecutionStatus, QualityStatus, RequestedTransition, ToolUsage, TransitionAction
from src.state.pipeline_state import PipelineIdentity, PipelineState, StageState, ArtifactState, DecisionLogEntry
from src.state.fingerprints import sha256_file
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from src.tools.verification.validation import validate_claim_verification_context
from test_multi_proposal_resolution_phase66 import bundle as real_bundle


def _real_agent06_handoff(tmp_path):
    src=Path(__file__).parents[1]/"fixtures"/"agent06_v17_e2e_snapshot"
    names=("state_of_art_draft.json","state_of_art_draft.md","draft_sections.csv","draft_rag_evidence.csv","draft_claim_evidence.csv","numeric_hallucination_check.csv","draft_validation_report.json","draft_generation_manifest.json")
    refs={}
    for name in names:
        target=tmp_path/name; target.write_bytes((src/name).read_bytes()); refs[name]=ArtifactReference(str(target),sha256_file(target))
    result=AgentResult(ExecutionStatus.COMPLETED,QualityStatus.APPROVED,DecisionInfo("OK","ok"),{},(),RequestedTransition(TransitionAction.ADVANCE,"07","OK",False),refs,ToolUsage(),1,"2026-01-01","",completed_at="2026-01-01")
    log=DecisionLogEntry("d06","2026-01-01","06_agente_redactor","06_agente_redactor",1,{}, {"code":"OK"},(),None,result.to_dict())
    state=PipelineState(PipelineIdentity("exp_synthetic","run_synthetic","2026-01-01","2026-01-01","v1"),stages={"06_agente_redactor":StageState(execution_status=ExecutionStatus.COMPLETED)},artifacts={name:ArtifactState(ref,"2026-01-01") for name,ref in refs.items()},decision_log=(log,))
    mapping=tmp_path/"outline_paper_mapping.csv"; mapping.write_bytes((src/"outline_paper_mapping.csv").read_bytes())
    return build_agent07_input_from_committed_agent06(store=SimpleNamespace(load=lambda:state),stage_name="06_agente_redactor",agent07_config={},policy_versions={"verification":"v1"},schema_versions={"runtime":"v5"},experiment_paths={"root":str(tmp_path)},outline_paper_mapping_path=mapping)


def test_s2_c1_real_handoff_adapts_and_validates_without_mutation(tmp_path):
    handoff=_real_agent06_handoff(tmp_path)
    source=next(x for x in handoff["claim_verification_contexts"] if x["claim_id"]=="S2_C1")
    before=deepcopy(source)
    adapted=build_claim_verification_context_from_agent06_handoff(source,verification_policy={})
    assert validate_claim_verification_context(adapted)["claim_id"]=="S2_C1"
    assert adapted["claim_text"]==source["original_claim_text"]
    assert adapted["claim_type"]=="QUANTITATIVE"
    assert adapted["verification_intensity"]=="STRICT"
    assert source==before
    assert all(pair[0] in source["authorized_source_filenames"] for pair in adapted["allowed_source_pairs"])


def test_real_verification_agent_runtime_processes_complete_agent06_handoff(tmp_path):
    handoff=_real_agent06_handoff(tmp_path)
    original=deepcopy(handoff)
    deps=VerificationRuntimeDependencies(
        verification_agent_factory=VerificationAgent,
        verification_llm=None,
        correction_context_factory=lambda context, result, config:{"claim_id":context["claim_id"]},
        reverification_input_factory=lambda *args:{},
        proposal_runner=lambda context, *, llm:{"correction_id":"none-"+context["claim_id"],"accepted_for_reverification":False},
        bundle_builder=lambda value:real_bundle(()),
        resolution_runner=resolve_multiple_correction_proposals,
    )
    runtime_input=Agent07RuntimeInput(handoff,{"verification_policy":{}},{"verification":"v1"},{"runtime":"v5","provisional_bundle":"v4","multi_proposal_resolution":"v1"},{"root":str(tmp_path)})
    result=run_agent07_in_memory(runtime_input,dependencies=deps)
    assert result.runtime_status=="COMPLETED"
    assert result.execution_metrics["claims_processed"]==len(handoff["expected_claim_ids"])
    assert result.execution_metrics["claims_processed"]>=1
    assert "CLAIM_VERIFICATION_INPUT_FIELDS_MISSING" not in str(result.to_dict())
    assert handoff==original


def test_claim_without_inherited_evidence_and_with_retrieved_evidence_are_valid(tmp_path):
    handoff=_real_agent06_handoff(tmp_path)
    source=deepcopy(handoff["claim_verification_contexts"][0])
    source["eligible_evidence"]=()
    empty=build_claim_verification_context_from_agent06_handoff(source,verification_policy={})
    assert empty["inherited_evidence_assessment"]["evidence_rows"]==()
    retrieved=deepcopy(source)
    retrieved["eligible_evidence"]=({"evidence_id":"rag-new","source_filename":source["authorized_source_filenames"][0],"chunk_id":"new_chunk","text":"Texto recuperado contractualmente.","authorized_for_section":True,"retrieval_origin":"AGENT07_INDEPENDENT_RAG"},)
    retrieved["agent07_independent_retrieval_executed"]=True
    retrieved["agent07_independent_retrieval_rounds"]=1
    retrieved["agent07_independent_retrieval_status"]="COMPLETED_WITH_RESULTS"
    adapted=build_claim_verification_context_from_agent06_handoff(retrieved,verification_policy={})
    assert adapted["retrieval_result"]["selected_candidates"][0]["evidence_id"]=="rag-new"
    assert adapted["inherited_evidence_assessment"]["evidence_rows"]==()


def test_versioned_claim_classification_covers_required_types():
    cases={
      "El sistema presenta resultados verificables.":"SUBSTANTIVE_FACTUAL",
      "El sistema obtuvo 91% de precisión.":"QUANTITATIVE",
      "El modelo A supera al modelo B.":"COMPARATIVE",
      "El método usa validación cruzada.":"METHODOLOGICAL",
      "Esta sección organiza el alcance de la revisión.":"ORGANIZATIONAL",
      "Finalmente, la siguiente sección presenta los resultados.":"TRANSITIONAL",
    }
    for text,expected in cases.items(): assert classify_claim_from_versioned_policy(text)==expected
