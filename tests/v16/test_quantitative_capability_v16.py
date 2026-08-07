from __future__ import annotations
import json, tempfile, unittest, hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from src.adapters.quantitative_extraction_runtime import build_quantitative_agent_input
from src.capabilities.quantitative_extraction import QuantitativeExtractionCapability, QuantitativeExtractionDependencies, build_quantitative_composite_fingerprint
from src.config.quantitative_extraction_policy_config import ARTIFACT_FILENAMES, DEFAULT_QUANTITATIVE_EXTRACTION_POLICY, STAGE_NAME
from src.contracts.agent_input import AgentContext, AgentInput, ArtifactReference, ExecutionMode
from src.contracts.agent_result import ExecutionStatus, QualityStatus, TransitionAction
from src.runtime.quantitative_extraction_protocol import build_quantitative_fingerprints, execute_quantitative_transaction, execute_quantitative_runtime_transaction
from src.state.fingerprints import sha256_file
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore

KB_COLUMNS=["source_filename","title","research_problem","objective","task_type","target_domain","target_variable_or_object","temporal_horizon_or_scope","methods_or_models","method_families","datasets_or_case_study","input_variables_or_data_sources","evaluation_metrics","main_results","reported_best_method_or_model","limitations_or_gaps","contribution","relevance_for_state_of_art","domain_specific_notes","include_in_state_of_art","retrieved_chunk_ids"]

class Message:
    def __init__(self, content): self.content=content
class Response:
    def __init__(self, content): self.content=content
class FakeLLM:
    def __init__(self): self.calls=0
    def invoke(self, messages):
        self.calls+=1
        return Response(json.dumps({"techniques":[{"technique_name":"LSTM","technique_family":"neural","source_text_evidence":"LSTM"}],"datasets":[{"dataset_name":"Wind","source_text_evidence":"Wind"}],"quantitative_results":[{"model_or_method":"LSTM","metric":"RMSE","value":"0.25","unit":"","dataset_or_case":"Wind","source_text_evidence":"RMSE 0.25"}],"notes":""}))
class FailingLLM:
    def invoke(self, messages): raise AssertionError("LLM should not be called")

def make_fixture(root:Path, *, manifest_experiment="exp", with_chunks=True):
    out=root/"out"; out.mkdir(parents=True)
    kb=root/"scientific_knowledge_base.csv"; kbj=root/"scientific_knowledge_base.jsonl"; manifest=root/"scientific_extraction_manifest.json"; chunks=root/"chunks_clean_for_rag.csv"
    row={c:"value" for c in KB_COLUMNS}; row.update({"source_filename":"paper.pdf","title":"Paper","include_in_state_of_art":True,"retrieved_chunk_ids":'["c1"]',"main_results":"RMSE 0.25 on Wind","evaluation_metrics":"RMSE","methods_or_models":"LSTM","datasets_or_case_study":"Wind"})
    pd.DataFrame([row],columns=KB_COLUMNS).to_csv(kb,index=False)
    kbj.write_text(json.dumps(row)+"\n",encoding="utf-8")
    manifest.write_text(json.dumps({"stage":"03_agente_extraccion_kb","experiment_id":manifest_experiment,"fingerprint":"fp03","safety_policy":{"uses_ground_truth":False,"uses_review_sections":False,"uses_bibliography":False}}),encoding="utf-8")
    if with_chunks:
        pd.DataFrame([{"chunk_id":"c1","source_filename":"paper.pdf","text":"LSTM achieved RMSE 0.25 on Wind","is_review_section_chunk":False,"is_bibliography_chunk":False,"excluded_from_rag":False}]).to_csv(chunks,index=False)
    paths={"scientific_knowledge_base_csv":kb,"scientific_knowledge_base_jsonl":kbj,"scientific_extraction_manifest":manifest}
    if with_chunks: paths["chunks_clean_for_rag_csv"]=chunks
    deps={k:ArtifactReference(path=str(v),hash=sha256_file(v)) for k,v in paths.items()}
    return out,deps,paths

def make_input(root:Path, *, deps=None, policy=None, model="fake-model"):
    out, default_deps, _=make_fixture(root) if deps is None else (root/"out",deps,{})
    return AgentInput(experiment_id="exp",run_id="run",stage_name=STAGE_NAME,attempt_number=1,mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=("llm",),output_directory=str(out),runtime_resources={"model":model}),dependencies=default_deps if deps is None else deps,policy=dict(DEFAULT_QUANTITATIVE_EXTRACTION_POLICY if policy is None else policy))

def capability(llm=None): return QuantitativeExtractionCapability(QuantitativeExtractionDependencies(llm=llm or FakeLLM(),human_message_factory=Message,json_parser=json.loads))

class QuantitativeCapabilityV16Tests(unittest.TestCase):
    def test_direct_call_writes_nine_atomic_artifacts_and_advances_neutrally(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=make_input(Path(tmp)); result=capability().execute(inp)
            self.assertEqual(result.execution_status,ExecutionStatus.COMPLETED)
            self.assertIn(result.quality_status,{QualityStatus.APPROVED,QualityStatus.APPROVED_WITH_WARNINGS})
            self.assertEqual(result.requested_transition.action,TransitionAction.ADVANCE)
            self.assertIsNone(result.requested_transition.target_stage)
            self.assertEqual(set(result.output_artifacts),set(ARTIFACT_FILENAMES))
            for ref in result.output_artifacts.values(): self.assertTrue(Path(ref.path).is_file()); self.assertEqual(sha256_file(ref.path),ref.hash)
    def test_verification_statuses_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=capability().execute(make_input(Path(tmp)))
            table=pd.read_csv(result.output_artifacts["quantitative_comparative_table.csv"].path)
            self.assertEqual(table.loc[0,"verification_status"],"confirmed_in_source_chunk")
            self.assertIn("source_chunk_scope",table.columns); self.assertIn("source_chunk_ids_checked",table.columns); self.assertIn("source_text_evidence",table.columns)
    def test_missing_kb_is_failed_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out,deps,paths=make_fixture(root); Path(paths["scientific_knowledge_base_csv"]).unlink()
            inp=make_input(root,deps=deps); result=capability().execute(inp)
            self.assertEqual(result.execution_status,ExecutionStatus.FAILED); self.assertIn("DEPENDENCY_NOT_FOUND",result.failure_reason_codes)
    def test_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out,deps,paths=make_fixture(root,manifest_experiment="other")
            result=capability().execute(make_input(root,deps=deps)); self.assertEqual(result.execution_status,ExecutionStatus.FAILED); self.assertIn("KB_MANIFEST_MISMATCH",result.failure_reason_codes)
    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out,deps,paths=make_fixture(root); deps=dict(deps); deps["scientific_knowledge_base_csv"]=ArtifactReference(path=str(paths["scientific_knowledge_base_csv"]),hash="bad")
            result=capability().execute(make_input(root,deps=deps)); self.assertIn("DEPENDENCY_HASH_MISMATCH",result.failure_reason_codes)
    def test_missing_required_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out,deps,paths=make_fixture(root,with_chunks=False)
            result=capability().execute(make_input(root,deps=deps)); self.assertIn("DEPENDENCY_NOT_FOUND",result.failure_reason_codes)
    def test_ground_truth_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=make_input(root); policy=dict(inp.policy); policy["ground_truth_path"]="secret.pdf"
            inp=AgentInput(experiment_id=inp.experiment_id,run_id=inp.run_id,stage_name=inp.stage_name,attempt_number=1,mode=inp.mode,agent_context=inp.agent_context,dependencies=inp.dependencies,policy=policy)
            result=capability().execute(inp); self.assertIn("GROUND_TRUTH_POLICY_VIOLATION",result.failure_reason_codes)
    def test_reuse_by_same_fingerprint_skips_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=make_input(root); first=capability().execute(inp); self.assertEqual(len(first.output_artifacts),9)
            second=capability(FailingLLM()).execute(inp); self.assertEqual(second.decision.code,"QUANTITATIVE_EXTRACTION_REUSED"); self.assertEqual(second.tool_usage.llm_calls,0)
    def test_kb_change_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=make_input(root); capability().execute(inp)
            kb=Path(inp.dependencies["scientific_knowledge_base_csv"].path); df=pd.read_csv(kb); df.loc[0,"main_results"]="changed RMSE 0.25"; df.to_csv(kb,index=False)
            deps=dict(inp.dependencies); deps["scientific_knowledge_base_csv"]=ArtifactReference(path=str(kb),hash=sha256_file(kb)); changed=AgentInput(experiment_id="exp",run_id="run",stage_name=STAGE_NAME,attempt_number=1,mode=ExecutionMode.FULL_RUN,agent_context=inp.agent_context,dependencies=deps,policy=inp.policy)
            llm=FakeLLM(); result=capability(llm).execute(changed); self.assertEqual(llm.calls,1); self.assertEqual(result.decision.code,"QUANTITATIVE_EXTRACTION_COMPLETED")
    def test_policy_model_prompt_manifest_and_chunks_change_fingerprint(self):
        from unittest.mock import patch
        import src.capabilities.quantitative_extraction as module
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=make_input(root); policy=dict(inp.policy)
            base=build_quantitative_composite_fingerprint(inp,policy).composite
            changed_policy=dict(policy); changed_policy["temperature"]=0.2
            self.assertNotEqual(base,build_quantitative_composite_fingerprint(inp,changed_policy).composite)
            model_input=AgentInput(experiment_id="exp",run_id="run",stage_name=STAGE_NAME,attempt_number=1,mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=("llm",),output_directory=inp.agent_context.output_directory,runtime_resources={"model":"other"}),dependencies=inp.dependencies,policy=policy)
            self.assertNotEqual(base,build_quantitative_composite_fingerprint(model_input,policy).composite)
            with patch.object(module,"QUANT_PROMPT_VERSION","changed-prompt"):
                self.assertNotEqual(base,module.build_quantitative_composite_fingerprint(inp,policy).composite)
            for dep_name in ("scientific_extraction_manifest","chunks_clean_for_rag_csv"):
                deps=dict(inp.dependencies); old=deps[dep_name]; deps[dep_name]=ArtifactReference(path=old.path,hash="changed-hash")
                changed=AgentInput(experiment_id="exp",run_id="run",stage_name=STAGE_NAME,attempt_number=1,mode=ExecutionMode.FULL_RUN,agent_context=inp.agent_context,dependencies=deps,policy=policy)
                self.assertNotEqual(base,build_quantitative_composite_fingerprint(changed,policy).composite)
    def test_credential_not_in_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=make_input(Path(tmp)); self.assertNotIn("OPENAI",json.dumps(inp.to_dict())); self.assertNotIn("sk-",build_quantitative_fingerprints(inp).composite)
    def test_no_import_of_extraction_agent_or_stage04(self):
        source=(Path(__file__).parents[2]/"src/capabilities/quantitative_extraction.py").read_text()
        self.assertNotIn("ExtractionAgent",source); self.assertNotIn("04_",source)
    def test_transaction_prepare_persist_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=make_input(root); state_path=root/"pipeline_state.json"; store=StateStore(state_path); now=datetime.now(timezone.utc).isoformat(); store.initialize(PipelineState(identity=PipelineIdentity(experiment_id="exp",run_id="run",created_at=now,updated_at=now,schema_version="1.0")))
            tx=execute_quantitative_transaction(store=store,capability=capability(),agent_input=inp)
            self.assertTrue(Path(tx.persisted_result_path).is_file()); self.assertIsNone(tx.committed_state.pending_execution); self.assertIn(STAGE_NAME,tx.committed_state.stages)
    def test_early_runtime_failure_is_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state_path=root/"pipeline_state.json"; store=StateStore(state_path); now=datetime.now(timezone.utc).isoformat(); store.initialize(PipelineState(identity=PipelineIdentity(experiment_id="exp",run_id="run",created_at=now,updated_at=now,schema_version="1.0")))
            tx=execute_quantitative_runtime_transaction(store=store,build_execution=lambda: (_ for _ in ()).throw(FileNotFoundError("KB ausente")))
            self.assertEqual(tx.agent_result.execution_status,ExecutionStatus.FAILED); self.assertIn("DEPENDENCY_NOT_FOUND",tx.agent_result.failure_reason_codes); self.assertIsNone(tx.committed_state.pending_execution)

if __name__=="__main__": unittest.main()
