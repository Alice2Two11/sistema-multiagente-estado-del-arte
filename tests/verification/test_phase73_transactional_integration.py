from copy import deepcopy
import json
from pathlib import Path
import pytest

from test_phase72_runtime_notebook_closure import runtime_input, deps
from src.adapters.verification_runtime import (
    Agent07RuntimeResult, validate_agent07_runtime_result_contract,
)
from src.adapters.verification_notebook import (
    AGENT07_ARTIFACT_NAMES, AGENT07_STAGE_NAME,
    commit_executed_agent07, execute_prepared_agent07,
    prepare_agent07_execution, resume_agent07_execution,
)
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore


def store_at(tmp_path):
    store=StateStore(tmp_path/"pipeline_state.json")
    store.initialize(PipelineState(identity=PipelineIdentity("exp","run","2026-01-01T00:00:00+00:00","2026-01-01T00:00:00+00:00","v1")))
    return store


def tx_input(tmp_path):
    base=runtime_input()
    return type(base)(base.committed_agent06_output,base.agent07_config,base.policy_versions,base.schema_versions,{"root":str(tmp_path),"agent07_output_dir":str(tmp_path/"official"),"agent07_staging_dir":str(tmp_path/"staging")})


def test_prepare_valid_and_does_not_execute(tmp_path):
    store=store_at(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path))
    assert store.load().pending_execution.decision_id==prepared.decision_id
    assert not (tmp_path/"official").exists()


def test_prepare_rejects_uncommitted_agent06_before_state_change(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); bad=dict(value.committed_agent06_output); bad["commit_status"]="DRAFT"
    with pytest.raises(ValueError,match="NOT_COMMITTED"):
        prepare_agent07_execution(store=store,runtime_input=type(value)(bad,value.agent07_config,value.policy_versions,value.schema_versions,value.experiment_paths))
    assert store.load().pending_execution is None


def test_prepare_rejects_bad_source_fingerprint(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); bad=dict(value.committed_agent06_output); bad["source_draft_fingerprint"]="bad"
    with pytest.raises(ValueError): prepare_agent07_execution(store=store,runtime_input=type(value)(bad,value.agent07_config,value.policy_versions,value.schema_versions,value.experiment_paths))


def test_execute_persists_candidate_not_official(tmp_path):
    store=store_at(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path)); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED"))
    assert Path(executed.persisted_result_path).is_file()
    assert Path(executed.staging_manifest_path).is_file()
    assert not (tmp_path/"official").exists()
    validate_agent07_runtime_result_contract(executed.runtime_result)


def test_operational_block_has_no_scientific_contracts():
    d=deps("COMPLETED")
    from src.adapters.verification_runtime import VerificationRuntimeDependencies, run_agent07_in_memory
    d=VerificationRuntimeDependencies(verification_agent_factory=d.verification_agent_factory,proposal_runner=lambda *a,**k:(_ for _ in()).throw(RuntimeError("secret")),bundle_builder=d.bundle_builder,resolution_runner=d.resolution_runner,correction_context_factory=d.correction_context_factory,reverification_input_factory=d.reverification_input_factory)
    result=run_agent07_in_memory(runtime_input(),dependencies=d)
    assert result.runtime_status=="BLOCKED"
    assert result.provisional_bundle is None and result.multi_proposal_resolution_result is None
    assert result.candidate_artifact_inventory==()
    assert result.blocked_runtime_audit_record is not None


def test_nested_bundle_tamper_rejected():
    result=deps("COMPLETED").bundle_builder({})
    resolution=deps("COMPLETED").resolution_runner(result)
    from src.adapters.verification_runtime import create_agent07_runtime_result,_candidate_inventory,_base_metrics
    terminal=create_agent07_runtime_result(provisional_bundle=result.to_dict(),multi_proposal_resolution_result=resolution.to_dict(),candidate_artifact_inventory=_candidate_inventory(result.to_dict(),resolution.to_dict(),{"provisional_bundle":"v2","multi_proposal_resolution":"v1"}),execution_metrics=_base_metrics(),runtime_warnings=(),runtime_issue_codes=(),runtime_error_records=(),blocked_runtime_audit_record=None,runtime_status="COMPLETED",correction_applied=False,official_artifacts_created=False,evaluation_ready_emitted=False).to_dict()
    terminal["provisional_bundle"]["aggregation_audit_fingerprint"]="f"*64
    with pytest.raises(ValueError): validate_agent07_runtime_result_contract(terminal)


def test_commit_writes_four_artifacts_and_updates_state_once(tmp_path):
    store=store_at(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path)); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED")); state=commit_executed_agent07(store=store,executed=executed)
    assert state.pending_execution is None
    assert state.stages[AGENT07_STAGE_NAME].execution_status.value=="COMPLETED"
    assert set(AGENT07_ARTIFACT_NAMES)<=set(state.artifacts)
    assert all((tmp_path/"official"/name).is_file() for name in AGENT07_ARTIFACT_NAMES)
    assert len([e for e in state.decision_log if e.stage==AGENT07_STAGE_NAME])==1


def test_failure_between_writes_never_commits_state(tmp_path):
    store=store_at(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path)); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED"))
    with pytest.raises(RuntimeError,match="INJECTED"):
        commit_executed_agent07(store=store,executed=executed,fail_after_writes=2)
    state=store.load(); assert state.pending_execution is not None; assert AGENT07_STAGE_NAME not in state.stages


def test_resume_after_execute_without_commit(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=value); execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED"))
    resumed=resume_agent07_execution(store=store,runtime_input=value)
    assert resumed.action=="EXECUTED_NOT_COMMITTED" and resumed.executed is not None


def test_resume_after_commit_and_double_commit_is_idempotent_via_resume(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=value); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED")); commit_executed_agent07(store=store,executed=executed)
    resumed=resume_agent07_execution(store=store,runtime_input=value)
    assert resumed.action=="COMMITTED"
    assert len(store.load().decision_log)==1


def test_resume_detects_tampered_artifact(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=value); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED")); commit_executed_agent07(store=store,executed=executed)
    (tmp_path/"official"/AGENT07_ARTIFACT_NAMES[0]).write_text("{}",encoding="utf-8")
    assert resume_agent07_execution(store=store,runtime_input=value).action=="ARTIFACT_MISMATCH"


def test_manifest_incomplete_detected_before_commit(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); prepared=prepare_agent07_execution(store=store,runtime_input=value); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED"))
    payloads=dict(executed.candidate_payloads); payloads.pop(AGENT07_ARTIFACT_NAMES[1])
    broken=type(executed)(executed.decision_id,executed.runtime_input,executed.runtime_result,payloads,executed.staging_manifest_path,executed.agent_result,executed.persisted_result_path,executed.stage_fingerprints)
    with pytest.raises(RuntimeError,match="MANIFEST_INCOMPLETE"): commit_executed_agent07(store=store,executed=broken)


def test_zero_draft_application_and_evaluation_ready(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path); before=deepcopy(value.committed_agent06_output); prepared=prepare_agent07_execution(store=store,runtime_input=value); executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED")); commit_executed_agent07(store=store,executed=executed)
    assert value.committed_agent06_output==before
    assert executed.runtime_result.correction_applied is False
    assert executed.runtime_result.evaluation_ready_emitted is False
