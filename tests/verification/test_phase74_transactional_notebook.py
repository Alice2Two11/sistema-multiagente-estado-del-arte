from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest

from test_phase72_runtime_notebook_closure import runtime_input, deps
from test_phase73_transactional_integration import store_at, tx_input
from src.adapters.verification_runtime import VerificationRuntimeDependencies
from src.adapters.verification_notebook import (
    AGENT07_ARTIFACT_NAMES, AGENT07_STAGE_NAME,
    Agent07ArtifactManifest, Agent07ManifestArtifact, Agent07ResumeResult,
    commit_executed_agent07, execute_prepared_agent07, prepare_agent07_execution,
    resume_agent07_execution, validate_agent07_artifact_manifest_contract,
    validate_agent07_resume_result_contract, validate_executed_agent07_execution_contract,
    validate_prepared_agent07_execution_contract,
)
from src.state.pipeline_state import StageState
from src.contracts.agent_result import ExecutionStatus


def test_operational_block_never_publishes_scientific_artifacts(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path)
    d=deps("COMPLETED")
    blocked=VerificationRuntimeDependencies(
        verification_agent_factory=d.verification_agent_factory,
        proposal_runner=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("secret")),
        bundle_builder=d.bundle_builder,resolution_runner=d.resolution_runner,
        correction_context_factory=d.correction_context_factory,
        reverification_input_factory=d.reverification_input_factory,
    )
    prepared=prepare_agent07_execution(store=store,runtime_input=value)
    executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=blocked)
    assert set(executed.candidate_payloads)=={"agent07_runtime_report.json","agent07_operational_audit.json"}
    with pytest.raises(RuntimeError,match="OPERATIONAL_BLOCK"):
        commit_executed_agent07(store=store,executed=executed)
    assert not (tmp_path/"official").exists()
    assert store.load().pending_execution is not None


def test_failed_commit_leaves_no_official_names(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path)
    prepared=prepare_agent07_execution(store=store,runtime_input=value)
    executed=execute_prepared_agent07(store=store,prepared=prepared,dependencies=deps("COMPLETED"))
    with pytest.raises(RuntimeError,match="INJECTED"):
        commit_executed_agent07(store=store,executed=executed,fail_after_writes=2)
    assert not (tmp_path/"official").exists()


def test_manifest_contract_tampering_hash_size_missing():
    arts=tuple(Agent07ManifestArtifact(n,"a"*64,10) for n in AGENT07_ARTIFACT_NAMES[:3])
    m=Agent07ArtifactManifest(AGENT07_STAGE_NAME,"d",1,"b"*64,{"bundle":"v2"},"c"*64,arts,False,False)
    validate_agent07_artifact_manifest_contract(m)
    bad=m.to_dict(); bad["artifacts"][0]["sha256"]="x"*64
    with pytest.raises(ValueError): validate_agent07_artifact_manifest_contract(bad)
    bad=m.to_dict(); bad["artifacts"][0]["size_bytes"]=-1
    with pytest.raises(ValueError): validate_agent07_artifact_manifest_contract(bad)
    bad=m.to_dict(); bad["artifacts"]=bad["artifacts"][:-1]
    with pytest.raises(ValueError): validate_agent07_artifact_manifest_contract(bad)


def test_attempt_two_uses_stage_attempts_plus_one(tmp_path):
    store=store_at(tmp_path); state=store.load()
    stages=dict(state.stages); stages[AGENT07_STAGE_NAME]=StageState(execution_status=ExecutionStatus.FAILED,attempts_used=1)
    store.save(replace(state,stages=stages))
    prepared=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path))
    assert prepared.attempt_number==2
    assert store.load().pending_execution.attempt_number==2


def test_transaction_uuid_changes_but_execution_fingerprint_stable(tmp_path):
    s1=store_at(tmp_path/"a"); s2=store_at(tmp_path/"b")
    v1=tx_input(tmp_path/"same"); v2=tx_input(tmp_path/"same")
    p1=prepare_agent07_execution(store=s1,runtime_input=v1); p2=prepare_agent07_execution(store=s2,runtime_input=v2)
    assert p1.decision_id!=p2.decision_id
    assert p1.execution_fingerprint==p2.execution_fingerprint


def test_operational_contract_validators_and_resume_catalog(tmp_path):
    store=store_at(tmp_path); p=prepare_agent07_execution(store=store,runtime_input=tx_input(tmp_path))
    validate_prepared_agent07_execution_contract(p)
    e=execute_prepared_agent07(store=store,prepared=p,dependencies=deps("COMPLETED"))
    validate_executed_agent07_execution_contract(e)
    validate_agent07_resume_result_contract(Agent07ResumeResult("EXECUTED_NOT_COMMITTED",None,e))
    with pytest.raises(ValueError): validate_agent07_resume_result_contract(Agent07ResumeResult("WHATEVER",None,None))


def test_resume_removes_partial_official_and_reuses_staging(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path)
    p=prepare_agent07_execution(store=store,runtime_input=value); e=execute_prepared_agent07(store=store,prepared=p,dependencies=deps("COMPLETED"))
    official=tmp_path/"official"; official.mkdir(); (official/AGENT07_ARTIFACT_NAMES[0]).write_text("{}")
    resumed=resume_agent07_execution(store=store,runtime_input=value)
    assert resumed.action=="EXECUTED_NOT_COMMITTED"
    assert not official.exists()


def test_manifest_real_bytes_validated_after_commit(tmp_path):
    store=store_at(tmp_path); value=tx_input(tmp_path)
    p=prepare_agent07_execution(store=store,runtime_input=value); e=execute_prepared_agent07(store=store,prepared=p,dependencies=deps("COMPLETED"))
    commit_executed_agent07(store=store,executed=e)
    official=tmp_path/"official"; manifest=json.loads((official/AGENT07_ARTIFACT_NAMES[3]).read_text())
    validate_agent07_artifact_manifest_contract(manifest,artifact_bytes={n:(official/n).read_bytes() for n in AGENT07_ARTIFACT_NAMES[:3]})
    assert e.runtime_result.correction_applied is False and e.runtime_result.evaluation_ready_emitted is False
