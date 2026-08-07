from copy import deepcopy
from dataclasses import FrozenInstanceError
import pytest

from src.adapters.verification_runtime import (
    Agent07RuntimeInput,
    VerificationRuntimeDependencies,
    create_agent07_runtime_result,
    run_agent07_in_memory,
    validate_agent07_runtime_input_contract,
    validate_agent07_runtime_result_contract,
)
from test_multi_proposal_resolution_phase66 import bundle as real_bundle, corr as real_corr
from src.tools.verification.resolution import resolve_multiple_correction_proposals

from src.adapters.verification_notebook import (
    Agent07NotebookRequest,
    execute_agent07_notebook_in_memory,
    prepare_agent07_notebook_execution,
)


def fake_bundle(status="VALID"):
    return {
        "aggregation_status": status,
        "normalized_bundle_fingerprint": None if status == "INVALID" else "1" * 64,
        "aggregation_audit_fingerprint": "2" * 64,
    }


def fake_resolution(status="COMPLETED"):
    return {
        "resolution_status": status,
        "multi_proposal_resolution_fingerprint": None if status == "BLOCKED" else "3" * 64,
        "multi_proposal_audit_fingerprint": "4" * 64,
    }


def runtime_input(contexts=None, config=None):
    return Agent07RuntimeInput(
        committed_agent06_output={"commit_status": "COMMITTED", "run_id":"run-06", "artifact_identity":"draft-manifest", "schema_version":"AGENT06_COMMITTED_V1", "source_draft_fingerprint":"a"*64, "claim_verification_contexts": tuple(contexts or ({"claim_id": "c1", "section_id": "s1", "eligible_evidence": [{"evidence_id": "e1"}]},))},
        agent07_config=deepcopy(config or {"policy": {"mode": "strict"}}),
        policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v2", "provisional_bundle": "v2", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": "/tmp/exp"},
    )


def deps(status="COMPLETED", mutate=False):
    class Agent:
        def __init__(self, **kwargs): pass
        def verify_claim(self, context):
            if mutate:
                context["eligible_evidence"].append({"evidence_id": "MUTATED"})
            return {"claim_id": context["claim_id"]}
    def correction_factory(context, verification, config):
        if mutate:
            context["eligible_evidence"].clear()
            config["policy"]["mode"] = "mutated"
        return {"claim_id": context["claim_id"]}
    def proposal(context, *, llm): return {"correction_id": "x1", "accepted_for_reverification": False}
    def build_bundle(value):
        if status == "BLOCKED":
            return real_bundle(status="INVALID")
        if status == "PARTIAL":
            row = real_corr("x1",6,10,"beta","B",decision="DEFER_TO_MANUAL_REVIEW")
            return real_bundle((row,))
        return real_bundle(())
    return VerificationRuntimeDependencies(
        verification_agent_factory=Agent,
        proposal_runner=proposal,
        bundle_builder=build_bundle,
        resolution_runner=resolve_multiple_correction_proposals,
        correction_context_factory=correction_factory,
        reverification_input_factory=lambda *args: {},
    )


@pytest.mark.parametrize("resolution_status,runtime_status", [
    ("COMPLETED", "COMPLETED"), ("BLOCKED", "BLOCKED"),
])
def test_runtime_status_preserves_resolution_status(resolution_status, runtime_status):
    result = run_agent07_in_memory(runtime_input(), dependencies=deps(resolution_status))
    assert result.runtime_status == runtime_status
    assert result.result_contract_valid is True
    validate_agent07_runtime_result_contract(result)


def test_runtime_result_status_tamper_rejected():
    result = run_agent07_in_memory(runtime_input(), dependencies=deps("COMPLETED")).to_dict()
    result["runtime_status"] = "BLOCKED"
    with pytest.raises(ValueError, match="STATUS_MISMATCH"):
        validate_agent07_runtime_result_contract(result)


def test_runtime_result_isolation_tamper_rejected():
    result = run_agent07_in_memory(runtime_input(), dependencies=deps()).to_dict()
    result["execution_metrics"]["official_writes"] = 1
    with pytest.raises(ValueError, match="ISOLATION"):
        validate_agent07_runtime_result_contract(result)


def test_candidate_fingerprint_mismatch_rejected():
    result = run_agent07_in_memory(runtime_input(), dependencies=deps()).to_dict()
    result["candidate_artifact_inventory"][0]["normalized_fingerprint"] = "9" * 64
    with pytest.raises(ValueError, match="ARTIFACT_FINGERPRINT_MISMATCH"):
        validate_agent07_runtime_result_contract(result)


def test_result_contract_valid_is_derived_and_frozen():
    result = run_agent07_in_memory(runtime_input(), dependencies=deps())
    assert result.result_contract_valid
    with pytest.raises(FrozenInstanceError):
        result.runtime_status = "BLOCKED"
    kwargs = result.to_dict(); kwargs.pop("result_contract_valid")
    with pytest.raises(TypeError, match="derived"):
        create_agent07_runtime_result(**kwargs, result_contract_valid=True)


def test_runtime_deep_copy_prevents_dependency_mutation():
    original = runtime_input()
    committed_before = deepcopy(original.committed_agent06_output)
    config_before = deepcopy(original.agent07_config)
    result = run_agent07_in_memory(original, dependencies=deps(mutate=True))
    assert original.committed_agent06_output == committed_before
    assert original.agent07_config == config_before
    assert result.runtime_status == "COMPLETED"


def test_input_validated_before_agent_instantiation():
    events = []
    bad = runtime_input(contexts=({"claim_id": "c1", "section_id": "s1"}, {"claim_id": "c1", "section_id": "s1"}))
    d = deps(); d = VerificationRuntimeDependencies(**{**d.__dict__, "verification_agent_factory": lambda **kwargs: events.append("agent")}) if hasattr(d, '__dict__') else d
    with pytest.raises(ValueError, match="DUPLICATE_CLAIM_CONTEXT"):
        run_agent07_in_memory(bad, dependencies=deps())
    assert events == []


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(commit_status="DRAFT"), "NOT_COMMITTED"),
    (lambda p: p.update(claim_verification_contexts="bad"), "CLAIM_CONTEXTS_INVALID"),
])
def test_runtime_input_contract_rejects_invalid_committed_output(mutation,error):
    value = runtime_input().committed_agent06_output.copy(); mutation(value)
    bad = runtime_input(); bad = Agent07RuntimeInput(value, bad.agent07_config, bad.policy_versions, bad.schema_versions, bad.experiment_paths)
    with pytest.raises(ValueError, match=error): validate_agent07_runtime_input_contract(bad)


def test_unknown_outer_input_field_rejected():
    payload = runtime_input().__dict__ if hasattr(runtime_input(), "__dict__") else {
        "committed_agent06_output": runtime_input().committed_agent06_output,
        "agent07_config": runtime_input().agent07_config,
        "policy_versions": runtime_input().policy_versions,
        "schema_versions": runtime_input().schema_versions,
        "experiment_paths": runtime_input().experiment_paths,
    }
    payload = dict(payload); payload["unknown"] = 1
    with pytest.raises(ValueError, match="INPUT_SCHEMA_INVALID"):
        validate_agent07_runtime_input_contract(payload)


def test_operational_failure_returns_closed_blocked_result_without_raw_message():
    d = deps()
    def explode(*args, **kwargs): raise RuntimeError("SECRET raw prompt and traceback")
    d = VerificationRuntimeDependencies(
        verification_agent_factory=d.verification_agent_factory,
        proposal_runner=explode,
        bundle_builder=d.bundle_builder,
        resolution_runner=d.resolution_runner,
        correction_context_factory=d.correction_context_factory,
        reverification_input_factory=d.reverification_input_factory,
    )
    result = run_agent07_in_memory(runtime_input(), dependencies=d)
    assert result.runtime_status == "BLOCKED"
    assert result.runtime_error_records[0]["error_code"] == "AGENT07_RUNTIME_STAGE_FAILURE:RuntimeError"
    assert "SECRET" not in str(result.to_dict())
    validate_agent07_runtime_result_contract(result)


def test_notebook_full_in_memory_integration_and_state_preservation():
    committed = runtime_input().committed_agent06_output
    committed_before = deepcopy(committed)
    request = Agent07NotebookRequest(
        configuration_source={"policy": {"mode": "strict"}}, committed_agent06_source=committed,
        experiment_paths={"root": "/tmp/exp"}, policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v2", "provisional_bundle": "v2", "multi_proposal_resolution": "v1"},
    )
    prepared = prepare_agent07_notebook_execution(
        request,
        configuration_loader=lambda source: source,
        committed_output_loader=lambda source: source,
        dependency_resolver=lambda config: deps("COMPLETED"),
    )
    assert prepared.preparation_status == "READY"
    executed = execute_agent07_notebook_in_memory(prepared)
    assert executed.runtime_result is not None
    assert executed.runtime_result.runtime_status == "COMPLETED"
    assert executed.runtime_result.result_contract_valid is True
    assert committed == committed_before
    assert executed.official_artifacts_created is False
    assert executed.correction_applied is False
    assert executed.evaluation_ready_emitted is False


def test_notebook_separates_configuration_input_and_dependency_errors():
    request = Agent07NotebookRequest({}, {}, {"root":"/tmp"}, {"p":"v"}, {"s":"v"})
    prepared = prepare_agent07_notebook_execution(
        request,
        configuration_loader=lambda source: (_ for _ in ()).throw(ValueError("bad config")),
        committed_output_loader=lambda source: {"commit_status":"DRAFT","claim_verification_contexts":()},
        dependency_resolver=lambda config: (_ for _ in ()).throw(RuntimeError("bad dep")),
    )
    assert prepared.preparation_status == "BLOCKED"
    assert prepared.configuration_errors
    assert prepared.input_contract_errors
    assert prepared.runtime_result is None


def test_runtime_has_zero_new_side_effect_metrics():
    result = run_agent07_in_memory(runtime_input(), dependencies=deps())
    assert result.execution_metrics["additional_llm_calls"] == 0
    assert result.execution_metrics["additional_retrieval_rounds"] == 0
    assert result.execution_metrics["official_writes"] == 0
    assert result.execution_metrics["physical_corrections"] == 0
    assert not result.correction_applied and not result.official_artifacts_created and not result.evaluation_ready_emitted
