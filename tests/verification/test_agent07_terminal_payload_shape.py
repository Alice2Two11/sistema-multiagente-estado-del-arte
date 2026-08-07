from dataclasses import replace
from pathlib import Path

import pytest

from test_phase72_runtime_notebook_closure import deps
from test_phase73_transactional_integration import store_at, tx_input
from src.adapters.verification_notebook import (
    AGENT07_ARTIFACT_NAMES,
    OPERATIONAL_AUDIT_NAME,
    commit_executed_agent07,
    execute_prepared_agent07,
    prepare_agent07_execution,
    validate_executed_agent07_execution_contract,
    _expected_candidate_payload_names,
)
from src.adapters.verification_runtime import VerificationRuntimeDependencies


def _execute(tmp_path, status):
    store = store_at(tmp_path)
    prepared = prepare_agent07_execution(store=store, runtime_input=tx_input(tmp_path))
    executed = execute_prepared_agent07(
        store=store,
        prepared=prepared,
        dependencies=deps(status),
    )
    return store, executed


def _operationally_blocked_dependencies():
    base = deps("COMPLETED")
    return VerificationRuntimeDependencies(
        verification_agent_factory=base.verification_agent_factory,
        proposal_runner=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked")),
        bundle_builder=base.bundle_builder,
        resolution_runner=base.resolution_runner,
        correction_context_factory=base.correction_context_factory,
        reverification_input_factory=base.reverification_input_factory,
    )


def test_early_operational_block_uses_audit_only_payload_set(tmp_path):
    store = store_at(tmp_path)
    prepared = prepare_agent07_execution(store=store, runtime_input=tx_input(tmp_path))
    executed = execute_prepared_agent07(
        store=store,
        prepared=prepared,
        dependencies=_operationally_blocked_dependencies(),
    )

    assert executed.runtime_result.runtime_status == "BLOCKED"
    assert executed.runtime_result.provisional_bundle is None
    assert executed.runtime_result.multi_proposal_resolution_result is None
    assert set(executed.candidate_payloads) == {
        "agent07_runtime_report.json",
        OPERATIONAL_AUDIT_NAME,
    }
    validate_executed_agent07_execution_contract(executed)


def test_terminal_scientific_block_uses_full_candidate_payload_set_but_cannot_commit(tmp_path):
    store, executed = _execute(tmp_path, "BLOCKED")

    assert executed.runtime_result.runtime_status == "BLOCKED"
    assert executed.runtime_result.provisional_bundle is not None
    assert executed.runtime_result.provisional_bundle["aggregation_status"] == "INVALID"
    assert executed.runtime_result.multi_proposal_resolution_result is not None
    assert executed.runtime_result.multi_proposal_resolution_result["resolution_status"] == "BLOCKED"
    assert set(executed.candidate_payloads) == set(AGENT07_ARTIFACT_NAMES)
    validate_executed_agent07_execution_contract(executed)

    with pytest.raises(RuntimeError, match="SCIENTIFIC_BLOCK_NOT_OFFICIAL_COMMITTABLE"):
        commit_executed_agent07(store=store, executed=executed)

    assert not (tmp_path / "official").exists()
    assert store.load().pending_execution is not None


def test_completed_result_uses_full_candidate_payload_set(tmp_path):
    _, executed = _execute(tmp_path, "COMPLETED")

    assert executed.runtime_result.runtime_status == "COMPLETED"
    assert set(executed.candidate_payloads) == set(AGENT07_ARTIFACT_NAMES)
    validate_executed_agent07_execution_contract(executed)


def test_partial_result_shape_uses_full_candidate_payload_set(tmp_path):
    _, executed = _execute(tmp_path, "COMPLETED")
    partial_shape = replace(executed.runtime_result, runtime_status="PARTIAL")

    assert _expected_candidate_payload_names(partial_shape) == set(AGENT07_ARTIFACT_NAMES)


def test_rejects_incomplete_scientific_candidate_payload_set(tmp_path):
    _, executed = _execute(tmp_path, "BLOCKED")
    payloads = dict(executed.candidate_payloads)
    payloads.pop("multi_proposal_resolution_result.json")
    broken = replace(executed, candidate_payloads=payloads)

    with pytest.raises(ValueError, match="AGENT07_EXECUTED_PAYLOAD_SET_INVALID"):
        validate_executed_agent07_execution_contract(broken)


def test_rejects_surplus_candidate_payload_and_staging_index_is_not_candidate_payload(tmp_path):
    _, executed = _execute(tmp_path, "COMPLETED")
    payloads = dict(executed.candidate_payloads)
    payloads["staging_index.json"] = Path(executed.staging_manifest_path).read_bytes()
    broken = replace(executed, candidate_payloads=payloads)

    with pytest.raises(ValueError, match="AGENT07_EXECUTED_PAYLOAD_SET_INVALID"):
        validate_executed_agent07_execution_contract(broken)
