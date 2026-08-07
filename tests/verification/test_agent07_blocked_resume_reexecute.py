from pathlib import Path

from test_phase72_runtime_notebook_closure import deps
from test_phase73_transactional_integration import store_at, tx_input
from src.adapters.verification_runtime import VerificationRuntimeDependencies
from src.adapters.verification_notebook import (
    AGENT07_STAGE_NAME,
    execute_prepared_agent07,
    prepare_agent07_execution,
    resume_agent07_execution,
)


def test_blocked_uncommitted_execution_is_reprepared_and_reexecuted(tmp_path):
    store = store_at(tmp_path)
    runtime_input = tx_input(tmp_path)

    first_prepared = prepare_agent07_execution(
        store=store,
        runtime_input=runtime_input,
    )
    base_dependencies = deps("COMPLETED")
    blocked_dependencies = VerificationRuntimeDependencies(
        verification_agent_factory=base_dependencies.verification_agent_factory,
        proposal_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("AGENT07_TEST_OPERATIONAL_BLOCK")
        ),
        bundle_builder=base_dependencies.bundle_builder,
        resolution_runner=base_dependencies.resolution_runner,
        correction_context_factory=base_dependencies.correction_context_factory,
        reverification_input_factory=base_dependencies.reverification_input_factory,
    )
    first_executed = execute_prepared_agent07(
        store=store,
        prepared=first_prepared,
        dependencies=blocked_dependencies,
    )

    assert first_executed.runtime_result.runtime_status == "BLOCKED"
    assert store.load().pending_execution.decision_id == first_prepared.decision_id

    resumed = resume_agent07_execution(
        store=store,
        runtime_input=runtime_input,
    )

    assert resumed.action == "REEXECUTE"
    assert resumed.executed is None

    failed_state = store.load()
    assert failed_state.pending_execution is None
    assert failed_state.stages[AGENT07_STAGE_NAME].attempts_used == 1
    assert failed_state.stages[AGENT07_STAGE_NAME].execution_status.value == "FAILED"
    assert first_executed.runtime_result.provisional_bundle is None
    assert not Path(runtime_input.experiment_paths["agent07_output_dir"]).exists()

    second_prepared = prepare_agent07_execution(
        store=store,
        runtime_input=runtime_input,
    )
    assert second_prepared.decision_id != first_prepared.decision_id
    assert second_prepared.attempt_number == 2

    second_executed = execute_prepared_agent07(
        store=store,
        prepared=second_prepared,
        dependencies=deps("COMPLETED"),
    )

    assert second_executed.decision_id == second_prepared.decision_id
    assert second_executed.attempt_number == 2
    assert second_executed.runtime_result.runtime_status == "COMPLETED"
    assert Path(second_executed.staging_manifest_path).parent.name == second_prepared.decision_id
    assert Path(first_executed.staging_manifest_path).parent.name == first_prepared.decision_id
    assert second_executed.staging_manifest_path != first_executed.staging_manifest_path

    # The blocked attempt remains auditable, but its scientific result and
    # staging-only artifacts are not reused by the second execution.
    state_after_second_execute = store.load()
    assert state_after_second_execute.pending_execution.decision_id == second_prepared.decision_id
    first_log = [
        entry for entry in state_after_second_execute.decision_log
        if entry.decision_id == first_prepared.decision_id
    ]
    assert len(first_log) == 1
    assert first_log[0].observations["scientific_result_reused"] is False
    assert first_log[0].result["output_artifacts"] == {}
