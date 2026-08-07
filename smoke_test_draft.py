import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, '.')
from src.orchestration.pipeline_orchestrator import (
    StageSpec, run_stage, ensure_pipeline_state, DRAFT_STAGE_NAME,
    _draft_runtime_transaction,
)
from src.runtime.draft_writing_protocol import resolve_draft_resume

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "active_experiment.json").write_text(json.dumps({
        "active_experiment_id": "exp1", "run_id": "run1",
    }), encoding="utf-8")

    def failing_build(project_dir, attempt_number):
        raise FileNotFoundError("draft_writing_agent.json no existe (simulado)")

    spec = StageSpec(
        key=DRAFT_STAGE_NAME, label="06 draft (fake failure)",
        build_execution=failing_build,
        runtime_transaction=_draft_runtime_transaction,
        resolve_resume=resolve_draft_resume,
    )
    store = ensure_pipeline_state(root)
    out = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
    assert out.status == "FAILED", out
    assert out.execution_status == "FAILED"
    assert out.error["type"] == "FileNotFoundError"
    print("PASS: build failure is committed as FAILED AgentResult:", out)

    # state must reflect it, and a subsequent call should NOT be treated as
    # already-completed (must remain retryable)
    state = store.load()
    assert state.stages[DRAFT_STAGE_NAME].execution_status.value == "FAILED"
    assert state.pending_execution is None
    print("PASS: pipeline_state.json correctly shows FAILED, no dangling pending_execution")
