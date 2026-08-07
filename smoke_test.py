import sys, json, tempfile
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, '.')

from src.orchestration.pipeline_orchestrator import (
    StageSpec, run_stage, ensure_pipeline_state, resolve_state_path,
)
from src.contracts.agent_input import AgentInput, AgentContext, ExecutionMode
from src.contracts.agent_result import (
    AgentResult, DecisionInfo, ExecutionStatus, QualityStatus,
    RequestedTransition, TransitionAction, ToolUsage,
)
from src.runtime.thematic_analysis_protocol import (
    execute_thematic_runtime_transaction, resolve_thematic_resume,
)

STAGE_KEY = "04_agente_analisis_tematico"

def make_agent_input(attempt=1):
    return AgentInput(
        experiment_id="exp1", run_id="run1", stage_name=STAGE_KEY,
        attempt_number=attempt, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("llm",), output_directory="/tmp/out"),
        dependencies={}, policy={"p": 1},
    )

class FakeAgent:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
    def execute(self, agent_input):
        self.calls += 1
        now = datetime.now(timezone.utc).isoformat()
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED,
            quality_status=QualityStatus.APPROVED,
            decision=DecisionInfo(code="OK", rationale="ok"),
            quality_metrics={}, warnings=(),
            requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, reason_code="OK"),
            output_artifacts={}, tool_usage=ToolUsage(),
            attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
        )

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "active_experiment.json").write_text(json.dumps({
        "active_experiment_id": "exp1", "run_id": "run1",
    }), encoding="utf-8")

    fake_agent = FakeAgent()
    build_calls = {"n": 0}
    def build_execution(project_dir, attempt_number):
        build_calls["n"] += 1
        return fake_agent, make_agent_input(attempt_number)

    spec = StageSpec(
        key=STAGE_KEY, label="fake stage",
        build_execution=build_execution,
        runtime_transaction=execute_thematic_runtime_transaction,
        resolve_resume=resolve_thematic_resume,
    )

    store = ensure_pipeline_state(root)

    # 1) first run: should COMMIT and execute the agent once
    out1 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
    assert out1.status == "COMMITTED", out1
    assert fake_agent.calls == 1, fake_agent.calls
    print("PASS: first run commits", out1)

    # 2) second run: already COMPLETED and fingerprints still match -> should
    #    SKIP as fresh, agent not called again. (El contrato real de
    #    run_stage usa "SKIPPED_FRESH" para este caso -- "SKIPPED_ALREADY_COMPLETED"
    #    no es un valor que StageOutcome.status produzca; se corrige aquí la
    #    aserción del smoke test, no la lógica productiva.)
    out2 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
    assert out2.status == "SKIPPED_FRESH", out2
    assert fake_agent.calls == 1
    print("PASS: second run skips", out2)

    # 3) force_rerun: should execute again
    out3 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1, force_rerun=True)
    assert out3.status == "COMMITTED", out3
    assert fake_agent.calls == 2
    print("PASS: force_rerun re-executes", out3)

    # 4) simulate a leftover PENDING execution from a crashed run, with a
    #    persisted result already on disk -> resume should COMMIT it without
    #    calling the agent again.
    state = store.load()
    prep = store.prepare_execution(target_stage=STAGE_KEY, intended_action="EXECUTE_THEMATIC_ANALYSIS", attempt_number=1)
    pending_result = fake_agent.execute(make_agent_input(1))
    store.persist_agent_result(prep.decision_id, pending_result)
    calls_before = fake_agent.calls
    out4 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1, force_rerun=True)
    assert out4.status == "COMMITTED", out4
    assert fake_agent.calls == calls_before, "resume should not re-invoke the agent"
    print("PASS: resume commits persisted result without re-executing", out4)

print("ALL SMOKE TESTS PASSED")
