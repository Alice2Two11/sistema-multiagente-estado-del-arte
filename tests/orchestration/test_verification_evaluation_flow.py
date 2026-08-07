"""Pruebas de integración del flujo 06 -> 07 -> 08 y del ciclo writer_verifier.

07 y 08 se representan aquí con StageSpec sintéticos (mismo enfoque que
test_decision_engine.py): un ``FakeAgent`` controlable en vez de
verification_notebook.py real / la evaluación real de 08 (que no tiene
adaptador -- ver el informe de esta iteración). Lo que SÍ es real en estas
pruebas es el ``StateStore`` (prepare_execution/persist_agent_result/
commit_execution/resolve_resume reales, sobre un directorio temporal) y
``decision_engine`` (validate_transition, apply_return_with_cycle,
invalidate_from, resolve_cycle_if_active) — no se mockea nada de eso.

No ejecuta notebooks ni llama a ningún LLM real.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode
from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration import decision_engine as de
from src.orchestration.pipeline_orchestrator import (
    StageSpec,
    ensure_pipeline_state,
    run_pipeline,
    run_stage,
)
from src.state.fingerprints import build_stage_fingerprints
from src.state.pipeline_state import StageFingerprints
from src.state.state_store import StateStore

DRAFT = "06_agente_redactor"
VERIFY = "07_agente_verificador"
EVAL = "08_evaluacion_experimental"


# ---------------------------------------------------------------------------
# Infraestructura de prueba (idéntica en espíritu a test_decision_engine.py,
# reutilizando el StateStore real para PREPARE/EXECUTE/COMMIT/RESUME en vez
# de mockearlo)
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprints_for(agent_input: AgentInput) -> StageFingerprints:
    return build_stage_fingerprints(
        input_data={
            "experiment_id": agent_input.experiment_id,
            "run_id": agent_input.run_id,
            "stage_name": agent_input.stage_name,
            "attempt_number": agent_input.attempt_number,
        },
        config_data=dict(agent_input.policy),
        dependencies_data={k: v.to_dict() for k, v in agent_input.dependencies.items()},
    )


@dataclass
class _TxResult:
    agent_result: AgentResult


def _generic_runtime_transaction(*, store, build_execution, attempt_number, observations=None):
    agent, agent_input = build_execution()
    prep = store.prepare_execution(
        target_stage=agent_input.stage_name,
        intended_action="TEST_EXECUTE",
        attempt_number=attempt_number,
    )
    result = agent.execute(agent_input)
    store.persist_agent_result(prep.decision_id, result)
    fp = _fingerprints_for(agent_input)
    store.commit_execution(
        decision_id=prep.decision_id,
        result=result,
        stage_name=agent_input.stage_name,
        fingerprints=fp,
        observations=dict(observations or {}),
    )
    return _TxResult(agent_result=result)


def _generic_resolve_resume(*, store, agent_input, observations=None):
    return store.resolve_resume(
        stage_name=agent_input.stage_name,
        fingerprints=_fingerprints_for(agent_input),
        observations=dict(observations or {}),
    )


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self._script: list = []

    def queue(self, *results):
        self._script.extend(results)

    def execute(self, agent_input: AgentInput) -> AgentResult:
        self.calls += 1
        if self._script:
            make_result = self._script.pop(0)
            return make_result(agent_input)
        return _completed_advance(agent_input)


def _completed_advance(agent_input, *, target_stage=None):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo(code="OK", rationale="ok"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE, target_stage=target_stage, reason_code="OK"
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def _completed_return_to_draft(agent_input):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code="CLAIMS_NEED_CORRECTION", rationale="requiere redraft"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.RETURN, target_stage=DRAFT, reason_code="CLAIMS_UNSUPPORTED"
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def _completed_pending_review(agent_input):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED_PENDING_MANUAL_REVIEW,
        decision=DecisionInfo(code="NEEDS_HUMAN", rationale="revisión manual"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE, reason_code="PENDING_REVIEW"
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def make_agent_input(stage_key: str, *, attempt=1, policy=None) -> AgentInput:
    return AgentInput(
        experiment_id="exp1",
        run_id="run1",
        stage_name=stage_key,
        attempt_number=attempt,
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("llm",), output_directory="/tmp/out"),
        dependencies={},
        policy=policy or {"v": 1},
    )


def make_spec(stage_key: str, agent: FakeAgent, **kwargs) -> StageSpec:
    def build_execution(project_dir, attempt_number):
        return agent, make_agent_input(stage_key, attempt=attempt_number)

    return StageSpec(
        key=stage_key,
        label=f"fake {stage_key}",
        build_execution=build_execution,
        runtime_transaction=_generic_runtime_transaction,
        resolve_resume=_generic_resolve_resume,
        build_fingerprints=_fingerprints_for,
        **kwargs,
    )


def new_project(tmp: Path) -> Path:
    root = tmp / "proj"
    root.mkdir(parents=True)
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return root


def patch_registry(specs: list[StageSpec]):
    import src.orchestration.pipeline_orchestrator as po

    orig_registry = po._stage_registry
    orig_order = po.STAGE_ORDER
    po._stage_registry = lambda: list(specs)
    po.STAGE_ORDER = tuple(s.key for s in specs)
    return po, orig_registry, orig_order


def unpatch_registry(po, orig_registry, orig_order):
    po._stage_registry = orig_registry
    po.STAGE_ORDER = orig_order


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:  # noqa: BLE001
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


@scenario("1. 06 -> 07 -> 08 (camino feliz completo)")
def test_happy_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        draft_agent, verify_agent, eval_agent = FakeAgent(), FakeAgent(), FakeAgent()
        specs = [
            make_spec(DRAFT, draft_agent),
            make_spec(VERIFY, verify_agent),
            make_spec(EVAL, eval_agent),
        ]
        po, orig_reg, orig_order = patch_registry(specs)
        try:
            outcomes = run_pipeline(root, start_stage=DRAFT)
        finally:
            unpatch_registry(po, orig_reg, orig_order)

        assert [o.key for o in outcomes] == [DRAFT, VERIFY, EVAL], outcomes
        assert outcomes[-1].next_action == "STOP_PIPELINE"
        assert outcomes[-1].reason_code == "PIPELINE_COMPLETE"
        assert draft_agent.calls == 1 and verify_agent.calls == 1 and eval_agent.calls == 1


@scenario("2. 07 -> RETURN a 06 -> 07 -> 08 (un ciclo de corrección)")
def test_one_correction_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        draft_agent, verify_agent, eval_agent = FakeAgent(), FakeAgent(), FakeAgent()
        verify_agent.queue(_completed_return_to_draft, _completed_advance)
        specs = [
            make_spec(DRAFT, draft_agent),
            make_spec(VERIFY, verify_agent),
            make_spec(EVAL, eval_agent),
        ]
        po, orig_reg, orig_order = patch_registry(specs)
        try:
            outcomes = run_pipeline(root, start_stage=DRAFT)
        finally:
            unpatch_registry(po, orig_reg, orig_order)

        assert [o.key for o in outcomes] == [DRAFT, VERIFY, DRAFT, VERIFY, EVAL], outcomes
        assert draft_agent.calls == 2
        assert verify_agent.calls == 2
        assert eval_agent.calls == 1
        store = ensure_pipeline_state(root)
        cycle = store.load().cycles[de.WRITER_VERIFIER_CYCLE_NAME]
        assert cycle.rounds_used == 1
        assert cycle.status == "RESOLVED"


@scenario("3. agotamiento de CycleState.max_rounds")
def test_cycle_exhaustion():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        draft_agent, verify_agent = FakeAgent(), FakeAgent()
        # 07 pide RETURN indefinidamente -> debe agotarse en max_rounds (3 por defecto)
        verify_agent.queue(*([_completed_return_to_draft] * 5))
        specs = [make_spec(DRAFT, draft_agent), make_spec(VERIFY, verify_agent)]
        po, orig_reg, orig_order = patch_registry(specs)
        try:
            outcomes = run_pipeline(root, start_stage=DRAFT)
        finally:
            unpatch_registry(po, orig_reg, orig_order)

        assert outcomes[-1].status == "CYCLE_EXHAUSTED", outcomes[-1]
        assert outcomes[-1].reason_code == "WRITER_VERIFIER_CYCLE_EXHAUSTED"
        # 3 rondas completas de 06->07 antes de agotar, más el intento 06/07
        # inicial no cuenta como ronda de retorno: verify_agent se llama una
        # vez por cada paso por 07 hasta agotar el presupuesto.
        store = ensure_pipeline_state(root)
        cycle = store.load().cycles[de.WRITER_VERIFIER_CYCLE_NAME]
        assert cycle.status == "EXHAUSTED"
        assert cycle.rounds_used == cycle.max_rounds == 3


@scenario("4. invalidación de 06, 07 y 08 tras un RETURN")
def test_invalidation_06_07_08():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        store = ensure_pipeline_state(root)
        draft_agent, verify_agent, eval_agent = FakeAgent(), FakeAgent(), FakeAgent()
        specs = [
            make_spec(DRAFT, draft_agent),
            make_spec(VERIFY, verify_agent),
            make_spec(EVAL, eval_agent),
        ]
        po, orig_reg, orig_order = patch_registry(specs)
        try:
            # Primero, un pipeline completo y exitoso hasta 08 (para tener
            # algo que invalidar).
            run_pipeline(root, start_stage=DRAFT)
            for key in (DRAFT, VERIFY, EVAL):
                assert store.load().stages[key].execution_status == ExecutionStatus.COMPLETED

            # Ahora se dispara un RETURN explícito 07->06 (simulando que,
            # tras revisar 08, se decide reabrir el ciclo).
            new_state, invalidated = de.invalidate_from(
                store, from_stage_inclusive=DRAFT, reason="TEST_MANUAL_INVALIDATION"
            )
        finally:
            unpatch_registry(po, orig_reg, orig_order)

        assert set(invalidated) == {DRAFT, VERIFY, EVAL}, invalidated
        for key in (DRAFT, VERIFY, EVAL):
            assert new_state.stages[key].execution_status == ExecutionStatus.INVALIDATED


@scenario("5. aprobación pendiente de revisión manual en 07")
def test_manual_review_in_verify():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        verify_agent = FakeAgent()
        verify_agent.queue(_completed_pending_review)
        spec = make_spec(VERIFY, verify_agent)
        store = ensure_pipeline_state(root)
        outcome = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert outcome.next_action == "HALT_STAGE"
        assert outcome.reason_code == "MANUAL_REVIEW_REQUIRED"


@scenario("6. finalización correcta después de 08 (última etapa canónica)")
def test_finalization_after_eval():
    result = de.validate_transition(
        current_stage=EVAL,
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE, reason_code="EVALUATION_COMPLETE"
        ),
        quality_status=QualityStatus.APPROVED,
        attempts_used=1,
        known_stages=frozenset(de.CANONICAL_STAGE_ORDER),
    )
    assert result.action == "STOP_PIPELINE"
    assert result.reason_code == "PIPELINE_COMPLETE"


@scenario("7. interrupción y RESUME en 07")
def test_interrupt_resume_verify():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        spec = make_spec(VERIFY, agent)
        store = ensure_pipeline_state(root)
        prep = store.prepare_execution(
            target_stage=VERIFY, intended_action="TEST_EXECUTE", attempt_number=1
        )
        result = _completed_advance(make_agent_input(VERIFY, attempt=1))
        store.persist_agent_result(prep.decision_id, result)

        outcome = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome
        assert agent.calls == 0, "el resume no debe volver a invocar al agente"
        state = store.load()
        assert state.stages[VERIFY].execution_status == ExecutionStatus.COMPLETED
        assert state.pending_execution is None


@scenario("8. interrupción y RESUME en 08")
def test_interrupt_resume_eval():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        spec = make_spec(EVAL, agent)
        store = ensure_pipeline_state(root)
        prep = store.prepare_execution(
            target_stage=EVAL, intended_action="TEST_EXECUTE", attempt_number=1
        )
        result = _completed_advance(make_agent_input(EVAL, attempt=1))
        store.persist_agent_result(prep.decision_id, result)

        outcome = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome
        assert agent.calls == 0, "el resume no debe volver a invocar al agente"
        state = store.load()
        assert state.stages[EVAL].execution_status == ExecutionStatus.COMPLETED
        assert state.pending_execution is None


if __name__ == "__main__":
    for fn in (
        test_happy_path,
        test_one_correction_cycle,
        test_cycle_exhaustion,
        test_invalidation_06_07_08,
        test_manual_review_in_verify,
        test_finalization_after_eval,
        test_interrupt_resume_verify,
        test_interrupt_resume_eval,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
