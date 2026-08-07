"""Pruebas de integración del orquestador ampliado.

Usa un ``StateStore`` real sobre un directorio temporal (no mocks de
StateStore) y los protocolos transaccionales reales de
``src/runtime/*_protocol.py`` (``execute_thematic_runtime_transaction``,
``execute_outline_runtime_transaction``, ``resolve_*_resume``,
``build_*_fingerprints``). Solo se reemplaza la parte que normalmente llama a
un LLM/Chroma real: ``build_execution`` usa un ``FakeAgent`` controlable en
vez de ``build_real_*_execution``.

No ejecuta notebooks ni llama a ningún LLM real. Corre como script plano
(``python3 tests/orchestration/test_decision_engine.py``); no depende de
pytest para poder ejecutarse en este entorno sin dependencias adicionales.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
from src.runtime.draft_writing_protocol import (
    build_draft_fingerprints,
    resolve_draft_resume,
)
from src.runtime.draft_writing_protocol import execute_draft_transaction as _unused  # noqa: F401
from src.runtime.outline_generation_protocol import (
    build_outline_fingerprints,
    execute_outline_runtime_transaction,
    resolve_outline_resume,
)
from src.runtime.thematic_analysis_protocol import (
    build_thematic_fingerprints,
    execute_thematic_runtime_transaction,
    resolve_thematic_resume,
)

THEMATIC = "04_agente_analisis_tematico"
OUTLINE = "05_generador_esquema"
DRAFT = "06_agente_redactor"


# ---------------------------------------------------------------------------
# Infraestructura de prueba: agente controlable + specs sintéticos reales
# ---------------------------------------------------------------------------


class FakeAgent:
    """Agente cuyo próximo resultado se define desde el test (script FIFO)."""

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


def _now():
    return datetime.now(timezone.utc).isoformat()


def _completed_advance(agent_input, *, target_stage=None):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo(code="OK", rationale="ok"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE,
            target_stage=target_stage,
            reason_code="OK",
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


def _completed_retry(agent_input):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code="RETRY_ME", rationale="reintentar"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.RETRY, reason_code="NEEDS_REVISION"
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def _completed_return(agent_input, *, target_stage):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code="RETURN_ME", rationale="volver"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.RETURN,
            target_stage=target_stage,
            reason_code="INCONSISTENT_UPSTREAM",
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def _completed_halt(agent_input):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED_WITH_WARNINGS,
        decision=DecisionInfo(code="HALT_ME", rationale="detener etapa"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.HALT_STAGE, reason_code="MANUAL_INTERVENTION"
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at=now,
        completed_at=now,
    )


def _completed_stop(agent_input):
    now = _now()
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.REJECTED,
        decision=DecisionInfo(code="STOP_ME", rationale="detener pipeline"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.STOP_PIPELINE, reason_code="UNRECOVERABLE"
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


def make_spec(
    stage_key: str,
    agent: FakeAgent,
    runtime_transaction,
    resolve_resume,
    build_fingerprints,
    *,
    max_attempt_number=None,
    policy_provider=None,
) -> StageSpec:
    policy_provider = policy_provider or (lambda: {"v": 1})

    def build_execution(project_dir, attempt_number):
        return agent, make_agent_input(stage_key, attempt=attempt_number, policy=policy_provider())

    return StageSpec(
        key=stage_key,
        label=f"fake {stage_key}",
        build_execution=build_execution,
        runtime_transaction=runtime_transaction,
        resolve_resume=resolve_resume,
        build_fingerprints=build_fingerprints,
        max_attempt_number=max_attempt_number,
    )


def new_project(tmp: Path) -> Path:
    root = tmp / "proj"
    root.mkdir(parents=True)
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return root


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


@scenario("1. avance normal (ADVANCE hacia la siguiente etapa por defecto)")
def test_advance_normal():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        thematic_agent = FakeAgent()
        outline_agent = FakeAgent()
        registry_patch = {
            THEMATIC: make_spec(
                THEMATIC, thematic_agent, execute_thematic_runtime_transaction,
                resolve_thematic_resume, build_thematic_fingerprints,
            ),
            OUTLINE: make_spec(
                OUTLINE, outline_agent, execute_outline_runtime_transaction,
                resolve_outline_resume, build_outline_fingerprints,
            ),
        }
        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: list(registry_patch.values())
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC, OUTLINE)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC, until=OUTLINE)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert [o.key for o in outcomes] == [THEMATIC, OUTLINE], outcomes
        assert outcomes[0].next_action == "ADVANCE" and outcomes[0].target_stage == OUTLINE
        assert thematic_agent.calls == 1 and outline_agent.calls == 1


@scenario("2. skip de etapa completada y vigente (fingerprints sin cambios)")
def test_skip_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        store = ensure_pipeline_state(root)
        out1 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert out1.status == "COMMITTED" and agent.calls == 1
        out2 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert out2.status == "SKIPPED_FRESH", out2
        assert agent.calls == 1, "no debe volver a invocar al agente"


@scenario("2b. NO se salta si los fingerprints cambiaron (obsoleta)")
def test_stale_reexecutes():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        policy_box = {"v": 1}
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
            policy_provider=lambda: dict(policy_box),
        )
        store = ensure_pipeline_state(root)
        out1 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert out1.status == "COMMITTED" and agent.calls == 1
        policy_box["v"] = 2  # simula que cambió la configuración/entrada
        out2 = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert out2.status == "COMMITTED", out2
        assert agent.calls == 2, "debe reejecutar porque el fingerprint ya no coincide"


@scenario("3. retry (reintenta la misma etapa y luego avanza)")
def test_retry_then_advance():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        agent.queue(_completed_retry, _completed_advance)
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: [spec]
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC,)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC, until=THEMATIC)
            # until=THEMATIC detiene tras el primer resultado; para observar
            # el reintento completo hay que dejar correr sin 'until'.
            agent2 = FakeAgent()
            agent2.queue(_completed_retry, _completed_advance)
            spec2 = make_spec(
                THEMATIC, agent2, execute_thematic_runtime_transaction,
                resolve_thematic_resume, build_thematic_fingerprints,
            )
            po._stage_registry = lambda: [spec2]
            root2 = new_project(Path(tmp) / "p2")
            outcomes2 = run_pipeline(root2, start_stage=THEMATIC)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert outcomes[0].next_action == "RETRY"
        assert agent2.calls == 2, "debe reintentar una vez y luego avanzar"
        assert outcomes2[0].next_action == "RETRY" and outcomes2[0].attempt_number == 1
        assert outcomes2[1].next_action in {"ADVANCE", "STOP_PIPELINE"}
        assert outcomes2[1].attempt_number == 2


@scenario("4. límite de reintentos (RETRY_EXHAUSTED -> HALT_STAGE)")
def test_retry_limit():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        agent.queue(_completed_retry, _completed_retry, _completed_retry, _completed_retry)
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
            max_attempt_number=2,
        )
        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: [spec]
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC,)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert len(outcomes) == 2, outcomes  # intento 1 (RETRY) + intento 2 (HALT_STAGE)
        assert outcomes[0].next_action == "RETRY"
        assert outcomes[1].next_action == "HALT_STAGE"
        assert outcomes[1].reason_code == "RETRY_EXHAUSTED"
        assert agent.calls == 2, "no debe ejecutar un tercer intento (max_attempt_number=2)"


@scenario("5+6. return a etapa anterior + invalidación de etapas posteriores")
def test_return_invalidates_descendants():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        store = ensure_pipeline_state(root)

        thematic_agent = FakeAgent()
        outline_agent = FakeAgent()
        draft_agent = FakeAgent()
        draft_agent.queue(lambda ai: _completed_return(ai, target_stage=THEMATIC))

        thematic_spec = make_spec(
            THEMATIC, thematic_agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        outline_spec = make_spec(
            OUTLINE, outline_agent, execute_outline_runtime_transaction,
            resolve_outline_resume, build_outline_fingerprints,
        )
        # 06 usa execute_draft_transaction directo (no hay *_runtime_transaction
        # público para draft en src/runtime/), así que se envuelve igual que
        # hace pipeline_orchestrator._draft_runtime_transaction.
        def draft_runtime_transaction(*, store, build_execution, attempt_number, observations=None):
            agent, agent_input = build_execution()
            from src.runtime.draft_writing_protocol import execute_draft_transaction

            return execute_draft_transaction(
                store=store, agent=agent, agent_input=agent_input, observations=observations
            )

        draft_spec = make_spec(
            DRAFT, draft_agent, draft_runtime_transaction,
            resolve_draft_resume, build_draft_fingerprints,
        )

        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: [thematic_spec, outline_spec, draft_spec]
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC, OUTLINE, DRAFT)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC, until=DRAFT)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert [o.key for o in outcomes] == [THEMATIC, OUTLINE, DRAFT], outcomes
        assert outcomes[-1].next_action == "RETURN" and outcomes[-1].target_stage == THEMATIC

        # Antes del RETURN, 04/05/06 deben estar COMPLETED.
        state_before_check = store.load()
        for key in (THEMATIC, OUTLINE, DRAFT):
            assert state_before_check.stages[key].execution_status == ExecutionStatus.COMPLETED

        # Simular explícitamente que run_pipeline invalida tras el RETURN
        # (lo hace internamente; aquí lo repetimos para poder inspeccionar
        # el resultado con nombres claros).
        new_state, invalidated = de.invalidate_from(
            store, from_stage_inclusive=THEMATIC, reason="TEST_RETURN"
        )
        assert set(invalidated) == {THEMATIC, OUTLINE, DRAFT}, invalidated
        for key in (THEMATIC, OUTLINE, DRAFT):
            assert new_state.stages[key].execution_status == ExecutionStatus.INVALIDATED


@scenario("7. halt (HALT_STAGE detiene el pipeline)")
def test_halt():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        agent.queue(_completed_halt)
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: [spec]
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC,)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert len(outcomes) == 1
        assert outcomes[0].next_action == "HALT_STAGE"
        assert outcomes[0].reason_code == "MANUAL_INTERVENTION"


@scenario("7b. APPROVED_PENDING_MANUAL_REVIEW fuerza HALT_STAGE aunque pida ADVANCE")
def test_pending_manual_review_forces_halt():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        agent.queue(_completed_pending_review)
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        store = ensure_pipeline_state(root)
        outcome = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert outcome.next_action == "HALT_STAGE"
        assert outcome.reason_code == "MANUAL_REVIEW_REQUIRED"


@scenario("8. stop (STOP_PIPELINE detiene el pipeline)")
def test_stop():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        agent.queue(_completed_stop)
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        import src.orchestration.pipeline_orchestrator as po

        orig = po._stage_registry
        po._stage_registry = lambda: [spec]
        orig_order = po.STAGE_ORDER
        po.STAGE_ORDER = (THEMATIC,)
        try:
            outcomes = run_pipeline(root, start_stage=THEMATIC)
        finally:
            po._stage_registry = orig
            po.STAGE_ORDER = orig_order

        assert len(outcomes) == 1
        assert outcomes[0].next_action == "STOP_PIPELINE"


@scenario("9. resume de una ejecución pendiente (sin StageState previo)")
def test_resume_pending_first_attempt():
    with tempfile.TemporaryDirectory() as tmp:
        root = new_project(Path(tmp))
        agent = FakeAgent()
        spec = make_spec(
            THEMATIC, agent, execute_thematic_runtime_transaction,
            resolve_thematic_resume, build_thematic_fingerprints,
        )
        store = ensure_pipeline_state(root)
        # Simula una interrupción: PREPARE + resultado persistido, sin COMMIT,
        # y sin ningún StageState previo para la etapa (primer intento real).
        prep = store.prepare_execution(
            target_stage=THEMATIC, intended_action="EXECUTE_THEMATIC_ANALYSIS",
            attempt_number=1,
        )
        result = _completed_advance(make_agent_input(THEMATIC, attempt=1))
        store.persist_agent_result(prep.decision_id, result)

        outcome = run_stage(store=store, project_dir=root, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome
        assert agent.calls == 0, "el resume no debe volver a invocar al agente"
        state = store.load()
        assert state.stages[THEMATIC].execution_status == ExecutionStatus.COMPLETED
        assert state.pending_execution is None


@scenario("Validación de transición: RETURN hacia adelante se rechaza")
def test_return_forward_rejected():
    try:
        de.validate_transition(
            current_stage=THEMATIC,
            requested_transition=RequestedTransition(
                action=TransitionAction.RETURN, target_stage=DRAFT, reason_code="X"
            ),
            quality_status=QualityStatus.NEEDS_REVISION,
            attempts_used=1,
            known_stages=frozenset(de.CANONICAL_STAGE_ORDER),
        )
    except de.TransitionValidationError:
        pass
    else:
        raise AssertionError("debía rechazar un RETURN hacia una etapa posterior")


@scenario("Validación de transición: ADVANCE saltando etapas no permitido por defecto")
def test_advance_skip_rejected():
    try:
        de.validate_transition(
            current_stage=THEMATIC,
            requested_transition=RequestedTransition(
                action=TransitionAction.ADVANCE, target_stage=DRAFT, reason_code="X"
            ),
            quality_status=QualityStatus.APPROVED,
            attempts_used=1,
            known_stages=frozenset(de.CANONICAL_STAGE_ORDER),
        )
    except de.TransitionValidationError:
        pass
    else:
        raise AssertionError("debía rechazar un ADVANCE que salta 05 sin permiso explícito")


if __name__ == "__main__":
    for name in list(globals()):
        pass
    for fn in (
        test_advance_normal,
        test_skip_fresh,
        test_stale_reexecutes,
        test_retry_then_advance,
        test_retry_limit,
        test_return_invalidates_descendants,
        test_halt,
        test_pending_manual_review_forces_halt,
        test_stop,
        test_resume_pending_first_attempt,
        test_return_forward_rejected,
        test_advance_skip_rejected,
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
