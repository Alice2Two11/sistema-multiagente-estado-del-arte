"""Prueba de regresión: la transición REAL de una etapa reconciliada
(``_reconcile_pending_execution_for_other_stage``) debe gobernar el
flujo principal del bucle de ``run_pipeline`` -- nunca debe ignorarse
para seguir el recorrido normal 02→03→04→05→06.

Causa raíz corregida: ``run_pipeline`` reconciliaba correctamente una
``pending_execution`` de otra etapa (parche 3), pero después de
reconciliarla el bucle simplemente CONTINUABA intentando
``current_stage`` (el que venía procesando en orden), sin importar si
la reconciliación había producido ``HALT_STAGE``, ``RETURN`` o
``ADVANCE``. Esto permitía que, tras reconciliar 07 con un resultado
``HALT_STAGE`` real, el bucle igual llegara a intentar 06 -- que, si su
ronda ya estaba ``REVISION_COMPLETED``, fallaba con
``DRAFT_REVISION_ROUND_ALREADY_COMPLETED`` (una segunda inconsistencia
encima de la primera).

Corrección: se factorizó la interpretación de ``next_action``
(``ADVANCE``/``RETRY``/``RETURN``/``HALT_STAGE``/``STOP_PIPELINE``) en
``_apply_stage_transition`` -- la MISMA función que ya gobernaba el
flujo normal del bucle ahora también se aplica al resultado de la
reconciliación. Si reconciliar produce ``HALT_STAGE``, el bucle se
detiene ahí (06 nunca se intenta). Si produce ``RETURN``, el bucle
continúa desde el ``target_stage`` real de esa transición (vía
``apply_return_with_cycle``, el mismo mecanismo oficial que cualquier
otro RETURN). Si produce ``ADVANCE``, continúa desde el ``target_stage``
real.

Como ``run_pipeline`` construye su propio registro con
``_stage_registry()`` (que exige credenciales OpenAI/red reales), estas
pruebas reproducen UNA vuelta completa del bucle (reconciliar +
despachar la transición real) llamando directamente a las dos funciones
productivas reales que ``run_pipeline`` invoca en ese orden --
``_reconcile_pending_execution_for_other_stage`` y
``_apply_stage_transition`` -- con un registro de ``StageSpec`` FALSOS
(mismo patrón que ``smoke_test.py`` y el parche anterior), en vez de
reescribir esa lógica para el test.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.pipeline_orchestrator import (  # noqa: E402
    StageSpec,
    _apply_stage_transition,
    _reconcile_pending_execution_for_other_stage,
    ensure_pipeline_state,
    run_stage,
)
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


STAGE_06 = "06_agente_redactor"
STAGE_07 = "07_agente_verificador"
STAGE_08 = "08_evaluacion_experimental"


def _make_agent_input(stage_key, attempt=1):
    return AgentInput(
        experiment_id="exp1", run_id="run1", stage_name=stage_key,
        attempt_number=attempt, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("llm",), output_directory="/tmp/out"),
        dependencies={}, policy={"p": 1},
    )


class _FakeAgent:
    """Doble determinista: la transición que produce es parametrizable,
    para poder reproducir HALT_STAGE, RETURN y ADVANCE reconciliados."""

    def __init__(self, *, action: TransitionAction, target_stage: str | None, reason_code: str):
        self.calls = 0
        self._action = action
        self._target_stage = target_stage
        self._reason_code = reason_code

    def execute(self, agent_input):
        self.calls += 1
        now = datetime.now(timezone.utc).isoformat()
        completed = self._action != TransitionAction.HALT_STAGE
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED if completed else ExecutionStatus.FAILED,
            quality_status=QualityStatus.APPROVED if completed else QualityStatus.NEEDS_REVISION,
            decision=DecisionInfo(code=self._reason_code, rationale="doble determinista"),
            quality_metrics={}, warnings=(),
            requested_transition=RequestedTransition(
                action=self._action, target_stage=self._target_stage, reason_code=self._reason_code,
            ),
            output_artifacts={}, tool_usage=ToolUsage(),
            attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
            error=None if completed else {"code": self._reason_code},
        )


def _generic_fingerprints(agent_input):
    return build_stage_fingerprints(
        input_data={"stage_name": agent_input.stage_name, "attempt_number": agent_input.attempt_number},
        config_data=dict(agent_input.policy), dependencies_data=dict(agent_input.dependencies),
    )


def _make_generic_transaction(stage_key):
    def runtime_transaction(*, store, build_execution, attempt_number=1, observations=None):
        prep = store.prepare_execution(target_stage=stage_key, intended_action="EXECUTE", attempt_number=attempt_number)
        agent, agent_input = build_execution()
        result = agent.execute(agent_input)
        fingerprints = _generic_fingerprints(agent_input)
        store.persist_agent_result(prep.decision_id, result)
        store.commit_execution(
            decision_id=prep.decision_id, result=result, stage_name=stage_key,
            fingerprints=fingerprints, observations=dict(observations or {}),
        )

        class _Result:
            pass

        out = _Result()
        out.agent_result = result
        return out

    return runtime_transaction


def _make_generic_resume(stage_key):
    def resolve_resume(*, store, agent_input, observations=None):
        return store.resolve_resume(
            stage_name=agent_input.stage_name, fingerprints=_generic_fingerprints(agent_input),
            observations=dict(observations or {}),
        )

    return resolve_resume


def _make_fake_spec(stage_key, agent):
    def build_execution(project_dir, attempt_number):
        return agent, _make_agent_input(stage_key, attempt_number)

    return StageSpec(
        key=stage_key, label=f"fake {stage_key}", build_execution=build_execution,
        runtime_transaction=_make_generic_transaction(stage_key), resolve_resume=_make_generic_resume(stage_key),
    )


def _seed_project(tmp: Path):
    root = Path(tmp)
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return root


def _leave_stale_pending_for_07(store, agent_07):
    """Reproduce el escenario real: 07 queda con un resultado preparado
    y persistido, pero nunca comprometido (equivalente al crash previo
    al parche 1)."""
    prep = store.prepare_execution(target_stage=STAGE_07, intended_action="EXECUTE", attempt_number=1)
    pending_result = agent_07.execute(_make_agent_input(STAGE_07, 1))
    store.persist_agent_result(prep.decision_id, pending_result)
    agent_07.calls = 0  # medir solo lo que pase durante la reconciliación en sí


def _run_one_loop_iteration(*, store, root, registry, current_stage, attempt_numbers, until=None):
    """Reproduce EXACTAMENTE lo que run_pipeline hace en una vuelta de su
    bucle: reconciliar primero, y si hubo reconciliación, despachar SU
    transición real -- llamando a las dos funciones productivas reales,
    en el mismo orden, con la misma interpretación de resultados."""

    reconcile_outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
        store=store, project_dir=root, registry=registry, current_stage=current_stage,
        attempt_numbers=attempt_numbers, observations=None,
    )
    outcomes = list(reconcile_outcomes)
    if reconcile_outcomes:
        reconcile_outcome = reconcile_outcomes[0]
        if must_stop:
            return outcomes, None, True
        reconciled_stage_key = reconcile_outcome.key
        reconciled_attempt_number = attempt_numbers.get(reconciled_stage_key, 1)
        new_stage, should_stop = _apply_stage_transition(
            reconcile_outcome, store=store, stage_key=reconciled_stage_key,
            attempt_number=reconciled_attempt_number, attempt_numbers=attempt_numbers,
            until=until, outcomes=outcomes,
        )
        return outcomes, new_stage, should_stop

    return outcomes, current_stage, False


@scenario("F01. Reconciliar 07 -> HALT_STAGE: el pipeline se detiene ahí, 06 NUNCA se ejecuta")
def test_reconciled_halt_stage_stops_pipeline_06_not_executed():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)

        agent_07 = _FakeAgent(action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED")
        spec_07 = _make_fake_spec(STAGE_07, agent_07)
        agent_06 = _FakeAgent(action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK")
        spec_06 = _make_fake_spec(STAGE_06, agent_06)
        registry = {STAGE_06: spec_06, STAGE_07: spec_07}

        _leave_stale_pending_for_07(store, agent_07)

        outcomes, new_stage, should_stop = _run_one_loop_iteration(
            store=store, root=root, registry=registry, current_stage=STAGE_06, attempt_numbers={},
        )

        assert should_stop is True
        assert new_stage is None
        assert outcomes[0].key == STAGE_07
        assert outcomes[0].next_action == "HALT_STAGE"
        # La aserción central del reporte: 06 NUNCA se invoca.
        assert agent_06.calls == 0


@scenario("F02. Reconciliar 07 -> RETURN a 06: el flujo continúa desde 06 vía la transición oficial (apply_return_with_cycle real)")
def test_reconciled_return_governs_flow_to_06():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)

        agent_07 = _FakeAgent(action=TransitionAction.RETURN, target_stage=STAGE_06, reason_code="AGENT07_CORRECTABLE_ISSUES")
        spec_07 = _make_fake_spec(STAGE_07, agent_07)
        agent_06 = _FakeAgent(action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK")
        spec_06 = _make_fake_spec(STAGE_06, agent_06)
        registry = {STAGE_06: spec_06, STAGE_07: spec_07}

        _leave_stale_pending_for_07(store, agent_07)

        outcomes, new_stage, should_stop = _run_one_loop_iteration(
            store=store, root=root, registry=registry, current_stage=STAGE_06, attempt_numbers={},
        )

        assert should_stop is False
        assert new_stage == STAGE_06
        assert outcomes[0].key == STAGE_07
        assert outcomes[0].next_action == "RETURN"
        # El ciclo writer_verifier real se actualizó vía apply_return_with_cycle.
        cycle = store.load().cycles.get("writer_verifier")
        assert cycle is not None and cycle.rounds_used >= 1


@scenario("F03. Reconciliar 07 -> ADVANCE a 08: el flujo continúa desde 08 vía la transición oficial")
def test_reconciled_advance_governs_flow_to_08():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)

        agent_07 = _FakeAgent(action=TransitionAction.ADVANCE, target_stage=STAGE_08, reason_code="AGENT07_ALL_CLAIMS_APPROVED")
        spec_07 = _make_fake_spec(STAGE_07, agent_07)
        agent_06 = _FakeAgent(action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK")
        spec_06 = _make_fake_spec(STAGE_06, agent_06)
        registry = {STAGE_06: spec_06, STAGE_07: spec_07}

        _leave_stale_pending_for_07(store, agent_07)

        outcomes, new_stage, should_stop = _run_one_loop_iteration(
            store=store, root=root, registry=registry, current_stage=STAGE_06, attempt_numbers={},
        )

        assert should_stop is False
        assert new_stage == STAGE_08
        assert outcomes[0].key == STAGE_07
        assert outcomes[0].next_action == "ADVANCE"
        assert agent_06.calls == 0  # nunca se intentó 06 -- la transición fue directo a 08


@scenario("F04. Sin pending: el flujo normal (current_stage sin cambios) sigue funcionando exactamente igual que antes")
def test_no_pending_normal_flow_unaffected():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)
        agent_06 = _FakeAgent(action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK")
        spec_06 = _make_fake_spec(STAGE_06, agent_06)
        registry = {STAGE_06: spec_06}

        outcomes, new_stage, should_stop = _run_one_loop_iteration(
            store=store, root=root, registry=registry, current_stage=STAGE_06, attempt_numbers={},
        )
        assert outcomes == []
        assert new_stage == STAGE_06  # nada que reconciliar -- el llamador sigue con run_stage normal
        assert should_stop is False


if __name__ == "__main__":
    for fn in (
        test_reconciled_halt_stage_stops_pipeline_06_not_executed,
        test_reconciled_return_governs_flow_to_06,
        test_reconciled_advance_governs_flow_to_08,
        test_no_pending_normal_flow_unaffected,
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
