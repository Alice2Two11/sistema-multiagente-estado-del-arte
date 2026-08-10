"""Prueba de regresión: una ``pending_execution`` GLOBAL (el ``StateStore``
solo tiene un slot, no uno por etapa) que quedó de OTRA etapa (ej. 07,
crasheando antes de comprometer) no debe bloquear silenciosamente la
preparación de la etapa que el pipeline está intentando ahora con
``RuntimeError("a pending execution already exists")``.

Causa raíz: ``StateStore.prepare_execution`` rechaza cualquier PREPARE
nuevo mientras exista OTRA ``pending_execution`` sin resolver, sin
importar a qué etapa pertenezca. El bucle de ``run_pipeline`` procesaba
las etapas en orden sin verificar primero si la pending actual
pertenecía a la etapa que estaba a punto de intentar -- si no coincidía,
el intento de PREPARE de la etapa actual simplemente reventaba.

Corrección: ``_reconcile_pending_execution_for_other_stage`` (nueva
función, llamada desde ``run_pipeline`` antes de cada intento de etapa)
reconcilia esa pending vía el protocolo OFICIAL de la etapa a la que
realmente pertenece -- ``run_stage()`` sobre su propio ``StageSpec`` --
antes de tocar la etapa actual. Nunca lee ni escribe
``pending_execution`` directamente; nunca resetea rondas; nunca requiere
``--force-rerun``.

Como ``_stage_registry()`` real usa builders que necesitan credenciales
OpenAI/red (``build_real_draft_execution``, etc.), estas pruebas usan un
registro de ``StageSpec`` FALSOS -- mismo patrón que ``smoke_test.py`` --
para poder probar el MECANISMO de reconciliación end-to-end sin red,
llamando a la función productiva real, no una reescrita para el test.
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


STAGE_A = "06_agente_redactor"  # etapa que el pipeline intenta ahora
STAGE_B = "07_agente_verificador"  # etapa dueña de la pending vieja


def _make_agent_input(stage_key, attempt=1):
    return AgentInput(
        experiment_id="exp1", run_id="run1", stage_name=stage_key,
        attempt_number=attempt, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("llm",), output_directory="/tmp/out"),
        dependencies={}, policy={"p": 1},
    )


class _FakeAgent:
    def __init__(self):
        self.calls = 0

    def execute(self, agent_input):
        self.calls += 1
        now = datetime.now(timezone.utc).isoformat()
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED, quality_status=QualityStatus.APPROVED,
            decision=DecisionInfo(code="OK", rationale="ok"), quality_metrics={}, warnings=(),
            requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, reason_code="OK"),
            output_artifacts={}, tool_usage=ToolUsage(),
            attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
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
        state = store.commit_execution(
            decision_id=prep.decision_id, result=result, stage_name=stage_key,
            fingerprints=fingerprints, observations=dict(observations or {}),
        )

        class _Result:
            pass

        out = _Result()
        out.agent_result = result
        out.state = state
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


@scenario("E01. pending de OTRA etapa (B) no bloquea la reconciliación al intentar la etapa actual (A) -- se resuelve vía el protocolo oficial de B")
def test_stale_pending_other_stage_reconciled():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)

        agent_b = _FakeAgent()
        spec_b = _make_fake_spec(STAGE_B, agent_b)
        agent_a = _FakeAgent()
        spec_a = _make_fake_spec(STAGE_A, agent_a)
        registry = {STAGE_A: spec_a, STAGE_B: spec_b}

        # Simula EXACTAMENTE el escenario real: B (07) quedó con una
        # ejecución preparada y persistida, pero nunca comprometida
        # (crash antes del COMMIT oficial) -- pending_execution queda
        # apuntando a B, no a A.
        prep = store.prepare_execution(target_stage=STAGE_B, intended_action="EXECUTE", attempt_number=1)
        pending_result = agent_b.execute(_make_agent_input(STAGE_B, 1))
        store.persist_agent_result(prep.decision_id, pending_result)
        agent_b.calls = 0  # resetear para medir solo lo que pase durante la reconciliación

        state_before = store.load()
        assert state_before.pending_execution is not None
        assert state_before.pending_execution.target_stage == STAGE_B

        # Intentar preparar A directamente (sin reconciliar) DEBE seguir
        # fallando -- confirma que el problema es real, no que lo estoy
        # inventando.
        try:
            store.prepare_execution(target_stage=STAGE_A, intended_action="EXECUTE", attempt_number=1)
        except RuntimeError as exc:
            assert "a pending execution already exists" in str(exc)
        else:
            raise AssertionError("se esperaba que el PREPARE directo de A siguiera fallando sin reconciliar")

        outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=root, registry=registry, current_stage=STAGE_A,
            attempt_numbers={}, observations=None,
        )

        assert must_stop is False
        assert len(outcomes) == 1
        assert outcomes[0].key == STAGE_B
        assert outcomes[0].status == "COMMITTED"
        assert agent_b.calls == 0  # el resultado ya persistido se comprometió sin reinvocar al agente

        state_after = store.load()
        assert state_after.pending_execution is None
        assert state_after.stages[STAGE_B].execution_status == ExecutionStatus.COMPLETED

        # Ahora SÍ debe poder prepararse A sin excepción.
        run_stage(store=store, project_dir=root, spec=spec_a, attempt_number=1)
        assert agent_a.calls == 1


@scenario("E02. Sin pending, o pending de la MISMA etapa actual: la reconciliación no hace nada (no interfiere con el flujo normal)")
def test_no_reconciliation_needed_cases():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)
        agent_a = _FakeAgent()
        spec_a = _make_fake_spec(STAGE_A, agent_a)
        registry = {STAGE_A: spec_a}

        # Caso 1: sin ninguna pending.
        outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=root, registry=registry, current_stage=STAGE_A,
            attempt_numbers={}, observations=None,
        )
        assert outcomes == []
        assert must_stop is False

        # Caso 2: pending de la MISMA etapa que se está por intentar --
        # eso lo resuelve el propio run_stage() más abajo, no esta función.
        store.prepare_execution(target_stage=STAGE_A, intended_action="EXECUTE", attempt_number=1)
        outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=root, registry=registry, current_stage=STAGE_A,
            attempt_numbers={}, observations=None,
        )
        assert outcomes == []
        assert must_stop is False
        assert store.load().pending_execution is not None  # no se tocó -- run_stage la resuelve después


@scenario("E03. pending que apunta a una etapa sin StageSpec registrado: se detiene explícitamente, no se oculta ni se fuerza")
def test_pending_unknown_target_stage_stops_explicitly():
    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)
        agent_a = _FakeAgent()
        spec_a = _make_fake_spec(STAGE_A, agent_a)
        registry = {STAGE_A: spec_a}  # STAGE_B deliberadamente NO está en el registro

        store.prepare_execution(target_stage=STAGE_B, intended_action="EXECUTE", attempt_number=1)

        outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=root, registry=registry, current_stage=STAGE_A,
            attempt_numbers={}, observations=None,
        )
        assert must_stop is True
        assert len(outcomes) == 1
        assert outcomes[0].reason_code == "PENDING_EXECUTION_UNKNOWN_TARGET_STAGE"
        # La pending real NO se tocó -- sigue exactamente como estaba.
        assert store.load().pending_execution is not None
        assert store.load().pending_execution.target_stage == STAGE_B


@scenario("E04. Escenario completo del reporte real: round_01 REVISION_COMPLETED (06 ya comprometido) + pending vieja de 07 -> el pipeline reanuda sin tocar el estado a mano")
def test_full_reported_scenario_end_to_end():
    """Reproduce, con dobles deterministas, la secuencia exacta reportada:
    06 ya comprometido (revisión de round_01 ya completada) + una
    pending_execution vieja de 07 (crash antes del parche 1) -- y
    confirma que un restart (sin --force-rerun, sin tocar round_01 a
    mano) primero reconcilia la pending de 07 vía su propio protocolo, y
    LUEGO deja pasar la preparación de 06 con normalidad."""

    with tempfile.TemporaryDirectory() as tmp:
        root = _seed_project(tmp)
        store = ensure_pipeline_state(root)

        agent_06 = _FakeAgent()
        spec_06 = _make_fake_spec(STAGE_A, agent_06)
        agent_07 = _FakeAgent()
        spec_07 = _make_fake_spec(STAGE_B, agent_07)
        registry = {STAGE_A: spec_06, STAGE_B: spec_07}

        # 1. 06 ya se ejecutó y comprometió una vez (equivalente real a
        #    "round_01 REVISION_COMPLETED, 06 ya comprometido").
        run_stage(store=store, project_dir=root, spec=spec_06, attempt_number=1)
        assert agent_06.calls == 1
        assert store.load().stages[STAGE_A].execution_status == ExecutionStatus.COMPLETED

        # 2. 07 queda con una ejecución preparada y persistida pero NUNCA
        #    comprometida -- equivalente real al crash anterior al parche 1
        #    (AGENT07_SCIENTIFIC_BLOCK_NOT_OFFICIAL_COMMITTABLE reventando
        #    antes de llegar a store.commit_execution).
        prep = store.prepare_execution(target_stage=STAGE_B, intended_action="EXECUTE", attempt_number=1)
        pending_result = agent_07.execute(_make_agent_input(STAGE_B, 1))
        store.persist_agent_result(prep.decision_id, pending_result)
        agent_07.calls = 0

        # 3. "Restart" del pipeline: intento de reconciliar antes de tocar
        #    06 de nuevo -- exactamente lo que run_pipeline() hace ahora en
        #    cada vuelta del bucle, sin --force-rerun y sin tocar
        #    pending_execution ni round_01 a mano.
        outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=root, registry=registry, current_stage=STAGE_A,
            attempt_numbers={}, observations=None,
        )
        assert must_stop is False
        assert outcomes[0].key == STAGE_B
        assert outcomes[0].status == "COMMITTED"
        assert store.load().pending_execution is None

        # 4. Con la pending ya reconciliada, 06 se resuelve con
        #    normalidad -- y como YA estaba COMPLETED (paso 1) con
        #    fingerprints vigentes, run_stage() debe reconocerlo
        #    SKIPPED_FRESH, sin reinvocar al agente ni tocar nada de 06.
        calls_before = agent_06.calls
        outcome_06 = run_stage(store=store, project_dir=root, spec=spec_06, attempt_number=1)
        assert outcome_06.status == "SKIPPED_FRESH"
        assert agent_06.calls == calls_before


if __name__ == "__main__":
    for fn in (
        test_stale_pending_other_stage_reconciled,
        test_no_reconciliation_needed_cases,
        test_pending_unknown_target_stage_stops_explicitly,
        test_full_reported_scenario_end_to_end,
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
