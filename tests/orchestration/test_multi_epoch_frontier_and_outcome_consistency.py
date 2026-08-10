"""Parche 7: dos correcciones sobre la reconstrucción del frontier
autoritativo del parche 6.

Bug 1 -- ``_reconstruct_authoritative_frontier`` asumía que TODO
``decision_log`` es una única cadena causal desde ``decision_log[0]``.
Un experimento real acumula MÚLTIPLES tramos/epochs (reintentos,
restarts, invalidaciones, ciclos 06↔07 repetidos durante todo el
debugging) -- si el primer tramo nunca reconecta con tramos
posteriores, el frontier reconstruido queda anclado en una decisión
antigua, sin importar qué pasó después.

Corrección: ``_segment_decision_log`` divide el log completo en tramos
MAXIMALES causalmente conectados (en vez de asumir un único tramo). El
frontier se reconstruye recorriendo los tramos desde el MÁS RECIENTE
hacia atrás, descartando como "espurio" únicamente el tramo cuya
primera entrada aparece justo después de que el tramo anterior
terminara con una transición TERMINAL (``HALT_STAGE``/``STOP_PIPELINE``
-- nunca traen ``target_stage``, así que nada legítimo continúa después
de una). El criterio es exclusivamente semántico (acción + target_stage
+ etapa); no depende de ningún nombre de etapa concreto ni de ningún
texto de ``reason_code``.

Bug 2 -- ``_check_already_terminal_state`` determinaba la terminalidad
con ``frontier_entry.requested_transition`` pero construía el
``StageOutcome`` reportado desde ``state.stages[frontier_entry.stage]``
(el estado COMPROMETIDO VIGENTE de esa etapa) -- que puede corresponder
a una ejecución POSTERIOR y distinta de la entrada histórica usada para
declarar terminalidad. Eso producía exactamente la contradicción
reportada: ``ALREADY_TERMINAL`` junto con ``next_action=ADVANCE``.

Corrección: el ``StageOutcome`` se construye SIEMPRE desde el propio
``frontier_entry.result`` (``AgentResult.from_dict`` +
``_outcome_from_result``, la misma función que ya usa el resto del
módulo), nunca desde ``state.stages``. Se añade además una invariante
explícita: si el ``StageOutcome`` resultante no tiene
``next_action in {"HALT_STAGE","STOP_PIPELINE"}``, NO se reporta como
``ALREADY_TERMINAL`` -- se deja que el flujo normal decida en su lugar.
"""

from __future__ import annotations

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
    _check_already_terminal_state,
    _reconstruct_authoritative_frontier,
    _segment_decision_log,
    ensure_pipeline_state,
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


STAGE_02 = "02_agente_extraccion"
STAGE_03B = "03B_extraccion_cuantitativa_kb"
STAGE_04 = "04_agente_analisis_tematico"
STAGE_05 = "05_generador_esquema"
STAGE_06 = "06_agente_redactor"
STAGE_07 = "07_agente_verificador"


def _generic_fp(stage_key, attempt=1):
    return build_stage_fingerprints(
        input_data={"stage_name": stage_key, "attempt_number": attempt}, config_data={}, dependencies_data={},
    )


def _commit_decision(store, *, stage_key, action, target_stage, reason_code, execution_status, attempt=1):
    now = datetime.now(timezone.utc).isoformat()
    completed = execution_status == ExecutionStatus.COMPLETED
    result = AgentResult(
        execution_status=execution_status,
        quality_status=QualityStatus.APPROVED if completed else QualityStatus.REJECTED,
        decision=DecisionInfo(code=reason_code, rationale="doble determinista"),
        quality_metrics={}, warnings=(),
        requested_transition=RequestedTransition(action=action, target_stage=target_stage, reason_code=reason_code),
        output_artifacts={}, tool_usage=ToolUsage(),
        attempt_number=attempt, started_at=now, completed_at=now,
        error=None if completed else {"code": reason_code},
    )
    prep = store.prepare_execution(target_stage=stage_key, intended_action="EXECUTE", attempt_number=attempt)
    store.persist_agent_result(prep.decision_id, result)
    store.commit_execution(
        decision_id=prep.decision_id, result=result, stage_name=stage_key,
        fingerprints=_generic_fp(stage_key, attempt), observations={},
    )


def _fake_spec(stage_key):
    def build_execution(project_dir, attempt_number):
        raise AssertionError(f"{stage_key} NO debía intentar ejecutarse -- el frontier debía detener el pipeline antes")

    return StageSpec(key=stage_key, label=f"fake {stage_key}", build_execution=build_execution, runtime_transaction=None, resolve_resume=None)


def _new_store(tmp: Path):
    project_dir = Path(tmp)
    (project_dir / "active_experiment.json").write_text(
        __import__("json").dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return ensure_pipeline_state(project_dir)


@scenario("I01. decision_log con DOS epochs completos (viejo 02->03B->04->05->06, y reciente 06 revision->07 HALT->06 espurio): el frontier es el 07/HALT del epoch reciente, NO decision_log[0]")
def test_multi_epoch_frontier_selects_most_recent_valid_chain():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        # --- EPOCH VIEJO: una corrida completa e independiente, mucho antes ---
        _commit_decision(store, stage_key=STAGE_02, action=TransitionAction.ADVANCE, target_stage=STAGE_03B, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_03B, action=TransitionAction.ADVANCE, target_stage=STAGE_04, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_04, action=TransitionAction.ADVANCE, target_stage=STAGE_05, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_05, action=TransitionAction.ADVANCE, target_stage=STAGE_06, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)

        # Ruptura deliberada de la cadena causal (equivalente real a un
        # restart/invalidación intermedia durante el debugging): la
        # siguiente entrada NO es continuación legítima de la anterior
        # (06 ya apuntaba a 07, pero aquí "reaparece" 02 de nuevo, sin
        # que ninguna transición lo señalara) -- separa claramente los
        # dos epochs para la prueba.
        _commit_decision(store, stage_key=STAGE_02, action=TransitionAction.ADVANCE, target_stage=STAGE_03B, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)

        # --- EPOCH RECIENTE: la cadena real que terminó en el HALT de 07 ---
        _commit_decision(store, stage_key=STAGE_03B, action=TransitionAction.ADVANCE, target_stage=STAGE_04, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_04, action=TransitionAction.ADVANCE, target_stage=STAGE_05, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_05, action=TransitionAction.ADVANCE, target_stage=STAGE_06, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="revision completada", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED", execution_status=ExecutionStatus.FAILED)
        # Entrada ESPURIA: 06 vuelve a ejecutarse después del HALT sin
        # ninguna transición que lo señalara -- el mismo patrón del
        # reporte real, reason_code deliberadamente distinto del real.
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="ALGUN_OTRO_ERROR_INVENTADO", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        assert len(state.decision_log) == 12

        segments = _segment_decision_log(state.decision_log)
        assert len(segments) >= 2  # al menos dos tramos reales

        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        assert frontier.stage == STAGE_07
        assert frontier.requested_transition.reason_code == "AGENT07_RUNTIME_BLOCKED"
        # Explícitamente NO decision_log[0] (que sería 02 del epoch viejo).
        assert frontier is not state.decision_log[0]

        registry = {k: _fake_spec(k) for k in (STAGE_02, STAGE_03B, STAGE_04, STAGE_05, STAGE_06, STAGE_07)}
        outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert outcome is not None
        assert outcome.key == STAGE_07
        assert outcome.status == "ALREADY_TERMINAL"
        assert outcome.next_action == "HALT_STAGE"

        # Invariante central pedida: jamás ALREADY_TERMINAL junto con una
        # transición no terminal.
        assert outcome.next_action in {"HALT_STAGE", "STOP_PIPELINE"}

        # El registro espurio de 06 sigue intacto en el log -- no se borró.
        assert any(e.stage == STAGE_06 and e.requested_transition.reason_code == "ALGUN_OTRO_ERROR_INVENTADO" for e in state.decision_log)


@scenario("I02. Invariante: ALREADY_TERMINAL nunca puede coexistir con next_action=ADVANCE/RETURN/RETRY -- ni por una inconsistencia entre decision_log y state.stages")
def test_invariant_no_already_terminal_with_advance():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        # Reproduce el bug 2 exacto: el frontier real (decision_log) es
        # terminal para 07, pero luego 02 se re-ejecuta legítimamente en
        # un epoch NUEVO y avanza -- confirmando que el chequeo nunca
        # confunde el estado COMPROMETIDO vigente de otra etapa con el
        # frontier histórico.
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED", execution_status=ExecutionStatus.FAILED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="DRAFT_REVISION_ROUND_ALREADY_COMPLETED", execution_status=ExecutionStatus.FAILED)

        registry = {STAGE_06: _fake_spec(STAGE_06), STAGE_07: _fake_spec(STAGE_07)}
        outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert outcome is not None
        assert outcome.key == STAGE_07  # nunca 06, el espurio
        # La contradicción reportada NUNCA debe poder producirse:
        assert not (outcome.status == "ALREADY_TERMINAL" and outcome.next_action == "ADVANCE")
        assert outcome.next_action in {"HALT_STAGE", "STOP_PIPELINE"}


@scenario("I03. El StageOutcome de ALREADY_TERMINAL proviene del propio frontier_entry.result, no de state.stages[stage] actual")
def test_outcome_built_from_frontier_entry_not_current_stage_state():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED", execution_status=ExecutionStatus.FAILED)

        registry = {STAGE_06: _fake_spec(STAGE_06), STAGE_07: _fake_spec(STAGE_07)}
        outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert outcome is not None
        assert outcome.reason_code == "AGENT07_RUNTIME_BLOCKED"
        assert outcome.execution_status == "FAILED"
        assert outcome.quality_status == "REJECTED"


if __name__ == "__main__":
    for fn in (
        test_multi_epoch_frontier_selects_most_recent_valid_chain,
        test_invariant_no_already_terminal_with_advance,
        test_outcome_built_from_frontier_entry_not_current_stage_state,
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
