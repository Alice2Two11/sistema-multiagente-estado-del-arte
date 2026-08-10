"""Prueba de regresión: el frontier autoritativo para reconocer un
estado TERMINAL en un restart debe reconstruirse siguiendo la cadena
CAUSAL de transiciones (``target_stage`` real de cada decisión), no
tomar ciegamente ``decision_log[-1]``.

Caso real que reproduce exactamente el reporte: el bug de control de
flujo que el parche 4 corrigió permitía que, tras comprometer 07 con
``HALT_STAGE`` (sin ``target_stage`` -- no hay continuación legítima),
el bucle igual reentrara a 06 por el recorrido normal, produciendo una
entrada ESPURIA en ``decision_log`` (06/FAILED/HALT_STAGE por
``DRAFT_REVISION_ROUND_ALREADY_COMPLETED``) que, cronológicamente, es
la más reciente -- pero que nunca debió existir, porque nada en la
transición de 07 la señalaba como continuación válida.

``_reconstruct_authoritative_frontier`` camina el log hacia adelante
desde el principio, y solo avanza el frontier cuando la siguiente
entrada coincide con el ``target_stage`` (o la misma etapa, para RETRY)
que la transición anterior señalaba. Una entrada que no está
causalmente conectada (como el 06 espurio, que aparece justo después de
un HALT_STAGE sin target_stage) no mueve el frontier -- queda en el log
como evidencia histórica, sin gobernar el restart.

El criterio es puramente semántico (acción + target_stage de cada
transición real) -- no depende de ningún texto de ``reason_code`` ni de
ninguna etapa específica; se prueba también con un ``reason_code``
inventado, distinto de ``DRAFT_REVISION_ROUND_ALREADY_COMPLETED``, para
confirmar que no hay ningún filtro de texto oculto.
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


STAGE_05 = "05_generador_esquema"
STAGE_06 = "06_agente_redactor"
STAGE_07 = "07_agente_verificador"


def _generic_fp(stage_key, attempt=1):
    return build_stage_fingerprints(
        input_data={"stage_name": stage_key, "attempt_number": attempt}, config_data={}, dependencies_data={},
    )


def _commit_decision(store, *, stage_key, action, target_stage, reason_code, execution_status, attempt=1):
    """Registra una entrada real de decision_log via
    prepare/persist/commit -- nunca se escribe pipeline_state.json a
    mano ni se inventa una forma de DecisionLogEntry aparte del
    protocolo real."""

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


@scenario("H01. Reconstrucción causal: 07/HALT_STAGE seguido de un 06 espurio (no conectado) -> el frontier autoritativo sigue siendo 07, no el 06 posterior")
def test_spurious_post_halt_entry_does_not_move_frontier():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        (project_dir / "active_experiment.json").write_text(
            __import__("json").dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
        )
        store = ensure_pipeline_state(project_dir)

        # Cadena causal real hasta el HALT de 07 (06 revision -> ADVANCE -> 07 -> HALT_STAGE).
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED", execution_status=ExecutionStatus.FAILED)

        # Entrada ESPURIA: 06 vuelve a ejecutarse (por el bug de control de
        # flujo ya corregido) DESPUÉS del HALT de 07, que no señalaba
        # ninguna continuación. Reason_code deliberadamente DISTINTO del
        # real reportado, para confirmar que no hay ningún filtro de texto.
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="ALGUN_OTRO_ERROR_INVENTADO", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        assert len(state.decision_log) == 3

        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        assert frontier.stage == STAGE_07
        assert frontier.requested_transition.reason_code == "AGENT07_RUNTIME_BLOCKED"

        registry = {STAGE_06: _fake_spec(STAGE_06), STAGE_07: _fake_spec(STAGE_07)}
        terminal_outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert terminal_outcome is not None
        assert terminal_outcome.key == STAGE_07  # NO el 06 espurio
        assert terminal_outcome.next_action == "HALT_STAGE"

        # El registro de 06 sigue en el log, intacto, como evidencia -- no se borró ni se alteró.
        assert any(e.stage == STAGE_06 and e.requested_transition.reason_code == "ALGUN_OTRO_ERROR_INVENTADO" for e in state.decision_log)


@scenario("H02. Un fallo 06 LEGÍTIMO (sin HALT previo que lo invalide causalmente) sí gobierna el frontier como terminal")
def test_legitimate_06_failure_without_prior_halt_is_authoritative():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        (project_dir / "active_experiment.json").write_text(
            __import__("json").dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
        )
        store = ensure_pipeline_state(project_dir)

        # Cadena causal real: 05 -> ADVANCE -> 06 -> 06 falla de verdad, sin ningún HALT anterior en el medio.
        _commit_decision(store, stage_key=STAGE_05, action=TransitionAction.ADVANCE, target_stage=STAGE_06, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="DRAFT_LEGITIMATE_FAILURE", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        assert frontier.stage == STAGE_06
        assert frontier.requested_transition.reason_code == "DRAFT_LEGITIMATE_FAILURE"

        registry = {STAGE_05: _fake_spec(STAGE_05), STAGE_06: _fake_spec(STAGE_06)}
        terminal_outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert terminal_outcome is not None
        assert terminal_outcome.key == STAGE_06
        assert terminal_outcome.next_action == "HALT_STAGE"


@scenario("H03. Cadena causal normal sin HALT (todo ADVANCE conectado): el frontier es la última entrada real, igual que antes")
def test_normal_advance_chain_frontier_is_last_entry():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        (project_dir / "active_experiment.json").write_text(
            __import__("json").dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
        )
        store = ensure_pipeline_state(project_dir)

        _commit_decision(store, stage_key=STAGE_05, action=TransitionAction.ADVANCE, target_stage=STAGE_06, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="OK", execution_status=ExecutionStatus.COMPLETED)

        state = store.load()
        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        # Contrato correcto (ver parche 8): sin ningún tramo terminal en
        # el log, no hay frontier autoritativo que reportar -- None, no
        # "la última entrada aunque no sea terminal".
        assert frontier is None

        registry = {STAGE_05: _fake_spec(STAGE_05), STAGE_06: _fake_spec(STAGE_06)}
        terminal_outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert terminal_outcome is None  # ADVANCE no es terminal -- el restart debe seguir con normalidad


if __name__ == "__main__":
    for fn in (
        test_spurious_post_halt_entry_does_not_move_frontier,
        test_legitimate_06_failure_without_prior_halt_is_authoritative,
        test_normal_advance_chain_frontier_is_last_entry,
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
