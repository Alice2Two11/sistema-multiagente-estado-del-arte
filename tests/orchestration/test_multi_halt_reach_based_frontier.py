"""Parche 8: corrige la cascada ciega de ``_reconstruct_authoritative_
frontier`` cuando el ``decision_log`` real contiene MÁS de un
``HALT_STAGE`` de 06 alrededor del verdadero terminal de 07.

Caso real confirmado con el ``decision_log`` productivo del reporte
(reproducido aquí con la misma estructura exacta, sin fixtures
inventados desde cero -- los mismos patrones de acción/etapa/
reason_code que el diagnóstico real mostró, en el mismo orden):

    tramo 20: 07/RETURN->06 ... 06/ADVANCE->07   (avanza hasta 07)
    tramo 21: 06/HALT_STAGE (RUNTIME_DEPENDENCY_FAILED) -- un fallo
              técnico real de 06, no relacionado con el bloqueo
              científico de 07
    tramo 22: 06/ADVANCE->07 ... 07/HALT_STAGE (AGENT07_BLOCKED) --
              06 se reintentó, esta vez avanzó de verdad hasta 07, y
              ESE es el terminal real
    tramo 23: 06/HALT_STAGE (RUNTIME_DEPENDENCY_FAILED) -- espurio,
              posterior al terminal real de 22
    tramo 24: 06/HALT_STAGE (RUNTIME_DEPENDENCY_FAILED) -- espurio
              también

Causa raíz del bug corregido: la versión anterior caminaba hacia atrás
descartando tramos EN CASCADA mientras el tramo INMEDIATAMENTE anterior
hubiera terminado en HALT_STAGE/STOP_PIPELINE -- sin verificar si el
tramo que se estaba descartando representaba, en sí mismo, un avance
real más allá de lo que el tramo anterior alcanzó. Eso hacía que el
tramo 22 (que SÍ llega hasta 07, más lejos que el tramo 21) se
descartara igual que 23/24 (que nunca pasan de 06), solo porque su
predecesor inmediato (21) también había terminado en HALT.

Corrección: se compara el "alcance" de cada tramo terminal -- la
posición de la etapa de su última entrada en ``CANONICAL_STAGE_ORDER``,
el orden canónico REAL del pipeline (ya usado en otras partes del mismo
módulo para razonar sobre progreso, ej. en el manejo de ``RETURN``) --
y se prefiere, entre TODOS los tramos terminales del log (sin importar
adyacencia), el que llegó ESTRICTAMENTE más lejos. El criterio sigue
siendo estructural: no se inspecciona ningún ``reason_code`` ni se
verifica el nombre de ninguna etapa por texto -- se usa exclusivamente
la posición en el orden canónico, aplicable a cualquier etapa.
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


@scenario("J01. Reproducción exacta de los tramos 20->21->22->23->24 del decision_log real: el frontier es el 07/HALT del tramo 22, no el 06/HALT de 21, 23 o 24")
def test_reproduce_real_segments_20_to_24():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        # tramo 20: 07/RETURN->06 ... 06/ADVANCE->07
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.RETURN, target_stage=STAGE_06, reason_code="AGENT07_CORRECTABLE_ISSUES", execution_status=ExecutionStatus.FAILED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED)

        # tramo 21: 06/HALT_STAGE (fallo técnico real de 06, no de 07)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)

        # tramo 22: 06/ADVANCE->07 ... 07/HALT_STAGE -- EL TERMINAL REAL
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED)

        # tramos 23 y 24: espurios, posteriores al terminal real de 22
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        assert len(state.decision_log) == 7

        segments = _segment_decision_log(state.decision_log)
        # Al menos 5 tramos reales: [20],[21],[22],[23],[24] (20 podría
        # fusionarse con lo anterior si hubiera más historia, pero aquí
        # es el inicio del log).
        assert len(segments) >= 5

        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        assert frontier.stage == STAGE_07
        assert frontier.requested_transition.reason_code == "AGENT07_BLOCKED"
        assert frontier.requested_transition.action == TransitionAction.HALT_STAGE

        registry = {STAGE_06: _fake_spec(STAGE_06), STAGE_07: _fake_spec(STAGE_07)}
        outcome = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert outcome is not None
        assert outcome.key == STAGE_07
        assert outcome.status == "ALREADY_TERMINAL"
        assert outcome.next_action == "HALT_STAGE"
        assert outcome.reason_code == "AGENT07_BLOCKED"

        # Los tramos espurios de 06 (21, 23, 24) siguen intactos en el
        # log -- nunca se borraron ni se modificaron.
        dependency_failures = [e for e in state.decision_log if e.stage == STAGE_06 and e.requested_transition.reason_code == "RUNTIME_DEPENDENCY_FAILED"]
        assert len(dependency_failures) == 3


@scenario("J02. Si el ÚLTIMO tramo (cronológicamente) es el que llega más lejos, se prefiere ese -- el criterio es 'alcance', no 'más reciente' ni 'más antiguo' a ciegas")
def test_reach_based_not_recency_based():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        # 06 falla primero (alcance menor)
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)
        # luego 06 avanza y 07 termina más lejos, cronológicamente después
        _commit_decision(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        frontier = _reconstruct_authoritative_frontier(state.decision_log)
        assert frontier.stage == STAGE_07  # el más reciente Y el que más lejos llega -- ambos coinciden aquí


if __name__ == "__main__":
    for fn in (
        test_reproduce_real_segments_20_to_24,
        test_reach_based_not_recency_based,
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
