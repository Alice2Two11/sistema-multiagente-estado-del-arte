"""Prueba de regresión: un ``HALT_STAGE`` (o ``STOP_PIPELINE``) ya
COMPROMETIDO como la última decisión real (``decision_log``) es
TERMINAL -- un restart sin ``start_stage`` explícito ni
``--force-rerun`` debe reconocerlo y detenerse ahí, sin recorrer
02→03→04→05→06 de nuevo.

Causa raíz: ``run_pipeline`` siempre empezaba en ``STAGE_ORDER[0]`` y
recorría hacia adelante sin preguntarse primero si el pipeline YA estaba
en un estado terminal. Cuando 07 quedó comprometido con
``execution=FAILED``/``HALT_STAGE`` (bloqueo científico real, ver
parche 1), un restart natural (mismo comando, sin flags) volvía a
recorrer las etapas -- y si por CUALQUIER motivo 06 no se reconocía
fresco (el mecanismo exacto no se pudo reproducir con dobles
deterministas -- ver más abajo la nota sobre el parche 2), terminaba
reintentando la revisión sobre ``round_01``, ya ``REVISION_COMPLETED``,
disparando ``DRAFT_REVISION_ROUND_ALREADY_COMPLETED``.

Corrección: ``_check_already_terminal_state`` (llamada al inicio de
``run_pipeline``, antes del bucle) lee la ÚLTIMA entrada de
``decision_log`` -- append-only y cronológico, la fuente de verdad real
sobre "qué fue lo último que realmente se comprometió". Si esa decisión
pidió ``HALT_STAGE``/``STOP_PIPELINE``, el pipeline reporta ese estado
terminal tal cual y NO recorre nada -- ni 06 ni ninguna otra etapa se
tocan. No aplica si el llamador pidió explícitamente ``start_stage`` o
``force_rerun=True`` -- ahí se respeta su intención sin cuestionarla.

Por qué la prueba del parche 2 (D01-D05) no cubrió esta ruta productiva
-------------------------------------------------------------------------
Las pruebas D01-D05 llamaban directamente a
``_resolve_draft_execution_mode`` y confirmaban que YA NO lanza
``DRAFT_REVISION_ROUND_UNEXPECTED_STATUS`` para ``REVISION_COMPLETED``,
y que devuelve el mismo ``AgentInput`` que para ``AWAITING_REVISION``
(mismo fingerprint "en papel"). Eso es necesario pero NO suficiente:
nunca ejercitaron el camino COMPLETO por el que ``run_stage()`` decide
si saltar la etapa (``spec.build_fingerprints(agent_input)`` seguido de
``is_stage_fresh``) en un restart genuino (proceso nuevo, sin memoria de
``attempt_numbers`` previos). Reproduje ese camino completo aquí
(escenario G02, más abajo) con un ``StageSpec`` falso que usa la función
productiva REAL ``build_draft_fingerprints`` -- y, con datos controlados
(dependencias vacías, sin Chroma/RAG real), el mecanismo SÍ produce
``SKIPPED_FRESH`` correctamente, incluso con un 07 comprometido con
``HALT_STAGE`` de por medio. No pude reproducir, sin acceso a Chroma/RAG
real, la condición exacta que hizo que el fingerprint de 06 no
coincidiera en el entorno productivo real -- por eso este parche agrega
una defensa INDEPENDIENTE de esa causa exacta (el chequeo de estado
terminal), que resuelve el síntoma reportado sin depender de diagnosticar
la causa raíz del posible desajuste de fingerprint.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import replace
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
    _resolve_draft_execution_mode,
    ensure_pipeline_state,
    run_stage,
)
from src.runtime.draft_writing_protocol import build_draft_fingerprints  # noqa: E402
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402
from src.state.pipeline_state import CycleState  # noqa: E402
from src.tools.verification.cycle_round_persistence import (  # noqa: E402
    complete_round_revision,
    create_round_awaiting_revision,
)

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


def _previous_draft():
    return {
        "sections": [{"section_id": "S1", "section_title": "M", "draft_text": "x [a.pdf | a1].", "claims": []}],
        "source_draft_fingerprint": "fp_previous_draft",
    }


def _revision_request(round_number=1):
    return {
        "schema_version": "writer_revision_request_v1", "experiment_id": "exp1", "cycle_id": "cyc1",
        "round_number": round_number, "source_draft_path": "draft.json",
        "source_draft_fingerprint": "fp_previous_draft", "verification_fingerprint": "fp_verification",
        "created_at": "2026-01-01T00:00:00Z", "transition_reason": "AGENT07_CORRECTABLE_ISSUES", "summary": "x",
        "issues": [
            {"issue_id": "issue_c2", "claim_id": "c2", "section_id": "S2", "claim_text": "x",
             "problem_type": "AUTO_CORRECTABLE", "verdict": "UNSUPPORTED", "severity": "medium",
             "requested_change": "x", "supporting_evidence": ()},
        ],
    }


def _revised_artifacts():
    return {
        "revised_draft.json": {
            "sections": [{"section_id": "S1", "section_title": "M", "draft_text": "x [a.pdf | a1].", "claims": []}],
            "source_draft_fingerprint": "fp_previous_draft",
        },
        "revision_changelog.json": {"round_number": 1, "changes": []},
        "revision_resolution_matrix.json": {"c2": "RESOLVED"},
    }


class _FakeAgent06:
    def __init__(self, *, target_stage=STAGE_07):
        self.calls = 0
        self._target_stage = target_stage

    def execute(self, agent_input):
        self.calls += 1
        now = datetime.now(timezone.utc).isoformat()
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED, quality_status=QualityStatus.APPROVED_WITH_WARNINGS,
            decision=DecisionInfo(code="OK", rationale="ok"), quality_metrics={}, warnings=(),
            requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, target_stage=self._target_stage, reason_code="OK"),
            output_artifacts={}, tool_usage=ToolUsage(),
            attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
        )


class _FakeAgent07Halt:
    def execute(self, agent_input):
        now = datetime.now(timezone.utc).isoformat()
        return AgentResult(
            execution_status=ExecutionStatus.FAILED, quality_status=QualityStatus.NEEDS_REVISION,
            decision=DecisionInfo(code="AGENT07_RUNTIME_BLOCKED", rationale="bloqueo real"), quality_metrics={}, warnings=(),
            requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_RUNTIME_BLOCKED"),
            output_artifacts={}, tool_usage=ToolUsage(),
            attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
            error={"code": "AGENT07_RUNTIME_BLOCKED"},
        )


def _generic_fp(agent_input):
    return build_stage_fingerprints(
        input_data={"stage_name": agent_input.stage_name, "attempt_number": agent_input.attempt_number},
        config_data=dict(agent_input.policy), dependencies_data={},
    )


def _seed_project_with_completed_revision(tmp: Path):
    """Reproduce exactamente el estado real reportado hasta el punto
    justo antes de comprometer 07: cycle ACTIVE, round_01 creada,
    06 completa su revisión real (complete_round_revision real), queda
    REVISION_COMPLETED."""

    project_dir = Path(tmp)
    experiment_id = "exp1"
    (project_dir / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
    )
    draft_dir = project_dir / experiment_id / "05_outputs" / "05_draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "state_of_art_draft.json").write_text(json.dumps(_previous_draft()), encoding="utf-8")

    store = ensure_pipeline_state(project_dir)
    state = store.load()
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)}))

    rr = _revision_request(1)
    create_round_awaiting_revision(
        project_dir=project_dir, experiment_id=experiment_id, cycle_id=rr["cycle_id"], round_number=1,
        writer_revision_request=rr, artifacts={"writer_revision_request.json": rr},
    )

    agent_06 = _FakeAgent06()

    def build_execution_06(pd, attempt_number):
        overrides = _resolve_draft_execution_mode(project_dir, store)
        agent_input = AgentInput(
            experiment_id=experiment_id, run_id="run1", stage_name=STAGE_06, attempt_number=attempt_number,
            mode=ExecutionMode.FULL_RUN, agent_context=AgentContext(allowed_tools=("llm",), output_directory=str(project_dir / "out")),
            dependencies={}, policy=overrides or {},
        )
        return agent_06, agent_input

    def runtime_transaction_06(*, store, build_execution, attempt_number=1, observations=None):
        prep = store.prepare_execution(target_stage=STAGE_06, intended_action="EXECUTE_DRAFT_WRITING", attempt_number=attempt_number)
        agent, agent_input = build_execution()
        result = agent.execute(agent_input)
        fp = build_draft_fingerprints(agent_input)
        store.persist_agent_result(prep.decision_id, result)
        store.commit_execution(decision_id=prep.decision_id, result=result, stage_name=STAGE_06, fingerprints=fp, observations=dict(observations or {}))

        class _R:
            pass

        out = _R()
        out.agent_result = result
        return out

    def resolve_resume_06(*, store, agent_input, observations=None):
        return store.resolve_resume(stage_name=agent_input.stage_name, fingerprints=build_draft_fingerprints(agent_input), observations=dict(observations or {}))

    spec_06 = StageSpec(
        key=STAGE_06, label="fake 06", build_execution=build_execution_06,
        runtime_transaction=runtime_transaction_06, resolve_resume=resolve_resume_06, build_fingerprints=build_draft_fingerprints,
    )

    run_stage(store=store, project_dir=project_dir, spec=spec_06, attempt_number=1)
    complete_round_revision(
        project_dir=project_dir, experiment_id=experiment_id, cycle_id=rr["cycle_id"], round_number=1,
        writer_revision_request=rr, artifacts=_revised_artifacts(),
    )
    return project_dir, store, spec_06, agent_06


def _commit_07_halt(store, project_dir):
    agent_07 = _FakeAgent07Halt()
    ai07 = AgentInput(
        experiment_id="exp1", run_id="run1", stage_name=STAGE_07, attempt_number=1, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("llm",), output_directory=str(project_dir / "out07")),
        dependencies={}, policy={},
    )
    prep07 = store.prepare_execution(target_stage=STAGE_07, intended_action="EXECUTE", attempt_number=1)
    result07 = agent_07.execute(ai07)
    store.persist_agent_result(prep07.decision_id, result07)
    store.commit_execution(decision_id=prep07.decision_id, result=result07, stage_name=STAGE_07, fingerprints=_generic_fp(ai07), observations={})


@scenario("G01 (caso A). pending=None + último comprometido 07/HALT_STAGE + round_01 REVISION_COMPLETED + restart -> 06 NO se ejecuta")
def test_case_a_terminal_halt_stops_before_06():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, spec_06, agent_06 = _seed_project_with_completed_revision(tmp)
        _commit_07_halt(store, project_dir)

        assert store.load().pending_execution is None  # tal como describe el caso A

        def build_execution_07(pd, attempt_number):
            return _FakeAgent07Halt(), AgentInput(
                experiment_id="exp1", run_id="run1", stage_name=STAGE_07, attempt_number=attempt_number,
                mode=ExecutionMode.FULL_RUN, agent_context=AgentContext(allowed_tools=("llm",), output_directory=str(project_dir / "out07")),
                dependencies={}, policy={},
            )
        spec_07 = StageSpec(key=STAGE_07, label="fake 07", build_execution=build_execution_07, runtime_transaction=None, resolve_resume=None)
        registry = {STAGE_06: spec_06, STAGE_07: spec_07}
        agent_06.calls = 0  # medir solo lo que pase DESPUÉS del setup (la revisión ya se ejecutó una vez, legítimamente, al construir el escenario)
        terminal_outcome = _check_already_terminal_state(
            store=store, registry=registry, start_stage=None, force_rerun=False
        )
        assert terminal_outcome is not None
        assert terminal_outcome.key == STAGE_07
        assert terminal_outcome.next_action == "HALT_STAGE"
        assert terminal_outcome.status == "ALREADY_TERMINAL"
        # La aserción central: como el llamador (run_pipeline) devuelve
        # apenas ve esto, 06 nunca se toca.
        assert agent_06.calls == 0


@scenario("G02 (caso B). pending=None + 06 revisión válida (ADVANCE->07) + round_01 REVISION_COMPLETED + sin 07 todavía + restart -> 06 SKIPPED_FRESH, continúa a 07")
def test_case_b_no_terminal_halt_continues_to_07():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, spec_06, agent_06 = _seed_project_with_completed_revision(tmp)
        # NO se compromete 07 -- exactamente el caso B: no existe todavía un 07 posterior.

        registry = {STAGE_06: spec_06}
        terminal_outcome = _check_already_terminal_state(
            store=store, registry=registry, start_stage=None, force_rerun=False
        )
        # El último comprometido fue 06 con ADVANCE -- no es terminal, no bloquea nada.
        assert terminal_outcome is None

        # Y el camino COMPLETO de run_stage() (el hueco real que el parche 2 no probaba)
        # SÍ reconoce 06 como fresco -- no se reinvoca al agente, se avanza a 07 vía su
        # propia transición comprometida (ADVANCE).
        agent_06.calls = 0
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec_06, attempt_number=1)
        assert outcome.status == "SKIPPED_FRESH"
        assert agent_06.calls == 0
        assert outcome.next_action == "ADVANCE"
        assert outcome.target_stage == STAGE_07


@scenario("G03 (caso C). round_01 REVISION_COMPLETED pero el borrador previo en disco cambió de verdad -> DRAFT_REVISION_FINGERPRINT_MISMATCH explícito, no se reutiliza silenciosamente")
def test_case_c_real_input_change_fails_closed_explicitly():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, spec_06, agent_06 = _seed_project_with_completed_revision(tmp)

        # Simula un cambio REAL de inputs: el borrador previo en disco ya
        # no coincide con el source_draft_fingerprint que el
        # writer_revision_request original declaraba -- exactamente lo
        # que _resolve_draft_execution_mode ya valida.
        experiment_id = "exp1"
        draft_path = project_dir / experiment_id / "05_outputs" / "05_draft" / "state_of_art_draft.json"
        changed_draft = json.loads(draft_path.read_text())
        changed_draft["source_draft_fingerprint"] = "fp_DISTINTO_de_verdad_cambio"
        draft_path.write_text(json.dumps(changed_draft), encoding="utf-8")

        try:
            _resolve_draft_execution_mode(project_dir, store)
        except RuntimeError as exc:
            assert "DRAFT_REVISION_FINGERPRINT_MISMATCH" in str(exc)
        else:
            raise AssertionError(
                "un cambio real de source_draft_fingerprint debía fallar explícitamente, "
                "no reutilizar en silencio una ronda incompatible"
            )


if __name__ == "__main__":
    for fn in (
        test_case_a_terminal_halt_stops_before_06,
        test_case_b_no_terminal_halt_continues_to_07,
        test_case_c_real_input_change_fails_closed_explicitly,
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
