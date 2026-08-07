"""Prueba integrada del conflicto transaccional de la misma ronda, vía
``run_stage()`` real (no las funciones de persistencia llamadas
directamente, como en T06 de ``test_writer_verifier_cycle_e2e.py`` —
aquí se pasa por ``StateStore.prepare_execution``/
``persist_agent_result``/``commit_execution`` reales, con ``StageSpec``
reales para 07 y 06).

Alcance deliberado: los ``custom_run`` de estos dos ``StageSpec`` invocan
directamente ``create_round_awaiting_revision``/``complete_round_revision``
(las funciones productivas reales que arreglan el conflicto) envueltas en
un ``AgentResult`` mínimo, en vez de reconstruir el ``DraftWritingAgent``/
``VerificationAgent`` completos (ya cubiertos con fixtures reales en
``test_verification_stagespec_integration.py`` y
``test_writer_verifier_cycle_e2e.py`` respectivamente). Lo que se prueba
aquí es específicamente que el CONTRATO TRANSACCIONAL
(``run_stage``/``StateStore``) no reintroduce el ``FileExistsError`` ni
permite una doble finalización cuando dos etapas escriben en la misma
ronda en secuencia.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.pipeline_orchestrator import StageSpec, run_stage
from src.state.fingerprints import build_stage_fingerprints
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore
from src.tools.verification.cycle_round_persistence import (
    complete_round_revision,
    create_round_awaiting_revision,
    read_round_status,
    round_directory,
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


VERIFY = "07_agente_verificador"
DRAFT = "06_agente_redactor"
EXPERIMENT_ID = "exp1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_store(project_dir: Path) -> StateStore:
    experiment_dir = project_dir / EXPERIMENT_ID
    state_path = experiment_dir / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    store = StateStore(state_path)
    store.initialize(PipelineState(identity=PipelineIdentity(EXPERIMENT_ID, "run_round_conflict", now, now, "v1")))
    return store


def _revision_request(round_number):
    return {
        "experiment_id": EXPERIMENT_ID,
        "cycle_id": "cycle_1",
        "round_number": round_number,
        "correctable_issue_ids": ["issue_1"],
    }


def _ok_result(attempt_number, decision_code):
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo(code=decision_code, rationale="prueba integrada"),
        quality_metrics={"technical": {}, "scientific": {}},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE, reason_code=decision_code, requires_human_confirmation=False
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=attempt_number,
        started_at=_now(),
        completed_at=_now(),
    )


def _make_seven_custom_run(request):
    """``custom_run`` mínimo de 07: crea round_01 (AWAITING_REVISION) con
    ``create_round_awaiting_revision`` real, luego comete un AgentResult
    COMPLETED vía StateStore real."""

    def _run(*, store, project_dir, spec, attempt_number=1, observations=None, force_rerun=False):
        state = store.load()
        committed = state.stages.get(spec.key)
        fingerprints = build_stage_fingerprints(
            input_data={"round": request["round_number"]}, config_data={}, dependencies_data={}
        )
        if committed is not None and committed.execution_status == ExecutionStatus.COMPLETED and not force_rerun:
            from src.orchestration.pipeline_orchestrator import _outcome_from_committed_stage

            return _outcome_from_committed_stage(spec, committed, status="SKIPPED_FRESH")

        prepared = store.prepare_execution(
            target_stage=spec.key, intended_action="AGENT07_CORRECTABLE_ISSUES", attempt_number=attempt_number
        )

        create_round_awaiting_revision(
            project_dir=project_dir,
            experiment_id=EXPERIMENT_ID,
            cycle_id=request["cycle_id"],
            round_number=request["round_number"],
            writer_revision_request=request,
            artifacts={
                "input_draft_reference.json": {"source_draft_path": "draft.json"},
                "agent07_result.json": {"decision_code": "AGENT07_CORRECTABLE_ISSUES"},
                "writer_revision_request.json": request,
                "transition.json": {"action": "RETURN", "target_stage": DRAFT},
                "fingerprints.json": {"verification_fingerprint": "fp_verif"},
            },
        )

        result = _ok_result(attempt_number, "AGENT07_CORRECTABLE_ISSUES")
        store.persist_agent_result(prepared.decision_id, result)
        store.commit_execution(
            decision_id=prepared.decision_id, result=result, stage_name=spec.key,
            fingerprints=fingerprints, observations=dict(observations or {}),
        )

        from src.orchestration.pipeline_orchestrator import _outcome_from_result

        state = store.load()
        attempts_used = state.stages[spec.key].attempts_used
        return _outcome_from_result(spec, result, "COMMITTED", attempts_used=attempts_used)

    return _run


def _make_six_custom_run(request):
    """``custom_run`` mínimo de 06: COMPLETA la misma round_01 con
    ``complete_round_revision`` real (nunca la crea), luego comete su
    propio AgentResult COMPLETED, en el MISMO StateStore."""

    def _run(*, store, project_dir, spec, attempt_number=1, observations=None, force_rerun=False):
        state = store.load()
        committed = state.stages.get(spec.key)
        fingerprints = build_stage_fingerprints(
            input_data={"round": request["round_number"]}, config_data={}, dependencies_data={}
        )
        if committed is not None and committed.execution_status == ExecutionStatus.COMPLETED and not force_rerun:
            from src.orchestration.pipeline_orchestrator import _outcome_from_committed_stage

            return _outcome_from_committed_stage(spec, committed, status="SKIPPED_FRESH")

        prepared = store.prepare_execution(
            target_stage=spec.key, intended_action="DRAFT_REVISION_COMPLETED", attempt_number=attempt_number
        )

        complete_round_revision(
            project_dir=project_dir,
            experiment_id=EXPERIMENT_ID,
            cycle_id=request["cycle_id"],
            round_number=request["round_number"],
            writer_revision_request=request,
            artifacts={
                "revised_draft.json": {"sections": []},
                "revision_changelog.json": [{"section_id": "S2", "action": "REVISED"}],
                "revision_resolution_matrix.json": [{"issue_id": "issue_1", "result": "RESOLVED"}],
                "unresolved_issues.json": [],
                "fingerprint.json": {"new_fingerprint": "fp_revised"},
            },
        )

        result = _ok_result(attempt_number, "DRAFT_REVISION_COMPLETED")
        store.persist_agent_result(prepared.decision_id, result)
        store.commit_execution(
            decision_id=prepared.decision_id, result=result, stage_name=spec.key,
            fingerprints=fingerprints, observations=dict(observations or {}),
        )

        from src.orchestration.pipeline_orchestrator import _outcome_from_result

        state = store.load()
        attempts_used = state.stages[spec.key].attempts_used
        return _outcome_from_result(spec, result, "COMMITTED", attempts_used=attempts_used)

    return _run


@scenario("R01. Integración real vía run_stage(): 07 crea round_01, 06 la completa en el MISMO StateStore, sin FileExistsError")
def test_run_stage_seven_then_six_same_round():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        store = _new_store(project_dir)
        request = _revision_request(1)

        seven_spec = StageSpec(
            key=VERIFY, label="07 · integración run_stage", build_execution=None,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_make_seven_custom_run(request),
        )
        six_spec = StageSpec(
            key=DRAFT, label="06 · integración run_stage", build_execution=None,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_make_six_custom_run(request),
        )

        outcome_07 = run_stage(store=store, project_dir=project_dir, spec=seven_spec, attempt_number=1)
        assert outcome_07.status == "COMMITTED", outcome_07

        status_after_07 = read_round_status(project_dir=project_dir, experiment_id=EXPERIMENT_ID, round_number=1)
        assert status_after_07["status"] == "AWAITING_REVISION"

        # 06 completa la MISMA ronda a través de run_stage() -- si hubiera
        # un FileExistsError real, este outcome sería FAILED, no COMMITTED.
        outcome_06 = run_stage(store=store, project_dir=project_dir, spec=six_spec, attempt_number=1)
        assert outcome_06.status == "COMMITTED", outcome_06

        final_status = read_round_status(project_dir=project_dir, experiment_id=EXPERIMENT_ID, round_number=1)
        assert final_status["status"] == "REVISION_COMPLETED"

        # Coexistencia de artefactos de ambas etapas en la misma ronda.
        directory = round_directory(project_dir, EXPERIMENT_ID, 1)
        for name in ("writer_revision_request.json", "transition.json"):
            assert (directory / name).is_file()
        for name in ("revised_draft.json", "revision_changelog.json"):
            assert (directory / name).is_file()

        state = store.load()
        assert state.pending_execution is None
        assert state.stages[VERIFY].execution_status == ExecutionStatus.COMPLETED
        assert state.stages[DRAFT].execution_status == ExecutionStatus.COMPLETED


@scenario("R02. RESUME de 07: segunda llamada a run_stage() sobre 07 ya comprometido -> SKIPPED_FRESH, sin recrear la ronda")
def test_resume_seven_after_commit():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        store = _new_store(project_dir)
        request = _revision_request(1)
        seven_spec = StageSpec(
            key=VERIFY, label="07 · integración run_stage", build_execution=None,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_make_seven_custom_run(request),
        )

        first = run_stage(store=store, project_dir=project_dir, spec=seven_spec, attempt_number=1)
        assert first.status == "COMMITTED"

        second = run_stage(store=store, project_dir=project_dir, spec=seven_spec, attempt_number=1)
        assert second.status == "SKIPPED_FRESH", second

        # RESUME no debe haber intentado recrear round_01 (create_round_awaiting_revision
        # habria lanzado FileExistsError si se hubiera vuelto a invocar).
        status = read_round_status(project_dir=project_dir, experiment_id=EXPERIMENT_ID, round_number=1)
        assert status["status"] == "AWAITING_REVISION"


@scenario("R03. RESUME de 06: segunda llamada a run_stage() sobre 06 ya comprometido -> SKIPPED_FRESH, sin recompletar la ronda")
def test_resume_six_after_commit():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        store = _new_store(project_dir)
        request = _revision_request(1)

        seven_spec = StageSpec(
            key=VERIFY, label="07 · integración run_stage", build_execution=None,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_make_seven_custom_run(request),
        )
        six_spec = StageSpec(
            key=DRAFT, label="06 · integración run_stage", build_execution=None,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_make_six_custom_run(request),
        )

        run_stage(store=store, project_dir=project_dir, spec=seven_spec, attempt_number=1)
        first_06 = run_stage(store=store, project_dir=project_dir, spec=six_spec, attempt_number=1)
        assert first_06.status == "COMMITTED"

        second_06 = run_stage(store=store, project_dir=project_dir, spec=six_spec, attempt_number=1)
        assert second_06.status == "SKIPPED_FRESH", second_06

        # No debe haber un segundo intento de completar la ronda (que
        # habria lanzado DRAFT_REVISION_ROUND_ALREADY_COMPLETED).
        status = read_round_status(project_dir=project_dir, experiment_id=EXPERIMENT_ID, round_number=1)
        assert status["status"] == "REVISION_COMPLETED"


@scenario("R04. Rechazo real de una segunda finalización si se fuerza force_rerun en 06 sobre la misma ronda ya completada")
def test_forced_second_completion_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        store = _new_store(project_dir)
        request = _revision_request(1)

        seven_spec = StageSpec(
            key=VERIFY, label="07", build_execution=None, runtime_transaction=None,
            resolve_resume=None, build_fingerprints=None, custom_run=_make_seven_custom_run(request),
        )
        six_spec = StageSpec(
            key=DRAFT, label="06", build_execution=None, runtime_transaction=None,
            resolve_resume=None, build_fingerprints=None, custom_run=_make_six_custom_run(request),
        )

        run_stage(store=store, project_dir=project_dir, spec=seven_spec, attempt_number=1)
        run_stage(store=store, project_dir=project_dir, spec=six_spec, attempt_number=1)

        # force_rerun=True obliga a _run al custom_run a re-invocar
        # complete_round_revision sobre una ronda ya REVISION_COMPLETED ->
        # RuntimeError real, capturado como FAILED por run_stage (no un
        # commit exitoso silencioso).
        outcome = run_stage(
            store=store, project_dir=project_dir, spec=six_spec, attempt_number=1, force_rerun=True
        )
        assert outcome.status == "FAILED", outcome


if __name__ == "__main__":
    for fn in (
        test_run_stage_seven_then_six_same_round,
        test_resume_seven_after_commit,
        test_resume_six_after_commit,
        test_forced_second_completion_rejected,
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
