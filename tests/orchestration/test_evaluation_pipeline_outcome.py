"""``resolve_pipeline_outcome_for_evaluation`` (nuevo,
``src/adapters/evaluation_pipeline_outcome.py``) -- decide si 07 es
evaluable por 08, y con qué metadatos, a partir del ``decision_log``
CAUSALMENTE VÁLIDO de 07 (nunca ``state.stages`` directo).

Reutiliza el harness real ya probado en
``test_evaluation_stagespec_integration.py`` (``_make_kwargs``/
``_seed_state``/``DEFAULT_POLICY`` -- las MISMAS fakes de traducción/
embeddings/BERTScore/judge, sin OpenAI real) -- la diferencia es que
aquí ``build_execution`` inyecta ``_pipeline_outcome_metadata`` real,
calculado contra un ``store`` con un commit REAL de 07 (``COMPLETED``/
``ADVANCE``, ``COMPLETED``/``HALT_STAGE`` científico, o ``FAILED``
técnico) -- exactamente el mecanismo nuevo que decide evaluabilidad,
sin necesitar la capa de wiring dependiente de OpenAI/langchain
(``evaluation_stagespec_wiring.py``, que solo ensambla inputs desde
disco -- no es donde vive la lógica nueva)."""

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
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_evaluation_stagespec_integration as E  # noqa: E402

from src.adapters.evaluation_orchestrator_runtime import EVALUATION_STAGE_NAME, _run_evaluation_stage  # noqa: E402
from src.adapters.evaluation_pipeline_outcome import resolve_pipeline_outcome_for_evaluation  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.pipeline_orchestrator import StageSpec, run_stage  # noqa: E402
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402
from src.state.pipeline_state import CycleState  # noqa: E402
from src.state.state_store import StateStore  # noqa: E402

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


AGENT07 = "07_agente_verificador"
AGENT06 = "06_agente_redactor"


def _generic_fp(stage_key, attempt=1):
    return build_stage_fingerprints(
        input_data={"stage_name": stage_key, "attempt_number": attempt}, config_data={}, dependencies_data={},
    )


def _commit_07(store: StateStore, *, execution_status, action, reason_code, target_stage=None, attempt=1):
    now = datetime.now(timezone.utc).isoformat()
    completed = execution_status == ExecutionStatus.COMPLETED
    result = AgentResult(
        execution_status=execution_status,
        quality_status=QualityStatus.APPROVED if action == TransitionAction.ADVANCE else QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code=reason_code, rationale="commit real de prueba"),
        quality_metrics={}, warnings=(),
        requested_transition=RequestedTransition(action=action, target_stage=target_stage, reason_code=reason_code),
        output_artifacts={}, tool_usage=ToolUsage(),
        attempt_number=attempt, started_at=now, completed_at=now,
        error=None if completed else {"code": reason_code},
    )
    prep = store.prepare_execution(target_stage=AGENT07, intended_action="EXECUTE", attempt_number=attempt)
    store.persist_agent_result(prep.decision_id, result)
    store.commit_execution(
        decision_id=prep.decision_id, result=result, stage_name=AGENT07,
        fingerprints=_generic_fp(AGENT07, attempt), observations={},
    )
    return prep.decision_id


def _seed_with_07(tmp: Path, *, execution_status, action, reason_code, target_stage=None, cycle=None):
    project_dir, store = E._seed_state(tmp)
    decision_id = _commit_07(
        store, execution_status=execution_status, action=action, reason_code=reason_code, target_stage=target_stage,
    )
    if cycle is not None:
        state = store.load()
        store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))
    return project_dir, store, decision_id


def _make_spec_with_outcome(tmp: Path, output_dir: Path, *, store, allow_partial_halt):
    def build_execution(project_dir, attempt_number):
        kwargs = E._make_kwargs(tmp)
        kwargs["_pipeline_outcome_metadata"] = resolve_pipeline_outcome_for_evaluation(
            store=store, allow_partial_halt=allow_partial_halt,
        )
        kwargs["output_dir"] = str(output_dir)
        kwargs["numeric_check_output_dir"] = str(output_dir)
        kwargs["backup_root"] = str(output_dir / ".backups")
        kwargs["_openai_model"] = "gpt-4.1-mini"
        return kwargs

    return StageSpec(
        key=EVALUATION_STAGE_NAME, label="08 · Evaluación (prueba pipeline_outcome)",
        build_execution=build_execution, runtime_transaction=None, resolve_resume=None,
        build_fingerprints=None, custom_run=_run_evaluation_stage,
    )


@scenario("Z01. 07 COMPLETED+APPROVED+ADVANCE->08 -> pipeline_outcome=SUCCESS, evaluable")
def test_success_evaluable():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=EVALUATION_STAGE_NAME,
        )
        outcome_meta = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        assert outcome_meta["pipeline_outcome"] == "SUCCESS"
        assert outcome_meta["verification_approved"] is True
        assert outcome_meta["autonomous_convergence"] is True
        assert outcome_meta["human_review_required"] is False

        output_dir = tmp / "eval_out"
        spec = _make_spec_with_outcome(tmp, output_dir, store=store, allow_partial_halt=False)
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome
        assert outcome.execution_status == "COMPLETED"

        manifest = json.loads((output_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["pipeline_outcome"]["pipeline_outcome"] == "SUCCESS"


@scenario("Z02. 07 COMPLETED+NEEDS_REVISION+HALT_STAGE (WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED) + allow_partial_halt=True -> PARTIAL_HALT evaluable, con rounds_used/max_rounds/reason_code")
def test_partial_halt_scientific_evaluable():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cycle = CycleState(rounds_used=3, max_rounds=3, status="EXHAUSTED", claim_identity_contract_version="LEGACY")
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
            reason_code="WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED", cycle=cycle,
        )
        outcome_meta = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        assert outcome_meta["pipeline_outcome"] == "PARTIAL_HALT"
        assert outcome_meta["verification_approved"] is False
        assert outcome_meta["autonomous_convergence"] is False
        assert outcome_meta["human_review_required"] is True
        assert outcome_meta["rounds_used"] == 3
        assert outcome_meta["max_rounds"] == 3
        assert outcome_meta["agent07_reason_code"] == "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED"

        output_dir = tmp / "eval_out"
        spec = _make_spec_with_outcome(tmp, output_dir, store=store, allow_partial_halt=True)
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome

        manifest = json.loads((output_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["pipeline_outcome"]["pipeline_outcome"] == "PARTIAL_HALT"
        assert manifest["pipeline_outcome"]["rounds_used"] == 3

        # Sin allow_partial_halt explícito, el MISMO estado de 07 NO es evaluable.
        try:
            resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        except ValueError as exc:
            assert "AGENT08_PARTIAL_HALT_NOT_EXPLICITLY_AUTHORIZED" in str(exc)
        else:
            raise AssertionError("debía fallar cerrado sin autorización explícita")


@scenario("Z03. 07 FAILED (fallo técnico real) -> no evaluable, sin importar allow_partial_halt")
def test_technical_halt_not_evaluable():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.FAILED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_RUNTIME_STAGE_FAILURE",
        )
        for allow in (False, True):
            try:
                resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=allow)
            except ValueError as exc:
                assert "AGENT08_UPSTREAM_07_TECHNICAL_FAILURE" in str(exc)
            else:
                raise AssertionError("un fallo técnico real de 07 nunca debe ser evaluable")


@scenario("Z04. Ground Truth: la resolución de pipeline_outcome nunca la toca -- confirmado por búsqueda, el módulo nuevo no importa nada de tools.evaluation.ground_truth")
def test_ground_truth_not_touched_by_outcome_resolution():
    import inspect
    from src.adapters import evaluation_pipeline_outcome as module

    source = inspect.getsource(module)
    assert "ground_truth" not in source.lower()


@scenario("Z05. 07 no se modifica: resolver el pipeline_outcome (varias veces, con y sin fallar) no cambia state.stages['07_agente_verificador'] ni decision_log")
def test_07_not_modified_by_outcome_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=EVALUATION_STAGE_NAME,
        )
        before_stage = store.load().stages[AGENT07]
        before_log = store.load().decision_log

        resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        try:
            resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        except ValueError:
            pass

        after_stage = store.load().stages[AGENT07]
        after_log = store.load().decision_log
        assert before_stage == after_stage
        assert before_log == after_log


@scenario("Z06. 06 no se modifica: idéntica verificación para el stage 06")
def test_06_not_modified_by_outcome_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=EVALUATION_STAGE_NAME,
        )
        before = dict(store.load().stages)
        resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        after = dict(store.load().stages)
        assert before.get(AGENT06) == after.get(AGENT06)  # ninguno de los dos existía ni se creó


@scenario("Z07. No 07C: ninguna ruta de resolve_pipeline_outcome_for_evaluation menciona ni lee agent07c_directory/07C")
def test_no_07c_reference():
    import inspect
    from src.adapters import evaluation_pipeline_outcome as module

    source = inspect.getsource(module)
    assert "07c" not in source.lower()
    assert "07C" not in source


@scenario("Z08. Commit real de 08 vía StateStore: SUCCESS produce una entrada COMPLETED real en decision_log, con pipeline_outcome persistido en el manifest y en quality_metrics")
def test_real_commit_of_08():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=EVALUATION_STAGE_NAME,
        )
        output_dir = tmp / "eval_out"
        spec = _make_spec_with_outcome(tmp, output_dir, store=store, allow_partial_halt=False)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)

        state = store.load()
        assert EVALUATION_STAGE_NAME in state.stages
        assert state.stages[EVALUATION_STAGE_NAME].execution_status == ExecutionStatus.COMPLETED
        eval_entries = [e for e in state.decision_log if e.stage == EVALUATION_STAGE_NAME]
        assert len(eval_entries) == 1
        committed_result = AgentResult.from_dict(eval_entries[0].result)
        assert committed_result.quality_metrics["technical"]["pipeline_outcome"] == "SUCCESS"


@scenario("Z09. Reejecución idempotente: mismo pipeline_outcome + misma política -> SKIPPED_FRESH, no reescribe")
def test_rerun_is_idempotent_via_freshness():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store, decision_id = _seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=EVALUATION_STAGE_NAME,
        )
        output_dir = tmp / "eval_out"
        spec = _make_spec_with_outcome(tmp, output_dir, store=store, allow_partial_halt=False)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        mtime_before = (output_dir / "evaluation_manifest.json").stat().st_mtime

        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome2.status == "SKIPPED_FRESH", outcome2
        mtime_after = (output_dir / "evaluation_manifest.json").stat().st_mtime
        assert mtime_before == mtime_after


if __name__ == "__main__":
    for fn in (
        test_success_evaluable,
        test_partial_halt_scientific_evaluable,
        test_technical_halt_not_evaluable,
        test_ground_truth_not_touched_by_outcome_resolution,
        test_07_not_modified_by_outcome_resolution,
        test_06_not_modified_by_outcome_resolution,
        test_no_07c_reference,
        test_real_commit_of_08,
        test_rerun_is_idempotent_via_freshness,
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
