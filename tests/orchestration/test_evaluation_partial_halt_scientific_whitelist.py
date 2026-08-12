"""Regresión real Exp04: 07 ``COMPLETED``/``NEEDS_REVISION``/
``HALT_STAGE``/``AGENT07_NON_CORRECTABLE_ISSUE`` (claims terminales
``DEFER_TO_MANUAL_REVIEW``/``NO_CORRECTION`` ya correctamente
trazados, sin fallo técnico) era rechazado por 08 con
``AGENT08_UPSTREAM_07_NOT_EVALUABLE:HALT_STAGE:AGENT07_NON_
CORRECTABLE_ISSUE``.

Causa raíz confirmada: NUNCA hubo pérdida ni desconexión del wiring
(``evaluation_stagespec_wiring.py``/``evaluation_orchestrator_
runtime.py`` seguían llamando a ``resolve_pipeline_outcome_for_
evaluation`` correctamente) -- ``SCIENTIFIC_EXHAUSTION_HALT_REASON_
CODES`` (``evaluation_pipeline_outcome.py``) simplemente nunca incluyó
``AGENT07_NON_CORRECTABLE_ISSUE`` ni ``AGENT07_CORRECTION_EVIDENCE_
INSUFFICIENT`` -- las otras dos ramas de ``HALT_STAGE`` que ``classify_
verification_transition`` alcanza DESPUÉS de procesar realmente cada
claim (la tercera, agotamiento de rondas, ya estaba incluida)."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_evaluation_pipeline_outcome as E  # noqa: E402
import test_evaluation_stagespec_integration as ES  # noqa: E402

from src.adapters.evaluation_pipeline_outcome import (  # noqa: E402
    SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES,
    resolve_pipeline_outcome_for_evaluation,
)
from src.contracts.agent_result import ExecutionStatus, TransitionAction  # noqa: E402

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


@scenario("DD01. 07 APPROVED (ADVANCE->08) -> pipeline_outcome=SUCCESS, approved_for_publication=True, usable_for_evaluation=True")
def test_approved_advance_yields_success():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.ADVANCE,
            reason_code="AGENT07_ALL_CLAIMS_APPROVED", target_stage=E.EVALUATION_STAGE_NAME,
        )
        outcome = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=False)
        assert outcome["pipeline_outcome"] == "SUCCESS"
        assert outcome["approved_for_publication"] is True
        assert outcome["usable_for_evaluation"] is True
        assert outcome["agent07_halt_reason"] is None


@scenario("DD02. Caso real Exp04 exacto: 07 COMPLETED+HALT_STAGE+AGENT07_NON_CORRECTABLE_ISSUE -> pipeline_outcome=PARTIAL_HALT, evaluable con allow_partial_halt=True")
def test_non_correctable_issue_yields_partial_halt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_NON_CORRECTABLE_ISSUE",
        )
        outcome = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        assert outcome["pipeline_outcome"] == "PARTIAL_HALT"
        assert outcome["agent07_reason_code"] == "AGENT07_NON_CORRECTABLE_ISSUE"
        # Requisito 6, explícito:
        assert outcome["approved_for_publication"] is False
        assert outcome["usable_for_evaluation"] is True
        assert outcome["agent07_halt_reason"] == "AGENT07_NON_CORRECTABLE_ISSUE"
        assert outcome["human_review_required"] is True
        assert outcome["verification_approved"] is False


@scenario("DD03. AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT (la otra rama científica tras procesar claims) también es PARTIAL_HALT evaluable")
def test_correction_evidence_insufficient_yields_partial_halt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT",
        )
        outcome = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        assert outcome["pipeline_outcome"] == "PARTIAL_HALT"
        assert outcome["approved_for_publication"] is False
        assert outcome["usable_for_evaluation"] is True


@scenario("DD04. 07 FAILED (fallo técnico real) -> bloqueado, incluso con allow_partial_halt=True")
def test_failed_execution_status_blocks_regardless_of_reason():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.FAILED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_NON_CORRECTABLE_ISSUE",  # aunque el reason_code "parezca" científico
        )
        try:
            resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        except ValueError as exc:
            assert "AGENT08_UPSTREAM_07_TECHNICAL_FAILURE" in str(exc)
        else:
            raise AssertionError("execution_status=FAILED debe bloquear siempre, sin importar el reason_code")


@scenario("DD05. 07 incompleto (sin commit en decision_log) -> bloqueado")
def test_no_commit_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = ES._seed_state(tmp)  # sin ningún commit de 07
        try:
            resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        except ValueError as exc:
            assert "AGENT08_UPSTREAM_07_NOT_COMMITTED" in str(exc)
        else:
            raise AssertionError("sin commit de 07, debe bloquear siempre")


@scenario("DD06. HALT_STAGE estructural NO reconocido como científico (ej. AGENT07_NO_CLAIMS) sigue bloqueado, incluso con allow_partial_halt=True")
def test_structural_halt_never_becomes_partial_halt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_NO_CLAIMS",
        )
        try:
            resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        except ValueError as exc:
            assert "AGENT08_UPSTREAM_07_NOT_EVALUABLE" in str(exc)
        else:
            raise AssertionError("un HALT_STAGE estructural nunca debe convertirse en PARTIAL_HALT")


@scenario("DD07. PARTIAL_HALT NUNCA marca approved_for_publication=true -- para las tres causas científicas reconocidas")
def test_partial_halt_never_approved_for_publication():
    for reason_code in SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, store, _ = E._seed_with_07(
                tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
                reason_code=reason_code,
            )
            outcome = resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
            assert outcome["pipeline_outcome"] == "PARTIAL_HALT", reason_code
            assert outcome["approved_for_publication"] is False, reason_code


@scenario("DD08. 07 no se modifica: decision_log/state.stages idénticos antes y después, con el nuevo whitelist ejercitado")
def test_07_not_modified_with_expanded_whitelist():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, store, _ = E._seed_with_07(
            tmp, execution_status=ExecutionStatus.COMPLETED, action=TransitionAction.HALT_STAGE,
            reason_code="AGENT07_NON_CORRECTABLE_ISSUE",
        )
        before_log = store.load().decision_log
        before_stage = store.load().stages["07_agente_verificador"]
        resolve_pipeline_outcome_for_evaluation(store=store, allow_partial_halt=True)
        assert store.load().decision_log == before_log
        assert store.load().stages["07_agente_verificador"] == before_stage


if __name__ == "__main__":
    for fn in (
        test_approved_advance_yields_success,
        test_non_correctable_issue_yields_partial_halt,
        test_correction_evidence_insufficient_yields_partial_halt,
        test_failed_execution_status_blocks_regardless_of_reason,
        test_no_commit_blocks,
        test_structural_halt_never_becomes_partial_halt,
        test_partial_halt_never_approved_for_publication,
        test_07_not_modified_with_expanded_whitelist,
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
