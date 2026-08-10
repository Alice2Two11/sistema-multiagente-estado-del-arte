"""Prueba de regresión: una ronda ``REVISION_COMPLETED`` (06 ya la
completó en una ejecución previa real) NUNCA debe volver a intentarse
"tomar" -- ``_resolve_draft_execution_mode`` no debe lanzar
``DRAFT_REVISION_ROUND_UNEXPECTED_STATUS`` para este caso.

Causa raíz: ``run_stage()`` (``src/orchestration/pipeline_orchestrator.py``)
llama a ``build_execution()`` -- que para 06 termina invocando
``_resolve_draft_execution_mode`` -- de forma INCONDICIONAL, incluso
cuando la etapa ya está ``COMPLETED``, solo para poder calcular el
fingerprint actual y compararlo contra el comprometido
(``spec.build_fingerprints(agent_input)`` seguido de ``is_stage_fresh``).
Antes de esta corrección, ``_resolve_draft_execution_mode`` solo aceptaba
``status["status"] == "AWAITING_REVISION"`` para la última ronda
persistida -- cualquier otro valor (incluido ``REVISION_COMPLETED``, el
estado correcto y esperado DESPUÉS de que 06 complete su revisión)
lanzaba ``RuntimeError`` antes de siquiera llegar a comparar
fingerprints. Esto rompía el simple RESUME del pipeline (sin
``--force-rerun``) apenas 06 terminaba su revisión, porque CUALQUIER
intento posterior de ``run_stage()`` sobre 06 (incluso uno que solo
necesitaba confirmar que sigue fresca) volvía a ejecutar
``_resolve_draft_execution_mode`` y crasheaba.

La corrección amplía el conjunto aceptado a
``{"AWAITING_REVISION", "REVISION_COMPLETED"}`` y devuelve el MISMO
``AgentInput`` de revisión en ambos casos (los archivos persistidos --
``writer_revision_request.json`` y el borrador previo -- son idénticos;
solo cambia el campo de estado de la ronda, que no es parte de este
``AgentInput``) -- así el fingerprint recalculado coincide con el ya
comprometido y ``is_stage_fresh`` reconoce la etapa como
``SKIPPED_FRESH`` sin volver a invocar al agente. La ronda en sí NUNCA
se toca aquí -- ``complete_round_revision`` sigue rechazando
explícitamente un segundo intento real de completarla, sin cambios.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

from src.orchestration.pipeline_orchestrator import _resolve_draft_execution_mode  # noqa: E402
from src.state.pipeline_state import CycleState, PipelineIdentity, PipelineState  # noqa: E402
from src.state.state_store import StateStore  # noqa: E402
from src.tools.verification.cycle_round_persistence import (  # noqa: E402
    complete_round_revision,
    create_round_awaiting_revision,
    read_round_status,
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


def _previous_draft():
    return {
        "sections": [
            {
                "section_id": "S1", "section_title": "Methods",
                "draft_text": "Original approved statement [a.pdf | a1].",
                "claims": [{"claim": "Original approved statement", "supporting_citations": ["[a.pdf | a1]"]}],
            },
            {
                "section_id": "S2", "section_title": "Results",
                "draft_text": "Wrong statement about error value [b.pdf | b1].",
                "claims": [{"claim": "Wrong statement about error value", "supporting_citations": ["[b.pdf | b1]"]}],
            },
        ],
        "source_draft_fingerprint": "fp_previous_draft",
    }


def _revision_request(round_number=1):
    return {
        "schema_version": "writer_revision_request_v1", "experiment_id": "exp1", "cycle_id": "cyc1",
        "round_number": round_number, "source_draft_path": "draft.json",
        "source_draft_fingerprint": "fp_previous_draft", "verification_fingerprint": "fp_verification",
        "created_at": "2026-01-01T00:00:00Z", "transition_reason": "AGENT07_CORRECTABLE_ISSUES",
        "summary": "1 observacion corregible.",
        "issues": [
            {
                "issue_id": "issue_c2", "claim_id": "c2", "section_id": "S2",
                "claim_text": "Wrong statement about error value", "problem_type": "AUTO_CORRECTABLE",
                "verdict": "UNSUPPORTED", "severity": "medium",
                "requested_change": "Eliminar o reescribir dentro del alcance soportado.",
                "supporting_evidence": (),
            },
        ],
    }


def _revised_artifacts():
    revised_draft = {
        "sections": [
            {
                "section_id": "S1", "section_title": "Methods",
                "draft_text": "Original approved statement [a.pdf | a1].",
                "claims": [{"claim": "Original approved statement", "supporting_citations": ["[a.pdf | a1]"]}],
            },
            {
                "section_id": "S2", "section_title": "Results",
                "draft_text": "Revised statement within supported scope [b.pdf | b1].",
                "claims": [{"claim": "Revised statement within supported scope", "supporting_citations": ["[b.pdf | b1]"]}],
            },
        ],
        "source_draft_fingerprint": "fp_previous_draft",
    }
    return {
        "revised_draft.json": revised_draft,
        "revision_changelog.json": {"round_number": 1, "changes": [{"claim_id": "c2", "action": "REWRITE_TO_SUPPORTED_SCOPE"}]},
        "revision_resolution_matrix.json": {"c2": "RESOLVED"},
    }


def _new_store(tmp):
    from datetime import datetime, timezone

    store = StateStore(Path(tmp) / "pipeline_state.json")
    now = datetime.now(timezone.utc).isoformat()
    store.initialize(
        PipelineState(
            identity=PipelineIdentity(
                experiment_id="exp1", run_id="run1", created_at=now, updated_at=now, schema_version="1.0"
            )
        )
    )
    return store


def _seed_active_cycle_with_completed_round(tmp: Path) -> tuple[Path, StateStore, str]:
    project_dir = Path(tmp)
    experiment_id = "exp1"
    (project_dir / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
    )
    draft_dir = project_dir / experiment_id / "05_outputs" / "05_draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "state_of_art_draft.json").write_text(json.dumps(_previous_draft()), encoding="utf-8")

    store = _new_store(tmp)
    state = store.load()
    cycle = CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))

    revision_request = _revision_request(1)
    create_round_awaiting_revision(
        project_dir=project_dir, experiment_id=experiment_id, cycle_id=revision_request["cycle_id"],
        round_number=1, writer_revision_request=revision_request,
        artifacts={"writer_revision_request.json": revision_request},
    )
    # 06 YA completó la ronda en una ejecución previa real -- reproduce
    # exactamente eso, no un estado inventado a mano.
    complete_round_revision(
        project_dir=project_dir, experiment_id=experiment_id, cycle_id=revision_request["cycle_id"],
        round_number=1, writer_revision_request=revision_request, artifacts=_revised_artifacts(),
    )
    return project_dir, store, experiment_id


@scenario("D01. round_01 REVISION_COMPLETED real: _resolve_draft_execution_mode NO lanza DRAFT_REVISION_ROUND_UNEXPECTED_STATUS")
def test_revision_completed_round_does_not_raise():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, _ = _seed_active_cycle_with_completed_round(tmp)
        overrides = _resolve_draft_execution_mode(project_dir, store)  # no debe lanzar
        assert overrides is not None
        assert overrides["mode"] == "REVISION"
        assert overrides["round_number"] == 1


@scenario("D02. El AgentInput devuelto para una ronda REVISION_COMPLETED es IDÉNTICO al de AWAITING_REVISION (mismo fingerprint)")
def test_revision_completed_returns_same_shape_as_awaiting():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        # Escenario A: ronda recién creada, AWAITING_REVISION.
        project_dir_a = Path(tmp1)
        experiment_id = "exp1"
        (project_dir_a / "active_experiment.json").write_text(
            json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
        )
        draft_dir_a = project_dir_a / experiment_id / "05_outputs" / "05_draft"
        draft_dir_a.mkdir(parents=True)
        (draft_dir_a / "state_of_art_draft.json").write_text(json.dumps(_previous_draft()), encoding="utf-8")
        store_a = _new_store(tmp1)
        state_a = store_a.load()
        store_a.save(replace(state_a, cycles={**state_a.cycles, "writer_verifier": CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)}))
        revision_request = _revision_request(1)
        create_round_awaiting_revision(
            project_dir=project_dir_a, experiment_id=experiment_id, cycle_id=revision_request["cycle_id"],
            round_number=1, writer_revision_request=revision_request,
            artifacts={"writer_revision_request.json": revision_request},
        )
        overrides_awaiting = _resolve_draft_execution_mode(project_dir_a, store_a)

        # Escenario B: la MISMA ronda, ya completada por 06.
        project_dir_b, store_b, _ = _seed_active_cycle_with_completed_round(tmp2)
        overrides_completed = _resolve_draft_execution_mode(project_dir_b, store_b)

        assert overrides_awaiting["mode"] == overrides_completed["mode"]
        assert overrides_awaiting["writer_revision_request"] == overrides_completed["writer_revision_request"]
        assert overrides_awaiting["previous_draft"] == overrides_completed["previous_draft"]
        assert overrides_awaiting["round_number"] == overrides_completed["round_number"]


@scenario("D03. La ronda REVISION_COMPLETED sigue REVISION_COMPLETED después de llamar _resolve_draft_execution_mode -- no se toca")
def test_round_status_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, experiment_id = _seed_active_cycle_with_completed_round(tmp)
        status_before = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=1)
        assert status_before["status"] == "REVISION_COMPLETED"

        _resolve_draft_execution_mode(project_dir, store)

        status_after = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=1)
        assert status_after["status"] == "REVISION_COMPLETED"
        assert status_after == status_before  # ni un campo cambió


@scenario("D04. Un segundo intento REAL de completar la misma ronda sigue rechazándose (red de seguridad de complete_round_revision intacta)")
def test_double_completion_still_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, experiment_id = _seed_active_cycle_with_completed_round(tmp)
        revision_request = _revision_request(1)
        try:
            complete_round_revision(
                project_dir=project_dir, experiment_id=experiment_id, cycle_id=revision_request["cycle_id"],
                round_number=1, writer_revision_request=revision_request, artifacts=_revised_artifacts(),
            )
        except RuntimeError:
            pass  # esperado -- la corrección de _resolve_draft_execution_mode no debilitó esta red de seguridad
        else:
            raise AssertionError("un segundo intento de completar la ronda debía seguir siendo rechazado")


@scenario("D05. Estado inesperado real (ni AWAITING_REVISION ni REVISION_COMPLETED) sigue lanzando la excepción, sin ocultarlo")
def test_genuinely_unexpected_status_still_raises():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, experiment_id = _seed_active_cycle_with_completed_round(tmp)
        from src.tools.verification.cycle_round_persistence import round_directory
        from src.io.atomic_write import atomic_write_json

        status_path = round_directory(project_dir, experiment_id, 1) / "_round_status.json"
        atomic_write_json(status_path, {"status": "CORRUPTED_UNKNOWN_STATE"})

        try:
            _resolve_draft_execution_mode(project_dir, store)
        except RuntimeError as exc:
            assert "DRAFT_REVISION_ROUND_UNEXPECTED_STATUS" in str(exc)
        else:
            raise AssertionError("un estado genuinamente inesperado debía seguir lanzando la excepción")


if __name__ == "__main__":
    for fn in (
        test_revision_completed_round_does_not_raise,
        test_revision_completed_returns_same_shape_as_awaiting,
        test_round_status_untouched,
        test_double_completion_still_rejected,
        test_genuinely_unexpected_status_still_raises,
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
