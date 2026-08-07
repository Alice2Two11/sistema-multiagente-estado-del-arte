"""Pruebas del contrato transaccional y del StageSpec real de 08:
PREPARE/EXECUTE/COMMIT/RESUME, persistencia de los 15 outputs, fingerprints,
force_rebuild, backup, integración 07->08 y finalización del pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.evaluation_orchestrator_runtime import (
    EVALUATION_STAGE_NAME,
    _run_evaluation_stage,
    build_experimental_evaluation_execution,
)
from src.adapters.evaluation_persistence import REQUIRED_OUTPUT_FILENAMES
from src.contracts.agent_result import QualityStatus, RequestedTransition, TransitionAction
from src.orchestration import decision_engine as de
from src.orchestration.pipeline_orchestrator import StageSpec, ensure_pipeline_state, run_stage
from src.tools.evaluation.llm_judge import JUDGE_CRITERIA
from tests.orchestration.test_evaluation_automatic_metrics_integration import FakeEmbeddingModel
from tests.orchestration.test_evaluation_bertscore_characterization import FakeTensor
from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

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


DEFAULT_POLICY = {
    "translate_for_rouge_if_language_differs": True,
    "max_translation_chars_per_chunk": 200,
    "semantic_chunk_chars": 60,
    "semantic_chunk_overlap_chars": 0,
    "max_semantic_chunks_per_text": 5,
    "evaluation_embedding_model": "fake-embedding-model",
    "bertscore_model": "fake-bertscore-model",
    "max_bertscore_pairs": 4,
    "minimum_ground_truth_words": 5,
    "require_explicit_ground_truth_end_heading": False,
    "minimum_generated_words": 3,
    "llm_judge_max_generated_chars": 2000,
    "llm_judge_max_ground_truth_chars": 2000,
    "llm_judge_max_attempts": 3,
    "fail_on_invalid_evaluation": True,
    "create_corpus_gap_suggestions": True,
    "run_llm_judge": True,
}

GENERATED_TEXT = (
    "El modelo alcanzó 91 puntos en el conjunto de datos evaluado [p.pdf | c1]. "
    "Los resultados confirman una mejora consistente frente a enfoques previos [p.pdf | c1]."
)
GT_TEXT = (
    "Estudios previos reportaron un desempeño de 90 puntos en tareas similares. "
    "El presente trabajo confirma mejoras consistentes en el area evaluada con nuevos metodos."
)


def _valid_judge_response():
    return {
        "scores": {
            criterion: {"score": 4, "justification": "Justificación válida.", "evidence_from_generated": []}
            for criterion in JUDGE_CRITERIA
        },
        "strengths": [],
        "organization_differences": [],
        "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general sólida.",
    }


def _bertscore_fn(candidates, references, **kwargs):
    values = [1.0 if c == r else 0.6 for c, r in zip(candidates, references)]
    return FakeTensor(values), FakeTensor(values), FakeTensor(values)


def _write_gt(tmp: Path) -> Path:
    gt_dir = tmp / "gt"
    gt_dir.mkdir(exist_ok=True)
    (gt_dir / "ground_truth_literature_review.txt").write_text(GT_TEXT, encoding="utf-8")
    return gt_dir


def _make_kwargs(tmp: Path, *, policy=None):
    draft_path = tmp / "draft.json"
    if not draft_path.exists():
        draft_path.write_text(
            json.dumps({"status": "EVALUATION_READY", "sections": [{"section_id": "s1", "draft_text": GENERATED_TEXT}]}),
            encoding="utf-8",
        )
    return build_experimental_evaluation_execution(
        generated_plain_text=GENERATED_TEXT,
        sections=[{"section_id": "s1", "draft_text": GENERATED_TEXT}],
        chunks=[{"source_filename": "p.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91 puntos, mejora consistente."}],
        traceability_rows=[
            {"claim_id": "1", "verdict": "supported", "claim": "El modelo alcanzó 91 puntos en el conjunto de datos evaluado.",
             "hallucination_risk": "low", "source_filename": "p.pdf", "chunk_id": "c1"}
        ],
        source_stage="AGENT07",
        upstream_runtime_status="COMPLETED",
        reverification_performed=False,
        reverification_reason=None,
        claims_verified=1,
        claims_requiring_manual_review=0,
        manual_review_claim_ids=[],
        generated_status="EVALUATION_READY",
        evaluation_ready_json_path=str(draft_path),
        experiment_id="exp_stagespec",
        topic_name="Tema de prueba",
        ground_truth_dir=str(_write_gt(tmp)),
        evaluation_policy=dict(policy or DEFAULT_POLICY),
        translation_llm_factory=FakeLLMFactory(),
        embedding_model_factory=lambda name: FakeEmbeddingModel(),
        bertscore_score_fn=_bertscore_fn,
        judge_llm_factory=FakeLLMFactory(responses=[json.dumps(_valid_judge_response())]),
    )


def _seed_state(tmp: Path):
    root = tmp / "proj"
    root.mkdir()
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": "exp_stagespec", "run_id": "run1"}), encoding="utf-8"
    )
    store = ensure_pipeline_state(root)
    return root, store


def _make_spec(tmp: Path, output_dir: Path, *, policy=None, judge_responses=None):
    def build_execution(project_dir, attempt_number):
        kwargs = _make_kwargs(tmp, policy=policy)
        if judge_responses is not None:
            kwargs["judge_llm_factory"] = FakeLLMFactory(responses=list(judge_responses))
        kwargs["output_dir"] = str(output_dir)
        kwargs["numeric_check_output_dir"] = str(output_dir)
        kwargs["backup_root"] = str(output_dir / ".backups")
        kwargs["_openai_model"] = "gpt-4.1-mini"
        return kwargs

    return StageSpec(
        key=EVALUATION_STAGE_NAME,
        label="08 · Evaluación (prueba real)",
        build_execution=build_execution,
        runtime_transaction=None,
        resolve_resume=None,
        build_fingerprints=None,
        custom_run=_run_evaluation_stage,
    )


@scenario("E01. StageSpec real de 08: PREPARE/EXECUTE/COMMIT completos, produce COMMITTED")
def test_real_stagespec_first_run():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED", outcome
        assert outcome.execution_status == "COMPLETED"
        state = store.load()
        assert EVALUATION_STAGE_NAME in state.stages
        assert state.pending_execution is None


@scenario("E02. Los 15 outputs obligatorios se escriben con los nombres reales")
def test_fifteen_outputs_written():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        for filename in REQUIRED_OUTPUT_FILENAMES:
            assert (output_dir / filename).is_file(), f"falta {filename}"
        assert len(REQUIRED_OUTPUT_FILENAMES) == 15


@scenario("E03. output_artifacts del AgentResult comprometido incluye los 15 archivos con hash")
def test_output_artifacts_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        state = store.load()
        committed_artifacts = {
            name for name, artifact in state.artifacts.items() if name in REQUIRED_OUTPUT_FILENAMES
        }
        assert committed_artifacts == set(REQUIRED_OUTPUT_FILENAMES)


@scenario("E04. Segunda llamada con el mismo fingerprint: SKIPPED_FRESH, no reescribe")
def test_fingerprint_reused_skips():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        mtimes_before = {f: (output_dir / f).stat().st_mtime for f in REQUIRED_OUTPUT_FILENAMES}

        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome2.status == "SKIPPED_FRESH", outcome2
        mtimes_after = {f: (output_dir / f).stat().st_mtime for f in REQUIRED_OUTPUT_FILENAMES}
        assert mtimes_before == mtimes_after  # no se reescribió nada


@scenario("E05. Fingerprint obsoleto (cambia la política) fuerza reejecución real")
def test_stale_fingerprint_reexecutes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)

        changed_policy = dict(DEFAULT_POLICY)
        changed_policy["max_bertscore_pairs"] = 2  # cambia el fingerprint de config
        spec2 = _make_spec(tmp, output_dir, policy=changed_policy)
        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec2, attempt_number=1)
        assert outcome2.status == "COMMITTED", outcome2  # se reejecutó de verdad, no SKIPPED_FRESH


@scenario("E06. force_rerun fuerza reejecución aunque el fingerprint no haya cambiado")
def test_force_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1, force_rerun=True)
        assert outcome2.status == "COMMITTED", outcome2


@scenario("E07. Backup: outputs previos se copian antes de sobrescribir en una reejecución real")
def test_backup_created_on_reexecution():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1, force_rerun=True)
        backups = list((output_dir / ".backups").glob("backup_*"))
        assert len(backups) == 1
        assert (backups[0] / "automatic_metrics.csv").is_file()


@scenario("E08. Excepción técnica (Ground Truth ausente) se compromete como FAILED, sin outputs")
def test_technical_exception_commits_failed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"

        def build_execution(pd_, attempt):
            kwargs = _make_kwargs(tmp)
            kwargs["ground_truth_dir"] = str(tmp / "no_existe")  # provoca FileNotFoundError real
            kwargs["output_dir"] = str(output_dir)
            kwargs["numeric_check_output_dir"] = str(output_dir)
            kwargs["backup_root"] = str(output_dir / ".backups")
            kwargs["_openai_model"] = "gpt-4.1-mini"
            return kwargs

        spec = StageSpec(
            key=EVALUATION_STAGE_NAME, label="08 fallo técnico", build_execution=build_execution,
            runtime_transaction=None, resolve_resume=None, build_fingerprints=None,
            custom_run=_run_evaluation_stage,
        )
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status == "FAILED", outcome
        assert outcome.execution_status == "FAILED"
        assert not (output_dir / "automatic_metrics.csv").exists()


@scenario("E09. No queda pending colgado tras una corrida normal")
def test_no_dangling_pending_after_normal_run():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert store.load().pending_execution is None


@scenario("E10. RESUME tras persistir: segunda corrida reutiliza vía fingerprint")
def test_resume_after_persisting_via_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome2.status == "SKIPPED_FRESH"


@scenario("E11. Integración 07->08: transición de 08 finaliza el pipeline (STOP_PIPELINE)")
def test_advance_transition_and_pipeline_finalization():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.next_action == "STOP_PIPELINE"
        assert outcome.target_stage is None  # 08 es la última etapa -> fin del pipeline

        validated = de.validate_transition(
            current_stage=EVALUATION_STAGE_NAME,
            requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, reason_code="EVALUATION_COMPLETED"),
            quality_status=QualityStatus.APPROVED,
            attempts_used=1,
            known_stages=frozenset(de.CANONICAL_STAGE_ORDER),
        )
        assert validated.action == "STOP_PIPELINE"
        assert validated.reason_code == "PIPELINE_COMPLETE"


@scenario("E12. 07C ausente del registro activo")
def test_07c_not_in_active_registry():
    for key in de.CANONICAL_STAGE_ORDER:
        assert "07C" not in key and "07c" not in key
    from src.orchestration.pipeline_orchestrator import _stage_registry

    for spec in _stage_registry():
        assert "07C" not in spec.key and "07c" not in spec.key


@scenario("E13. Output faltante: find_missing_outputs detecta un archivo borrado tras persistir")
def test_missing_output_detected():
    from src.adapters.evaluation_persistence import find_missing_outputs

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status == "COMMITTED"

        missing_before = find_missing_outputs(output_dir=output_dir)
        assert missing_before == []

        (output_dir / "automatic_metrics.csv").unlink()
        missing_after = find_missing_outputs(output_dir=output_dir)
        assert missing_after == ["automatic_metrics.csv"]


@scenario("A2. run_stage() tras borrar un output NO devuelve SKIPPED_FRESH y reconstruye el archivo")
def test_missing_output_forces_rebuild_via_run_stage():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project_dir, store = _seed_state(tmp)
        output_dir = tmp / "eval_out"
        spec = _make_spec(tmp, output_dir)

        outcome1 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome1.status == "COMMITTED"
        assert (output_dir / "automatic_metrics.csv").is_file()

        (output_dir / "automatic_metrics.csv").unlink()
        assert not (output_dir / "automatic_metrics.csv").is_file()

        outcome2 = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome2.status != "SKIPPED_FRESH", outcome2
        assert outcome2.status == "COMMITTED"
        assert (output_dir / "automatic_metrics.csv").is_file()  # reconstruido


if __name__ == "__main__":
    for fn in (
        test_real_stagespec_first_run,
        test_fifteen_outputs_written,
        test_output_artifacts_recorded,
        test_fingerprint_reused_skips,
        test_stale_fingerprint_reexecutes,
        test_force_rerun,
        test_backup_created_on_reexecution,
        test_technical_exception_commits_failed,
        test_no_dangling_pending_after_normal_run,
        test_resume_after_persisting_via_fingerprint,
        test_advance_transition_and_pipeline_finalization,
        test_07c_not_in_active_registry,
        test_missing_output_detected,
        test_missing_output_forces_rebuild_via_run_stage,
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
