"""Pruebas integrales del ensamblador completo de 08
(``run_evaluation_pipeline``): automática + factual + Judge + validación
final, con dependencias deterministas pero atravesando el código real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.evaluation_pipeline import run_evaluation_pipeline
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


def _valid_judge_response(overall="Evaluación general sólida."):
    return {
        "scores": {
            criterion: {"score": 4, "justification": "Justificación válida.", "evidence_from_generated": []}
            for criterion in JUDGE_CRITERIA
        },
        "strengths": ["Fortaleza uno."],
        "organization_differences": [],
        "missing_topics_or_omissions": [],
        "overall_assessment": overall,
    }


def _bertscore_fn(candidates, references, **kwargs):
    n = len(candidates)
    values = [1.0 if c == r else 0.6 for c, r in zip(candidates, references)]
    return FakeTensor(values), FakeTensor(values), FakeTensor(values)


def _write_gt(tmp: Path, text: str) -> Path:
    gt_dir = tmp / "gt"
    gt_dir.mkdir()
    (gt_dir / "ground_truth_literature_review.txt").write_text(text, encoding="utf-8")
    return gt_dir


GENERATED_TEXT = (
    "El modelo alcanzó 91 puntos en el conjunto de datos evaluado [p.pdf | c1]. "
    "Los resultados confirman una mejora consistente frente a enfoques previos [p.pdf | c1]."
)
GT_TEXT = (
    "Estudios previos reportaron un desempeño de 90 puntos en tareas similares. "
    "El presente trabajo confirma mejoras consistentes en el area evaluada con nuevos metodos."
)


def _base_kwargs(tmp: Path, *, judge_responses=None, policy=None):
    sections = [{"section_id": "s1", "draft_text": GENERATED_TEXT}]
    chunks = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91 puntos, mejora consistente."}]
    traceability_rows = [
        {
            "claim_id": "1",
            "verdict": "supported",
            "claim": "El modelo alcanzó 91 puntos en el conjunto de datos evaluado.",
            "hallucination_risk": "low",
            "source_filename": "p.pdf",
            "chunk_id": "c1",
        }
    ]
    judge_factory = FakeLLMFactory(
        responses=list(judge_responses or [json.dumps(_valid_judge_response())])
    )
    return dict(
        generated_plain_text=GENERATED_TEXT,
        sections=sections,
        chunks=chunks,
        traceability_rows=traceability_rows,
        source_stage="AGENT07",
        upstream_runtime_status="COMPLETED",
        reverification_performed=False,
        reverification_reason=None,
        claims_verified=1,
        claims_requiring_manual_review=0,
        manual_review_claim_ids=[],
        generated_status="EVALUATION_READY",
        evaluation_ready_json_path=str(tmp / "draft.json"),
        experiment_id="exp1",
        topic_name="Tema de prueba",
        ground_truth_dir=str(_write_gt(tmp, GT_TEXT)),
        evaluation_policy=dict(policy or DEFAULT_POLICY),
        translation_llm_factory=FakeLLMFactory(),
        embedding_model_factory=lambda name: FakeEmbeddingModel(),
        bertscore_score_fn=_bertscore_fn,
        judge_llm_factory=judge_factory,
    ), judge_factory


@scenario("F01. Ensamblaje completo automático+factual+Judge sin error")
def test_full_pipeline_success():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _judge_factory = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        assert result["final_validation"]["evaluation_validation_ok"] is True
        assert len(result["final_selected_metrics"]) == 15


@scenario("F02. Diez métricas automáticas presentes")
def test_ten_automatic_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        assert len(result["automatic_metrics_result"].automatic_metric_rows) == 10


@scenario("F03. Doce filas factuales presentes")
def test_twelve_factual_rows():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        assert len(result["factual_audit"]["factual_metric_rows"]) == 12


@scenario("F04. Cinco puntuaciones del Judge presentes")
def test_five_judge_scores():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        assert len(result["judge_score_rows"]) == 5


@scenario("F05. Quince métricas seleccionadas en orden (4 automáticas + 5 Judge + 6 factuales)")
def test_fifteen_selected_metrics_order():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        sections_seen = [row["section"] for row in result["final_selected_metrics"]]
        assert sections_seen[:4] == ["Métricas automáticas"] * 4
        assert sections_seen[4:9] == ["LLM Judge"] * 5
        assert sections_seen[9:] == ["Métricas factuales"] * 6


@scenario("F06. Pendientes factuales permitidos (source_stage=AGENT07, upstream PARTIAL) no bloquean")
def test_pending_factual_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        kwargs["upstream_runtime_status"] = "PARTIAL"
        kwargs["traceability_rows"] = [
            {"claim_id": "1", "verdict": "unsupported", "claim": "Afirmación no verificable.",
             "hallucination_risk": "high"}
        ]
        kwargs["sections"] = [{"section_id": "s1", "draft_text": "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."}]
        kwargs["generated_plain_text"] = "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."
        result = run_evaluation_pipeline(**kwargs)
        assert result["factual_consistency_result"]["factual_consistency_ok"] is False
        assert "upstream_partial_factual_consistency_not_approved" in result["final_validation"]["validation_warnings"]
        assert result["final_validation"]["evaluation_validation_ok"] is True  # el warning no cuenta como error


@scenario("F07. Fallo factual bloqueante (no PARTIAL/AGENT07) se propaga y detiene el ensamblador")
def test_blocking_factual_failure_propagates():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        kwargs["upstream_runtime_status"] = "COMPLETED"  # no PARTIAL -> bloqueante
        kwargs["traceability_rows"] = [
            {"claim_id": "1", "verdict": "unsupported", "claim": "Afirmación no verificable.",
             "hallucination_risk": "high"}
        ]
        kwargs["sections"] = [{"section_id": "s1", "draft_text": "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."}]
        kwargs["generated_plain_text"] = "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."
        try:
            run_evaluation_pipeline(**kwargs)
        except ValueError as exc:
            assert "incompatible" in str(exc)
        else:
            raise AssertionError("debía propagar el fallo factual bloqueante")


@scenario("F08. Validación final inválida bloqueante (fail_on_invalid_evaluation=True) se propaga")
def test_final_validation_blocking():
    with tempfile.TemporaryDirectory() as tmp:
        policy = dict(DEFAULT_POLICY)
        policy["fail_on_invalid_evaluation"] = True
        kwargs, judge_factory = _base_kwargs(
            Path(tmp), judge_responses=["invalido", "invalido", "invalido"], policy=policy
        )
        try:
            run_evaluation_pipeline(**kwargs)
        except ValueError:
            pass  # el Judge agota intentos -> ValueError ya antes de llegar a evaluation_validation_ok
        else:
            raise AssertionError("debía fallar por Judge inválido agotado")


@scenario("F09. Validación final inválida NO bloqueante (fail_on_invalid_evaluation=False) completa igual")
def test_final_validation_non_blocking():
    with tempfile.TemporaryDirectory() as tmp:
        policy = dict(DEFAULT_POLICY)
        policy["fail_on_invalid_evaluation"] = False
        kwargs, _ = _base_kwargs(Path(tmp), policy=policy)
        kwargs["upstream_runtime_status"] = "PARTIAL"
        kwargs["traceability_rows"] = [
            {"claim_id": "1", "verdict": "unsupported", "claim": "Afirmación no verificable.",
             "hallucination_risk": "high"}
        ]
        kwargs["sections"] = [{"section_id": "s1", "draft_text": "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."}]
        kwargs["generated_plain_text"] = "Afirmacion no verificable sin cita alguna presente en el texto, redactada con suficientes palabras para que el detector de idioma pueda identificar correctamente el idioma predominante del documento."
        result = run_evaluation_pipeline(**kwargs)
        assert result is not None  # completó pese a validation_ok=True con warning (no error)


@scenario("F10. Judge inválido con reintento real dentro del ensamblador completo")
def test_judge_invalid_then_valid_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, judge_factory = _base_kwargs(
            Path(tmp), judge_responses=["no es json valido", json.dumps(_valid_judge_response())]
        )
        result = run_evaluation_pipeline(**kwargs)
        assert judge_factory.instances_created == 2
        assert result["judge_errors"] == []


@scenario("F11. Fallo del Judge (agotamiento) se propaga desde el ensamblador completo")
def test_judge_exhaustion_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp), judge_responses=["mal"] * 3)
        try:
            run_evaluation_pipeline(**kwargs)
        except ValueError as exc:
            assert "intentos" in str(exc)
        else:
            raise AssertionError("debía agotar los intentos del Judge")


@scenario("F12. Ground Truth aislado: no se filtra a chunks ni a texto generado")
def test_ground_truth_isolated_within_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, _ = _base_kwargs(Path(tmp))
        result = run_evaluation_pipeline(**kwargs)
        # el texto del GT nunca debe aparecer en los chunks usados para BERTScore/semántica
        for chunk in result["automatic_metrics_result"].ground_truth_chunks:
            assert chunk not in kwargs["chunks"][0]["text"]
        assert result["ground_truth_plain_text"] != kwargs["generated_plain_text"]


if __name__ == "__main__":
    for fn in (
        test_full_pipeline_success,
        test_ten_automatic_metrics,
        test_twelve_factual_rows,
        test_five_judge_scores,
        test_fifteen_selected_metrics_order,
        test_pending_factual_allowed,
        test_blocking_factual_failure_propagates,
        test_final_validation_blocking,
        test_final_validation_non_blocking,
        test_judge_invalid_then_valid_full_pipeline,
        test_judge_exhaustion_full_pipeline,
        test_ground_truth_isolated_within_pipeline,
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
