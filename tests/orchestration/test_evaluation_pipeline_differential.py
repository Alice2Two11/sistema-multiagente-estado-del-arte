"""Prueba diferencial del FLUJO INTEGRAL de 08: ``run_evaluation_pipeline``
completo (Ground Truth + idioma + traducción + métricas automáticas +
auditoría factual + LLM Judge + validación final + resumen) comparado
contra una reproducción independiente que llama a las MISMAS piezas ya
probadas por separado, pero ensambladas por su cuenta (sin compartir
código con ``evaluation_pipeline.py``), usando los mismos dobles
deterministas en ambos caminos.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.automatic_metrics import build_automatic_metrics
from src.tools.evaluation.evaluation_pipeline import run_evaluation_pipeline
from src.tools.evaluation.factual_assembly import (
    build_factual_audit,
    evaluate_factual_consistency,
    resolve_factual_gate,
)
from src.tools.evaluation.final_report import (
    build_corpus_gap_markdown,
    build_corpus_gap_rows,
    build_evaluation_summary,
    build_final_selected_metrics,
)
from src.tools.evaluation.final_validation import (
    evaluate_final_validation,
    resolve_final_validation_gate,
)
from src.tools.evaluation.ground_truth import resolve_ground_truth_comparable_text
from src.tools.evaluation.language_preprocessing import detect_language_code
from src.tools.evaluation.llm_judge import build_judge_score_rows, run_llm_judge, validate_judge_result

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


class FakeLLMFactory:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.instances_created = 0

    def __call__(self):
        self.instances_created += 1
        return self

    def invoke(self, messages):
        content = self.responses.pop(0) if self.responses else "[respuesta simulada]"
        return SimpleNamespace(content=content)


class FakeEmbeddingModel:
    def encode(self, chunks, *, normalize_embeddings, show_progress_bar):
        import numpy as np

        def vec(text):
            vowels = sum(1 for c in text.lower() if c in "aeiouáéíóú")
            length = max(len(text), 1)
            v = np.array([vowels / length, 1 - vowels / length])
            n = (v[0] ** 2 + v[1] ** 2) ** 0.5
            return v / n if n > 0 else v

        return np.array([vec(c) for c in chunks])


class FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


def _bertscore_fn(candidates, references, **kwargs):
    values = [1.0 if c == r else 0.5 for c, r in zip(candidates, references)]
    return FakeTensor(values), FakeTensor(values), FakeTensor(values)


def _valid_judge_response():
    from src.tools.evaluation.llm_judge import JUDGE_CRITERIA

    return {
        "scores": {
            c: {"score": 4, "justification": "Justificación válida.", "evidence_from_generated": []}
            for c in JUDGE_CRITERIA
        },
        "strengths": [],
        "organization_differences": [],
        "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general aceptable.",
    }


POLICY = {
    "minimum_generated_words": 5,
    "minimum_ground_truth_words": 5,
    "require_explicit_ground_truth_end_heading": False,
    "translate_for_rouge_if_language_differs": True,
    "max_translation_chars_per_chunk": 500,
    "semantic_chunk_chars": 80,
    "semantic_chunk_overlap_chars": 0,
    "max_semantic_chunks_per_text": 5,
    "evaluation_embedding_model": "fake-model",
    "bertscore_model": "fake-bertscore",
    "max_bertscore_pairs": 4,
    "llm_judge_max_generated_chars": 2000,
    "llm_judge_max_ground_truth_chars": 2000,
    "llm_judge_max_attempts": 3,
    "fail_on_invalid_evaluation": False,
    "create_corpus_gap_suggestions": True,
    "run_llm_judge": True,
}

GENERATED_TEXT = (
    "El modelo alcanzó 91 puntos en el conjunto de datos evaluado [p.pdf | c1]. "
    "Los resultados confirman una mejora consistente frente a enfoques previos [p.pdf | c1]."
)
CHUNKS = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91 puntos, mejora consistente."}]
SECTIONS = [{"section_id": "s1", "draft_text": GENERATED_TEXT}]
TRACEABILITY = [
    {"claim_id": "1", "verdict": "supported", "claim": "El modelo alcanzó 91 puntos en el conjunto de datos evaluado.",
     "hallucination_risk": "low", "source_filename": "p.pdf", "chunk_id": "c1"}
]


def _oracle_run(*, ground_truth_dir, translation_factory, embedding_model, bertscore_fn, judge_factory):
    ground_truth_plain_text, gt_metadata, gt_source_path = resolve_ground_truth_comparable_text(
        ground_truth_dir=ground_truth_dir,
        minimum_words=POLICY["minimum_ground_truth_words"],
        require_explicit_end_heading=POLICY["require_explicit_ground_truth_end_heading"],
    )
    generated_language = detect_language_code(GENERATED_TEXT)
    ground_truth_language = detect_language_code(ground_truth_plain_text)

    automatic = build_automatic_metrics(
        generated_plain_text=GENERATED_TEXT,
        ground_truth_plain_text=ground_truth_plain_text,
        generated_language=generated_language,
        ground_truth_language=ground_truth_language,
        evaluation_policy=POLICY,
        translation_llm_factory=translation_factory,
        embedding_model_factory=lambda name: embedding_model,
        bertscore_score_fn=bertscore_fn,
    )

    factual_audit = build_factual_audit(
        sections=SECTIONS, chunks=CHUNKS, traceability_rows=TRACEABILITY,
        generated_content_text=GENERATED_TEXT,
    )
    factual_consistency = evaluate_factual_consistency(factual_audit)
    resolve_factual_gate(
        factual_consistency_result=factual_consistency, source_stage="AGENT07", upstream_runtime_status="COMPLETED"
    )

    automatic_dict = {r["metric"]: float(r["value"]) for r in automatic.automatic_metric_rows}
    factual_dict = {r["metric"]: r["value"] for r in factual_audit["factual_metric_rows"]}

    judge_run = run_llm_judge(
        topic_name="Tema", source_stage="AGENT07", automatic_metrics=automatic_dict, factual_metrics=factual_dict,
        generated_plain_text=GENERATED_TEXT, ground_truth_plain_text=ground_truth_plain_text,
        max_generated_chars=POLICY["llm_judge_max_generated_chars"],
        max_ground_truth_chars=POLICY["llm_judge_max_ground_truth_chars"],
        max_attempts=POLICY["llm_judge_max_attempts"], llm_factory=judge_factory,
    )
    judge_result = judge_run["result"]
    judge_rows = build_judge_score_rows(judge_result)
    judge_errors = validate_judge_result(judge_result)

    selected = build_final_selected_metrics(
        automatic_metric_rows=automatic.automatic_metric_rows, judge_score_rows=judge_rows,
        factual_metric_rows=factual_audit["factual_metric_rows"],
    )

    final_validation = evaluate_final_validation(
        factual_consistency_ok=factual_consistency["factual_consistency_ok"],
        factual_consistency_status=factual_consistency["factual_consistency_status"],
        source_stage="AGENT07", upstream_runtime_status="COMPLETED", judge_errors=judge_errors,
        final_selected_metrics=selected, experiment_id="exp1", reverification_performed=False,
        reverification_reason=None, claims_requiring_manual_review=0, manual_review_claim_ids=[],
    )

    gap_rows = build_corpus_gap_rows(
        missing_topics_or_omissions=judge_result.get("missing_topics_or_omissions", []),
        create_corpus_gap_suggestions=POLICY["create_corpus_gap_suggestions"],
    )
    gap_md = build_corpus_gap_markdown(gap_rows)

    summary = build_evaluation_summary(
        experiment_id="exp1", topic_name="Tema", evaluation_ready_json_path="draft.json",
        source_stage="AGENT07", reverification_performed=False, reverification_reason=None,
        upstream_runtime_status="COMPLETED", claims_verified=1, claims_requiring_manual_review=0,
        manual_review_claim_ids=[], generated_status="EVALUATION_READY",
        ground_truth_source_path=str(gt_source_path), generated_plain_text=GENERATED_TEXT,
        ground_truth_plain_text=ground_truth_plain_text, generated_language=generated_language,
        ground_truth_language=ground_truth_language, translation_mode=automatic.translation_mode,
        automatic_metric_rows=automatic.automatic_metric_rows, judge_score_rows=judge_rows,
        factual_metric_rows=factual_audit["factual_metric_rows"],
        factual_consistency_status=factual_consistency["factual_consistency_status"],
        overall_assessment=judge_result["overall_assessment"], corpus_gap_count=len(gap_rows),
    )

    resolve_final_validation_gate(
        final_validation_result=final_validation, fail_on_invalid_evaluation=POLICY["fail_on_invalid_evaluation"],
        validation_report_path="evaluation_validation_report.json",
    )

    return {
        "ground_truth_plain_text": ground_truth_plain_text,
        "generated_language": generated_language,
        "ground_truth_language": ground_truth_language,
        "automatic_metric_rows": automatic.automatic_metric_rows,
        "factual_metric_rows": factual_audit["factual_metric_rows"],
        "judge_score_rows": judge_rows,
        "final_selected_metrics": selected,
        "final_validation": final_validation,
        "corpus_gap_rows": gap_rows,
        "corpus_gap_markdown": gap_md,
        "evaluation_summary": summary,
    }


@scenario("O01. Diferencial COMPLETA del flujo integral de 08 (run_evaluation_pipeline vs. oráculo independiente)")
def test_full_integral_pipeline_differential():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_dir = tmp / "gt"
        gt_dir.mkdir()
        (gt_dir / "ground_truth_literature_review.txt").write_text(
            "Estudios previos reportaron un desempeño de 90 puntos en tareas similares. "
            "El presente trabajo confirma mejoras consistentes en el area evaluada con nuevos metodos.",
            encoding="utf-8",
        )

        judge_response = json.dumps(_valid_judge_response())

        real_result = run_evaluation_pipeline(
            generated_plain_text=GENERATED_TEXT,
            sections=SECTIONS,
            chunks=CHUNKS,
            traceability_rows=TRACEABILITY,
            source_stage="AGENT07",
            upstream_runtime_status="COMPLETED",
            reverification_performed=False,
            reverification_reason=None,
            claims_verified=1,
            claims_requiring_manual_review=0,
            manual_review_claim_ids=[],
            generated_status="EVALUATION_READY",
            evaluation_ready_json_path="draft.json",
            experiment_id="exp1",
            topic_name="Tema",
            ground_truth_dir=str(gt_dir),
            evaluation_policy=POLICY,
            translation_llm_factory=FakeLLMFactory(),
            embedding_model_factory=lambda name: FakeEmbeddingModel(),
            bertscore_score_fn=_bertscore_fn,
            judge_llm_factory=FakeLLMFactory(responses=[judge_response]),
        )

        oracle_result = _oracle_run(
            ground_truth_dir=str(gt_dir),
            translation_factory=FakeLLMFactory(),
            embedding_model=FakeEmbeddingModel(),
            bertscore_fn=_bertscore_fn,
            judge_factory=FakeLLMFactory(responses=[judge_response]),
        )

        assert real_result["ground_truth_plain_text"] == oracle_result["ground_truth_plain_text"]
        assert real_result["generated_language"] == oracle_result["generated_language"]
        assert real_result["ground_truth_language"] == oracle_result["ground_truth_language"]
        assert (
            real_result["automatic_metrics_result"].automatic_metric_rows
            == oracle_result["automatic_metric_rows"]
        )
        assert real_result["factual_audit"]["factual_metric_rows"] == oracle_result["factual_metric_rows"]
        assert real_result["judge_score_rows"] == oracle_result["judge_score_rows"]
        assert real_result["final_selected_metrics"] == oracle_result["final_selected_metrics"]
        assert (
            real_result["final_validation"]["evaluation_validation_ok"]
            == oracle_result["final_validation"]["evaluation_validation_ok"]
        )
        assert real_result["corpus_gap_rows"] == oracle_result["corpus_gap_rows"]
        assert real_result["corpus_gap_markdown"] == oracle_result["corpus_gap_markdown"]
        # evaluation_summary difiere solo en created_at (timestamp) -- se
        # comparan las demás claves.
        real_summary = dict(real_result["evaluation_summary"])
        oracle_summary = dict(oracle_result["evaluation_summary"])
        real_summary.pop("created_at")
        oracle_summary.pop("created_at")
        assert real_summary == oracle_summary


if __name__ == "__main__":
    for fn in (test_full_integral_pipeline_differential,):
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
