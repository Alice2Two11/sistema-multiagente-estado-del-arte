"""Prueba de equivalencia A3: ``run_llm_judge`` es bloqueante, no un salto
silencioso — mismo mensaje real de la celda 21."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.evaluation_pipeline import run_evaluation_pipeline

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


POLICY = {
    "minimum_generated_words": 5,
    "minimum_ground_truth_words": 5,
    "require_explicit_ground_truth_end_heading": False,
    "translate_for_rouge_if_language_differs": True,
    "max_translation_chars_per_chunk": 500,
    "semantic_chunk_chars": 80,
    "semantic_chunk_overlap_chars": 0,
    "max_semantic_chunks_per_text": 5,
    "evaluation_embedding_model": "fake",
    "bertscore_model": "fake",
    "max_bertscore_pairs": 4,
    "llm_judge_max_generated_chars": 2000,
    "llm_judge_max_ground_truth_chars": 2000,
    "llm_judge_max_attempts": 3,
    "fail_on_invalid_evaluation": False,
    "create_corpus_gap_suggestions": True,
    "run_llm_judge": False,
}


@scenario("A3. run_llm_judge=False lanza ValueError bloqueante (no un salto silencioso)")
def test_run_llm_judge_false_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_dir = tmp / "gt"
        gt_dir.mkdir()
        (gt_dir / "ground_truth_literature_review.txt").write_text(
            "Texto de referencia suficientemente largo para pasar la validación real.",
            encoding="utf-8",
        )
        try:
            run_evaluation_pipeline(
                generated_plain_text="Texto generado suficientemente largo para la prueba real.",
                sections=[{"section_id": "s1", "draft_text": "Texto."}],
                chunks=[],
                traceability_rows=[{"claim_id": "1", "verdict": "supported", "claim": "Texto."}],
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
                translation_llm_factory=lambda: None,
                embedding_model_factory=None,
                bertscore_score_fn=None,
                judge_llm_factory=lambda: None,
            )
        except ValueError as exc:
            assert "desactivó el LLM Judge" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError bloqueante")


if __name__ == "__main__":
    for fn in (test_run_llm_judge_false_raises,):
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
