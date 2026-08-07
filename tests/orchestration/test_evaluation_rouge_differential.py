"""Prueba diferencial del Bloque 4A: oráculo reproducido vs. módulo real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.rouge import build_rouge_metric_rows, build_rouge_scorer, compute_rouge_l

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


def _oracle_rouge_l(ground_truth_plain_text, generated_for_rouge, ground_truth_language):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=(ground_truth_language == "en"))
    return scorer.score(ground_truth_plain_text, generated_for_rouge)["rougeL"]


def _oracle_rows(rouge_result):
    method = "global_text_after_translation_to_ground_truth_language"
    return [
        {"metric": "rougeL_precision", "value": float(rouge_result.precision), "method": method},
        {"metric": "rougeL_recall", "value": float(rouge_result.recall), "method": method},
        {"metric": "rougeL_fmeasure", "value": float(rouge_result.fmeasure), "method": method},
    ]


@scenario("P01. Diferencial ROUGE-L: idioma español, textos parcialmente solapados")
def test_diff_rouge_spanish():
    gt = "El modelo previo logró resultados moderados en el conjunto de datos."
    generated = "El modelo previo logró resultados moderados en varios experimentos."

    real_scorer = build_rouge_scorer(ground_truth_language="es")
    real_result = compute_rouge_l(
        ground_truth_plain_text=gt, generated_for_rouge=generated, scorer=real_scorer
    )
    real_rows = build_rouge_metric_rows(real_result)

    oracle_result = _oracle_rouge_l(gt, generated, "es")
    oracle_rows = _oracle_rows(oracle_result)

    assert real_rows == oracle_rows
    assert (real_result.precision, real_result.recall, real_result.fmeasure) == (
        oracle_result.precision,
        oracle_result.recall,
        oracle_result.fmeasure,
    )


@scenario("P02. Diferencial ROUGE-L: idioma inglés (use_stemmer=True), textos distintos")
def test_diff_rouge_english_stemmer():
    gt = "The proposed method achieved better results than previous approaches."
    generated = "The proposed methods achieve improved results compared to prior approaches."

    real_scorer = build_rouge_scorer(ground_truth_language="en")
    real_result = compute_rouge_l(
        ground_truth_plain_text=gt, generated_for_rouge=generated, scorer=real_scorer
    )
    real_rows = build_rouge_metric_rows(real_result)

    oracle_result = _oracle_rouge_l(gt, generated, "en")
    oracle_rows = _oracle_rows(oracle_result)

    assert real_rows == oracle_rows


@scenario("P03. Diferencial ROUGE-L: textos idénticos, ambos caminos dan 1.0 en las 3 métricas")
def test_diff_rouge_identical():
    text = "Texto exactamente igual en ambos lados de la comparación."
    real_scorer = build_rouge_scorer(ground_truth_language="es")
    real_result = compute_rouge_l(
        ground_truth_plain_text=text, generated_for_rouge=text, scorer=real_scorer
    )
    oracle_result = _oracle_rouge_l(text, text, "es")
    assert build_rouge_metric_rows(real_result) == _oracle_rows(oracle_result)
    assert real_result.fmeasure == oracle_result.fmeasure == 1.0


if __name__ == "__main__":
    for fn in (test_diff_rouge_spanish, test_diff_rouge_english_stemmer, test_diff_rouge_identical):
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
