"""Pruebas de caracterización y diferenciales del Bloque 4A (ROUGE-L).

Usa ``rouge_score`` REAL (no un doble) — es una librería determinista local,
sin red ni modelo descargado en el caso base (``use_stemmer=False``); no
hay razón para no ejercitarla de verdad, a diferencia de OpenAI/embeddings.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.rouge import (
    METHOD_LABEL,
    build_rouge_metric_rows,
    build_rouge_scorer,
    compute_rouge_l,
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


# ---------------------------------------------------------------------------
# build_rouge_scorer
# ---------------------------------------------------------------------------


@scenario("R01. build_rouge_scorer: use_stemmer=True cuando el idioma del GT es inglés")
def test_scorer_stemmer_english():
    calls = []

    def fake_factory(metrics, use_stemmer):
        calls.append((metrics, use_stemmer))
        return object()

    build_rouge_scorer(ground_truth_language="en", scorer_factory=fake_factory)
    assert calls == [(["rougeL"], True)]


@scenario("R02. build_rouge_scorer: use_stemmer=False cuando el idioma del GT es español")
def test_scorer_stemmer_spanish():
    calls = []

    def fake_factory(metrics, use_stemmer):
        calls.append((metrics, use_stemmer))
        return object()

    build_rouge_scorer(ground_truth_language="es", scorer_factory=fake_factory)
    assert calls == [(["rougeL"], False)]


@scenario("R03. build_rouge_scorer: use_stemmer=False para cualquier idioma que no sea 'en' (no solo 'es')")
def test_scorer_stemmer_other_language():
    calls = []

    def fake_factory(metrics, use_stemmer):
        calls.append((metrics, use_stemmer))
        return object()

    build_rouge_scorer(ground_truth_language="fr", scorer_factory=fake_factory)
    assert calls == [(["rougeL"], False)]


@scenario("R04. build_rouge_scorer real (sin factory inyectada) construye un RougeScorer de verdad")
def test_scorer_real_construction():
    scorer = build_rouge_scorer(ground_truth_language="es")
    assert hasattr(scorer, "score")


# ---------------------------------------------------------------------------
# compute_rouge_l — con rouge_score REAL
# ---------------------------------------------------------------------------


@scenario("R05. compute_rouge_l: textos idénticos -> precisión/recall/F1 = 1.0")
def test_compute_identical_texts():
    text = "El modelo alcanzó un desempeño superior en el conjunto de datos evaluado."
    scorer = build_rouge_scorer(ground_truth_language="es")
    result = compute_rouge_l(
        ground_truth_plain_text=text, generated_for_rouge=text, scorer=scorer
    )
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.fmeasure == 1.0


@scenario("R06. compute_rouge_l: textos completamente distintos -> puntajes bajos, no lanza")
def test_compute_completely_different_texts():
    scorer = build_rouge_scorer(ground_truth_language="es")
    result = compute_rouge_l(
        ground_truth_plain_text="Perro gato pájaro casa árbol.",
        generated_for_rouge="Física cuántica termodinámica electromagnetismo óptica.",
        scorer=scorer,
    )
    assert result.fmeasure < 0.3


@scenario("R07. compute_rouge_l: orden de argumentos preservado (referencia primero, hipótesis segundo)")
def test_compute_argument_order_matters():
    # Con longitudes distintas, invertir el orden cambia precisión<->recall
    # -- se confirma que el módulo preserva el orden real (GT primero).
    reference = "Uno dos tres cuatro cinco."
    hypothesis = "Uno dos tres cuatro cinco seis siete ocho."
    scorer = build_rouge_scorer(ground_truth_language="es")

    real_order = compute_rouge_l(
        ground_truth_plain_text=reference, generated_for_rouge=hypothesis, scorer=scorer
    )
    inverted_order = compute_rouge_l(
        ground_truth_plain_text=hypothesis, generated_for_rouge=reference, scorer=scorer
    )
    assert real_order.precision != inverted_order.precision
    assert real_order.recall != inverted_order.recall
    # el módulo real usa (ground_truth, generated_for_rouge) -- confirmar
    # que coincide con llamar directamente a scorer.score en ese orden
    direct = scorer.score(reference, hypothesis)["rougeL"]
    assert real_order == direct


@scenario("R08. compute_rouge_l: texto generado vacío no lanza, produce puntajes bajos/cero")
def test_compute_empty_generated():
    scorer = build_rouge_scorer(ground_truth_language="es")
    result = compute_rouge_l(
        ground_truth_plain_text="Texto de referencia con contenido real.",
        generated_for_rouge="",
        scorer=scorer,
    )
    assert result.fmeasure == 0.0


# ---------------------------------------------------------------------------
# build_rouge_metric_rows
# ---------------------------------------------------------------------------


@scenario("R09. build_rouge_metric_rows: exactamente 3 filas con las claves y el method reales")
def test_metric_rows_shape():
    scorer = build_rouge_scorer(ground_truth_language="es")
    result = compute_rouge_l(
        ground_truth_plain_text="Texto de prueba con contenido.",
        generated_for_rouge="Texto de prueba con contenido similar.",
        scorer=scorer,
    )
    rows = build_rouge_metric_rows(result)
    assert len(rows) == 3
    assert [r["metric"] for r in rows] == ["rougeL_precision", "rougeL_recall", "rougeL_fmeasure"]
    assert all(r["method"] == METHOD_LABEL for r in rows)
    assert all(r["method"] == "global_text_after_translation_to_ground_truth_language" for r in rows)


@scenario("R10. build_rouge_metric_rows: valores coinciden exactamente con precision/recall/fmeasure")
def test_metric_rows_values_match():
    scorer = build_rouge_scorer(ground_truth_language="en")
    result = compute_rouge_l(
        ground_truth_plain_text="The model achieved strong results on the dataset.",
        generated_for_rouge="The model achieved strong results on this dataset today.",
        scorer=scorer,
    )
    rows = build_rouge_metric_rows(result)
    values = {r["metric"]: r["value"] for r in rows}
    assert values["rougeL_precision"] == float(result.precision)
    assert values["rougeL_recall"] == float(result.recall)
    assert values["rougeL_fmeasure"] == float(result.fmeasure)


@scenario("R11. build_rouge_metric_rows: method idéntico aunque no hubo traducción real (mismo idioma)")
def test_metric_rows_method_same_even_without_translation():
    # El notebook real no distingue "hubo traducción" vs "no hizo falta" en
    # el campo method -- se preserva tal cual, sin "corregirlo".
    scorer = build_rouge_scorer(ground_truth_language="es")
    result = compute_rouge_l(
        ground_truth_plain_text="Mismo idioma en ambos textos aquí.",
        generated_for_rouge="Mismo idioma en ambos textos aquí también.",
        scorer=scorer,
    )
    rows = build_rouge_metric_rows(result)
    assert all(r["method"] == METHOD_LABEL for r in rows)


if __name__ == "__main__":
    for fn in (
        test_scorer_stemmer_english,
        test_scorer_stemmer_spanish,
        test_scorer_stemmer_other_language,
        test_scorer_real_construction,
        test_compute_identical_texts,
        test_compute_completely_different_texts,
        test_compute_argument_order_matters,
        test_compute_empty_generated,
        test_metric_rows_shape,
        test_metric_rows_values_match,
        test_metric_rows_method_same_even_without_translation,
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
