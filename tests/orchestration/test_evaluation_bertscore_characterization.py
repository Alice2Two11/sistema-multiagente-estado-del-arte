"""Pruebas de caracterización del Bloque 4C: BERTScore sobre pares alineados."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.bertscore import (
    METHOD_LABEL,
    aggregate_bertscore,
    build_bertscore_metric_rows,
    build_bertscore_pairs,
    enrich_bertscore_pair_metadata,
    run_bertscore,
    select_bertscore_pair_indices,
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


class FakeTensor:
    """Doble mínimo de un tensor: soporta indexado, .mean() y len()."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


def fake_bertscore_score_fn(responses):
    """Doble determinista de bert_score.score: registra los argumentos
    recibidos y devuelve 3 FakeTensor con valores prefijados."""

    calls = []

    def _fn(candidates, references, *, model_type, verbose, batch_size, rescale_with_baseline):
        calls.append(
            {
                "candidates": list(candidates),
                "references": list(references),
                "model_type": model_type,
                "verbose": verbose,
                "batch_size": batch_size,
                "rescale_with_baseline": rescale_with_baseline,
            }
        )
        n = len(candidates)
        p, r, f1 = responses
        assert len(p) == len(r) == len(f1) == n, "el doble debe recibir tantos valores como pares"
        return FakeTensor(p), FakeTensor(r), FakeTensor(f1)

    _fn.calls = calls
    return _fn


# ---------------------------------------------------------------------------
# select_bertscore_pair_indices
# ---------------------------------------------------------------------------


@scenario("B01. select_bertscore_pair_indices: un chunk por lado")
def test_select_indices_single_chunk_each_side():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=["G0"], ground_truth_chunks=["T0"], max_bertscore_pairs=4
    )
    assert list(precision_idx) == [0]
    assert list(recall_idx) == [0]


@scenario("B02. select_bertscore_pair_indices: cantidades iguales")
def test_select_indices_equal_counts():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=["G0", "G1", "G2"],
        ground_truth_chunks=["T0", "T1", "T2"],
        max_bertscore_pairs=4,
    )
    assert len(precision_idx) == 2  # max(1, 4//2) = 2
    assert len(recall_idx) == 2  # max(1, 4-2) = 2


@scenario("B03. select_bertscore_pair_indices: cantidades desiguales")
def test_select_indices_unequal_counts():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=["G0", "G1", "G2", "G3", "G4"],
        ground_truth_chunks=["T0", "T1"],
        max_bertscore_pairs=6,
    )
    assert len(precision_idx) == 3  # min(5, max(1,6//2=3)) = 3
    assert len(recall_idx) == 2  # min(2, max(1,6-3=3)) = 2 (capado por len(gt)=2)


@scenario("B04. select_bertscore_pair_indices: MAX_BERTSCORE_PAIRS par")
def test_select_indices_even_limit():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=list("ABCDEFGH"), ground_truth_chunks=list("ABCDEFGH"), max_bertscore_pairs=4
    )
    assert len(precision_idx) == 2
    assert len(recall_idx) == 2


@scenario("B05. select_bertscore_pair_indices: MAX_BERTSCORE_PAIRS impar -> reparto asimétrico (recall recibe uno más)")
def test_select_indices_odd_limit_asymmetric():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=list("ABCDEFGH"), ground_truth_chunks=list("ABCDEFGH"), max_bertscore_pairs=5
    )
    assert len(precision_idx) == 2  # 5 // 2 = 2
    assert len(recall_idx) == 3  # 5 - 2 = 3


@scenario("B06. select_bertscore_pair_indices: MAX_BERTSCORE_PAIRS=1 -> igual produce 1 índice por rama (min 1+1=2 pares totales)")
def test_select_indices_max_pairs_one():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=list("ABCDE"), ground_truth_chunks=list("ABCDE"), max_bertscore_pairs=1
    )
    assert len(precision_idx) == 1  # max(1, 1//2=0) = 1
    assert len(recall_idx) == 1  # max(1, 1-1=0) = 1
    # Confirma el hallazgo documentado: con MAX_BERTSCORE_PAIRS=1 se generan
    # 2 pares totales (1+1), no 1 -- comportamiento real preservado.


@scenario("B07. select_bertscore_pair_indices: máximo mayor que el número de chunks disponibles")
def test_select_indices_max_greater_than_available():
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=["G0", "G1", "G2"], ground_truth_chunks=["T0", "T1"], max_bertscore_pairs=100
    )
    assert len(precision_idx) == 3  # capado por len(generated_chunks)
    assert len(recall_idx) == 2  # capado por len(ground_truth_chunks)


# ---------------------------------------------------------------------------
# build_bertscore_pairs
# ---------------------------------------------------------------------------


@scenario("B08. build_bertscore_pairs: candidato/referencia y metadata correctos, alineación por fila y columna")
def test_build_pairs_basic():
    generated_chunks = ["Gen 0", "Gen 1"]
    ground_truth_chunks = ["GT 0", "GT 1"]
    matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    precision_idx = np.array([0, 1])
    recall_idx = np.array([0, 1])

    candidates, references, metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=precision_idx,
        recall_pair_indices=recall_idx,
    )
    assert len(candidates) == len(references) == len(metadata) == 4

    # Primer par (precision, generated_index=0): mejor match en fila 0 -> col 1
    assert metadata[0]["direction"] == "generated_to_ground_truth"
    assert metadata[0]["ground_truth_chunk_index"] == 1
    assert candidates[0] == "Gen 0" and references[0] == "GT 1"

    # Tercer par (recall, gt_index=0): mejor match en columna 0 -> fila 1
    assert metadata[2]["direction"] == "ground_truth_to_generated"
    assert metadata[2]["generated_chunk_index"] == 1
    assert candidates[2] == "Gen 1" and references[2] == "GT 0"


@scenario("B09. build_bertscore_pairs: candidato siempre es generado y referencia siempre es Ground Truth, en ambas direcciones")
def test_build_pairs_candidate_reference_order_never_swaps():
    generated_chunks = ["Gen 0"]
    ground_truth_chunks = ["GT 0"]
    matrix = np.array([[0.5]])
    candidates, references, _ = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=np.array([0]),
        recall_pair_indices=np.array([0]),
    )
    assert all(c == "Gen 0" for c in candidates)
    assert all(r == "GT 0" for r in references)


@scenario("B10. build_bertscore_pairs: no deduplica pares repetidos entre direcciones")
def test_build_pairs_no_deduplication():
    # Con esta matriz, el mejor match de Gen0 es GT0 y el mejor match de GT0
    # es Gen0 -> el mismo par (0,0) aparece en ambas direcciones.
    generated_chunks = ["Gen 0"]
    ground_truth_chunks = ["GT 0"]
    matrix = np.array([[1.0]])
    candidates, references, metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=np.array([0]),
        recall_pair_indices=np.array([0]),
    )
    assert len(candidates) == 2  # NO deduplicado: aparece dos veces
    assert candidates == ["Gen 0", "Gen 0"]
    assert references == ["GT 0", "GT 0"]


@scenario("B11. build_bertscore_pairs: empates en la matriz -> argmax determinista (primer índice)")
def test_build_pairs_ties_in_matrix():
    generated_chunks = ["Gen 0"]
    ground_truth_chunks = ["GT 0", "GT 1"]
    matrix = np.array([[0.5, 0.5]])
    candidates, references, metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=np.array([0]),
        recall_pair_indices=np.array([]),
    )
    assert metadata[0]["ground_truth_chunk_index"] == 0  # numpy.argmax rompe empates por el primer índice


@scenario("B12. build_bertscore_pairs: conserva el orden (precision primero, luego recall)")
def test_build_pairs_order_preserved():
    generated_chunks = ["G0", "G1"]
    ground_truth_chunks = ["T0", "T1"]
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    _, _, metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=np.array([0, 1]),
        recall_pair_indices=np.array([0, 1]),
    )
    directions = [m["direction"] for m in metadata]
    assert directions == [
        "generated_to_ground_truth",
        "generated_to_ground_truth",
        "ground_truth_to_generated",
        "ground_truth_to_generated",
    ]


# ---------------------------------------------------------------------------
# run_bertscore
# ---------------------------------------------------------------------------


@scenario("B13. run_bertscore: ausencia de candidatos lanza ValueError, sin invocar el scorer")
def test_run_bertscore_empty_candidates_raises():
    factory_calls = []

    def scorer(*args, **kwargs):
        factory_calls.append(1)
        return None

    try:
        run_bertscore(candidates=[], references=[], bertscore_model="x", bertscore_score_fn=scorer)
    except ValueError as exc:
        assert "No se construyeron pares" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError")
    assert factory_calls == []


@scenario("B14. run_bertscore: argumentos exactos enviados al scorer inyectado (batch_size=8, verbose=True, rescale_with_baseline=False)")
def test_run_bertscore_exact_arguments():
    fn = fake_bertscore_score_fn(([0.9], [0.8], [0.85]))
    run_bertscore(
        candidates=["cand"], references=["ref"], bertscore_model="bert-base-multilingual-cased", bertscore_score_fn=fn
    )
    assert fn.calls == [
        {
            "candidates": ["cand"],
            "references": ["ref"],
            "model_type": "bert-base-multilingual-cased",
            "verbose": True,
            "batch_size": 8,
            "rescale_with_baseline": False,
        }
    ]


@scenario("B15. run_bertscore: propaga errores del scorer inyectado, sin silenciarlos")
def test_run_bertscore_propagates_scorer_errors():
    def failing_scorer(*args, **kwargs):
        raise RuntimeError("Fallo simulado del scorer BERTScore.")

    try:
        run_bertscore(candidates=["a"], references=["b"], bertscore_model="x", bertscore_score_fn=failing_scorer)
    except RuntimeError as exc:
        assert "simulado" in str(exc)
    else:
        raise AssertionError("debía propagar el error del scorer")


@scenario("B16. run_bertscore: devuelve tensores/arrays tal cual (sin post-procesar)")
def test_run_bertscore_returns_raw_tensors():
    fn = fake_bertscore_score_fn(([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]))
    p, r, f1 = run_bertscore(
        candidates=["a", "b"], references=["c", "d"], bertscore_model="x", bertscore_score_fn=fn
    )
    assert list(p) == [0.1, 0.2]
    assert list(r) == [0.3, 0.4]
    assert list(f1) == [0.5, 0.6]


# ---------------------------------------------------------------------------
# aggregate_bertscore
# ---------------------------------------------------------------------------


@scenario("B17. aggregate_bertscore: promedios correctos")
def test_aggregate_bertscore_means():
    p = FakeTensor([0.8, 0.6])
    r = FakeTensor([0.9, 0.7])
    f1 = FakeTensor([0.85, 0.65])
    result = aggregate_bertscore(p, r, f1)
    assert result == (0.7, 0.8, 0.75)


# ---------------------------------------------------------------------------
# enrich_bertscore_pair_metadata
# ---------------------------------------------------------------------------


@scenario("B18. enrich_bertscore_pair_metadata: valores individuales agregados correctamente por índice")
def test_enrich_metadata_individual_values():
    pair_metadata = [
        {"direction": "generated_to_ground_truth", "generated_chunk_index": 0, "ground_truth_chunk_index": 0, "semantic_similarity": 0.9},
        {"direction": "ground_truth_to_generated", "generated_chunk_index": 0, "ground_truth_chunk_index": 0, "semantic_similarity": 0.9},
    ]
    candidates = ["Cand 0", "Cand 1"]
    references = ["Ref 0", "Ref 1"]
    p = FakeTensor([0.1, 0.2])
    r = FakeTensor([0.3, 0.4])
    f1 = FakeTensor([0.5, 0.6])

    enriched = enrich_bertscore_pair_metadata(
        pair_metadata=pair_metadata, candidates=candidates, references=references,
        precision_values=p, recall_values=r, f1_values=f1,
    )
    assert enriched[0]["bertscore_precision"] == 0.1
    assert enriched[0]["bertscore_recall"] == 0.3
    assert enriched[0]["bertscore_f1"] == 0.5
    assert enriched[1]["bertscore_precision"] == 0.2


@scenario("B19. enrich_bertscore_pair_metadata: previews truncados a 300 caracteres")
def test_enrich_metadata_preview_truncation():
    long_text = "y" * 500
    pair_metadata = [{"direction": "generated_to_ground_truth", "generated_chunk_index": 0, "ground_truth_chunk_index": 0, "semantic_similarity": 1.0}]
    p, r, f1 = FakeTensor([1.0]), FakeTensor([1.0]), FakeTensor([1.0])
    enriched = enrich_bertscore_pair_metadata(
        pair_metadata=pair_metadata, candidates=[long_text], references=[long_text],
        precision_values=p, recall_values=r, f1_values=f1,
    )
    assert len(enriched[0]["generated_preview"]) == 300
    assert len(enriched[0]["ground_truth_preview"]) == 300


@scenario("B20. enrich_bertscore_pair_metadata: conserva las claves originales (direction, índices, semantic_similarity)")
def test_enrich_metadata_preserves_original_keys():
    pair_metadata = [{"direction": "generated_to_ground_truth", "generated_chunk_index": 3, "ground_truth_chunk_index": 5, "semantic_similarity": 0.42}]
    p, r, f1 = FakeTensor([0.1]), FakeTensor([0.2]), FakeTensor([0.3])
    enriched = enrich_bertscore_pair_metadata(
        pair_metadata=pair_metadata, candidates=["c"], references=["r"], precision_values=p, recall_values=r, f1_values=f1,
    )
    assert enriched[0]["direction"] == "generated_to_ground_truth"
    assert enriched[0]["generated_chunk_index"] == 3
    assert enriched[0]["ground_truth_chunk_index"] == 5
    assert enriched[0]["semantic_similarity"] == 0.42


# ---------------------------------------------------------------------------
# build_bertscore_metric_rows
# ---------------------------------------------------------------------------


@scenario("B21. build_bertscore_metric_rows: 3 filas exactas con el method literal del notebook")
def test_metric_rows_shape():
    rows = build_bertscore_metric_rows(bertscore_precision=0.8, bertscore_recall=0.7, bertscore_f1=0.746)
    assert [r["metric"] for r in rows] == ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    assert all(r["method"] == METHOD_LABEL for r in rows)
    assert all(r["method"] == "bidirectional_semantically_aligned_chunks" for r in rows)
    assert [r["value"] for r in rows] == [0.8, 0.7, 0.746]


if __name__ == "__main__":
    for fn in (
        test_select_indices_single_chunk_each_side,
        test_select_indices_equal_counts,
        test_select_indices_unequal_counts,
        test_select_indices_even_limit,
        test_select_indices_odd_limit_asymmetric,
        test_select_indices_max_pairs_one,
        test_select_indices_max_greater_than_available,
        test_build_pairs_basic,
        test_build_pairs_candidate_reference_order_never_swaps,
        test_build_pairs_no_deduplication,
        test_build_pairs_ties_in_matrix,
        test_build_pairs_order_preserved,
        test_run_bertscore_empty_candidates_raises,
        test_run_bertscore_exact_arguments,
        test_run_bertscore_propagates_scorer_errors,
        test_run_bertscore_returns_raw_tensors,
        test_aggregate_bertscore_means,
        test_enrich_metadata_individual_values,
        test_enrich_metadata_preview_truncation,
        test_enrich_metadata_preserves_original_keys,
        test_metric_rows_shape,
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
