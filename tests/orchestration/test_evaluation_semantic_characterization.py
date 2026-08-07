"""Pruebas de caracterización del Bloque 4B: similitud semántica por embeddings.

Usa vectores/matrices sintéticos de baja dimensión y un ``model_factory``
inyectado (nunca el modelo productivo real) para probar la lógica de
alineación/cómputo sin descargar ningún modelo de embeddings.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.semantic_similarity import (
    build_document_embedding,
    build_embedding_model,
    build_semantic_alignment_rows,
    build_semantic_chunks,
    build_semantic_metric_rows,
    compute_global_semantic_similarity,
    compute_semantic_precision_recall_f1,
    compute_similarity_matrix,
    encode_chunks,
    evenly_spaced_items,
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


class FakeEmbeddingModel:
    """Doble determinista: mapea cada texto a un vector fijo predefinido
    (por contenido exacto), sin red ni descarga de modelo."""

    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text
        self.encode_calls = []

    def encode(self, chunks, *, normalize_embeddings, show_progress_bar):
        self.encode_calls.append((list(chunks), normalize_embeddings, show_progress_bar))
        return np.array([self.vectors_by_text[c] for c in chunks], dtype=float)


# ---------------------------------------------------------------------------
# evenly_spaced_items
# ---------------------------------------------------------------------------


@scenario("S01. evenly_spaced_items: si hay menos o igual que el máximo, devuelve todo tal cual")
def test_evenly_spaced_under_max():
    items = ["a", "b", "c"]
    assert evenly_spaced_items(items, maximum=5) == items
    assert evenly_spaced_items(items, maximum=3) == items


@scenario("S02. evenly_spaced_items: muestreo uniforme, conserva primer y último elemento")
def test_evenly_spaced_uniform_sampling():
    items = list(range(10))
    result = evenly_spaced_items(items, maximum=3)
    assert result[0] == 0
    assert result[-1] == 9
    assert len(result) == 3


@scenario("S03. evenly_spaced_items: conserva el orden original")
def test_evenly_spaced_preserves_order():
    items = list(range(20))
    result = evenly_spaced_items(items, maximum=5)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# build_semantic_chunks
# ---------------------------------------------------------------------------


@scenario("S04. build_semantic_chunks: respeta semantic_chunk_chars y max_semantic_chunks_per_text")
def test_build_semantic_chunks_respects_limits():
    text = "Oracion uno aqui. " * 20
    chunks = build_semantic_chunks(
        text,
        semantic_chunk_chars=30,
        semantic_chunk_overlap_chars=0,
        max_semantic_chunks_per_text=3,
    )
    assert len(chunks) <= 3


@scenario("S05. build_semantic_chunks: texto vacío produce lista vacía (sin lanzar)")
def test_build_semantic_chunks_empty_text():
    chunks = build_semantic_chunks(
        "", semantic_chunk_chars=100, semantic_chunk_overlap_chars=0, max_semantic_chunks_per_text=5
    )
    assert chunks == []


# ---------------------------------------------------------------------------
# compute_similarity_matrix / compute_semantic_precision_recall_f1
# ---------------------------------------------------------------------------


@scenario("S06. matriz de similitud: textos idénticos -> diagonal principal en 1.0")
def test_similarity_matrix_identical_vectors():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    matrix = compute_similarity_matrix(vectors, vectors)
    assert np.allclose(np.diag(matrix), 1.0)


@scenario("S07. precision/recall/F1 semánticos: un chunk por lado, similitud perfecta -> 1.0/1.0/1.0")
def test_precision_recall_f1_single_chunk_perfect():
    matrix = np.array([[1.0]])
    precision, recall, f1 = compute_semantic_precision_recall_f1(matrix)
    assert precision == recall == f1 == 1.0


@scenario("S08. precision/recall/F1 semánticos: vectores ortogonales (similitud 0) -> F1 = 0.0 sin división por cero")
def test_precision_recall_f1_zero_similarity_no_division_error():
    matrix = np.array([[0.0]])
    precision, recall, f1 = compute_semantic_precision_recall_f1(matrix)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0  # rama explícita "else 0.0", no división por cero


@scenario("S09. precision/recall/F1 semánticos: múltiples chunks, cantidades desiguales (N != M)")
def test_precision_recall_f1_unequal_chunk_counts():
    # 3 chunks generados, 2 chunks de Ground Truth
    matrix = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    precision, recall, f1 = compute_semantic_precision_recall_f1(matrix)
    # precision = media de max por fila (axis=1): [0.9, 0.8, 0.5] -> 0.7333...
    assert abs(precision - (0.9 + 0.8 + 0.5) / 3) < 1e-9
    # recall = media de max por columna (axis=0): [0.9, 0.8] -> 0.85
    assert abs(recall - (0.9 + 0.8) / 2) < 1e-9


@scenario("S10. precision/recall/F1 semánticos: matriz con empates, argmax determinista (primer índice)")
def test_precision_recall_f1_matrix_with_ties():
    matrix = np.array([[0.5, 0.5]])
    precision, recall, f1 = compute_semantic_precision_recall_f1(matrix)
    assert precision == 0.5
    assert np.argmax(matrix[0]) == 0  # numpy.argmax rompe empates por el primer índice


@scenario("S11. build_document_embedding: promedio correcto sobre múltiples chunks")
def test_document_embedding_mean():
    chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    doc_embedding = build_document_embedding(chunk_embeddings)
    assert np.allclose(doc_embedding, [2 / 3, 2 / 3])


@scenario("S12. build_document_embedding: vector cero entre los chunks no lanza, se promedia igual")
def test_document_embedding_with_zero_vector():
    chunk_embeddings = np.array([[1.0, 0.0], [0.0, 0.0]])
    doc_embedding = build_document_embedding(chunk_embeddings)
    assert np.allclose(doc_embedding, [0.5, 0.0])


@scenario("S13. compute_global_semantic_similarity: documentos idénticos -> 1.0")
def test_global_similarity_identical():
    doc = np.array([1.0, 0.0])
    assert compute_global_semantic_similarity(doc, doc) == 1.0


@scenario("S14. compute_global_semantic_similarity: vector cero produce 0.0, no lanza ni da NaN")
def test_global_similarity_zero_vector():
    doc_generated = np.array([0.0, 0.0])
    doc_gt = np.array([1.0, 0.0])
    result = compute_global_semantic_similarity(doc_generated, doc_gt)
    assert result == 0.0  # comportamiento real de sklearn.cosine_similarity ante un vector cero
    assert not np.isnan(result)


# ---------------------------------------------------------------------------
# build_semantic_alignment_rows
# ---------------------------------------------------------------------------


@scenario("S15. build_semantic_alignment_rows: un chunk por lado, ambas direcciones presentes")
def test_alignment_rows_single_chunk_each_side():
    matrix = np.array([[0.9]])
    generated_chunks = ["Chunk generado."]
    ground_truth_chunks = ["Chunk de referencia."]
    rows = build_semantic_alignment_rows(
        similarity_matrix=matrix,
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        generated_best_scores=matrix.max(axis=1),
        ground_truth_best_scores=matrix.max(axis=0),
    )
    assert len(rows) == 2
    directions = {r["direction"] for r in rows}
    assert directions == {"generated_to_ground_truth", "ground_truth_to_generated"}
    assert all(r["similarity"] == 0.9 for r in rows)


@scenario("S16. build_semantic_alignment_rows: múltiples chunks, matched_chunk_index correcto por argmax")
def test_alignment_rows_multiple_chunks_correct_argmax():
    matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    generated_chunks = ["Gen 0", "Gen 1"]
    ground_truth_chunks = ["GT 0", "GT 1"]
    rows = build_semantic_alignment_rows(
        similarity_matrix=matrix,
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        generated_best_scores=matrix.max(axis=1),
        ground_truth_best_scores=matrix.max(axis=0),
    )
    gen_to_gt = {r["source_chunk_index"]: r for r in rows if r["direction"] == "generated_to_ground_truth"}
    assert gen_to_gt[0]["matched_chunk_index"] == 1  # fila 0: max en columna 1 (0.9)
    assert gen_to_gt[1]["matched_chunk_index"] == 0  # fila 1: max en columna 0 (0.8)


@scenario("S17. build_semantic_alignment_rows: previews truncados a 300 caracteres")
def test_alignment_rows_preview_truncation():
    long_text = "x" * 500
    matrix = np.array([[1.0]])
    rows = build_semantic_alignment_rows(
        similarity_matrix=matrix,
        generated_chunks=[long_text],
        ground_truth_chunks=[long_text],
        generated_best_scores=matrix.max(axis=1),
        ground_truth_best_scores=matrix.max(axis=0),
    )
    assert all(len(r["source_preview"]) == 300 for r in rows)
    assert all(len(r["matched_preview"]) == 300 for r in rows)


@scenario("S18. build_semantic_alignment_rows: conserva el orden (generated primero, luego ground_truth)")
def test_alignment_rows_order_preserved():
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    rows = build_semantic_alignment_rows(
        similarity_matrix=matrix,
        generated_chunks=["G0", "G1"],
        ground_truth_chunks=["T0", "T1"],
        generated_best_scores=matrix.max(axis=1),
        ground_truth_best_scores=matrix.max(axis=0),
    )
    directions = [r["direction"] for r in rows]
    assert directions == [
        "generated_to_ground_truth",
        "generated_to_ground_truth",
        "ground_truth_to_generated",
        "ground_truth_to_generated",
    ]


# ---------------------------------------------------------------------------
# build_semantic_metric_rows
# ---------------------------------------------------------------------------


@scenario("S19. build_semantic_metric_rows: 4 filas exactas con los method literales del notebook")
def test_metric_rows_shape_and_labels():
    rows = build_semantic_metric_rows(
        semantic_precision=0.8, semantic_recall=0.7, semantic_f1=0.746, global_semantic_similarity=0.75
    )
    assert [r["metric"] for r in rows] == [
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "global_semantic_similarity",
    ]
    assert [r["method"] for r in rows] == [
        "mean_best_generated_to_ground_truth_chunk_similarity",
        "mean_best_ground_truth_to_generated_chunk_similarity",
        "harmonic_mean_semantic_precision_recall",
        "cosine_similarity_mean_document_embeddings",
    ]
    assert [r["value"] for r in rows] == [0.8, 0.7, 0.746, 0.75]


# ---------------------------------------------------------------------------
# build_embedding_model / encode_chunks — con doble inyectado
# ---------------------------------------------------------------------------


@scenario("S20. build_embedding_model: factory inyectada recibe el nombre real del modelo")
def test_build_embedding_model_injected_factory():
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        return object()

    build_embedding_model(
        evaluation_embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        model_factory=fake_factory,
    )
    assert calls == ["paraphrase-multilingual-MiniLM-L12-v2"]


@scenario("S21. encode_chunks: pasa normalize_embeddings=True y show_progress_bar=True (igual que el notebook)")
def test_encode_chunks_real_arguments():
    model = FakeEmbeddingModel({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    encode_chunks(model, ["a", "b"])
    assert model.encode_calls == [(["a", "b"], True, True)]


if __name__ == "__main__":
    for fn in (
        test_evenly_spaced_under_max,
        test_evenly_spaced_uniform_sampling,
        test_evenly_spaced_preserves_order,
        test_build_semantic_chunks_respects_limits,
        test_build_semantic_chunks_empty_text,
        test_similarity_matrix_identical_vectors,
        test_precision_recall_f1_single_chunk_perfect,
        test_precision_recall_f1_zero_similarity_no_division_error,
        test_precision_recall_f1_unequal_chunk_counts,
        test_precision_recall_f1_matrix_with_ties,
        test_document_embedding_mean,
        test_document_embedding_with_zero_vector,
        test_global_similarity_identical,
        test_global_similarity_zero_vector,
        test_alignment_rows_single_chunk_each_side,
        test_alignment_rows_multiple_chunks_correct_argmax,
        test_alignment_rows_preview_truncation,
        test_alignment_rows_order_preserved,
        test_metric_rows_shape_and_labels,
        test_build_embedding_model_injected_factory,
        test_encode_chunks_real_arguments,
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
