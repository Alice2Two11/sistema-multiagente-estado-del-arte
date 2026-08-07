"""Pruebas diferenciales del Bloque 4B: oráculo reproducido vs. módulo real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.semantic_similarity import (
    build_document_embedding,
    build_semantic_alignment_rows,
    build_semantic_chunks,
    build_semantic_metric_rows,
    compute_global_semantic_similarity,
    compute_semantic_precision_recall_f1,
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


def _oracle_evenly_spaced_items(items, maximum):
    if len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=int)
    return [items[int(index)] for index in indices]


def _oracle_precision_recall_f1(similarity_matrix):
    generated_best_scores = similarity_matrix.max(axis=1)
    ground_truth_best_scores = similarity_matrix.max(axis=0)
    precision = float(generated_best_scores.mean())
    recall = float(ground_truth_best_scores.mean())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _oracle_alignment_rows(similarity_matrix, generated_chunks, ground_truth_chunks):
    generated_best_scores = similarity_matrix.max(axis=1)
    ground_truth_best_scores = similarity_matrix.max(axis=0)
    rows = []
    for generated_index, score in enumerate(generated_best_scores):
        best_gt_index = int(similarity_matrix[generated_index].argmax())
        rows.append(
            {
                "direction": "generated_to_ground_truth",
                "source_chunk_index": generated_index,
                "matched_chunk_index": best_gt_index,
                "similarity": float(score),
                "source_preview": generated_chunks[generated_index][:300],
                "matched_preview": ground_truth_chunks[best_gt_index][:300],
            }
        )
    for gt_index, score in enumerate(ground_truth_best_scores):
        best_generated_index = int(similarity_matrix[:, gt_index].argmax())
        rows.append(
            {
                "direction": "ground_truth_to_generated",
                "source_chunk_index": gt_index,
                "matched_chunk_index": best_generated_index,
                "similarity": float(score),
                "source_preview": ground_truth_chunks[gt_index][:300],
                "matched_preview": generated_chunks[best_generated_index][:300],
            }
        )
    return rows


def _oracle_metric_rows(semantic_precision, semantic_recall, semantic_f1, global_semantic_similarity):
    return [
        {
            "metric": "semantic_precision",
            "value": semantic_precision,
            "method": "mean_best_generated_to_ground_truth_chunk_similarity",
        },
        {
            "metric": "semantic_recall",
            "value": semantic_recall,
            "method": "mean_best_ground_truth_to_generated_chunk_similarity",
        },
        {
            "metric": "semantic_f1",
            "value": semantic_f1,
            "method": "harmonic_mean_semantic_precision_recall",
        },
        {
            "metric": "global_semantic_similarity",
            "value": global_semantic_similarity,
            "method": "cosine_similarity_mean_document_embeddings",
        },
    ]


@scenario("Q01. Diferencial: evenly_spaced_items idéntico en varios tamaños")
def test_diff_evenly_spaced():
    for items, maximum in [
        (list(range(10)), 3),
        (list(range(5)), 10),
        (list(range(1)), 1),
        (list(range(7)), 7),
    ]:
        assert evenly_spaced_items(items, maximum) == _oracle_evenly_spaced_items(items, maximum)


@scenario("Q02. Diferencial: precision/recall/F1 semánticos idénticos, incluida la rama F1=0.0")
def test_diff_precision_recall_f1():
    matrices = [
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]),
        np.array([[0.5, 0.5]]),
    ]
    for matrix in matrices:
        real = compute_semantic_precision_recall_f1(matrix)
        oracle = _oracle_precision_recall_f1(matrix)
        assert real == oracle, matrix


@scenario("Q03. Diferencial: filas de alineación idénticas, incluidos previews truncados")
def test_diff_alignment_rows():
    matrix = np.array([[0.1, 0.9, 0.3], [0.8, 0.2, 0.4]])
    generated_chunks = ["Generado A" * 40, "Generado B"]
    ground_truth_chunks = ["Referencia A", "Referencia B" * 40, "Referencia C"]

    real = build_semantic_alignment_rows(
        similarity_matrix=matrix,
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        generated_best_scores=matrix.max(axis=1),
        ground_truth_best_scores=matrix.max(axis=0),
    )
    oracle = _oracle_alignment_rows(matrix, generated_chunks, ground_truth_chunks)
    assert real == oracle


@scenario("Q04. Diferencial: filas de métricas semánticas idénticas")
def test_diff_metric_rows():
    real = build_semantic_metric_rows(
        semantic_precision=0.62, semantic_recall=0.71, semantic_f1=0.662, global_semantic_similarity=0.58
    )
    oracle = _oracle_metric_rows(0.62, 0.71, 0.662, 0.58)
    assert real == oracle


@scenario("Q05. Diferencial: similitud global y embedding de documento idénticos, incluido vector cero")
def test_diff_global_similarity_and_document_embedding():
    chunk_embeddings = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    doc_embedding = build_document_embedding(chunk_embeddings)
    oracle_doc_embedding = chunk_embeddings.mean(axis=0)
    assert np.allclose(doc_embedding, oracle_doc_embedding)

    other_doc = np.array([0.0, 0.0])
    real_similarity = compute_global_semantic_similarity(doc_embedding, other_doc)
    from sklearn.metrics.pairwise import cosine_similarity

    oracle_similarity = float(cosine_similarity([doc_embedding], [other_doc])[0][0])
    assert real_similarity == oracle_similarity == 0.0


@scenario("Q06. Diferencial: build_semantic_chunks idéntico en tamaño y contenido")
def test_diff_semantic_chunks():
    text = "Primera oracion aqui. " * 15 + "Segunda oracion diferente aqui. " * 15
    real = build_semantic_chunks(
        text,
        semantic_chunk_chars=50,
        semantic_chunk_overlap_chars=10,
        max_semantic_chunks_per_text=4,
    )

    from src.tools.evaluation.language_preprocessing import chunk_text_by_sentences

    all_chunks = chunk_text_by_sentences(text, 50, 10)
    oracle = _oracle_evenly_spaced_items(all_chunks, 4)
    assert real == oracle


if __name__ == "__main__":
    for fn in (
        test_diff_evenly_spaced,
        test_diff_precision_recall_f1,
        test_diff_alignment_rows,
        test_diff_metric_rows,
        test_diff_global_similarity_and_document_embedding,
        test_diff_semantic_chunks,
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
