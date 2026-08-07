"""Pruebas diferenciales del Bloque 4C: oráculo reproducido vs. módulo real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.bertscore import (
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


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


# ---------------------------------------------------------------------------
# Oráculo independiente (celda 17), sin compartir código con el módulo.
# ---------------------------------------------------------------------------


def _oracle_select_indices(generated_chunks, ground_truth_chunks, max_bertscore_pairs):
    precision_pair_indices = np.linspace(
        0,
        len(generated_chunks) - 1,
        num=min(len(generated_chunks), max(1, max_bertscore_pairs // 2)),
        dtype=int,
    )
    recall_pair_indices = np.linspace(
        0,
        len(ground_truth_chunks) - 1,
        num=min(
            len(ground_truth_chunks),
            max(1, max_bertscore_pairs - len(precision_pair_indices)),
        ),
        dtype=int,
    )
    return precision_pair_indices, recall_pair_indices


def _oracle_build_pairs(generated_chunks, ground_truth_chunks, semantic_matrix, precision_pair_indices, recall_pair_indices):
    candidates, references, metadata = [], [], []
    for generated_index in precision_pair_indices:
        generated_index = int(generated_index)
        gt_index = int(semantic_matrix[generated_index].argmax())
        candidates.append(generated_chunks[generated_index])
        references.append(ground_truth_chunks[gt_index])
        metadata.append(
            {
                "direction": "generated_to_ground_truth",
                "generated_chunk_index": generated_index,
                "ground_truth_chunk_index": gt_index,
                "semantic_similarity": float(semantic_matrix[generated_index, gt_index]),
            }
        )
    for gt_index in recall_pair_indices:
        gt_index = int(gt_index)
        generated_index = int(semantic_matrix[:, gt_index].argmax())
        candidates.append(generated_chunks[generated_index])
        references.append(ground_truth_chunks[gt_index])
        metadata.append(
            {
                "direction": "ground_truth_to_generated",
                "generated_chunk_index": generated_index,
                "ground_truth_chunk_index": gt_index,
                "semantic_similarity": float(semantic_matrix[generated_index, gt_index]),
            }
        )
    return candidates, references, metadata


def _oracle_run_bertscore(candidates, references, bertscore_model, score_fn):
    if not candidates:
        raise ValueError("No se construyeron pares para BERTScore.")
    return score_fn(
        candidates, references, model_type=bertscore_model, verbose=True, batch_size=8, rescale_with_baseline=False
    )


def _oracle_aggregate(p, r, f1):
    return float(p.mean()), float(r.mean()), float(f1.mean())


def _oracle_enrich(pair_metadata, candidates, references, p, r, f1):
    for index, metadata in enumerate(pair_metadata):
        metadata.update(
            {
                "bertscore_precision": float(p[index]),
                "bertscore_recall": float(r[index]),
                "bertscore_f1": float(f1[index]),
                "generated_preview": candidates[index][:300],
                "ground_truth_preview": references[index][:300],
            }
        )
    return pair_metadata


def _oracle_metric_rows(bertscore_precision, bertscore_recall, bertscore_f1):
    method = "bidirectional_semantically_aligned_chunks"
    return [
        {"metric": "bertscore_precision", "value": bertscore_precision, "method": method},
        {"metric": "bertscore_recall", "value": bertscore_recall, "method": method},
        {"metric": "bertscore_f1", "value": bertscore_f1, "method": method},
    ]


@scenario("K01. Diferencial: índices idénticos en varios repartos (par/impar/uno/máximo alto)")
def test_diff_select_indices():
    cases = [
        (["G0", "G1", "G2"], ["T0", "T1", "T2"], 4),
        (list("ABCDEFGH"), list("ABCDEFGH"), 5),
        (list("ABCDE"), list("ABCDE"), 1),
        (["G0", "G1", "G2"], ["T0", "T1"], 100),
    ]
    for generated, ground_truth, max_pairs in cases:
        real = select_bertscore_pair_indices(
            generated_chunks=generated, ground_truth_chunks=ground_truth, max_bertscore_pairs=max_pairs
        )
        oracle = _oracle_select_indices(generated, ground_truth, max_pairs)
        assert [list(x) for x in real] == [list(x) for x in oracle]


@scenario("K02. Diferencial: candidatos, referencias y metadata inicial idénticos")
def test_diff_build_pairs():
    generated_chunks = ["Gen 0", "Gen 1", "Gen 2"]
    ground_truth_chunks = ["GT 0", "GT 1"]
    matrix = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    precision_idx, recall_idx = select_bertscore_pair_indices(
        generated_chunks=generated_chunks, ground_truth_chunks=ground_truth_chunks, max_bertscore_pairs=4
    )

    real_candidates, real_references, real_metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix,
        precision_pair_indices=precision_idx,
        recall_pair_indices=recall_idx,
    )
    oracle_candidates, oracle_references, oracle_metadata = _oracle_build_pairs(
        generated_chunks, ground_truth_chunks, matrix, precision_idx, recall_idx
    )
    assert real_candidates == oracle_candidates
    assert real_references == oracle_references
    assert real_metadata == oracle_metadata


@scenario("K03. Diferencial: argumentos enviados al scorer idénticos en ambos caminos")
def test_diff_scorer_arguments():
    real_calls = []
    oracle_calls = []

    def real_scorer(candidates, references, **kwargs):
        real_calls.append((list(candidates), list(references), kwargs))
        return _FakeTensor([0.5] * len(candidates)), _FakeTensor([0.5] * len(candidates)), _FakeTensor([0.5] * len(candidates))

    def oracle_scorer(candidates, references, **kwargs):
        oracle_calls.append((list(candidates), list(references), kwargs))
        return _FakeTensor([0.5] * len(candidates)), _FakeTensor([0.5] * len(candidates)), _FakeTensor([0.5] * len(candidates))

    run_bertscore(candidates=["a", "b"], references=["c", "d"], bertscore_model="modelX", bertscore_score_fn=real_scorer)
    _oracle_run_bertscore(["a", "b"], ["c", "d"], "modelX", oracle_scorer)

    assert real_calls == oracle_calls


@scenario("K04. Diferencial: promedios idénticos")
def test_diff_aggregate():
    p, r, f1 = _FakeTensor([0.1, 0.9]), _FakeTensor([0.2, 0.8]), _FakeTensor([0.3, 0.7])
    real = aggregate_bertscore(p, r, f1)
    oracle = _oracle_aggregate(p, r, f1)
    assert real == oracle


@scenario("K05. Diferencial: metadata enriquecida idéntica")
def test_diff_enrich_metadata():
    pair_metadata_real = [
        {"direction": "generated_to_ground_truth", "generated_chunk_index": 0, "ground_truth_chunk_index": 1, "semantic_similarity": 0.7}
    ]
    pair_metadata_oracle = [dict(pair_metadata_real[0])]
    candidates = ["Texto candidato de prueba" * 20]
    references = ["Texto referencia de prueba" * 20]
    p, r, f1 = _FakeTensor([0.6]), _FakeTensor([0.65]), _FakeTensor([0.62])

    real = enrich_bertscore_pair_metadata(
        pair_metadata=pair_metadata_real, candidates=candidates, references=references,
        precision_values=p, recall_values=r, f1_values=f1,
    )
    oracle = _oracle_enrich(pair_metadata_oracle, candidates, references, p, r, f1)
    assert real == oracle


@scenario("K06. Diferencial: filas finales idénticas")
def test_diff_metric_rows():
    real = build_bertscore_metric_rows(bertscore_precision=0.55, bertscore_recall=0.61, bertscore_f1=0.578)
    oracle = _oracle_metric_rows(0.55, 0.61, 0.578)
    assert real == oracle


@scenario("K07. Diferencial COMPLETA: flujo end-to-end con dobles, cantidades desiguales")
def test_diff_full_pipeline():
    generated_chunks = ["Gen A", "Gen B", "Gen C", "Gen D"]
    ground_truth_chunks = ["GT X", "GT Y"]
    matrix = np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
    max_bertscore_pairs = 5

    def make_scorer():
        def scorer(candidates, references, **kwargs):
            n = len(candidates)
            return _FakeTensor([0.5] * n), _FakeTensor([0.6] * n), _FakeTensor([0.55] * n)

        return scorer

    # Camino real
    p_idx, r_idx = select_bertscore_pair_indices(
        generated_chunks=generated_chunks, ground_truth_chunks=ground_truth_chunks, max_bertscore_pairs=max_bertscore_pairs
    )
    cands, refs, meta = build_bertscore_pairs(
        generated_chunks=generated_chunks, ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=matrix, precision_pair_indices=p_idx, recall_pair_indices=r_idx,
    )
    pv, rv, fv = run_bertscore(candidates=cands, references=refs, bertscore_model="m", bertscore_score_fn=make_scorer())
    agg = aggregate_bertscore(pv, rv, fv)
    enriched = enrich_bertscore_pair_metadata(
        pair_metadata=meta, candidates=cands, references=refs, precision_values=pv, recall_values=rv, f1_values=fv
    )
    rows = build_bertscore_metric_rows(bertscore_precision=agg[0], bertscore_recall=agg[1], bertscore_f1=agg[2])

    # Camino oráculo
    o_p_idx, o_r_idx = _oracle_select_indices(generated_chunks, ground_truth_chunks, max_bertscore_pairs)
    o_cands, o_refs, o_meta = _oracle_build_pairs(generated_chunks, ground_truth_chunks, matrix, o_p_idx, o_r_idx)
    o_pv, o_rv, o_fv = _oracle_run_bertscore(o_cands, o_refs, "m", make_scorer())
    o_agg = _oracle_aggregate(o_pv, o_rv, o_fv)
    o_enriched = _oracle_enrich(o_meta, o_cands, o_refs, o_pv, o_rv, o_fv)
    o_rows = _oracle_metric_rows(o_agg[0], o_agg[1], o_agg[2])

    assert cands == o_cands
    assert refs == o_refs
    assert enriched == o_enriched
    assert agg == o_agg
    assert rows == o_rows


if __name__ == "__main__":
    for fn in (
        test_diff_select_indices,
        test_diff_build_pairs,
        test_diff_scorer_arguments,
        test_diff_aggregate,
        test_diff_enrich_metadata,
        test_diff_metric_rows,
        test_diff_full_pipeline,
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
