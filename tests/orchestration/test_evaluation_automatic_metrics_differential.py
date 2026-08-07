"""Prueba diferencial del flujo COMPLETO del ensamblador de métricas automáticas.

Reproduce, de forma independiente (sin importar ``src.tools.evaluation.
automatic_metrics``), el mismo orden de la celda 17 -- traducción -> ROUGE-L
-> chunks semánticos -> embeddings -> matriz -> alineación -> BERTScore --
y compara el resultado contra el módulo real, usando los MISMOS dobles
deterministas para que ambos caminos reciban idénticos estímulos externos.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.automatic_metrics import build_automatic_metrics
from tests.orchestration.test_evaluation_automatic_metrics_integration import (
    DEFAULT_POLICY,
    TEXT_A,
    TEXT_B,
    FakeEmbeddingModel,
    FakeTensor,
    make_bertscore_score_fn,
)
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


# ---------------------------------------------------------------------------
# Oráculo independiente: reproduce la celda 17 completa por su cuenta.
# ---------------------------------------------------------------------------


def _oracle_safe_str(value):
    return "" if value is None else str(value).strip()


def _oracle_split_sentences(text):
    candidates = re.split(r"(?<=[.!?])\s+", _oracle_safe_str(text))
    return [re.sub(r"\s+", " ", c).strip() for c in candidates if c.strip()]


def _oracle_chunk_text_by_sentences(text, max_chars, overlap_chars=0):
    sentences = _oracle_split_sentences(text)
    chunks, current = [], []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            overlap_sentences, overlap_length = [], 0
            for previous in reversed(current):
                if overlap_length + len(previous) > overlap_chars:
                    break
                overlap_sentences.insert(0, previous)
                overlap_length += len(previous) + 1
            current = overlap_sentences + [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def _oracle_evenly_spaced(items, maximum):
    if len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=int)
    return [items[int(i)] for i in indices]


def _oracle_build_translation_prompt(chunk, target_language_code):
    return f"""
Translate the following academic text into language code
"{target_language_code}".

Rules:
1. Preserve the scientific meaning and technical terminology.
2. Do not summarize, expand, explain, or add facts.
3. Preserve paragraph and sentence boundaries when possible.
4. Return only the translated text.

TEXT:
{chunk}
""".strip()


def _oracle_translate(text, target_language_code, *, llm_factory, max_chars_per_chunk):
    chunks = _oracle_chunk_text_by_sentences(text, max_chars_per_chunk, overlap_chars=0)
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = _oracle_build_translation_prompt(chunk, target_language_code)
        llm = llm_factory()
        response = llm.invoke([type("M", (), {"content": prompt})()])
        translated = _oracle_safe_str(response.content)
        if not translated:
            raise ValueError(f"La traducción devolvió un fragmento vacío: {index}.")
        parts.append(translated)
    translated_text = "\n\n".join(parts).strip()
    ratio = len(translated_text.split()) / max(len(text.split()), 1)
    if not 0.35 <= ratio <= 2.75:
        raise ValueError(f"La traducción tiene una proporción de longitud anómala: {ratio:.3f}.")
    return translated_text


def _oracle_rouge(ground_truth_text, generated_text, ground_truth_language):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=(ground_truth_language == "en"))
    return scorer.score(ground_truth_text, generated_text)["rougeL"]


def _oracle_build_automatic_metrics(
    *,
    generated_plain_text,
    ground_truth_plain_text,
    generated_language,
    ground_truth_language,
    evaluation_policy,
    translation_llm_factory,
    embedding_model,
    bertscore_score_fn,
):
    if evaluation_policy["translate_for_rouge_if_language_differs"] and generated_language != ground_truth_language:
        generated_for_rouge = _oracle_translate(
            generated_plain_text,
            ground_truth_language,
            llm_factory=translation_llm_factory,
            max_chars_per_chunk=evaluation_policy["max_translation_chars_per_chunk"],
        )
        translation_mode = "new_translation"
    else:
        generated_for_rouge = generated_plain_text
        translation_mode = "not_required_same_language"

    rouge_result = _oracle_rouge(ground_truth_plain_text, generated_for_rouge, ground_truth_language)
    rouge_rows = [
        {"metric": "rougeL_precision", "value": float(rouge_result.precision), "method": "global_text_after_translation_to_ground_truth_language"},
        {"metric": "rougeL_recall", "value": float(rouge_result.recall), "method": "global_text_after_translation_to_ground_truth_language"},
        {"metric": "rougeL_fmeasure", "value": float(rouge_result.fmeasure), "method": "global_text_after_translation_to_ground_truth_language"},
    ]

    generated_chunks = _oracle_evenly_spaced(
        _oracle_chunk_text_by_sentences(
            generated_plain_text,
            evaluation_policy["semantic_chunk_chars"],
            evaluation_policy["semantic_chunk_overlap_chars"],
        ),
        evaluation_policy["max_semantic_chunks_per_text"],
    )
    ground_truth_chunks = _oracle_evenly_spaced(
        _oracle_chunk_text_by_sentences(
            ground_truth_plain_text,
            evaluation_policy["semantic_chunk_chars"],
            evaluation_policy["semantic_chunk_overlap_chars"],
        ),
        evaluation_policy["max_semantic_chunks_per_text"],
    )
    if not generated_chunks or not ground_truth_chunks:
        raise ValueError("No se pudieron construir chunks para la evaluación semántica.")

    generated_embeddings = embedding_model.encode(generated_chunks, normalize_embeddings=True, show_progress_bar=True)
    ground_truth_embeddings = embedding_model.encode(ground_truth_chunks, normalize_embeddings=True, show_progress_bar=True)

    from sklearn.metrics.pairwise import cosine_similarity

    semantic_matrix = cosine_similarity(generated_embeddings, ground_truth_embeddings)

    generated_best_scores = semantic_matrix.max(axis=1)
    ground_truth_best_scores = semantic_matrix.max(axis=0)
    semantic_precision = float(generated_best_scores.mean())
    semantic_recall = float(ground_truth_best_scores.mean())
    semantic_f1 = (
        2 * semantic_precision * semantic_recall / (semantic_precision + semantic_recall)
        if (semantic_precision + semantic_recall) > 0
        else 0.0
    )
    generated_doc_embedding = generated_embeddings.mean(axis=0)
    ground_truth_doc_embedding = ground_truth_embeddings.mean(axis=0)
    global_semantic_similarity = float(
        cosine_similarity([generated_doc_embedding], [ground_truth_doc_embedding])[0][0]
    )

    semantic_alignment_rows = []
    for generated_index, score in enumerate(generated_best_scores):
        best_gt_index = int(semantic_matrix[generated_index].argmax())
        semantic_alignment_rows.append(
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
        best_generated_index = int(semantic_matrix[:, gt_index].argmax())
        semantic_alignment_rows.append(
            {
                "direction": "ground_truth_to_generated",
                "source_chunk_index": gt_index,
                "matched_chunk_index": best_generated_index,
                "similarity": float(score),
                "source_preview": ground_truth_chunks[gt_index][:300],
                "matched_preview": generated_chunks[best_generated_index][:300],
            }
        )

    semantic_rows = [
        {"metric": "semantic_precision", "value": semantic_precision, "method": "mean_best_generated_to_ground_truth_chunk_similarity"},
        {"metric": "semantic_recall", "value": semantic_recall, "method": "mean_best_ground_truth_to_generated_chunk_similarity"},
        {"metric": "semantic_f1", "value": semantic_f1, "method": "harmonic_mean_semantic_precision_recall"},
        {"metric": "global_semantic_similarity", "value": global_semantic_similarity, "method": "cosine_similarity_mean_document_embeddings"},
    ]

    max_bertscore_pairs = evaluation_policy["max_bertscore_pairs"]
    precision_pair_indices = np.linspace(
        0, len(generated_chunks) - 1, num=min(len(generated_chunks), max(1, max_bertscore_pairs // 2)), dtype=int
    )
    recall_pair_indices = np.linspace(
        0,
        len(ground_truth_chunks) - 1,
        num=min(len(ground_truth_chunks), max(1, max_bertscore_pairs - len(precision_pair_indices))),
        dtype=int,
    )

    bertscore_candidates, bertscore_references, bertscore_pair_metadata = [], [], []
    for generated_index in precision_pair_indices:
        generated_index = int(generated_index)
        gt_index = int(semantic_matrix[generated_index].argmax())
        bertscore_candidates.append(generated_chunks[generated_index])
        bertscore_references.append(ground_truth_chunks[gt_index])
        bertscore_pair_metadata.append(
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
        bertscore_candidates.append(generated_chunks[generated_index])
        bertscore_references.append(ground_truth_chunks[gt_index])
        bertscore_pair_metadata.append(
            {
                "direction": "ground_truth_to_generated",
                "generated_chunk_index": generated_index,
                "ground_truth_chunk_index": gt_index,
                "semantic_similarity": float(semantic_matrix[generated_index, gt_index]),
            }
        )

    if not bertscore_candidates:
        raise ValueError("No se construyeron pares para BERTScore.")

    bertscore_precision_values, bertscore_recall_values, bertscore_f1_values = bertscore_score_fn(
        bertscore_candidates,
        bertscore_references,
        model_type=evaluation_policy["bertscore_model"],
        verbose=True,
        batch_size=8,
        rescale_with_baseline=False,
    )
    bertscore_precision = float(bertscore_precision_values.mean())
    bertscore_recall = float(bertscore_recall_values.mean())
    bertscore_f1 = float(bertscore_f1_values.mean())

    for index, metadata in enumerate(bertscore_pair_metadata):
        metadata.update(
            {
                "bertscore_precision": float(bertscore_precision_values[index]),
                "bertscore_recall": float(bertscore_recall_values[index]),
                "bertscore_f1": float(bertscore_f1_values[index]),
                "generated_preview": bertscore_candidates[index][:300],
                "ground_truth_preview": bertscore_references[index][:300],
            }
        )

    bertscore_rows = [
        {"metric": "bertscore_precision", "value": bertscore_precision, "method": "bidirectional_semantically_aligned_chunks"},
        {"metric": "bertscore_recall", "value": bertscore_recall, "method": "bidirectional_semantically_aligned_chunks"},
        {"metric": "bertscore_f1", "value": bertscore_f1, "method": "bidirectional_semantically_aligned_chunks"},
    ]

    automatic_metric_rows = rouge_rows + semantic_rows + bertscore_rows

    return {
        "generated_for_rouge": generated_for_rouge,
        "translation_mode": translation_mode,
        "generated_chunks": generated_chunks,
        "ground_truth_chunks": ground_truth_chunks,
        "semantic_matrix": semantic_matrix,
        "semantic_alignment_rows": semantic_alignment_rows,
        "bertscore_pair_metadata": bertscore_pair_metadata,
        "automatic_metric_rows": automatic_metric_rows,
    }


@scenario("N01. Diferencial COMPLETA: mismo idioma, sin traducción")
def test_full_diff_same_language():
    model = FakeEmbeddingModel()
    scorer = make_bertscore_score_fn()
    translation_factory_real = FakeLLMFactory()
    translation_factory_oracle = FakeLLMFactory()

    real = build_automatic_metrics(
        generated_plain_text=TEXT_A,
        ground_truth_plain_text=TEXT_B,
        generated_language="es",
        ground_truth_language="es",
        evaluation_policy=DEFAULT_POLICY,
        translation_llm_factory=translation_factory_real,
        embedding_model_factory=lambda name: model,
        bertscore_score_fn=scorer,
    )
    oracle = _oracle_build_automatic_metrics(
        generated_plain_text=TEXT_A,
        ground_truth_plain_text=TEXT_B,
        generated_language="es",
        ground_truth_language="es",
        evaluation_policy=DEFAULT_POLICY,
        translation_llm_factory=translation_factory_oracle,
        embedding_model=model,
        bertscore_score_fn=scorer,
    )

    assert real.generated_for_rouge == oracle["generated_for_rouge"]
    assert real.translation_mode == oracle["translation_mode"]
    assert real.generated_chunks == oracle["generated_chunks"]
    assert real.ground_truth_chunks == oracle["ground_truth_chunks"]
    assert np.allclose(real.semantic_matrix, oracle["semantic_matrix"])
    assert real.semantic_alignment_rows == oracle["semantic_alignment_rows"]
    assert real.bertscore_pair_metadata == oracle["bertscore_pair_metadata"]
    assert real.automatic_metric_rows == oracle["automatic_metric_rows"]


@scenario("N02. Diferencial COMPLETA: idiomas distintos, con traducción real (doble)")
def test_full_diff_translation_branch():
    model = FakeEmbeddingModel()
    scorer = make_bertscore_score_fn()
    responses = [
        "Translated academic content with enough words to satisfy the length ratio check for this test case."
    ]
    translation_factory_real = FakeLLMFactory(responses=list(responses))
    translation_factory_oracle = FakeLLMFactory(responses=list(responses))

    real = build_automatic_metrics(
        generated_plain_text=TEXT_A,
        ground_truth_plain_text=TEXT_B,
        generated_language="es",
        ground_truth_language="en",
        evaluation_policy=DEFAULT_POLICY,
        translation_llm_factory=translation_factory_real,
        embedding_model_factory=lambda name: model,
        bertscore_score_fn=scorer,
    )
    oracle = _oracle_build_automatic_metrics(
        generated_plain_text=TEXT_A,
        ground_truth_plain_text=TEXT_B,
        generated_language="es",
        ground_truth_language="en",
        evaluation_policy=DEFAULT_POLICY,
        translation_llm_factory=translation_factory_oracle,
        embedding_model=model,
        bertscore_score_fn=scorer,
    )

    assert real.translation_mode == oracle["translation_mode"] == "new_translation"
    assert real.generated_for_rouge == oracle["generated_for_rouge"]
    assert real.automatic_metric_rows == oracle["automatic_metric_rows"]
    assert real.bertscore_pair_metadata == oracle["bertscore_pair_metadata"]


if __name__ == "__main__":
    for fn in (test_full_diff_same_language, test_full_diff_translation_branch):
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
