"""Pruebas de integración del ensamblador de métricas automáticas.

Usa dobles deterministas para el LLM de traducción, el modelo de embeddings
y BERTScore -- ninguno de los tres se llama de verdad (ni OpenAI, ni
descarga de modelo). Reutiliza ``FakeLLMFactory`` del Bloque 3
(``test_evaluation_language_characterization.py``) para no duplicar ese
doble.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.automatic_metrics import build_automatic_metrics
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
# Dobles deterministas
# ---------------------------------------------------------------------------


def _simple_vector(text: str) -> np.ndarray:
    """Embedding sintético 2D, determinista: fracción de vocales vs. resto,
    normalizado. No es el modelo productivo -- solo permite calcular
    similitudes reales y reproducibles en las pruebas."""

    vowels = sum(1 for c in text.lower() if c in "aeiouáéíóú")
    length = max(len(text), 1)
    vector = np.array([vowels / length, 1 - vowels / length])
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class FakeEmbeddingModel:
    def __init__(self, *, raise_on_encode=False):
        self.raise_on_encode = raise_on_encode
        self.encode_calls = []

    def encode(self, chunks, *, normalize_embeddings, show_progress_bar):
        self.encode_calls.append(list(chunks))
        if self.raise_on_encode:
            raise RuntimeError("Fallo simulado del encoder de embeddings.")
        return np.array([_simple_vector(c) for c in chunks])


def make_embedding_model_factory(model, factory_calls=None):
    def factory(model_name):
        if factory_calls is not None:
            factory_calls.append(model_name)
        return model

    return factory


class FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


def make_bertscore_score_fn(*, raise_error=False):
    calls = []

    def _fn(candidates, references, **kwargs):
        calls.append({"candidates": list(candidates), "references": list(references), "kwargs": kwargs})
        if raise_error:
            raise RuntimeError("Fallo simulado de BERTScore.")
        values = [1.0 if c == r else 0.4 for c, r in zip(candidates, references)]
        return FakeTensor(values), FakeTensor(values), FakeTensor(values)

    _fn.calls = calls
    return _fn


DEFAULT_POLICY = {
    "translate_for_rouge_if_language_differs": True,
    "max_translation_chars_per_chunk": 200,
    "semantic_chunk_chars": 60,
    "semantic_chunk_overlap_chars": 0,
    "max_semantic_chunks_per_text": 5,
    "evaluation_embedding_model": "fake-embedding-model",
    "bertscore_model": "fake-bertscore-model",
    "max_bertscore_pairs": 4,
}

TEXT_A = (
    "El modelo alcanzó resultados destacados en el conjunto de datos evaluado. "
    "Los experimentos confirmaron una mejora consistente frente a los métodos previos."
)
TEXT_B = (
    "El modelo obtuvo resultados notables en el conjunto de datos analizado. "
    "Las pruebas mostraron una mejora sostenida respecto a los enfoques anteriores."
)


def _run(
    *,
    generated=TEXT_A,
    ground_truth=TEXT_B,
    generated_language="es",
    ground_truth_language="es",
    policy=None,
    translation_responses=None,
    embedding_model=None,
    embedding_factory_calls=None,
    bertscore_fn=None,
):
    policy = dict(policy or DEFAULT_POLICY)
    translation_factory = FakeLLMFactory(responses=list(translation_responses or []))
    model = embedding_model if embedding_model is not None else FakeEmbeddingModel()
    embedding_factory = make_embedding_model_factory(model, embedding_factory_calls)
    scorer = bertscore_fn if bertscore_fn is not None else make_bertscore_score_fn()

    result = build_automatic_metrics(
        generated_plain_text=generated,
        ground_truth_plain_text=ground_truth,
        generated_language=generated_language,
        ground_truth_language=ground_truth_language,
        evaluation_policy=policy,
        translation_llm_factory=translation_factory,
        embedding_model_factory=embedding_factory,
        bertscore_score_fn=scorer,
    )
    return result, translation_factory, model, scorer


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------


@scenario("I01. Mismo idioma -> no traduce, factory de traducción nunca invocada")
def test_same_language_no_translation():
    result, translation_factory, _model, _scorer = _run(
        generated_language="es", ground_truth_language="es"
    )
    assert result.translation_mode == "not_required_same_language"
    assert translation_factory.instances_created == 0
    assert result.generated_for_rouge == TEXT_A


@scenario("I02. Idiomas distintos con traducción activada -> traduce, factory invocada")
def test_different_languages_translates():
    result, translation_factory, _model, _scorer = _run(
        generated_language="es",
        ground_truth_language="en",
        translation_responses=["Translated content with enough words to pass the length ratio check now."],
    )
    assert result.translation_mode == "new_translation"
    assert translation_factory.instances_created >= 1
    assert result.generated_for_rouge != TEXT_A


@scenario("I03. Traducción desactivada por política -> no traduce aunque los idiomas difieran")
def test_translation_disabled_by_policy():
    policy = dict(DEFAULT_POLICY)
    policy["translate_for_rouge_if_language_differs"] = False
    result, translation_factory, _model, _scorer = _run(
        generated_language="es", ground_truth_language="en", policy=policy
    )
    assert result.translation_mode == "not_required_same_language"
    assert translation_factory.instances_created == 0
    assert result.generated_for_rouge == TEXT_A


@scenario("I04. Textos idénticos -> ROUGE-L F1 = 1.0")
def test_identical_texts():
    result, _tf, _model, _scorer = _run(generated=TEXT_A, ground_truth=TEXT_A)
    rouge_rows = {r["metric"]: r["value"] for r in result.automatic_metric_rows[:3]}
    assert rouge_rows["rougeL_fmeasure"] == 1.0


@scenario("I05. Textos diferentes -> pipeline completo sin errores, F1 < 1.0")
def test_different_texts_no_errors():
    result, _tf, _model, _scorer = _run(generated=TEXT_A, ground_truth=TEXT_B)
    rouge_rows = {r["metric"]: r["value"] for r in result.automatic_metric_rows[:3]}
    assert 0.0 <= rouge_rows["rougeL_fmeasure"] < 1.0


@scenario("I06. Conjuntos desiguales de chunks (textos de longitudes distintas)")
def test_unequal_chunk_counts():
    short_text = "Frase corta."
    long_text = TEXT_A * 3
    result, _tf, _model, _scorer = _run(generated=long_text, ground_truth=short_text)
    assert len(result.generated_chunks) != len(result.ground_truth_chunks)


@scenario("I07. Chunks vacíos -> ValueError con el mensaje original, antes de cargar el modelo")
def test_empty_chunks_raises_before_model():
    factory_calls = []
    try:
        _run(
            generated="",
            ground_truth=TEXT_B,
            embedding_factory_calls=factory_calls,
        )
    except ValueError as exc:
        assert "No se pudieron construir chunks" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por chunks vacíos")
    assert factory_calls == [], "no debía construirse el modelo de embeddings tras el fallo"


@scenario("I08. Fallo de traducción se propaga sin silenciarse")
def test_translation_failure_propagates():
    try:
        _run(
            generated_language="es",
            ground_truth_language="en",
            translation_responses=[""],  # respuesta vacía -> ValueError real del Bloque 3
        )
    except ValueError as exc:
        assert "vacío" in str(exc)
    else:
        raise AssertionError("debía propagar el fallo de traducción")


@scenario("I09. Fallo del encoder se propaga sin silenciarse")
def test_encoder_failure_propagates():
    failing_model = FakeEmbeddingModel(raise_on_encode=True)
    try:
        _run(embedding_model=failing_model)
    except RuntimeError as exc:
        assert "encoder" in str(exc)
    else:
        raise AssertionError("debía propagar el fallo del encoder")


@scenario("I10. Fallo de BERTScore se propaga sin silenciarse")
def test_bertscore_failure_propagates():
    failing_scorer = make_bertscore_score_fn(raise_error=True)
    try:
        _run(bertscore_fn=failing_scorer)
    except RuntimeError as exc:
        assert "BERTScore" in str(exc)
    else:
        raise AssertionError("debía propagar el fallo de BERTScore")


@scenario("I11. Orden exacto de las 10 filas: 3 ROUGE + 4 semánticas + 3 BERTScore")
def test_exact_row_order():
    result, _tf, _model, _scorer = _run()
    metrics_in_order = [r["metric"] for r in result.automatic_metric_rows]
    assert metrics_in_order == [
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_fmeasure",
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "global_semantic_similarity",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    assert len(result.automatic_metric_rows) == 10


@scenario("I12. Propagación de metadata: alineación semántica y pares BERTScore tienen las claves reales")
def test_metadata_propagation():
    result, _tf, _model, _scorer = _run()
    assert result.semantic_alignment_rows
    for row in result.semantic_alignment_rows:
        assert set(row) == {
            "direction",
            "source_chunk_index",
            "matched_chunk_index",
            "similarity",
            "source_preview",
            "matched_preview",
        }
    assert result.bertscore_pair_metadata
    for row in result.bertscore_pair_metadata:
        assert {
            "direction",
            "generated_chunk_index",
            "ground_truth_chunk_index",
            "semantic_similarity",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "generated_preview",
            "ground_truth_preview",
        }.issubset(set(row))


@scenario("I13. Reutilización de la misma matriz semántica para similitud y BERTScore (no se recalcula)")
def test_same_matrix_reused_for_bertscore():
    result, _tf, _model, _scorer = _run()
    # Comportamiento observable: si BERTScore hubiese usado una matriz
    # distinta, el ground_truth_chunk_index de cada par
    # "generated_to_ground_truth" no coincidiría necesariamente con el
    # argmax de result.semantic_matrix.
    for row in result.bertscore_pair_metadata:
        if row["direction"] == "generated_to_ground_truth":
            expected_gt_index = int(
                result.semantic_matrix[row["generated_chunk_index"]].argmax()
            )
            assert row["ground_truth_chunk_index"] == expected_gt_index
        else:
            expected_gen_index = int(
                result.semantic_matrix[:, row["ground_truth_chunk_index"]].argmax()
            )
            assert row["generated_chunk_index"] == expected_gen_index

    # Prueba concluyente (monkeypatch, sin tocar el módulo productivo):
    # envuelve compute_similarity_matrix y build_bertscore_pairs, tal como
    # están vinculados en el namespace de automatic_metrics.py (import por
    # nombre -- reasignar el atributo del módulo SÍ afecta a la llamada
    # interna, porque Python resuelve el nombre en el namespace del módulo
    # en el momento de la llamada, no al definir la función). Se restaura
    # el original en un ``finally`` para no dejar el módulo parcheado.
    import src.tools.evaluation.automatic_metrics as automatic_metrics_module

    original_compute_similarity_matrix = automatic_metrics_module.compute_similarity_matrix
    original_build_bertscore_pairs = automatic_metrics_module.build_bertscore_pairs

    compute_call_results = []
    captured_matrix_at_build_pairs = {}

    def counting_compute_similarity_matrix(*args, **kwargs):
        matrix = original_compute_similarity_matrix(*args, **kwargs)
        compute_call_results.append(matrix)
        return matrix

    def spying_build_bertscore_pairs(*, semantic_matrix, **kwargs):
        captured_matrix_at_build_pairs["value"] = semantic_matrix
        return original_build_bertscore_pairs(semantic_matrix=semantic_matrix, **kwargs)

    automatic_metrics_module.compute_similarity_matrix = counting_compute_similarity_matrix
    automatic_metrics_module.build_bertscore_pairs = spying_build_bertscore_pairs
    try:
        monkeypatched_result, _tf2, _model2, _scorer2 = _run()
    finally:
        automatic_metrics_module.compute_similarity_matrix = original_compute_similarity_matrix
        automatic_metrics_module.build_bertscore_pairs = original_build_bertscore_pairs

    assert len(compute_call_results) == 1, (
        f"compute_similarity_matrix debía llamarse exactamente 1 vez; "
        f"se llamó {len(compute_call_results)} veces"
    )
    # La matriz que build_bertscore_pairs recibió es, por identidad de
    # objeto (no solo por igualdad de valores), la MISMA que produjo la
    # única llamada a compute_similarity_matrix -- y la misma que queda en
    # el resultado final.
    assert captured_matrix_at_build_pairs["value"] is compute_call_results[0]
    assert captured_matrix_at_build_pairs["value"] is monkeypatched_result.semantic_matrix


@scenario("I14. Ausencia de escrituras de archivos: el módulo no referencia I/O de disco")
def test_no_file_writes():
    import inspect

    from src.tools.evaluation import automatic_metrics as module

    source = inspect.getsource(module)
    for forbidden in ("open(", ".write_text(", ".to_csv(", "os.makedirs", "Path("):
        assert forbidden not in source, f"no debería referenciar {forbidden!r}"


if __name__ == "__main__":
    for fn in (
        test_same_language_no_translation,
        test_different_languages_translates,
        test_translation_disabled_by_policy,
        test_identical_texts,
        test_different_texts_no_errors,
        test_unequal_chunk_counts,
        test_empty_chunks_raises_before_model,
        test_translation_failure_propagates,
        test_encoder_failure_propagates,
        test_bertscore_failure_propagates,
        test_exact_row_order,
        test_metadata_propagation,
        test_same_matrix_reused_for_bertscore,
        test_no_file_writes,
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
