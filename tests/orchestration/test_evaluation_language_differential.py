"""Pruebas diferenciales del Bloque 3: oráculo reproducido vs. módulos reales.

Para las funciones puras (``split_sentences``, ``chunk_text_by_sentences``),
compara oráculo independiente vs. módulo real. Para la traducción, usa el
mismo doble determinista de la caracterización y compara chunks enviados,
orden, código de idioma objetivo, texto reconstruido, metadata
(``translation_mode``) y excepciones.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.language_preprocessing import (
    chunk_text_by_sentences,
    split_sentences,
)
from src.tools.evaluation.translation import (
    build_translation_prompt,
    resolve_generated_text_for_rouge,
    translate_text_to_language,
)

# ---------------------------------------------------------------------------
# Oráculo independiente (celda 17), sin compartir código con los módulos.
# ---------------------------------------------------------------------------


def _oracle_safe_str(value):
    import json

    import pandas as pd

    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _oracle_split_sentences(text):
    candidates = re.split(r"(?<=[.!?])\s+", _oracle_safe_str(text))
    return [
        re.sub(r"\s+", " ", candidate).strip()
        for candidate in candidates
        if candidate.strip()
    ]


def _oracle_chunk_text_by_sentences(text, max_chars, overlap_chars=0):
    sentences = _oracle_split_sentences(text)
    chunks = []
    current = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            overlap_sentences = []
            overlap_length = 0
            for previous_sentence in reversed(current):
                if overlap_length + len(previous_sentence) > overlap_chars:
                    break
                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence) + 1
            current = overlap_sentences + [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _oracle_build_prompt(chunk, target_language_code):
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


class _OracleFakeLLMInstance:
    def __init__(self, factory, index):
        self._factory = factory
        self.index = index

    def invoke(self, messages):
        self._factory.calls.append(messages[0].content)
        return SimpleNamespace(content=self._factory.responses.pop(0))


class _OracleFakeLLMFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.instances_created = 0

    def __call__(self):
        self.instances_created += 1
        return _OracleFakeLLMInstance(self, self.instances_created)


def _oracle_translate(text, target_language_code, *, llm_factory, max_chars_per_chunk):
    chunks = _oracle_chunk_text_by_sentences(text, max_chars_per_chunk, overlap_chars=0)
    translated_parts = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        prompt = _oracle_build_prompt(chunk, target_language_code)
        llm = llm_factory()  # instancia nueva por chunk, igual que el módulo real corregido
        response = llm.invoke([SimpleNamespace(content=prompt)])
        translated = _oracle_safe_str(response.content)
        if not translated:
            raise ValueError(f"La traducción devolvió un fragmento vacío: {chunk_index}.")
        translated_parts.append(translated)
    translated_text = "\n\n".join(translated_parts).strip()
    source_word_count = max(len(text.split()), 1)
    ratio = len(translated_text.split()) / source_word_count
    if not 0.35 <= ratio <= 2.75:
        raise ValueError(f"La traducción tiene una proporción de longitud anómala: {ratio:.3f}.")
    return translated_text


def _oracle_resolve_generated_text_for_rouge(
    *,
    generated_plain_text,
    generated_language,
    ground_truth_language,
    translate_for_rouge,
    llm_factory,
    max_chars_per_chunk,
):
    if translate_for_rouge and generated_language != ground_truth_language:
        generated_for_rouge = _oracle_translate(
            generated_plain_text,
            ground_truth_language,
            llm_factory=llm_factory,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        translation_mode = "new_translation"
    else:
        generated_for_rouge = generated_plain_text
        translation_mode = "not_required_same_language"
    return generated_for_rouge, translation_mode


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


@scenario("M01. Diferencial: split_sentences idéntico en textos variados")
def test_diff_split_sentences():
    samples = [
        "Una oracion. Otra oracion.",
        "",
        None,
        "El Dr. Perez llegó. Bien.",
        "91.5 por ciento. Es alto.",
        "Con salto\n\nde parrafo.",
        "日本語です。テストです。",
    ]
    for text in samples:
        assert split_sentences(text) == _oracle_split_sentences(text), text


@scenario("M02. Diferencial: chunk_text_by_sentences idéntico con y sin overlap")
def test_diff_chunk_text():
    text = "Uno dos tres. Cuatro cinco seis. Siete ocho nueve. Diez once doce."
    for max_chars, overlap in [(15, 0), (15, 5), (100, 0), (5, 0)]:
        real = chunk_text_by_sentences(text, max_chars=max_chars, overlap_chars=overlap)
        oracle = _oracle_chunk_text_by_sentences(text, max_chars, overlap_chars=overlap)
        assert real == oracle, (max_chars, overlap)


@scenario("M03. Diferencial traducción: chunks enviados, orden, código de idioma objetivo e instancias idénticos")
def test_diff_translation_chunks_sent():
    text = "Primera oracion aqui. " * 4 + "Segunda oracion aqui tambien. " * 4
    expected_chunks = chunk_text_by_sentences(text, max_chars=60, overlap_chars=0)
    responses = [
        f"Translated segment {i} with sufficient word count included here now."
        for i in range(1, len(expected_chunks) + 1)
    ]

    from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

    real_factory = FakeLLMFactory(responses=list(responses))
    oracle_factory = _OracleFakeLLMFactory(list(responses))

    real_result = translate_text_to_language(
        text, "en", llm_factory=real_factory, max_chars_per_chunk=60
    )
    oracle_result = _oracle_translate(
        text, "en", llm_factory=oracle_factory, max_chars_per_chunk=60
    )

    assert real_result == oracle_result
    assert (
        len(real_factory.calls)
        == len(oracle_factory.calls)
        == real_factory.instances_created
        == oracle_factory.instances_created
        == len(expected_chunks)
    )
    for real_prompt, oracle_prompt in zip(real_factory.calls, oracle_factory.calls):
        assert real_prompt == oracle_prompt


@scenario("M04. Diferencial traducción: texto reconstruido idéntico")
def test_diff_translation_reconstruction():
    from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

    text = "Texto corto de origen para traducir."
    response = "Short source text to be translated right here now today."
    real = translate_text_to_language(
        text, "en", llm_factory=FakeLLMFactory(responses=[response]), max_chars_per_chunk=1000
    )
    oracle = _oracle_translate(
        text, "en", llm_factory=_OracleFakeLLMFactory([response]), max_chars_per_chunk=1000
    )
    assert real == oracle == response


@scenario("M05. Diferencial traducción: misma excepción ante respuesta vacía")
def test_diff_translation_empty_response_exception():
    from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

    text = "Texto de origen."
    real_exc = oracle_exc = None
    try:
        translate_text_to_language(
            text, "en", llm_factory=FakeLLMFactory(responses=[""]), max_chars_per_chunk=1000
        )
    except ValueError as exc:
        real_exc = str(exc)
    try:
        _oracle_translate(
            text, "en", llm_factory=_OracleFakeLLMFactory([""]), max_chars_per_chunk=1000
        )
    except ValueError as exc:
        oracle_exc = str(exc)
    assert real_exc is not None and real_exc == oracle_exc


@scenario("M06. Diferencial: metadata (translation_mode) idéntica en ramas sin traducción (mismo idioma / política desactivada)")
def test_diff_translation_mode_metadata_no_translation():
    from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

    cases = [
        ("es", "es", True),  # mismo idioma
        ("es", "en", False),  # política desactivada
    ]
    for gen_lang, gt_lang, policy in cases:
        real_text, real_mode = resolve_generated_text_for_rouge(
            generated_plain_text="Texto generado.",
            generated_language=gen_lang,
            ground_truth_language=gt_lang,
            translate_for_rouge=policy,
            llm_factory=FakeLLMFactory(),
            max_chars_per_chunk=1000,
        )
        oracle_text, oracle_mode = _oracle_resolve_generated_text_for_rouge(
            generated_plain_text="Texto generado.",
            generated_language=gen_lang,
            ground_truth_language=gt_lang,
            translate_for_rouge=policy,
            llm_factory=FakeLLMFactory(),
            max_chars_per_chunk=1000,
        )
        assert real_mode == oracle_mode == "not_required_same_language"
        assert real_text == oracle_text == "Texto generado."


@scenario("M07. Diferencial COMPLETA de la rama new_translation (idiomas distintos, translate_for_rouge=True)")
def test_diff_new_translation_branch_complete():
    """Cubre exactamente lo pedido: generated_language != ground_truth_language
    y translate_for_rouge=True, comparando texto reconstruido,
    translation_mode, idioma objetivo (vía el prompt enviado), chunks
    enviados, orden de prompts, número de instancias del LLM creadas, y
    ausencia de excepción en ambos caminos."""

    from tests.orchestration.test_evaluation_language_characterization import FakeLLMFactory

    generated_text = (
        "Este es el texto generado en español que necesita traducirse "
        "porque el idioma del Ground Truth es distinto del idioma del "
        "texto generado, activando la rama new_translation del notebook."
    )
    expected_chunks = chunk_text_by_sentences(generated_text, max_chars=90, overlap_chars=0)
    assert len(expected_chunks) >= 1
    responses = [
        f"This is translated segment number {i} with enough words to satisfy the length ratio check."
        for i in range(1, len(expected_chunks) + 1)
    ]

    real_factory = FakeLLMFactory(responses=list(responses))
    oracle_factory = _OracleFakeLLMFactory(list(responses))

    real_text, real_mode = resolve_generated_text_for_rouge(
        generated_plain_text=generated_text,
        generated_language="es",
        ground_truth_language="en",
        translate_for_rouge=True,
        llm_factory=real_factory,
        max_chars_per_chunk=90,
    )
    oracle_text, oracle_mode = _oracle_resolve_generated_text_for_rouge(
        generated_plain_text=generated_text,
        generated_language="es",
        ground_truth_language="en",
        translate_for_rouge=True,
        llm_factory=oracle_factory,
        max_chars_per_chunk=90,
    )

    # texto reconstruido
    assert real_text == oracle_text == "\n\n".join(responses)
    # translation_mode
    assert real_mode == oracle_mode == "new_translation"
    # idioma objetivo: el prompt enviado debe pedir "en" explícitamente, en ambos caminos
    for prompt in real_factory.calls + oracle_factory.calls:
        assert '"en"' in prompt
    # chunks enviados y orden de prompts
    assert real_factory.calls == oracle_factory.calls
    assert real_factory.calls == [build_translation_prompt(c, "en") for c in expected_chunks]
    # número de instancias del LLM: una por chunk, igual en ambos caminos
    assert (
        real_factory.instances_created
        == oracle_factory.instances_created
        == len(expected_chunks)
    )
    # ninguna excepción se disparó en ningún camino (llegar hasta aquí ya lo confirma)


if __name__ == "__main__":
    for fn in (
        test_diff_split_sentences,
        test_diff_chunk_text,
        test_diff_translation_chunks_sent,
        test_diff_translation_reconstruction,
        test_diff_translation_empty_response_exception,
        test_diff_translation_mode_metadata_no_translation,
        test_diff_new_translation_branch_complete,
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
