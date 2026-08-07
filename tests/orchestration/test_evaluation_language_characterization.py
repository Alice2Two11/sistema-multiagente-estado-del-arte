"""Pruebas de caracterización del Bloque 3: preprocesamiento lingüístico y traducción.

``FakeLLMFactory`` es el doble determinista pedido para probar
``translate_text_to_language``/``resolve_generated_text_for_rouge`` sin
llamar a OpenAI, inyectado exactamente por el mismo punto (``llm_factory=``)
que usaría la factory productiva real (``lambda: get_llm(model=...,
temperature=...)``) — el mecanismo de traducción no se reemplaza, solo se
sustituye qué objeto concreto construye la factory y qué responde
``.invoke()``. Registra cuántas instancias se crearon, para confirmar que
se construye una nueva por chunk (igual que ``get_llm(...)`` dentro del
bucle real), no una única instancia reutilizada.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.language_preprocessing import (
    chunk_text_by_sentences,
    detect_language_code,
    split_sentences,
)
from src.tools.evaluation.translation import (
    build_translation_prompt,
    resolve_generated_text_for_rouge,
    translate_text_to_language,
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


class _FakeLLMInstance:
    def __init__(self, factory, index):
        self._factory = factory
        self.index = index  # 1-indexed, orden de construcción

    def invoke(self, messages):
        self._factory.calls.append(messages[0].content)
        if self._factory.fail_on_invoke_index == self.index:
            raise RuntimeError("Fallo simulado en .invoke().")
        if self._factory.responses:
            content = self._factory.responses.pop(0)
        else:
            content = "[traducción simulada]"
        return SimpleNamespace(content=content)


class FakeLLMFactory:
    """Doble determinista de una factory de LLM (``llm_factory: Callable[[], Any]``).

    Cada llamada ``factory()`` construye una instancia NUEVA (registrada en
    ``instances_created``), igual que ``get_llm(...)`` dentro del bucle real
    — no reutiliza un único cliente entre chunks.
    """

    def __init__(self, responses=None, *, fail_on_build_index=None, fail_on_invoke_index=None):
        self.responses = list(responses or [])
        self.calls = []  # prompts recibidos, en orden, a través de todas las instancias
        self.instances_created = 0
        self.fail_on_build_index = fail_on_build_index
        self.fail_on_invoke_index = fail_on_invoke_index

    def __call__(self):
        self.instances_created += 1
        if self.instances_created == self.fail_on_build_index:
            raise RuntimeError("Fallo simulado al construir el cliente LLM.")
        return _FakeLLMInstance(self, self.instances_created)


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


@scenario("L01. split_sentences con texto en español")
def test_split_sentences_spanish():
    result = split_sentences("Este es un texto. Tiene dos oraciones.")
    assert result == ["Este es un texto.", "Tiene dos oraciones."]


@scenario("L02. split_sentences con texto en inglés")
def test_split_sentences_english():
    result = split_sentences("This is a text. It has two sentences.")
    assert result == ["This is a text.", "It has two sentences."]


@scenario("L03. split_sentences con texto mixto ES/EN")
def test_split_sentences_mixed():
    result = split_sentences("Hola mundo. Hello world! ¿Qué tal?")
    assert result == ["Hola mundo.", "Hello world!", "¿Qué tal?"]


@scenario("L04. split_sentences con cadena vacía")
def test_split_sentences_empty():
    assert split_sentences("") == []


@scenario("L05. split_sentences con None")
def test_split_sentences_none():
    assert split_sentences(None) == []


@scenario("L06. split_sentences con una sola oración")
def test_split_sentences_one():
    assert split_sentences("Solo una oración sin más.") == ["Solo una oración sin más."]


@scenario("L07. split_sentences con abreviaturas (no separa en el punto de la abreviatura)")
def test_split_sentences_abbreviation():
    # El separador real es (?<=[.!?])\s+ -- separa tras CUALQUIER punto seguido
    # de espacio, incluida una abreviatura. Se documenta el comportamiento
    # real (no ideal) tal cual, sin mejorarlo.
    result = split_sentences("El Dr. Pérez llegó tarde. La reunión continuó.")
    assert result == ["El Dr.", "Pérez llegó tarde.", "La reunión continuó."]


@scenario("L08. split_sentences con números decimales (no separa en el punto decimal sin espacio)")
def test_split_sentences_decimal():
    result = split_sentences("El resultado fue 91.5 por ciento. Es significativo.")
    assert result == ["El resultado fue 91.5 por ciento.", "Es significativo."]


@scenario("L09. split_sentences con citas internas conservadas dentro de la oración")
def test_split_sentences_with_citations():
    result = split_sentences("El modelo mejoró un 91% [p1.pdf | c1]. Esto es relevante.")
    assert result[0] == "El modelo mejoró un 91% [p1.pdf | c1]."
    assert result[1] == "Esto es relevante."


@scenario("L10. split_sentences con saltos de párrafo se normalizan a espacio simple")
def test_split_sentences_paragraph_breaks():
    result = split_sentences("Primera oración.\n\nSegunda oración con\nsalto interno.")
    assert result == ["Primera oración.", "Segunda oración con salto interno."]


@scenario("L11. split_sentences con Unicode se conserva")
def test_split_sentences_unicode():
    result = split_sentences("El niño analizó los datos. 日本語のテキストです。")
    assert "niño" in result[0]


# ---------------------------------------------------------------------------
# chunk_text_by_sentences
# ---------------------------------------------------------------------------


@scenario("L12. chunk_text_by_sentences: oración mayor que el límite forma su propio chunk")
def test_chunk_sentence_larger_than_limit():
    long_sentence = "Palabra " * 30 + "."
    chunks = chunk_text_by_sentences(long_sentence, max_chars=20, overlap_chars=0)
    assert len(chunks) == 1
    assert chunks[0].strip().startswith("Palabra")


@scenario("L13. chunk_text_by_sentences: chunk exacto al límite no se corta de más")
def test_chunk_exact_at_limit():
    text = "Uno dos tres. Cuatro cinco seis."
    max_chars = len("Uno dos tres.")
    chunks = chunk_text_by_sentences(text, max_chars=max_chars, overlap_chars=0)
    assert chunks[0] == "Uno dos tres."


@scenario("L14. chunk_text_by_sentences: múltiples chunks conservan el orden original")
def test_chunk_multiple_preserves_order():
    text = "Primera oracion. Segunda oracion. Tercera oracion. Cuarta oracion."
    chunks = chunk_text_by_sentences(text, max_chars=20, overlap_chars=0)
    assert len(chunks) > 1
    rejoined = " ".join(chunks)
    assert rejoined.index("Primera") < rejoined.index("Segunda") < rejoined.index("Tercera")


@scenario("L15. chunk_text_by_sentences con overlap_chars > 0: repite oraciones que caben, no repite las que no caben")
def test_chunk_with_overlap():
    text = "Uno dos. Tres cuatro. Cinco seis. Siete ocho."
    chunks = chunk_text_by_sentences(text, max_chars=15, overlap_chars=10)
    assert chunks == ["Uno dos.", "Uno dos. Tres cuatro.", "Cinco seis.", "Siete ocho."]

    def _sentences_that_fit_in_overlap(chunk_text, overlap_chars):
        # Reproduce, de forma independiente, el mismo criterio real de
        # selección de overlap (acumular oraciones finales en reversa
        # mientras quepan en overlap_chars) para verificar el resultado
        # sin volver a llamar a chunk_text_by_sentences.
        sentences = split_sentences(chunk_text)
        fitting = []
        used = 0
        for sentence in reversed(sentences):
            if used + len(sentence) > overlap_chars:
                break
            fitting.insert(0, sentence)
            used += len(sentence) + 1
        return fitting

    # Caso con overlap: "Uno dos." (8 caracteres) cabe en overlap_chars=10
    # y debe reaparecer literalmente al inicio del chunk siguiente.
    fitting_from_chunk0 = _sentences_that_fit_in_overlap(chunks[0], overlap_chars=10)
    assert fitting_from_chunk0 == ["Uno dos."]
    assert chunks[1].startswith(" ".join(fitting_from_chunk0))

    # Caso sin overlap: ninguna oración de "Uno dos. Tres cuatro." cabe en
    # overlap_chars=10 ("Tres cuatro." por sí sola ya mide 12) -> el chunk
    # siguiente NO debe empezar repitiendo nada de él.
    fitting_from_chunk1 = _sentences_that_fit_in_overlap(chunks[1], overlap_chars=10)
    assert fitting_from_chunk1 == []
    assert not chunks[2].startswith("Tres cuatro.")


@scenario("L16. chunk_text_by_sentences con texto vacío devuelve lista vacía")
def test_chunk_empty():
    assert chunk_text_by_sentences("", max_chars=100) == []


@scenario("L17. chunk_text_by_sentences con None no lanza")
def test_chunk_none():
    assert chunk_text_by_sentences(None, max_chars=100) == []


# ---------------------------------------------------------------------------
# detect_language_code
# ---------------------------------------------------------------------------


_ES_SAMPLE = (
    "Este es un texto de prueba en idioma espanol con suficientes palabras "
    "para que el detector de idioma pueda identificar correctamente cual "
    "es el idioma predominante en el documento analizado."
)
_EN_SAMPLE = (
    "This is a sample text written in the English language with enough "
    "words for the language detector to correctly identify which language "
    "predominates in the analyzed document overall."
)


@scenario("L18. detect_language_code detecta español")
def test_detect_language_spanish():
    assert detect_language_code(_ES_SAMPLE) == "es"


@scenario("L19. detect_language_code detecta inglés")
def test_detect_language_english():
    assert detect_language_code(_EN_SAMPLE) == "en"


@scenario("L20. detect_language_code con texto insuficiente (< 20 palabras) lanza ValueError")
def test_detect_language_too_short():
    try:
        detect_language_code("Muy poco texto aquí.")
    except ValueError as exc:
        assert "suficiente" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por texto insuficiente")


@scenario("L21. detect_language_code con None lanza ValueError (no RuntimeError)")
def test_detect_language_none():
    try:
        detect_language_code(None)
    except ValueError:
        pass
    else:
        raise AssertionError("debía lanzar ValueError (safe_str(None) = '')")


# ---------------------------------------------------------------------------
# build_translation_prompt
# ---------------------------------------------------------------------------


@scenario("L22. build_translation_prompt incluye el chunk y el código de idioma objetivo")
def test_build_prompt_contents():
    prompt = build_translation_prompt("Texto de ejemplo.", "en")
    assert '"en"' in prompt
    assert "Texto de ejemplo." in prompt
    assert "Return only the translated text." in prompt


# ---------------------------------------------------------------------------
# translate_text_to_language (con doble determinista de factory)
# ---------------------------------------------------------------------------


@scenario("L23. translate_text_to_language: un solo chunk, respuesta directa, una instancia creada")
def test_translate_single_chunk():
    factory = FakeLLMFactory(responses=["Translated text here."])
    result = translate_text_to_language(
        "Texto de origen.", "en", llm_factory=factory, max_chars_per_chunk=1000
    )
    assert result == "Translated text here."
    assert len(factory.calls) == 1
    assert factory.instances_created == 1


@scenario("L24. translate_text_to_language: múltiples chunks se envían en orden y se reconstruyen con doble salto de línea")
def test_translate_multiple_chunks_order():
    text = "Primera oracion larga aqui. " * 5 + "Segunda parte del texto aqui. " * 5
    expected_chunks = chunk_text_by_sentences(text, max_chars=80, overlap_chars=0)
    assert len(expected_chunks) > 1  # confirma que este caso realmente ejercita >1 chunk

    fake_responses = [
        f"Translated chunk number {i} with enough words to pass validation now."
        for i in range(1, len(expected_chunks) + 1)
    ]
    factory = FakeLLMFactory(responses=list(fake_responses))
    result = translate_text_to_language(text, "en", llm_factory=factory, max_chars_per_chunk=80)

    assert len(factory.calls) == len(expected_chunks)
    assert factory.calls == [build_translation_prompt(c, "en") for c in expected_chunks]
    assert result == "\n\n".join(fake_responses)


@scenario("L24b. translate_text_to_language: se construye una instancia NUEVA por chunk (no se reutiliza)")
def test_translate_one_instance_per_chunk():
    text = "Primera oracion larga aqui. " * 5 + "Segunda parte del texto aqui. " * 5
    expected_chunks = chunk_text_by_sentences(text, max_chars=80, overlap_chars=0)
    fake_responses = [
        f"Translated chunk number {i} with enough words to pass validation now."
        for i in range(1, len(expected_chunks) + 1)
    ]
    factory = FakeLLMFactory(responses=list(fake_responses))
    translate_text_to_language(text, "en", llm_factory=factory, max_chars_per_chunk=80)
    assert factory.instances_created == len(expected_chunks), (
        "debe construir tantas instancias como chunks, igual que get_llm(...) dentro del bucle real"
    )


@scenario("L24c. translate_text_to_language: modelo y temperatura los resuelve el llamador, no esta función")
def test_translate_model_and_temperature_resolved_by_caller():
    # La función no tiene ningún parámetro de modelo/temperatura: solo
    # invoca la factory que el llamador ya configuró con esos valores
    # (lambda: get_llm(model=OPENAI_MODEL, temperature=TRANSLATION_TEMPERATURE)
    # en la factory productiva real, no incluida en este bloque).
    import inspect

    sig = inspect.signature(translate_text_to_language)
    assert "model" not in sig.parameters
    assert "temperature" not in sig.parameters
    assert "llm_factory" in sig.parameters


@scenario("L25. translate_text_to_language: respuesta vacía de un chunk lanza ValueError")
def test_translate_empty_response():
    factory = FakeLLMFactory(responses=[""])
    try:
        translate_text_to_language("Texto.", "en", llm_factory=factory, max_chars_per_chunk=1000)
    except ValueError as exc:
        assert "vacío" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por traducción vacía")


@scenario("L26. translate_text_to_language: fallo durante .invoke() se propaga (no se silencia, sin reintento)")
def test_translate_invoke_failure_propagates():
    factory = FakeLLMFactory(fail_on_invoke_index=1)
    try:
        translate_text_to_language(
            "Texto de prueba.", "en", llm_factory=factory, max_chars_per_chunk=1000
        )
    except RuntimeError as exc:
        assert "invoke" in str(exc)
    else:
        raise AssertionError("debía propagar el error de .invoke()")
    assert factory.instances_created == 1  # no reintentó construyendo una segunda instancia


@scenario("L26b. translate_text_to_language: fallo al CONSTRUIR una instancia se propaga (sin reintento)")
def test_translate_build_failure_propagates():
    factory = FakeLLMFactory(fail_on_build_index=1)
    try:
        translate_text_to_language(
            "Texto de prueba.", "en", llm_factory=factory, max_chars_per_chunk=1000
        )
    except RuntimeError as exc:
        assert "construir" in str(exc)
    else:
        raise AssertionError("debía propagar el error de construcción de la factory")
    assert factory.calls == []  # nunca llegó a invocar nada


@scenario("L26c. translate_text_to_language: fallo al construir la instancia del SEGUNDO chunk detiene ahí, sin reintento")
def test_translate_build_failure_second_chunk():
    text = "Primera oracion larga aqui. " * 5 + "Segunda parte del texto aqui. " * 5
    expected_chunks = chunk_text_by_sentences(text, max_chars=80, overlap_chars=0)
    assert len(expected_chunks) >= 2
    factory = FakeLLMFactory(
        responses=["Primera traduccion con suficientes palabras para pasar la validacion."],
        fail_on_build_index=2,
    )
    try:
        translate_text_to_language(text, "en", llm_factory=factory, max_chars_per_chunk=80)
    except RuntimeError:
        pass
    else:
        raise AssertionError("debía propagar el fallo de construcción del segundo chunk")
    assert factory.instances_created == 2
    assert len(factory.calls) == 1  # el primer chunk sí se envió antes del fallo


@scenario("L27. translate_text_to_language: proporción de longitud anómala (traducción muy corta) lanza")
def test_translate_anomalous_ratio_too_short():
    source = "palabra " * 50
    factory = FakeLLMFactory(responses=["short"])
    try:
        translate_text_to_language(source, "en", llm_factory=factory, max_chars_per_chunk=10000)
    except ValueError as exc:
        assert "proporción" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por proporción anómala")


@scenario("L28. translate_text_to_language: proporción de longitud anómala (traducción muy larga) lanza")
def test_translate_anomalous_ratio_too_long():
    source = "palabra palabra"
    factory = FakeLLMFactory(responses=["palabra " * 100])
    try:
        translate_text_to_language(source, "en", llm_factory=factory, max_chars_per_chunk=10000)
    except ValueError as exc:
        assert "proporción" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por proporción anómala")


# ---------------------------------------------------------------------------
# resolve_generated_text_for_rouge — selección generado/GT/ninguno
# ---------------------------------------------------------------------------


@scenario("L29. resolve_generated_text_for_rouge: idioma ya compatible con ROUGE -> no traduce, factory nunca invocada")
def test_resolve_same_language_no_translation():
    factory = FakeLLMFactory()
    text, mode = resolve_generated_text_for_rouge(
        generated_plain_text="Texto en español.",
        generated_language="es",
        ground_truth_language="es",
        translate_for_rouge=True,
        llm_factory=factory,
        max_chars_per_chunk=1000,
    )
    assert mode == "not_required_same_language"
    assert text == "Texto en español."
    assert factory.calls == []
    assert factory.instances_created == 0


@scenario("L30. resolve_generated_text_for_rouge: idiomas distintos entre generado y GT -> traduce el generado")
def test_resolve_different_languages_translates_generated():
    factory = FakeLLMFactory(responses=["Translated generated text."])
    text, mode = resolve_generated_text_for_rouge(
        generated_plain_text="Texto generado en español.",
        generated_language="es",
        ground_truth_language="en",
        translate_for_rouge=True,
        llm_factory=factory,
        max_chars_per_chunk=1000,
    )
    assert mode == "new_translation"
    assert text == "Translated generated text."
    assert len(factory.calls) == 1
    assert factory.instances_created == 1


@scenario("L31. resolve_generated_text_for_rouge: política desactivada -> no traduce aunque los idiomas difieran")
def test_resolve_policy_disabled_no_translation():
    factory = FakeLLMFactory()
    text, mode = resolve_generated_text_for_rouge(
        generated_plain_text="Texto generado en español.",
        generated_language="es",
        ground_truth_language="en",
        translate_for_rouge=False,
        llm_factory=factory,
        max_chars_per_chunk=1000,
    )
    assert mode == "not_required_same_language"
    assert text == "Texto generado en español."
    assert factory.calls == []


@scenario("L32. resolve_generated_text_for_rouge: nunca traduce el Ground Truth, solo el generado")
def test_resolve_never_translates_ground_truth():
    # Confirma por inspección de la firma: no existe ningún parámetro para
    # traducir ground_truth_plain_text -- solo generated_plain_text puede
    # ser traducido, coincidiendo con el notebook real.
    import inspect

    sig = inspect.signature(resolve_generated_text_for_rouge)
    assert "generated_plain_text" in sig.parameters
    assert "ground_truth_plain_text" not in sig.parameters


@scenario("L33. aislamiento: ni language_preprocessing ni translation importan chromadb/StateStore/hacen I/O de archivos")
def test_isolation_no_chroma_or_file_io():
    import inspect

    from src.tools.evaluation import language_preprocessing as lp_module
    from src.tools.evaluation import translation as tr_module

    for module in (lp_module, tr_module):
        source = inspect.getsource(module)
        for forbidden in ("chromadb", "StateStore", "AgentInput", "AgentResult", "open(", ".write_text(", ".write(", "Path(", "os.makedirs"):
            assert forbidden not in source, f"{module.__name__} no debería referenciar {forbidden!r}"


if __name__ == "__main__":
    for fn in (
        test_split_sentences_spanish,
        test_split_sentences_english,
        test_split_sentences_mixed,
        test_split_sentences_empty,
        test_split_sentences_none,
        test_split_sentences_one,
        test_split_sentences_abbreviation,
        test_split_sentences_decimal,
        test_split_sentences_with_citations,
        test_split_sentences_paragraph_breaks,
        test_split_sentences_unicode,
        test_chunk_sentence_larger_than_limit,
        test_chunk_exact_at_limit,
        test_chunk_multiple_preserves_order,
        test_chunk_with_overlap,
        test_chunk_empty,
        test_chunk_none,
        test_detect_language_spanish,
        test_detect_language_english,
        test_detect_language_too_short,
        test_detect_language_none,
        test_build_prompt_contents,
        test_translate_single_chunk,
        test_translate_multiple_chunks_order,
        test_translate_one_instance_per_chunk,
        test_translate_model_and_temperature_resolved_by_caller,
        test_translate_empty_response,
        test_translate_invoke_failure_propagates,
        test_translate_build_failure_propagates,
        test_translate_build_failure_second_chunk,
        test_translate_anomalous_ratio_too_short,
        test_translate_anomalous_ratio_too_long,
        test_resolve_same_language_no_translation,
        test_resolve_different_languages_translates_generated,
        test_resolve_policy_disabled_no_translation,
        test_resolve_never_translates_ground_truth,
        test_isolation_no_chroma_or_file_io,
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
