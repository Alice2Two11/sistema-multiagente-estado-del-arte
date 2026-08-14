"""Fase 2A -- contrato canónico mínimo (``sentences[]``), sin conectar
todavía al pipeline completo de Agent06 ni materializar ``state_of_
art_draft.json``. Solo prueba ``validate_and_parse_sentences_v2`` y sus
funciones auxiliares (``canonical_sentences.py``), reutilizando -- sin
modificarlas -- las funciones puras ya existentes de
``normalization.py``.

Multidominio y genérico por diseño: ningún test usa contenido de
Exp07/S6/biocapacidad ni ningún experimento real -- los ejemplos son
deliberadamente de dominios distintos entre sí (métodos, ecología,
medicina) para confirmar que nada depende de un vocabulario o tema
específico."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
    EMPTY_SENTENCE_ITEM,
    INLINE_CITATION_NOT_ALLOWED,
    INVALID_CITATION,
    INVALID_SENTENCES_STRUCTURE,
    MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE,
    SECTION_ID_MISMATCH,
    SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES,
    UNEXPECTED_SENTENCE_FIELD,
    validate_and_parse_sentences_v2,
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


def _errors_have_code(errors, code):
    return any(str(e).startswith(code) for e in errors)


@scenario("V2A01. Una oración válida con una cita válida -> aceptada, sentence_id=0")
def test_single_valid_sentence_with_valid_citation():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El modelo alcanza una precisión estable en el conjunto de validación experimental completo.",
            "supporting_citations": ["[paper_alpha.pdf | chunk_0001]"],
        }],
    }
    allowed = {("paper_alpha.pdf", "chunk_0001")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is True
    assert result["errors"] == []
    assert len(result["sentences"]) == 1
    assert result["sentences"][0]["sentence_id"] == 0
    assert result["sentences"][0]["supporting_citations"] == ["[paper_alpha.pdf | chunk_0001]"]


@scenario("V2A02. Dos oraciones dentro de un solo text -> SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES, rechazo total")
def test_two_sentences_in_one_text_item_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El tratamiento redujo la inflamación significativamente. El efecto se mantuvo durante seis semanas consecutivas.",
            "supporting_citations": ["[study_beta.pdf | chunk_0007]"],
        }],
    }
    allowed = {("study_beta.pdf", "chunk_0007")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES)


@scenario("V2A03. Texto vacío -> EMPTY_SENTENCE_ITEM, rechazo total")
def test_empty_text_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{"text": "   ", "supporting_citations": []}],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], EMPTY_SENTENCE_ITEM)


@scenario("V2A04. Oración sustantiva sin cita -> MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE")
def test_substantive_sentence_without_citation_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "La reducción observada en la tasa de fallos mecánicos fue considerable durante todo el período evaluado.",
            "supporting_citations": [],
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE)


@scenario("V2A05. Cita inexistente (no resuelve contra allowed_pairs) -> INVALID_CITATION, rechazo total")
def test_nonexistent_citation_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El algoritmo propuesto converge más rápido que los métodos de referencia comparados en el estudio.",
            "supporting_citations": ["[paper_gamma.pdf | chunk_9999]"],
        }],
    }
    allowed = {("paper_gamma.pdf", "chunk_0001")}  # chunk distinto -- no resuelve
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)


@scenario("V2A06. Mezcla de una cita válida y una inválida en la misma oración -> rechazo COMPLETO, no parcial")
def test_mixed_valid_and_invalid_citations_rejects_whole_section():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "Los resultados combinados muestran una mejora consistente en ambos escenarios experimentales evaluados.",
            "supporting_citations": ["[paper_delta.pdf | chunk_0002]", "[paper_delta.pdf | chunk_unknown]"],
        }],
    }
    allowed = {("paper_delta.pdf", "chunk_0002")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)


@scenario("V2A07. Dos oraciones válidas en dos elementos distintos -> aceptación completa")
def test_two_valid_sentences_in_separate_items_accepted():
    payload = {
        "section_id": "S1",
        "sentences": [
            {
                "text": "El primer experimento confirmó la hipótesis inicial sobre el comportamiento térmico del material.",
                "supporting_citations": ["[paper_epsilon.pdf | chunk_0010]"],
            },
            {
                "text": "El segundo experimento replicó estos hallazgos bajo condiciones ambientales considerablemente distintas.",
                "supporting_citations": ["[paper_zeta.pdf | chunk_0003]"],
            },
        ],
    }
    allowed = {("paper_epsilon.pdf", "chunk_0010"), ("paper_zeta.pdf", "chunk_0003")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is True
    assert len(result["sentences"]) == 2


@scenario("V2A08. sentence_id asignados 0 y 1 por el sistema, puramente posicional, nunca UUID")
def test_sentence_id_assigned_positionally():
    payload = {
        "section_id": "S1",
        "sentences": [
            {
                "text": "La primera afirmación describe un fenómeno observado consistentemente en múltiples estudios independientes.",
                "supporting_citations": ["[paper_eta.pdf | chunk_0001]"],
            },
            {
                "text": "La segunda afirmación describe un fenómeno relacionado pero mecánicamente distinto al anterior.",
                "supporting_citations": ["[paper_theta.pdf | chunk_0002]"],
            },
        ],
    }
    allowed = {("paper_eta.pdf", "chunk_0001"), ("paper_theta.pdf", "chunk_0002")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is True
    ids = [s["sentence_id"] for s in result["sentences"]]
    assert ids == [0, 1]
    for sid in ids:
        assert isinstance(sid, int)


@scenario("V2A09. Ninguna cita inválida se elimina silenciosamente -- queda registrada en errors, nunca desaparece sin rastro")
def test_invalid_citation_never_silently_dropped():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El análisis comparativo reveló diferencias estadísticamente significativas entre los grupos estudiados.",
            "supporting_citations": ["[paper_iota.pdf | chunk_fantasma]"],
        }],
    }
    allowed = {("paper_iota.pdf", "chunk_real")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    # La cita inválida EXACTA aparece en el mensaje de error -- no solo
    # un código genérico sin rastro de cuál cita fue.
    assert any("chunk_fantasma" in str(e) for e in result["errors"])


@scenario("V2A10. Importar canonical_sentences.py no produce efectos secundarios")
def test_import_has_no_side_effects():
    import importlib
    import src.tools.draft_writing.canonical_sentences as module

    # Reimportar no debe registrar nada nuevo, no debe tocar el
    # sistema de archivos, no debe modificar ningún estado global --
    # simplemente recarga el módulo y confirma que sigue exponiendo
    # exactamente las mismas funciones puras, sin ningún efecto
    # colateral observable.
    importlib.reload(module)
    assert callable(module.validate_and_parse_sentences_v2)
    assert callable(module.materialize_initial_section_v2)
    assert callable(module.generate_section_canonical_v2)
    # Nota de fase 3: generate_section_canonical_v2 ya NO es un stub
    # (implementación real desde Fase 3, ver test_canonical_sentences_
    # v2_phase3_generation.py) -- esta prueba de fase 2A ya no verifica
    # NotImplementedError; solo confirma que el import en sí es limpio
    # y expone las funciones esperadas, sin ejecutar ninguna lógica que
    # requiera un runtime real.


@scenario("V2A11. supporting_citations es string en vez de list -> rechazo (INVALID_CITATION, no se filtra en silencio)")
def test_supporting_citations_string_instead_of_list_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El estudio observacional confirmó una asociación consistente entre ambas variables clínicas medidas.",
            "supporting_citations": "[paper_kappa.pdf | chunk_0004]",
        }],
    }
    allowed = {("paper_kappa.pdf", "chunk_0004")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)


@scenario("V2A12. Lista contiene una cita malformada (no matchea CITATION_RE) -> INVALID_CITATION, nunca MISSING_CITATIONS")
def test_malformed_citation_in_list_reports_invalid_not_missing():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "La respuesta inmunológica varió significativamente entre los distintos grupos etarios analizados.",
            "supporting_citations": ["paper_lambda.pdf | chunk_0005"],  # sin corchetes -- malformada
        }],
    }
    result = validate_and_parse_sentences_v2(payload, {("paper_lambda.pdf", "chunk_0005")})
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)
    # Nunca debe reportarse como "faltan citas" -- la cita SÍ fue
    # declarada, solo está malformada; extract_claim_pairs la habría
    # descartado en silencio, produciendo MISSING_CITATIONS por error.
    assert not _errors_have_code(result["errors"], MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE)


@scenario("V2A13. Mezcla de una cita bien formada válida y un string malformado -> rechazo TOTAL de la sección")
def test_mix_of_valid_and_malformed_citation_rejects_whole_section():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "Ambos indicadores mostraron una correlación positiva bajo las condiciones experimentales controladas.",
            "supporting_citations": ["[paper_mu.pdf | chunk_0006]", "paper_mu.pdf chunk_0007"],
        }],
    }
    allowed = {("paper_mu.pdf", "chunk_0006"), ("paper_mu.pdf", "chunk_0007")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)


@scenario("V2A14. Oración NO sustantiva con cita malformada -> también rechazo; ninguna cita declarada puede desaparecer sin error")
def test_non_substantive_sentence_with_malformed_citation_also_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "Sí.",  # no sustantiva -- menos de 8 palabras
            "supporting_citations": ["paper_nu.pdf chunk_0008"],  # malformada
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_CITATION)


@scenario("V2A15. Cita inline dentro de text + supporting_citation válida -> rechazo por INLINE_CITATION_NOT_ALLOWED")
def test_inline_citation_in_text_rejected_even_with_valid_supporting_citation():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El modelo alcanza una precisión notable [paper_xi.pdf | chunk_0009] en el conjunto de prueba completo.",
            "supporting_citations": ["[paper_xi.pdf | chunk_0009]"],
        }],
    }
    allowed = {("paper_xi.pdf", "chunk_0009")}
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INLINE_CITATION_NOT_ALLOWED)


@scenario("V2A16. El valor crudo de la cita inválida queda visible en errors -- nunca solo un código genérico sin rastro")
def test_raw_invalid_citation_value_visible_in_errors():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El análisis longitudinal reveló cambios progresivos en los marcadores biológicos evaluados.",
            "supporting_citations": ["paper_omicron.pdf__chunk_0010_SIN_FORMATO"],
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert any("paper_omicron.pdf__chunk_0010_SIN_FORMATO" in str(e) for e in result["errors"])


@scenario("V2A17. section_id ausente -> INVALID_SENTENCES_STRUCTURE")
def test_missing_section_id_rejected():
    payload = {
        "sentences": [{
            "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
            "supporting_citations": [],
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert _errors_have_code(result["errors"], INVALID_SENTENCES_STRUCTURE)


@scenario("V2A18. section_id vacío/no-string -> INVALID_SENTENCES_STRUCTURE")
def test_empty_or_non_string_section_id_rejected():
    base_sentences = [{
        "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
        "supporting_citations": [],
    }]
    for bad_section_id in ("", "   ", 5, None, []):
        payload = {"section_id": bad_section_id, "sentences": base_sentences}
        result = validate_and_parse_sentences_v2(payload, set())
        assert result["validation_ok"] is False, bad_section_id
        assert result["sentences"] is None, bad_section_id
        assert _errors_have_code(result["errors"], INVALID_SENTENCES_STRUCTURE), bad_section_id


@scenario("V2A19. expected_section_id='S2' pero payload declara 'S1' -> SECTION_ID_MISMATCH:S2:S1, nunca se infiere ni se sobrescribe")
def test_section_id_mismatch_against_expected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
            "supporting_citations": [],
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set(), expected_section_id="S2")
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert result["errors"] == ["SECTION_ID_MISMATCH:S2:S1"]


@scenario("V2A20. El LLM intenta enviar sentence_id (campo que el sistema asigna después) -> rechazo, UNEXPECTED_SENTENCE_FIELD")
def test_llm_sending_sentence_id_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
            "supporting_citations": [],
            "sentence_id": 0,
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert result["errors"] == ["UNEXPECTED_SENTENCE_FIELD:0:sentence_id"]


@scenario("V2A21. El LLM intenta enviar identity_action/parent_claim_uids -> rechazo, UNEXPECTED_SENTENCE_FIELD")
def test_llm_sending_identity_fields_rejected():
    for field, value in (("identity_action", "NEW"), ("parent_claim_uids", []), ("claim_uid", "x"), ("claim", "x"), ("claim_id", "S1_C1")):
        payload = {
            "section_id": "S1",
            "sentences": [{
                "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
                "supporting_citations": [],
                field: value,
            }],
        }
        result = validate_and_parse_sentences_v2(payload, set())
        assert result["validation_ok"] is False, field
        assert result["sentences"] is None, field
        assert result["errors"] == [f"UNEXPECTED_SENTENCE_FIELD:0:{field}"], field


@scenario("V2A22. Campo desconocido cualquiera dentro de una oración -> rechazo genérico, nunca ignorado en silencio")
def test_any_unknown_field_rejected():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
            "supporting_citations": [],
            "confidence_score": 0.87,
        }],
    }
    result = validate_and_parse_sentences_v2(payload, set())
    assert result["validation_ok"] is False
    assert result["sentences"] is None
    assert result["errors"] == ["UNEXPECTED_SENTENCE_FIELD:0:confidence_score"]


@scenario("V2A23. Payload correcto con expected_section_id coincidente -> aceptación normal")
def test_correct_payload_with_matching_expected_section_id_accepted():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El proceso metabólico estudiado mostró variaciones consistentes bajo distintas condiciones experimentales.",
            "supporting_citations": ["[paper_pi.pdf | chunk_0011]"],
        }],
    }
    result = validate_and_parse_sentences_v2(
        payload, {("paper_pi.pdf", "chunk_0011")}, expected_section_id="S1",
    )
    assert result["validation_ok"] is True
    assert result["errors"] == []
    assert len(result["sentences"]) == 1
    assert result["sentences"][0]["sentence_id"] == 0


if __name__ == "__main__":
    for fn in (
        test_single_valid_sentence_with_valid_citation,
        test_two_sentences_in_one_text_item_rejected,
        test_empty_text_rejected,
        test_substantive_sentence_without_citation_rejected,
        test_nonexistent_citation_rejected,
        test_mixed_valid_and_invalid_citations_rejects_whole_section,
        test_two_valid_sentences_in_separate_items_accepted,
        test_sentence_id_assigned_positionally,
        test_invalid_citation_never_silently_dropped,
        test_import_has_no_side_effects,
        test_supporting_citations_string_instead_of_list_rejected,
        test_malformed_citation_in_list_reports_invalid_not_missing,
        test_mix_of_valid_and_malformed_citation_rejects_whole_section,
        test_non_substantive_sentence_with_malformed_citation_also_rejected,
        test_inline_citation_in_text_rejected_even_with_valid_supporting_citation,
        test_raw_invalid_citation_value_visible_in_errors,
        test_missing_section_id_rejected,
        test_empty_or_non_string_section_id_rejected,
        test_section_id_mismatch_against_expected,
        test_llm_sending_sentence_id_rejected,
        test_llm_sending_identity_fields_rejected,
        test_any_unknown_field_rejected,
        test_correct_payload_with_matching_expected_section_id_accepted,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    raise SystemExit(1 if failed else 0)
