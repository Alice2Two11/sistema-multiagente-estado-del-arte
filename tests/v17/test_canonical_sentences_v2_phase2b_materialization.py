"""Fase 2B -- materialización determinista GENERACIÓN INICIAL:
``sentences[]`` (ya validado por ``validate_and_parse_sentences_v2``,
CONTRATO EVIDENCE HANDLES) -> ``draft_text`` + ``claims[]``. Todavía
desconectada de ``execute()`` de Agent06 en el sentido de que estas
pruebas ejercitan la materialización aisladamente -- ``materialize_
initial_section_v2`` en sí NO cambió con el contrato de evidence
handles (sigue consumiendo ``sentences[i].supporting_citations`` ya
resuelto, exactamente igual que antes).

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
    build_evidence_handle_map,
    materialize_initial_section_v2,
    validate_and_parse_sentences_v2,
)
from src.tools.draft_writing.normalization import CITATION_RE  # noqa: E402

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


def _map(*evidence_rows):
    return build_evidence_handle_map(list(evidence_rows))


def _row(source, chunk):
    return {"source_filename": source, "chunk_id": chunk}


def _parse(payload, handle_map):
    result = validate_and_parse_sentences_v2(payload, handle_map)
    assert result["validation_ok"] is True, result["errors"]
    return result["sentences"]


@scenario("V2B01. Una oración sustantiva -> un claim")
def test_one_substantive_sentence_one_claim():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El compuesto reduce la actividad enzimática de forma dependiente de la concentración administrada.",
            "supporting_evidence_ids": ["E1"],
        }],
    }
    parsed = _parse(payload, _map(_row("study_one.pdf", "chunk_0001")))
    result = materialize_initial_section_v2(parsed, "S1")
    assert len(result["claims"]) == 1


@scenario("V2B02. Tres oraciones sustantivas -> tres claims en el mismo orden")
def test_three_substantive_sentences_three_claims_in_order():
    payload = {
        "section_id": "S2",
        "sentences": [
            {"text": "La primera observación describe un incremento sostenido en la variable medida durante el ensayo.", "supporting_evidence_ids": ["E1"]},
            {"text": "La segunda observación describe un patrón inverso bajo las mismas condiciones experimentales evaluadas.", "supporting_evidence_ids": ["E2"]},
            {"text": "La tercera observación confirma la consistencia de ambos patrones en ensayos independientes repetidos.", "supporting_evidence_ids": ["E3"]},
        ],
    }
    handle_map = _map(_row("a.pdf", "c1"), _row("b.pdf", "c2"), _row("c.pdf", "c3"))
    parsed = _parse(payload, handle_map)
    result = materialize_initial_section_v2(parsed, "S2")
    assert [c["claim_id"] for c in result["claims"]] == ["S2_C1", "S2_C2", "S2_C3"]
    assert result["claims"][0]["claim"].startswith("La primera")
    assert result["claims"][1]["claim"].startswith("La segunda")
    assert result["claims"][2]["claim"].startswith("La tercera")


@scenario("V2B03. Oración NO sustantiva permanece en draft_text pero no genera claim")
def test_non_substantive_sentence_stays_in_draft_text_without_claim():
    payload = {
        "section_id": "S3",
        "sentences": [
            {"text": "Introducción.", "supporting_evidence_ids": []},
            {"text": "El modelo propuesto supera consistentemente a los métodos de referencia evaluados en el estudio.", "supporting_evidence_ids": ["E1"]},
        ],
    }
    parsed = _parse(payload, _map(_row("d.pdf", "c4")))
    result = materialize_initial_section_v2(parsed, "S3")
    assert "Introducción." in result["draft_text"]
    assert len(result["claims"]) == 1


@scenario("V2B04. claim == sentence.text SIN puntuación final -- ajuste de Fase 4: el consumidor real de 07 exige que claim sea subcadena exacta de draft_text (ver hallazgo AGENT07_AGENT06_CLAIM_SPAN_AMBIGUOUS), lo que excluye la puntuación final, igual que ya hacía legacy")
def test_claim_text_exactly_equals_sentence_text():
    text = "El proceso de degradación térmica ocurre a una tasa constante bajo las condiciones controladas del experimento."
    payload = {"section_id": "S4", "sentences": [{"text": text, "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("e.pdf", "c5")))
    result = materialize_initial_section_v2(parsed, "S4")
    assert result["claims"][0]["claim"] == text.rstrip(".")
    assert parsed[0]["text"] == text


@scenario("V2B05. Citas de sentence == citas del claim, exactamente")
def test_sentence_citations_equal_claim_citations():
    payload = {
        "section_id": "S5",
        "sentences": [{
            "text": "El análisis multivariado reveló interacciones significativas entre las tres variables consideradas conjuntamente.",
            "supporting_evidence_ids": ["E1", "E2"],
        }],
    }
    handle_map = _map(_row("f.pdf", "c6"), _row("g.pdf", "c7"))
    parsed = _parse(payload, handle_map)
    result = materialize_initial_section_v2(parsed, "S5")
    assert result["claims"][0]["supporting_citations"] == parsed[0]["supporting_citations"]
    assert result["claims"][0]["supporting_citations"] == ["[f.pdf | c6]", "[g.pdf | c7]"]


@scenario("V2B06. Citas aparecen correctamente en draft_text y CITATION_RE las detecta")
def test_citations_appear_in_draft_text_and_are_detected_by_citation_re():
    payload = {
        "section_id": "SX",
        "sentences": [{
            "text": "El tratamiento combinado mostró eficacia superior respecto a las monoterapias evaluadas en el ensayo clínico.",
            "supporting_evidence_ids": ["E1"],
        }],
    }
    parsed = _parse(payload, _map(_row("h.pdf", "c8")))
    result = materialize_initial_section_v2(parsed, "SX")
    matches = CITATION_RE.findall(result["draft_text"])
    assert matches == [("h.pdf", "c8")]


@scenario("V2B07. claim_id sigue exactamente la convención productiva vigente ({section_id}_C{n}, n desde 1)")
def test_claim_id_follows_exact_production_convention():
    payload = {
        "section_id": "S7",
        "sentences": [
            {"text": "El primer resultado confirma la hipótesis original planteada al inicio del estudio experimental.", "supporting_evidence_ids": ["E1"]},
            {"text": "El segundo resultado extiende ese hallazgo a un contexto experimental distinto y más amplio.", "supporting_evidence_ids": ["E2"]},
        ],
    }
    handle_map = _map(_row("i.pdf", "c9"), _row("j.pdf", "c10"))
    parsed = _parse(payload, handle_map)
    result = materialize_initial_section_v2(parsed, "S7")
    assert result["claims"][0]["claim_id"] == "S7_C1"
    assert result["claims"][1]["claim_id"] == "S7_C2"


@scenario("V2B08. Todos los claims iniciales son NEW")
def test_all_initial_claims_are_new():
    payload = {
        "section_id": "S8",
        "sentences": [
            {"text": "La primera afirmación describe un fenómeno biológico observado repetidamente en el laboratorio.", "supporting_evidence_ids": ["E1"]},
            {"text": "La segunda afirmación describe un fenómeno relacionado pero mecánicamente distinto al primero.", "supporting_evidence_ids": ["E2"]},
        ],
    }
    handle_map = _map(_row("k.pdf", "c11"), _row("l.pdf", "c12"))
    parsed = _parse(payload, handle_map)
    result = materialize_initial_section_v2(parsed, "S8")
    assert all(c["identity_action"] == "NEW" for c in result["claims"])


@scenario("V2B09. Todos tienen parent_claim_uids=[]")
def test_all_initial_claims_have_no_parents():
    payload = {"section_id": "S9", "sentences": [{"text": "El fenómeno observado no había sido reportado previamente en la literatura especializada del área.", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("m.pdf", "c13")))
    result = materialize_initial_section_v2(parsed, "S9")
    assert result["claims"][0]["parent_claim_uids"] == []


@scenario("V2B10. claim_uid generado mediante la infraestructura real actual (resolve_claim_identity/default_mint_claim_uid)")
def test_claim_uid_generated_via_real_infrastructure():
    import uuid as uuid_module

    payload = {"section_id": "S10", "sentences": [{"text": "El compuesto sintetizado presentó una estabilidad térmica notablemente superior a la esperada inicialmente.", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("n.pdf", "c14")))
    result = materialize_initial_section_v2(parsed, "S10")
    claim_uid = result["claims"][0]["claim_uid"]
    parsed_uuid = uuid_module.UUID(claim_uid)
    assert str(parsed_uuid) == claim_uid
    assert result["claims"][0]["claim_version"] == 1
    assert result["claims"][0]["created_round"] == 1
    assert result["claims"][0]["updated_round"] == 1


@scenario("V2B11. Mismo input materializado dos veces -> mismos resultados deterministas (salvo claim_uid, que es intencionalmente único por diseño)")
def test_same_input_materialized_twice_is_deterministic():
    payload = {
        "section_id": "S11",
        "sentences": [{
            "text": "El experimento replicado confirmó los resultados originales bajo condiciones ligeramente distintas del entorno.",
            "supporting_evidence_ids": ["E1"],
        }],
    }
    handle_map = _map(_row("o.pdf", "c15"))
    parsed1 = _parse(payload, handle_map)
    parsed2 = _parse(payload, handle_map)
    result1 = materialize_initial_section_v2(parsed1, "S11")
    result2 = materialize_initial_section_v2(parsed2, "S11")

    assert result1["draft_text"] == result2["draft_text"]
    assert len(result1["claims"]) == len(result2["claims"]) == 1
    c1, c2 = result1["claims"][0], result2["claims"][0]
    for key in ("claim_id", "claim", "supporting_citations", "identity_action",
                "parent_claim_uids", "claim_text_fingerprint", "claim_version",
                "created_round", "updated_round"):
        assert c1[key] == c2[key], key
    assert c1["claim_uid"] != c2["claim_uid"]

    counter = {"n": 0}

    def deterministic_mint():
        counter["n"] += 1
        return f"fixed-uid-{counter['n']}"

    result3 = materialize_initial_section_v2(_parse(payload, handle_map), "S11", mint_uid=deterministic_mint)
    counter["n"] = 0
    result4 = materialize_initial_section_v2(_parse(payload, handle_map), "S11", mint_uid=deterministic_mint)
    assert result3 == result4


@scenario("V2B12. Ninguna oración sustantiva queda sin claim")
def test_no_substantive_sentence_without_claim():
    payload = {
        "section_id": "S12",
        "sentences": [
            {"text": "Contexto general.", "supporting_evidence_ids": []},
            {"text": "El primer hallazgo relevante fue confirmado mediante análisis estadístico riguroso del conjunto completo.", "supporting_evidence_ids": ["E1"]},
            {"text": "Cierre.", "supporting_evidence_ids": []},
            {"text": "El segundo hallazgo relevante amplía la comprensión del fenómeno estudiado en profundidad considerable.", "supporting_evidence_ids": ["E2"]},
        ],
    }
    handle_map = _map(_row("p.pdf", "c16"), _row("q.pdf", "c17"))
    parsed = _parse(payload, handle_map)
    result = materialize_initial_section_v2(parsed, "S12")
    assert len(result["claims"]) == 2
    claim_texts = {c["claim"] for c in result["claims"]}
    assert "El primer hallazgo relevante fue confirmado mediante análisis estadístico riguroso del conjunto completo" in claim_texts
    assert "El segundo hallazgo relevante amplía la comprensión del fenómeno estudiado en profundidad considerable" in claim_texts


@scenario("V2B13. Ningún claim aparece sin oración sustantiva correspondiente (mismo número, mismo texto de origen)")
def test_no_claim_without_corresponding_substantive_sentence():
    payload = {
        "section_id": "S13",
        "sentences": [
            {"text": "El único hallazgo sustantivo de esta sección fue confirmado en tres réplicas experimentales independientes.", "supporting_evidence_ids": ["E1"]},
            {"text": "Fin.", "supporting_evidence_ids": []},
        ],
    }
    parsed = _parse(payload, _map(_row("r.pdf", "c18")))
    result = materialize_initial_section_v2(parsed, "S13")
    assert len(result["claims"]) == 1
    origin_texts_no_punct = {s["text"].rstrip(".") for s in parsed}
    for claim in result["claims"]:
        assert claim["claim"] in origin_texts_no_punct


@scenario("V2B14. sentence.text no se modifica en ningún punto de la materialización")
def test_sentence_text_never_modified():
    payload = {
        "section_id": "S14",
        "sentences": [{
            "text": "El resultado obtenido difiere significativamente de lo reportado en estudios previos similares.",
            "supporting_evidence_ids": ["E1"],
        }],
    }
    parsed = _parse(payload, _map(_row("s.pdf", "c19")))
    original_texts = [dict(s) for s in parsed]
    materialize_initial_section_v2(parsed, "S14")
    assert parsed == original_texts


@scenario("V2B15. El materializador no llama al LLM -- función pura, sin runtime/invoke")
def test_materializer_never_calls_llm():
    import inspect

    from src.tools.draft_writing import canonical_sentences as module

    source = inspect.getsource(module.materialize_initial_section_v2)
    assert "runtime" not in source
    assert "invoke" not in source
    assert ".invoke(" not in source


@scenario("V2B16. Final '.' -> puntuación preservada exacta, cita insertada antes")
def test_punctuation_period_preserved():
    payload = {"section_id": "S16", "sentences": [{"text": "El experimento confirmó la hipótesis planteada al inicio del estudio.", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("t.pdf", "c20")))
    result = materialize_initial_section_v2(parsed, "S16")
    assert result["draft_text"].endswith("[t.pdf | c20].")


@scenario("V2B17. Final '?' -> puntuación preservada exacta")
def test_punctuation_question_mark_preserved():
    payload = {"section_id": "S17", "sentences": [{"text": "¿El resultado observado es realmente concluyente para la hipótesis planteada?", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("u.pdf", "c21")))
    result = materialize_initial_section_v2(parsed, "S17")
    assert result["draft_text"].endswith("[u.pdf | c21]?")


@scenario("V2B18. Final '!' -> puntuación preservada exacta")
def test_punctuation_exclamation_preserved():
    payload = {"section_id": "S18", "sentences": [{"text": "¡El resultado obtenido superó ampliamente las expectativas iniciales del equipo de investigación!", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("v.pdf", "c22")))
    result = materialize_initial_section_v2(parsed, "S18")
    assert result["draft_text"].endswith("[v.pdf | c22]!")


@scenario("V2B19. Final '?!' -> preservado COMPLETO, nunca normalizado a un solo carácter (ejemplo exacto pedido)")
def test_punctuation_question_exclamation_preserved_exactly():
    payload = {"section_id": "S19", "sentences": [{"text": "¿El resultado observado es realmente concluyente para toda la hipótesis?!", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("paper.pdf", "chunk_1")))
    result = materialize_initial_section_v2(parsed, "S19")
    assert result["draft_text"].endswith("[paper.pdf | chunk_1]?!")
    assert "?!" in result["draft_text"]
    assert not result["draft_text"].rstrip().endswith("!.") and not result["draft_text"].rstrip().endswith("!!")


@scenario("V2B20. Final '!!' -> preservado COMPLETO, nunca recortado a un solo '!'")
def test_punctuation_double_exclamation_preserved_exactly():
    payload = {"section_id": "S20", "sentences": [{"text": "El descubrimiento realizado por el equipo resultó absolutamente inesperado para toda la comunidad científica!!", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("w.pdf", "c23")))
    result = materialize_initial_section_v2(parsed, "S20")
    assert result["draft_text"].endswith("[w.pdf | c23]!!")


@scenario("V2B21. Final '...' -> preservado COMPLETO, nunca normalizado a un solo '.'")
def test_punctuation_ellipsis_preserved_exactly():
    payload = {"section_id": "S21", "sentences": [{"text": "El proceso de degradación observado continuó durante todo el período experimental evaluado...", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("x.pdf", "c24")))
    result = materialize_initial_section_v2(parsed, "S21")
    assert result["draft_text"].endswith("[x.pdf | c24]...")


@scenario("V2B22. Oración SIN puntuación final -> ninguna se inventa")
def test_no_trailing_punctuation_none_invented():
    payload = {"section_id": "S22", "sentences": [{"text": "El resultado observado carece de puntuación final en el texto original recibido", "supporting_evidence_ids": ["E1"]}]}
    parsed = _parse(payload, _map(_row("y.pdf", "c25")))
    result = materialize_initial_section_v2(parsed, "S22")
    assert result["draft_text"].endswith("[y.pdf | c25]")
    assert not result["draft_text"].rstrip().endswith((".", "?", "!"))


@scenario("V2B23. Quitando únicamente las citas insertadas, el fragmento conserva EXACTAMENTE el contenido y puntuación de la oración de entrada")
def test_removing_only_inserted_citations_preserves_exact_original_content():
    from src.tools.draft_writing.canonical_sentences import _materialize_sentence_fragment
    from src.tools.draft_writing.normalization import CITATION_RE

    originals = [
        "El experimento confirmó la hipótesis planteada al inicio del estudio.",
        "¿El resultado observado es realmente concluyente para la hipótesis?",
        "¡El resultado obtenido superó ampliamente las expectativas del equipo!",
        "¿El resultado observado es realmente concluyente para la hipótesis?!",
        "El descubrimiento resultó absolutamente inesperado para la comunidad!!",
        "El proceso de degradación continuó durante todo el período evaluado...",
        "El resultado observado carece de puntuación final en el texto recibido",
    ]
    for text in originals:
        fragment = _materialize_sentence_fragment(text, ["[z.pdf | c26]"])
        without_citation = CITATION_RE.sub("", fragment)
        without_citation = " ".join(without_citation.split())
        import re as _re
        without_citation = _re.sub(r"\s+([.?!]+)$", r"\1", without_citation)
        expected = " ".join(text.split())
        assert without_citation == expected, (text, without_citation)


@scenario("V2B24. materialización final mantiene el contrato que consume 07: claim subcadena exacta de draft_text, supporting_citations en formato histórico")
def test_materialization_preserves_downstream_07_contract():
    payload = {
        "section_id": "S24",
        "sentences": [{
            "text": "El nuevo procedimiento reduce el tiempo de procesamiento respecto a las técnicas previamente establecidas.",
            "supporting_evidence_ids": ["E1"],
        }],
    }
    parsed = _parse(payload, _map(_row("aa.pdf", "c27")))
    result = materialize_initial_section_v2(parsed, "S24")
    claim = result["claims"][0]
    assert claim["claim"] in result["draft_text"]
    assert claim["supporting_citations"] == ["[aa.pdf | c27]"]
    for key in ("claim_id", "claim", "supporting_citations", "claim_uid", "identity_action", "parent_claim_uids", "claim_text_fingerprint"):
        assert key in claim


if __name__ == "__main__":
    for fn in (
        test_one_substantive_sentence_one_claim,
        test_three_substantive_sentences_three_claims_in_order,
        test_non_substantive_sentence_stays_in_draft_text_without_claim,
        test_claim_text_exactly_equals_sentence_text,
        test_sentence_citations_equal_claim_citations,
        test_citations_appear_in_draft_text_and_are_detected_by_citation_re,
        test_claim_id_follows_exact_production_convention,
        test_all_initial_claims_are_new,
        test_all_initial_claims_have_no_parents,
        test_claim_uid_generated_via_real_infrastructure,
        test_same_input_materialized_twice_is_deterministic,
        test_no_substantive_sentence_without_claim,
        test_no_claim_without_corresponding_substantive_sentence,
        test_sentence_text_never_modified,
        test_materializer_never_calls_llm,
        test_punctuation_period_preserved,
        test_punctuation_question_mark_preserved,
        test_punctuation_exclamation_preserved,
        test_punctuation_question_exclamation_preserved_exactly,
        test_punctuation_double_exclamation_preserved_exactly,
        test_punctuation_ellipsis_preserved_exactly,
        test_no_trailing_punctuation_none_invented,
        test_removing_only_inserted_citations_preserves_exact_original_content,
        test_materialization_preserves_downstream_07_contract,
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
