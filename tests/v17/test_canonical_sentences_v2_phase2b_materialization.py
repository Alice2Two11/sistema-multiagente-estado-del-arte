"""Fase 2B -- materialización determinista GENERACIÓN INICIAL:
``sentences[]`` (ya validado por ``validate_and_parse_sentences_v2``)
-> ``draft_text`` + ``claims[]``. Todavía desconectada de ``execute()``
de Agent06 -- ``generate_section_canonical_v2`` sigue en
``NotImplementedError``.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
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


def _parse(payload, allowed):
    result = validate_and_parse_sentences_v2(payload, allowed)
    assert result["validation_ok"] is True, result["errors"]
    return result["sentences"]


@scenario("V2B01. Una oración sustantiva -> un claim")
def test_one_substantive_sentence_one_claim():
    payload = {
        "section_id": "S1",
        "sentences": [{
            "text": "El compuesto reduce la actividad enzimática de forma dependiente de la concentración administrada.",
            "supporting_citations": ["[study_one.pdf | chunk_0001]"],
        }],
    }
    parsed = _parse(payload, {("study_one.pdf", "chunk_0001")})
    result = materialize_initial_section_v2(parsed, "S1")
    assert len(result["claims"]) == 1


@scenario("V2B02. Tres oraciones sustantivas -> tres claims en el mismo orden")
def test_three_substantive_sentences_three_claims_in_order():
    payload = {
        "section_id": "S2",
        "sentences": [
            {"text": "La primera observación describe un incremento sostenido en la variable medida durante el ensayo.", "supporting_citations": ["[a.pdf | c1]"]},
            {"text": "La segunda observación describe un patrón inverso bajo las mismas condiciones experimentales evaluadas.", "supporting_citations": ["[b.pdf | c2]"]},
            {"text": "La tercera observación confirma la consistencia de ambos patrones en ensayos independientes repetidos.", "supporting_citations": ["[c.pdf | c3]"]},
        ],
    }
    allowed = {("a.pdf", "c1"), ("b.pdf", "c2"), ("c.pdf", "c3")}
    parsed = _parse(payload, allowed)
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
            {"text": "Introducción.", "supporting_citations": []},
            {"text": "El modelo propuesto supera consistentemente a los métodos de referencia evaluados en el estudio.", "supporting_citations": ["[d.pdf | c4]"]},
        ],
    }
    parsed = _parse(payload, {("d.pdf", "c4")})
    result = materialize_initial_section_v2(parsed, "S3")
    assert "Introducción." in result["draft_text"]
    assert len(result["claims"]) == 1


@scenario("V2B04. claim == sentence.text exacto, carácter por carácter")
def test_claim_text_exactly_equals_sentence_text():
    text = "El proceso de degradación térmica ocurre a una tasa constante bajo las condiciones controladas del experimento."
    payload = {"section_id": "S4", "sentences": [{"text": text, "supporting_citations": ["[e.pdf | c5]"]}]}
    parsed = _parse(payload, {("e.pdf", "c5")})
    result = materialize_initial_section_v2(parsed, "S4")
    assert result["claims"][0]["claim"] == text


@scenario("V2B05. Citas de sentence == citas del claim, exactamente")
def test_sentence_citations_equal_claim_citations():
    payload = {
        "section_id": "S5",
        "sentences": [{
            "text": "El análisis multivariado reveló interacciones significativas entre las tres variables consideradas conjuntamente.",
            "supporting_citations": ["[f.pdf | c6]", "[g.pdf | c7]"],
        }],
    }
    allowed = {("f.pdf", "c6"), ("g.pdf", "c7")}
    parsed = _parse(payload, allowed)
    result = materialize_initial_section_v2(parsed, "S5")
    assert result["claims"][0]["supporting_citations"] == parsed[0]["supporting_citations"]
    assert result["claims"][0]["supporting_citations"] == ["[f.pdf | c6]", "[g.pdf | c7]"]


@scenario("V2B06. Citas aparecen correctamente en draft_text y CITATION_RE las detecta")
def test_citations_appear_in_draft_text_and_are_detected_by_citation_re():
    payload = {
        "section_id": "S6",
        "sentences": [{
            "text": "El tratamiento combinado mostró eficacia superior respecto a las monoterapias evaluadas en el ensayo clínico.",
            "supporting_citations": ["[h.pdf | c8]"],
        }],
    }
    parsed = _parse(payload, {("h.pdf", "c8")})
    result = materialize_initial_section_v2(parsed, "S6")
    matches = CITATION_RE.findall(result["draft_text"])
    assert matches == [("h.pdf", "c8")]


@scenario("V2B07. claim_id sigue exactamente la convención productiva vigente ({section_id}_C{n}, n desde 1)")
def test_claim_id_follows_exact_production_convention():
    payload = {
        "section_id": "S7",
        "sentences": [
            {"text": "El primer resultado confirma la hipótesis original planteada al inicio del estudio experimental.", "supporting_citations": ["[i.pdf | c9]"]},
            {"text": "El segundo resultado extiende ese hallazgo a un contexto experimental distinto y más amplio.", "supporting_citations": ["[j.pdf | c10]"]},
        ],
    }
    allowed = {("i.pdf", "c9"), ("j.pdf", "c10")}
    parsed = _parse(payload, allowed)
    result = materialize_initial_section_v2(parsed, "S7")
    assert result["claims"][0]["claim_id"] == "S7_C1"  # empieza en 1, no en 0
    assert result["claims"][1]["claim_id"] == "S7_C2"


@scenario("V2B08. Todos los claims iniciales son NEW")
def test_all_initial_claims_are_new():
    payload = {
        "section_id": "S8",
        "sentences": [
            {"text": "La primera afirmación describe un fenómeno biológico observado repetidamente en el laboratorio.", "supporting_citations": ["[k.pdf | c11]"]},
            {"text": "La segunda afirmación describe un fenómeno relacionado pero mecánicamente distinto al primero.", "supporting_citations": ["[l.pdf | c12]"]},
        ],
    }
    allowed = {("k.pdf", "c11"), ("l.pdf", "c12")}
    parsed = _parse(payload, allowed)
    result = materialize_initial_section_v2(parsed, "S8")
    assert all(c["identity_action"] == "NEW" for c in result["claims"])


@scenario("V2B09. Todos tienen parent_claim_uids=[]")
def test_all_initial_claims_have_no_parents():
    payload = {"section_id": "S9", "sentences": [{"text": "El fenómeno observado no había sido reportado previamente en la literatura especializada del área.", "supporting_citations": ["[m.pdf | c13]"]}]}
    parsed = _parse(payload, {("m.pdf", "c13")})
    result = materialize_initial_section_v2(parsed, "S9")
    assert result["claims"][0]["parent_claim_uids"] == []


@scenario("V2B10. claim_uid generado mediante la infraestructura real actual (resolve_claim_identity/default_mint_claim_uid)")
def test_claim_uid_generated_via_real_infrastructure():
    import uuid as uuid_module

    payload = {"section_id": "S10", "sentences": [{"text": "El compuesto sintetizado presentó una estabilidad térmica notablemente superior a la esperada inicialmente.", "supporting_citations": ["[n.pdf | c14]"]}]}
    parsed = _parse(payload, {("n.pdf", "c14")})
    result = materialize_initial_section_v2(parsed, "S10")
    claim_uid = result["claims"][0]["claim_uid"]
    # Confirma que es un UUID real y válido -- producido por
    # default_mint_claim_uid (claim_identity.py), no inventado aquí.
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
            "supporting_citations": ["[o.pdf | c15]"],
        }],
    }
    allowed = {("o.pdf", "c15")}
    parsed1 = _parse(payload, allowed)
    parsed2 = _parse(payload, allowed)
    result1 = materialize_initial_section_v2(parsed1, "S11")
    result2 = materialize_initial_section_v2(parsed2, "S11")

    assert result1["draft_text"] == result2["draft_text"]
    assert len(result1["claims"]) == len(result2["claims"]) == 1
    c1, c2 = result1["claims"][0], result2["claims"][0]
    for key in ("claim_id", "claim", "supporting_citations", "identity_action",
                "parent_claim_uids", "claim_text_fingerprint", "claim_version",
                "created_round", "updated_round"):
        assert c1[key] == c2[key], key
    # claim_uid es la única excepción deliberada: cada NEW mintea una
    # identidad real y única -- dos materializaciones del mismo texto
    # no deben compartir identidad por casualidad.
    assert c1["claim_uid"] != c2["claim_uid"]

    # Con un mint_uid inyectado y determinista, incluso claim_uid
    # coincide -- confirma que la única fuente de no-determinismo es
    # el minteo de UUID en sí, no la lógica de materialización.
    counter = {"n": 0}

    def deterministic_mint():
        counter["n"] += 1
        return f"fixed-uid-{counter['n']}"

    result3 = materialize_initial_section_v2(_parse(payload, allowed), "S11", mint_uid=deterministic_mint)
    counter["n"] = 0
    result4 = materialize_initial_section_v2(_parse(payload, allowed), "S11", mint_uid=deterministic_mint)
    assert result3 == result4


@scenario("V2B12. Ninguna oración sustantiva queda sin claim")
def test_no_substantive_sentence_without_claim():
    payload = {
        "section_id": "S12",
        "sentences": [
            {"text": "Contexto general.", "supporting_citations": []},
            {"text": "El primer hallazgo relevante fue confirmado mediante análisis estadístico riguroso del conjunto completo.", "supporting_citations": ["[p.pdf | c16]"]},
            {"text": "Cierre.", "supporting_citations": []},
            {"text": "El segundo hallazgo relevante amplía la comprensión del fenómeno estudiado en profundidad considerable.", "supporting_citations": ["[q.pdf | c17]"]},
        ],
    }
    allowed = {("p.pdf", "c16"), ("q.pdf", "c17")}
    parsed = _parse(payload, allowed)
    result = materialize_initial_section_v2(parsed, "S12")
    substantive_count = sum(1 for s in parsed if len(s["text"].split()) >= 8 or "hallazgo" in s["text"])
    # Verificación directa: ambas oraciones largas producen claim.
    assert len(result["claims"]) == 2
    claim_texts = {c["claim"] for c in result["claims"]}
    assert "El primer hallazgo relevante fue confirmado mediante análisis estadístico riguroso del conjunto completo." in claim_texts
    assert "El segundo hallazgo relevante amplía la comprensión del fenómeno estudiado en profundidad considerable." in claim_texts


@scenario("V2B13. Ningún claim aparece sin oración sustantiva correspondiente (mismo número, mismo texto de origen)")
def test_no_claim_without_corresponding_substantive_sentence():
    payload = {
        "section_id": "S13",
        "sentences": [
            {"text": "El único hallazgo sustantivo de esta sección fue confirmado en tres réplicas experimentales independientes.", "supporting_citations": ["[r.pdf | c18]"]},
            {"text": "Fin.", "supporting_citations": []},
        ],
    }
    parsed = _parse(payload, {("r.pdf", "c18")})
    result = materialize_initial_section_v2(parsed, "S13")
    assert len(result["claims"]) == 1
    origin_texts = {s["text"] for s in parsed}
    for claim in result["claims"]:
        assert claim["claim"] in origin_texts


@scenario("V2B14. sentence.text no se modifica en ningún punto de la materialización")
def test_sentence_text_never_modified():
    payload = {
        "section_id": "S14",
        "sentences": [{
            "text": "El resultado obtenido difiere significativamente de lo reportado en estudios previos similares.",
            "supporting_citations": ["[s.pdf | c19]"],
        }],
    }
    parsed = _parse(payload, {("s.pdf", "c19")})
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
    payload = {"section_id": "S16", "sentences": [{"text": "El experimento confirmó la hipótesis planteada al inicio del estudio.", "supporting_citations": ["[t.pdf | c20]"]}]}
    parsed = _parse(payload, {("t.pdf", "c20")})
    result = materialize_initial_section_v2(parsed, "S16")
    assert result["draft_text"].endswith("[t.pdf | c20].")


@scenario("V2B17. Final '?' -> puntuación preservada exacta")
def test_punctuation_question_mark_preserved():
    payload = {"section_id": "S17", "sentences": [{"text": "¿El resultado observado es realmente concluyente para la hipótesis planteada?", "supporting_citations": ["[u.pdf | c21]"]}]}
    parsed = _parse(payload, {("u.pdf", "c21")})
    result = materialize_initial_section_v2(parsed, "S17")
    assert result["draft_text"].endswith("[u.pdf | c21]?")


@scenario("V2B18. Final '!' -> puntuación preservada exacta")
def test_punctuation_exclamation_preserved():
    payload = {"section_id": "S18", "sentences": [{"text": "¡El resultado obtenido superó ampliamente las expectativas iniciales del equipo de investigación!", "supporting_citations": ["[v.pdf | c22]"]}]}
    parsed = _parse(payload, {("v.pdf", "c22")})
    result = materialize_initial_section_v2(parsed, "S18")
    assert result["draft_text"].endswith("[v.pdf | c22]!")


@scenario("V2B19. Final '?!' -> preservado COMPLETO, nunca normalizado a un solo carácter (ejemplo exacto pedido)")
def test_punctuation_question_exclamation_preserved_exactly():
    payload = {"section_id": "S19", "sentences": [{"text": "¿El resultado observado es realmente concluyente para toda la hipótesis?!", "supporting_citations": ["[paper.pdf | chunk_1]"]}]}
    parsed = _parse(payload, {("paper.pdf", "chunk_1")})
    result = materialize_initial_section_v2(parsed, "S19")
    assert result["draft_text"].endswith("[paper.pdf | chunk_1]?!")
    assert "?!" in result["draft_text"]
    assert not result["draft_text"].rstrip().endswith("!.") and not result["draft_text"].rstrip().endswith("!!")


@scenario("V2B20. Final '!!' -> preservado COMPLETO, nunca recortado a un solo '!'")
def test_punctuation_double_exclamation_preserved_exactly():
    payload = {"section_id": "S20", "sentences": [{"text": "El descubrimiento realizado por el equipo resultó absolutamente inesperado para toda la comunidad científica!!", "supporting_citations": ["[w.pdf | c23]"]}]}
    parsed = _parse(payload, {("w.pdf", "c23")})
    result = materialize_initial_section_v2(parsed, "S20")
    assert result["draft_text"].endswith("[w.pdf | c23]!!")


@scenario("V2B21. Final '...' -> preservado COMPLETO, nunca normalizado a un solo '.'")
def test_punctuation_ellipsis_preserved_exactly():
    payload = {"section_id": "S21", "sentences": [{"text": "El proceso de degradación observado continuó durante todo el período experimental evaluado...", "supporting_citations": ["[x.pdf | c24]"]}]}
    parsed = _parse(payload, {("x.pdf", "c24")})
    result = materialize_initial_section_v2(parsed, "S21")
    assert result["draft_text"].endswith("[x.pdf | c24]...")


@scenario("V2B22. Oración SIN puntuación final -> ninguna se inventa")
def test_no_trailing_punctuation_none_invented():
    payload = {"section_id": "S22", "sentences": [{"text": "El resultado observado carece de puntuación final en el texto original recibido", "supporting_citations": ["[y.pdf | c25]"]}]}
    parsed = _parse(payload, {("y.pdf", "c25")})
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
        # La cita removida deja un espacio entre el texto base y la
        # puntuación final (el fragmento real es "...base [cita]punt.",
        # remover [cita] da "...base  punt." tras colapsar espacios
        # dobles arriba) -- se retira ese único espacio residual antes
        # de un signo de puntuación para comparar el CONTENIDO, exactamente
        # como quedaría reconstruido a mano quitando solo la cita.
        import re as _re
        without_citation = _re.sub(r"\s+([.?!]+)$", r"\1", without_citation)
        expected = " ".join(text.split())
        assert without_citation == expected, (text, without_citation)


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
