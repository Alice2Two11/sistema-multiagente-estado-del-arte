"""Pruebas de caracterización del Bloque 5A: validación numérica."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.numeric_validation import (
    NUMERIC_ROW_COLUMNS,
    aggregate_numeric_metrics,
    build_chunk_text_by_pair,
    build_numeric_metric_rows,
    build_section_text_by_id,
    extract_numeric_rows,
    split_section_sentences,
    to_bool,
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


def _chunks(rows):
    return [{"source_filename": s, "chunk_id": c, "text": t} for s, c, t in rows]


def _sections(rows):
    return [{"section_id": sid, "draft_text": text} for sid, text in rows]


# ---------------------------------------------------------------------------
# Extracción de tipos numéricos individuales
# ---------------------------------------------------------------------------


@scenario("V01. Entero: se extrae tal cual, sin normalizar el valor crudo")
def test_extract_integer():
    section_text_by_id = {"s1": "El modelo procesó 42 documentos [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "Se procesaron 42 documentos en total.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert len(rows) == 1
    assert rows[0]["numeric_value"] == "42"
    assert rows[0]["found_in_cited_chunks"] is True


@scenario("V02. Decimal con punto")
def test_extract_decimal_dot():
    section_text_by_id = {"s1": "La precisión fue 91.5 en el experimento [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "La precisión alcanzó 91.5 puntos.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["numeric_value"] == "91.5"
    assert rows[0]["found_in_cited_chunks"] is True


@scenario("V03. Decimal con coma: variante con punto matchea el chunk")
def test_extract_decimal_comma_variant_match():
    section_text_by_id = {"s1": "La precisión fue 91,5 en el experimento [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "La precisión alcanzó 91.5 puntos.")])  # chunk usa punto
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["numeric_value"] == "91,5"
    assert rows[0]["found_in_cited_chunks"] is True  # matchea vía numeric_search_variants


@scenario("V04. Porcentaje")
def test_extract_percentage():
    section_text_by_id = {"s1": "El error fue del 5% en la prueba [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "Se observó un error del 5%.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["numeric_value"] == "5%"
    assert rows[0]["found_in_cited_chunks"] is True


@scenario("V05. Signo negativo")
def test_extract_negative_sign():
    section_text_by_id = {"s1": "La variación fue de -3.2 puntos [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "Se registró una variación de -3.2 puntos.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["numeric_value"] == "-3.2"


@scenario("V06. Rango numérico: se detectan dos números separados, no un rango como unidad")
def test_extract_range_as_two_numbers():
    section_text_by_id = {"s1": "El rango fue de 10 a 20 unidades [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El rango observado fue de 10 a 20 unidades.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    values = [r["numeric_value"] for r in rows]
    assert values == ["10", "20"]  # el patrón real no reconoce rangos como una unidad


@scenario("V07. Unidad pegada al número (ej. '5kg'): el patrón NO matchea la unidad")
def test_extract_number_with_attached_unit():
    section_text_by_id = {"s1": "El peso fue de 5kg en la muestra [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El peso registrado fue de 5kg.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["numeric_value"] == "5"  # el patrón no incluye la unidad "kg"


@scenario("V08. Número repetido en la misma oración: genera dos filas con numeric_index consecutivo")
def test_extract_repeated_number_same_sentence():
    section_text_by_id = {"s1": "El valor 10 se repitió como 10 en el mismo párrafo [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El valor 10 aparece dos veces: 10 y 10.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert len(rows) == 2
    assert [r["numeric_index"] for r in rows] == [1, 2]


@scenario("V09. Número dentro de una cita (chunk_id numérico) no se cuenta como valor del texto")
def test_number_inside_citation_not_counted():
    section_text_by_id = {"s1": "El experimento confirmó el resultado [p.pdf | 123]."}
    chunks = _chunks([("p.pdf", "123", "Texto del chunk sin números relevantes.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows == []  # la cita se elimina antes de buscar números; "123" no cuenta


# ---------------------------------------------------------------------------
# Coincidencia exacta / por variante / sin evidencia
# ---------------------------------------------------------------------------


@scenario("V10. Coincidencia exacta")
def test_exact_match():
    section_text_by_id = {"s1": "El resultado fue 100 [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El resultado exacto fue 100.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["found_in_cited_chunks"] is True
    assert rows[0]["matched_pairs"] == '[["p.pdf", "c1"]]'


@scenario("V11. Coincidencia por variante (mayúsculas/minúsculas y forma con/sin %)")
def test_variant_match_casefold_and_percent():
    section_text_by_id = {"s1": "El error fue 8% [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "SE OBSERVÓ UN ERROR DE 8 PUNTOS.")])  # sin "%", en mayúsculas
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["found_in_cited_chunks"] is True  # matchea "8" (variante sin %) en texto en mayúsculas


@scenario("V12. Número sin evidencia: no aparece en ningún chunk citado")
def test_number_without_evidence():
    section_text_by_id = {"s1": "El resultado fue 999 [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El resultado no menciona ese valor.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert rows[0]["found_in_cited_chunks"] is False
    assert rows[0]["matched_pairs"] == "[]"


@scenario("V13. Evidencia 'ambigua' (dos chunks citados, coincide en uno, no en otro): sigue siendo found=True")
def test_ambiguous_evidence_still_binary():
    section_text_by_id = {"s1": "El resultado fue 50 [p.pdf | c1][q.pdf | c2]."}
    chunks = _chunks(
        [("p.pdf", "c1", "El resultado fue 50."), ("q.pdf", "c2", "Sin ese valor mencionado.")]
    )
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    # El notebook real NO tiene una categoría "ambiguo": si al menos un
    # chunk citado matchea, found_in_cited_chunks es True sin más matiz.
    assert rows[0]["found_in_cited_chunks"] is True
    assert rows[0]["matched_pairs"] == '[["p.pdf", "c1"]]'


@scenario("V14. Valor vacío / oración sin números: no genera filas")
def test_empty_value_no_numbers():
    section_text_by_id = {"s1": "Esta oración no contiene ningún valor numérico relevante."}
    rows = extract_numeric_rows(section_text_by_id=section_text_by_id, chunk_text_by_pair={})
    assert rows == []


@scenario("V15. Oración sin ninguna cita: evaluation_status = NO_CITATION_IN_SENTENCE, no cuenta en la tasa")
def test_sentence_without_citation():
    section_text_by_id = {"s1": "El resultado fue 77 sin ninguna referencia."}
    rows = extract_numeric_rows(section_text_by_id=section_text_by_id, chunk_text_by_pair={})
    assert rows[0]["evaluation_status"] == "NO_CITATION_IN_SENTENCE"
    assert rows[0]["cited_pair_count"] == 0
    metrics = aggregate_numeric_metrics(rows)
    assert metrics["total_numeric_values"] == 0  # se excluye de la tasa
    assert metrics["numeric_error_rate"] is None


# ---------------------------------------------------------------------------
# Archivo upstream ausente / columnas faltantes (a nivel de entradas)
# ---------------------------------------------------------------------------


@scenario("V16. build_chunk_text_by_pair con lista de chunks vacía (equivalente a upstream ausente)")
def test_missing_upstream_empty_chunks():
    rows = extract_numeric_rows(
        section_text_by_id={"s1": "El resultado fue 10 [p.pdf | c1]."}, chunk_text_by_pair={}
    )
    assert rows[0]["found_in_cited_chunks"] is False  # sin chunks, ninguna cita puede matchear


@scenario("V17. build_chunk_text_by_pair: columna faltante en un dict de chunk lanza KeyError (no se silencia)")
def test_missing_column_raises():
    try:
        build_chunk_text_by_pair([{"source_filename": "p.pdf", "chunk_id": "c1"}])  # falta "text"
    except KeyError:
        pass
    else:
        raise AssertionError("debía lanzar KeyError por columna faltante")


# ---------------------------------------------------------------------------
# Múltiples filas para la misma afirmación / conteos y tasas
# ---------------------------------------------------------------------------


@scenario("V18. Múltiples filas para la misma oración (varios números): cada uno con su propia fila")
def test_multiple_rows_same_sentence():
    section_text_by_id = {"s1": "El modelo mejoró de 40 a 55 puntos, un 15% más [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "Mejora de 40 a 55, equivalente a 15%.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert [r["numeric_value"] for r in rows] == ["40", "55", "15%"]
    assert all(r["found_in_cited_chunks"] for r in rows)


@scenario("V19. Conteos y tasas: 3 comprobables, 1 fallo -> error_rate = 1/3")
def test_counts_and_rate():
    section_text_by_id = {
        "s1": "Valor 10 correcto [p.pdf | c1]. Valor 20 correcto [p.pdf | c1]. Valor 30 incorrecto [p.pdf | c1]."
    }
    chunks = _chunks([("p.pdf", "c1", "Contiene 10 y 20 pero no el tercer valor.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    metrics = aggregate_numeric_metrics(rows)
    assert metrics["total_numeric_values"] == 3
    assert metrics["numeric_failures"] == 1
    assert abs(metrics["numeric_error_rate"] - (1 / 3)) < 1e-9


@scenario("V20. build_numeric_metric_rows: fila con la clave 'description' real, no 'method'")
def test_metric_row_shape():
    rows = build_numeric_metric_rows(0.25)
    assert len(rows) == 1
    assert rows[0]["metric"] == "numeric_error_rate"
    assert rows[0]["value"] == 0.25
    assert "description" in rows[0]
    assert "method" not in rows[0]


@scenario("V21. build_numeric_metric_rows: value=None se preserva tal cual (no se convierte a 0.0)")
def test_metric_row_none_preserved():
    rows = build_numeric_metric_rows(None)
    assert rows[0]["value"] is None


# ---------------------------------------------------------------------------
# Orden y esquema exactos
# ---------------------------------------------------------------------------


@scenario("V22. Esquema exacto de columnas y orden en cada fila numérica")
def test_exact_schema_and_order():
    section_text_by_id = {"s1": "El resultado fue 5 [p.pdf | c1]."}
    chunks = _chunks([("p.pdf", "c1", "El resultado fue 5.")])
    rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=build_chunk_text_by_pair(chunks)
    )
    assert list(rows[0].keys()) == NUMERIC_ROW_COLUMNS


@scenario("V23. build_section_text_by_id: prioriza verified_text sobre draft_text cuando existe")
def test_section_text_prioritizes_verified():
    sections = [{"section_id": "s1", "verified_text": "Texto verificado.", "draft_text": "Texto borrador."}]
    result = build_section_text_by_id(sections)
    assert result["s1"] == "Texto verificado."


@scenario("V24. build_section_text_by_id: cae a draft_text si verified_text es None")
def test_section_text_falls_back_to_draft():
    sections = [{"section_id": "s1", "verified_text": None, "draft_text": "Texto borrador."}]
    result = build_section_text_by_id(sections)
    assert result["s1"] == "Texto borrador."


@scenario("V25. to_bool: reconoce variantes true/1/yes/si/sí, sin distinguir mayúsculas")
def test_to_bool_variants():
    for value in [True, "true", "1", "yes", "si", "sí", "SI", "TRUE"]:
        assert to_bool(value) is True
    for value in [False, "false", "0", "no", ""]:
        assert to_bool(value) is False


@scenario("V26. split_section_sentences: corta también por saltos de línea sueltos (distinto de split_sentences)")
def test_split_section_sentences_newlines():
    result = split_section_sentences("Primera línea sin punto\nSegunda línea.")
    assert result == ["Primera línea sin punto", "Segunda línea."]


if __name__ == "__main__":
    for fn in (
        test_extract_integer,
        test_extract_decimal_dot,
        test_extract_decimal_comma_variant_match,
        test_extract_percentage,
        test_extract_negative_sign,
        test_extract_range_as_two_numbers,
        test_extract_number_with_attached_unit,
        test_extract_repeated_number_same_sentence,
        test_number_inside_citation_not_counted,
        test_exact_match,
        test_variant_match_casefold_and_percent,
        test_number_without_evidence,
        test_ambiguous_evidence_still_binary,
        test_empty_value_no_numbers,
        test_sentence_without_citation,
        test_missing_upstream_empty_chunks,
        test_missing_column_raises,
        test_multiple_rows_same_sentence,
        test_counts_and_rate,
        test_metric_row_shape,
        test_metric_row_none_preserved,
        test_exact_schema_and_order,
        test_section_text_prioritizes_verified,
        test_section_text_falls_back_to_draft,
        test_to_bool_variants,
        test_split_section_sentences_newlines,
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
