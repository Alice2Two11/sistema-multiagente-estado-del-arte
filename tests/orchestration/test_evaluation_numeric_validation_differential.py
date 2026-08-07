"""Pruebas diferenciales del Bloque 5A: oráculo reproducido vs. módulo real."""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.numeric_validation import (
    aggregate_numeric_metrics,
    build_chunk_text_by_pair,
    build_numeric_metric_rows,
    build_section_text_by_id,
    extract_numeric_rows,
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


# ---------------------------------------------------------------------------
# Oráculo independiente (celda 19), sin compartir código con el módulo.
# ---------------------------------------------------------------------------

_ORACLE_CITATION_PATTERN = re.compile(r"\[([^\[\]\|]+?)\s*\|\s*([^\[\]\|]+?)\]")
_ORACLE_NUMERIC_PATTERN = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*%)?")


def _oracle_safe_str(value):
    return "" if value is None else str(value).strip()


def _oracle_strip_internal_citations(text):
    return _ORACLE_CITATION_PATTERN.sub("", _oracle_safe_str(text))


def _oracle_normalize_numeric_token(value):
    token = _oracle_safe_str(value)
    token = re.sub(r"\s+", "", token)
    token = token.replace(",", ".")
    return token.casefold()


def _oracle_numeric_search_variants(value):
    normalized = _oracle_normalize_numeric_token(value)
    variants = {normalized, normalized.replace("%", "")}
    if "." in normalized:
        variants.add(normalized.replace(".", ","))
        variants.add(normalized.replace(".", ",").replace("%", ""))
    return {item for item in variants if item}


def _oracle_split_section_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", _oracle_safe_str(text)) if s.strip()]


def _oracle_to_bool(value):
    if isinstance(value, bool):
        return value
    return _oracle_safe_str(value).casefold() in {"true", "1", "yes", "si", "sí"}


def _oracle_build_section_text_by_id(sections):
    result = {}
    for section in sections:
        section_id = _oracle_safe_str(section.get("section_id"))
        text = section.get("verified_text") if section.get("verified_text") is not None else section.get("draft_text")
        result[section_id] = _oracle_safe_str(text)
    return result


def _oracle_build_chunk_text_by_pair(chunks):
    return {
        (_oracle_safe_str(c["source_filename"]), _oracle_safe_str(c["chunk_id"])): _oracle_safe_str(c["text"])
        for c in chunks
    }


def _oracle_extract_numeric_rows(*, section_text_by_id, chunk_text_by_pair):
    numeric_rows = []
    for section_id, section_text in section_text_by_id.items():
        sentences = _oracle_split_section_sentences(section_text)
        for sentence_index, sentence in enumerate(sentences, start=1):
            cited_pairs = [
                (s.strip(), c.strip()) for s, c in _ORACLE_CITATION_PATTERN.findall(sentence)
            ]
            sentence_without_citations = _oracle_strip_internal_citations(sentence)
            numeric_values = _ORACLE_NUMERIC_PATTERN.findall(sentence_without_citations)
            for numeric_index, numeric_value in enumerate(numeric_values, start=1):
                variants = _oracle_numeric_search_variants(numeric_value)
                matched_pairs = []
                for pair in cited_pairs:
                    chunk_text = chunk_text_by_pair.get(pair, "")
                    normalized_chunk = chunk_text.replace(",", ".").casefold()
                    if any(v in normalized_chunk for v in variants):
                        matched_pairs.append(pair)
                numeric_rows.append(
                    {
                        "section_id": section_id,
                        "sentence_index": sentence_index,
                        "numeric_index": numeric_index,
                        "numeric_value": numeric_value,
                        "context": sentence_without_citations,
                        "cited_pair_count": len(cited_pairs),
                        "cited_pairs": json.dumps(cited_pairs, ensure_ascii=False),
                        "matched_pairs": json.dumps(matched_pairs, ensure_ascii=False),
                        "found_in_cited_chunks": bool(matched_pairs),
                        "evaluation_status": "CHECKED" if cited_pairs else "NO_CITATION_IN_SENTENCE",
                    }
                )
    return numeric_rows


def _oracle_aggregate_numeric_metrics(numeric_rows):
    checked = [r for r in numeric_rows if r["evaluation_status"] == "CHECKED"]
    total = int(len(checked))
    failures = int(sum(1 for r in checked if not _oracle_to_bool(r["found_in_cited_chunks"]))) if total else 0
    rate = failures / total if total else None
    return {"total_numeric_values": total, "numeric_failures": failures, "numeric_error_rate": rate}


def _oracle_build_numeric_metric_rows(numeric_error_rate):
    return [
        {
            "metric": "numeric_error_rate",
            "value": numeric_error_rate,
            "description": (
                "Proporción de valores numéricos que no aparecen "
                "en los chunks citados. Es null si no hubo valores comprobables."
            ),
        }
    ]


def _chunks(rows):
    return [{"source_filename": s, "chunk_id": c, "text": t} for s, c, t in rows]


@scenario("W01. Diferencial: tokens y variantes numéricas idénticos (reutilizados del Bloque 1)")
def test_diff_tokens_and_variants():
    from src.tools.evaluation.text_normalization import (
        normalize_numeric_token,
        numeric_search_variants,
    )

    for value in ["91,5%", "100", "-3.2", "5kg", "", None, "no numérico"]:
        assert normalize_numeric_token(value) == _oracle_normalize_numeric_token(value)
        assert numeric_search_variants(value) == _oracle_numeric_search_variants(value)


@scenario("W02. Diferencial: extracción y clasificación por fila idénticas, texto complejo con múltiples secciones")
def test_diff_extraction_classification():
    section_text_by_id = {
        "intro": "El modelo alcanzó 91.5% de precisión [p1.pdf | c1]. Sin cita este otro dato: 42.",
        "results": "La mejora fue de 10 a 20 puntos [p2.pdf | c2][p1.pdf | c1]. El costo bajó -5,3% [p1.pdf | c1].",
    }
    chunks = _chunks(
        [
            ("p1.pdf", "c1", "Se alcanzó 91.5% de precisión con una reducción de -5.3% en costo."),
            ("p2.pdf", "c2", "No contiene los valores mencionados."),
        ]
    )
    chunk_text_by_pair = build_chunk_text_by_pair(chunks)
    oracle_chunk_text_by_pair = _oracle_build_chunk_text_by_pair(chunks)
    assert chunk_text_by_pair == oracle_chunk_text_by_pair

    real_rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=chunk_text_by_pair
    )
    oracle_rows = _oracle_extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=oracle_chunk_text_by_pair
    )
    assert real_rows == oracle_rows


@scenario("W03. Diferencial: conteos y tasas idénticos")
def test_diff_counts_and_rates():
    section_text_by_id = {
        "s1": "Valor 10 [p.pdf | c1]. Valor 20 [p.pdf | c1]. Valor 30 sin evidencia [p.pdf | c1]. Valor 40 sin cita."
    }
    chunks = _chunks([("p.pdf", "c1", "Contiene 10 y 20.")])
    chunk_text_by_pair = build_chunk_text_by_pair(chunks)

    real_rows = extract_numeric_rows(section_text_by_id=section_text_by_id, chunk_text_by_pair=chunk_text_by_pair)
    oracle_rows = _oracle_extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=chunk_text_by_pair
    )
    assert real_rows == oracle_rows

    real_metrics = aggregate_numeric_metrics(real_rows)
    oracle_metrics = _oracle_aggregate_numeric_metrics(oracle_rows)
    assert real_metrics == oracle_metrics


@scenario("W04. Diferencial: columnas y valores finales de la fila de métrica idénticos, incluida value=None")
def test_diff_final_metric_row():
    for rate in [0.0, 0.5, 1.0, None]:
        real = build_numeric_metric_rows(rate)
        oracle = _oracle_build_numeric_metric_rows(rate)
        assert real == oracle


@scenario("W05. Diferencial: build_section_text_by_id idéntico (prioridad verified_text/draft_text)")
def test_diff_section_text_by_id():
    sections = [
        {"section_id": "s1", "verified_text": "Verificado.", "draft_text": "Borrador."},
        {"section_id": "s2", "verified_text": None, "draft_text": "Solo borrador."},
    ]
    real = build_section_text_by_id(sections)
    oracle = _oracle_build_section_text_by_id(sections)
    assert real == oracle


if __name__ == "__main__":
    for fn in (
        test_diff_tokens_and_variants,
        test_diff_extraction_classification,
        test_diff_counts_and_rates,
        test_diff_final_metric_row,
        test_diff_section_text_by_id,
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
