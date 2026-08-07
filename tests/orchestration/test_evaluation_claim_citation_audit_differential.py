"""Pruebas diferenciales del Bloque 5B: oráculo reproducido vs. módulo real."""

from __future__ import annotations

import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.claim_citation_audit import (
    build_citation_rows,
    build_claim_audit_rows,
    build_factual_metric_rows,
    build_valid_source_chunk_pairs,
    compute_citation_metrics,
    compute_claim_factual_metrics,
    count_removed_claims,
    select_active_claims,
)
from src.tools.evaluation.numeric_validation import build_section_text_by_id

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
_PROBLEM_VERDICTS = {"partially_supported", "unclear", "unsupported"}
_PROBLEM_RISK_LEVELS = {"medium", "high"}


def _oracle_safe_str(value):
    return "" if value is None else str(value).strip()


def _oracle_strip_internal_citations(text):
    return _ORACLE_CITATION_PATTERN.sub("", _oracle_safe_str(text))


def _oracle_normalize_claim_text(text):
    value = _oracle_strip_internal_citations(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .;:!?").casefold()


def _oracle_to_bool(value):
    if isinstance(value, bool):
        return value
    return _oracle_safe_str(value).casefold() in {"true", "1", "yes", "si", "sí"}


def _oracle_first_non_empty(values):
    for value in values:
        text = _oracle_safe_str(value)
        if text:
            return text
    return ""


def _oracle_build_claim_audit_rows(*, traceability_rows, generated_content_text, valid_source_chunk_pairs):
    all_columns = set()
    for row in traceability_rows:
        all_columns.update(row.keys())
    missing = sorted({"claim_id", "verdict"} - all_columns)
    if missing:
        raise ValueError(f"post_correction_traceability_matrix.csv está incompleto. Faltan: {missing}")

    groups = defaultdict(list)
    for row in traceability_rows:
        groups[str(row.get("claim_id"))].append(row)

    rows = []
    for key in sorted(groups):
        group = groups[key]
        claim_id = _oracle_safe_str(key)
        if not claim_id:
            continue
        claim_text = _oracle_first_non_empty(r.get("claim") for r in group) if "claim" in all_columns else ""
        verdict = _oracle_first_non_empty(r.get("verdict") for r in group).casefold()
        risk = (
            _oracle_first_non_empty(r.get("hallucination_risk") for r in group).casefold()
            if "hallucination_risk" in all_columns
            else ""
        )
        correction_needed = False
        if "correction_needed" in all_columns:
            correction_needed = any(_oracle_to_bool(r.get("correction_needed")) for r in group)
        evidence_pairs = set()
        if {"source_filename", "chunk_id"}.issubset(all_columns):
            for r in group:
                s = _oracle_safe_str(r.get("source_filename"))
                c = _oracle_safe_str(r.get("chunk_id"))
                if s and c:
                    evidence_pairs.add((s, c))
        active_claim = verdict != "removed" and bool(_oracle_normalize_claim_text(claim_text))
        claim_in_final_text = (
            _oracle_normalize_claim_text(claim_text) in _oracle_normalize_claim_text(generated_content_text)
            if active_claim
            else True
        )
        invalid_evidence_pairs = sorted(p for p in evidence_pairs if p not in valid_source_chunk_pairs)
        rows.append(
            {
                "claim_id": claim_id,
                "claim": claim_text,
                "verdict": verdict,
                "hallucination_risk": risk,
                "correction_needed": correction_needed,
                "active_claim": active_claim,
                "claim_in_final_text": claim_in_final_text,
                "evidence_pair_count": len(evidence_pairs),
                "evidence_present": bool(evidence_pairs),
                "invalid_evidence_pair_count": len(invalid_evidence_pairs),
                "invalid_evidence_pairs": json.dumps(invalid_evidence_pairs, ensure_ascii=False),
            }
        )
    if not rows:
        raise ValueError("No se encontraron claims finales para evaluar.")
    return rows


def _oracle_select_active(rows):
    active = [r for r in rows if _oracle_to_bool(r["active_claim"])]
    if not active:
        raise ValueError("No existen claims activos en la entrada seleccionada para el Agente 08.")
    return active


def _oracle_claim_metrics(active_claims):
    supported = sum(1 for r in active_claims if r["verdict"] == "supported")
    problems = sum(
        1
        for r in active_claims
        if r["verdict"] in _PROBLEM_VERDICTS
        or r["hallucination_risk"] in _PROBLEM_RISK_LEVELS
        or _oracle_to_bool(r["correction_needed"])
    )
    total = len(active_claims)
    return {
        "total_active_claims": total,
        "supported_claims": supported,
        "problem_claims": problems,
        "factual_precision": supported / total,
        "hallucination_rate": problems / total,
        "evidence_coverage": sum(1 for r in active_claims if _oracle_to_bool(r["evidence_present"])) / total,
        "traceability_text_coverage": sum(1 for r in active_claims if _oracle_to_bool(r["claim_in_final_text"])) / total,
        "invalid_traceability_pairs": sum(r["invalid_evidence_pair_count"] for r in active_claims),
    }


def _oracle_build_citation_rows(*, section_text_by_id, valid_source_chunk_pairs):
    rows = []
    for section_id, section_text in section_text_by_id.items():
        for idx, (source, chunk) in enumerate(_ORACLE_CITATION_PATTERN.findall(section_text), start=1):
            pair = (source.strip(), chunk.strip())
            rows.append(
                {
                    "section_id": section_id,
                    "citation_index": idx,
                    "source_filename": pair[0],
                    "chunk_id": pair[1],
                    "citation": f"[{pair[0]} | {pair[1]}]",
                    "exists_in_clean_chunks": pair in valid_source_chunk_pairs,
                }
            )
    return rows


def _oracle_citation_metrics(citation_rows):
    total = len(citation_rows)
    invalid = sum(1 for r in citation_rows if not _oracle_to_bool(r["exists_in_clean_chunks"])) if total else 0
    rate = invalid / total if total else None
    return {"total_citations": total, "invalid_citations": invalid, "citation_error_rate": rate}


def _chunks(rows):
    return [{"source_filename": s, "chunk_id": c} for s, c in rows]


@scenario("X01. Diferencial: filas de auditoría de claims idénticas (múltiples claims, evidencias, riesgos)")
def test_diff_claim_audit_rows():
    traceability_rows = [
        {"claim_id": "1", "verdict": "supported", "claim": "A funciona bien.", "hallucination_risk": "low",
         "source_filename": "p.pdf", "chunk_id": "c1"},
        {"claim_id": "2", "verdict": "unsupported", "claim": "B no funciona.", "hallucination_risk": "high",
         "correction_needed": True},
        {"claim_id": "3", "verdict": "removed", "claim": "C eliminado."},
    ]
    valid_pairs = build_valid_source_chunk_pairs(_chunks([("p.pdf", "c1")]))
    real = build_claim_audit_rows(
        traceability_rows=traceability_rows, generated_content_text="A funciona bien.", valid_source_chunk_pairs=valid_pairs
    )
    oracle = _oracle_build_claim_audit_rows(
        traceability_rows=traceability_rows, generated_content_text="A funciona bien.", valid_source_chunk_pairs=valid_pairs
    )
    assert real == oracle


@scenario("X02. Diferencial: claims activos y máscara de problemas idénticos")
def test_diff_active_claims_and_problem_mask():
    traceability_rows = [
        {"claim_id": "1", "verdict": "supported", "claim": "A."},
        {"claim_id": "2", "verdict": "partially_supported", "claim": "B."},
        {"claim_id": "3", "verdict": "removed", "claim": "C."},
    ]
    audit_real = build_claim_audit_rows(
        traceability_rows=traceability_rows, generated_content_text="A. B.", valid_source_chunk_pairs=set()
    )
    audit_oracle = _oracle_build_claim_audit_rows(
        traceability_rows=traceability_rows, generated_content_text="A. B.", valid_source_chunk_pairs=set()
    )
    active_real = select_active_claims(audit_real)
    active_oracle = _oracle_select_active(audit_oracle)
    assert active_real == active_oracle


@scenario("X03. Diferencial: conteos y tasas factuales idénticos")
def test_diff_counts_and_rates():
    traceability_rows = [
        {"claim_id": str(i), "verdict": v, "claim": f"Claim {i}.", "hallucination_risk": r}
        for i, (v, r) in enumerate(
            [
                ("supported", "low"),
                ("supported", "medium"),
                ("unsupported", "high"),
                ("partially_supported", "low"),
                ("removed", "low"),
            ],
            start=1,
        )
    ]
    audit_real = build_claim_audit_rows(
        traceability_rows=traceability_rows,
        generated_content_text="Claim 1. Claim 2. Claim 3. Claim 4.",
        valid_source_chunk_pairs=set(),
    )
    audit_oracle = _oracle_build_claim_audit_rows(
        traceability_rows=traceability_rows,
        generated_content_text="Claim 1. Claim 2. Claim 3. Claim 4.",
        valid_source_chunk_pairs=set(),
    )
    real_metrics = compute_claim_factual_metrics(select_active_claims(audit_real))
    oracle_metrics = _oracle_claim_metrics(_oracle_select_active(audit_oracle))
    assert real_metrics == oracle_metrics
    assert count_removed_claims(audit_real) == sum(1 for r in audit_oracle if r["verdict"] == "removed")


@scenario("X04. Diferencial: filas de citas y citation_error_rate idénticos")
def test_diff_citation_rows_and_rate():
    sections = [
        {"section_id": "intro", "draft_text": "Texto con [p.pdf | c1] y [x.pdf | c9]."},
        {"section_id": "results", "draft_text": "Más contenido [p.pdf | c1]."},
    ]
    section_text_by_id = build_section_text_by_id(sections)
    valid_pairs = build_valid_source_chunk_pairs(_chunks([("p.pdf", "c1")]))

    real_rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=valid_pairs)
    oracle_rows = _oracle_build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=valid_pairs)
    assert real_rows == oracle_rows

    real_metrics = compute_citation_metrics(real_rows)
    oracle_metrics = _oracle_citation_metrics(oracle_rows)
    assert real_metrics == oracle_metrics


@scenario("X05. Diferencial: filas factuales finales idénticas (12 filas, valores mixtos incluido None)")
def test_diff_factual_rows():
    kwargs = dict(
        total_active_claims=4, supported_claims=2, factual_precision=0.5, hallucination_rate=0.5,
        evidence_coverage=0.75, traceability_text_coverage=1.0, citation_error_rate=None,
        numeric_error_rate=0.2, invalid_traceability_pairs=1, removed_claims=1,
        total_final_citations=2, total_numeric_values_checked=5,
    )
    real = build_factual_metric_rows(**kwargs)

    def oracle_build(**k):
        return [
            {"metric": "total_active_claims", "value": int(k["total_active_claims"]), "description": "Claims presentes en el texto final, excluyendo los fragmentos eliminados."},
            {"metric": "supported_claims", "value": k["supported_claims"], "description": "Claims finales con veredicto supported."},
            {"metric": "factual_precision", "value": k["factual_precision"], "description": "Claims supported dividido para claims activos."},
            {"metric": "hallucination_rate", "value": k["hallucination_rate"], "description": "Claims parcialmente soportados, ambiguos, no soportados, de riesgo medio/alto o pendientes."},
            {"metric": "evidence_coverage", "value": k["evidence_coverage"], "description": "Claims activos con al menos un par fuente-chunk."},
            {"metric": "traceability_text_coverage", "value": k["traceability_text_coverage"], "description": "Claims activos localizados en el texto final."},
            {"metric": "citation_error_rate", "value": k["citation_error_rate"], "description": "Proporción de citas del texto final que no existen en chunks_clean_for_rag.csv. Es null si no hay citas detectadas."},
            {"metric": "numeric_error_rate", "value": k["numeric_error_rate"], "description": "Proporción de valores numéricos que no aparecen en los chunks citados. Es null si no hubo valores comprobables."},
            {"metric": "invalid_traceability_pairs", "value": k["invalid_traceability_pairs"], "description": "Pares fuente-chunk de la matriz final que no existen en los chunks limpios."},
            {"metric": "removed_claims_after_correction", "value": k["removed_claims"], "description": "Claims problemáticos eliminados antes de la evaluación."},
            {"metric": "total_final_citations", "value": k["total_final_citations"], "description": "Citas internas revisadas en el texto final."},
            {"metric": "total_numeric_values_checked", "value": k["total_numeric_values_checked"], "description": "Valores numéricos disponibles en la ruta upstream seleccionada."},
        ]

    oracle = oracle_build(**kwargs)
    assert real == oracle


@scenario("X06. Diferencial: excepciones idénticas ante columnas faltantes y cero claims activos")
def test_diff_exceptions():
    real_exc = oracle_exc = None
    try:
        build_claim_audit_rows(traceability_rows=[{"claim_id": "1"}], generated_content_text="", valid_source_chunk_pairs=set())
    except ValueError as exc:
        real_exc = str(exc)
    try:
        _oracle_build_claim_audit_rows(
            traceability_rows=[{"claim_id": "1"}], generated_content_text="", valid_source_chunk_pairs=set()
        )
    except ValueError as exc:
        oracle_exc = str(exc)
    assert real_exc is not None and real_exc == oracle_exc

    rows = [{"claim_id": "1", "verdict": "removed", "claim": "X."}]
    audit = build_claim_audit_rows(traceability_rows=rows, generated_content_text="", valid_source_chunk_pairs=set())
    oracle_audit = _oracle_build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="", valid_source_chunk_pairs=set()
    )
    real_exc2 = oracle_exc2 = None
    try:
        select_active_claims(audit)
    except ValueError as exc:
        real_exc2 = str(exc)
    try:
        _oracle_select_active(oracle_audit)
    except ValueError as exc:
        oracle_exc2 = str(exc)
    assert real_exc2 is not None and real_exc2 == oracle_exc2


if __name__ == "__main__":
    for fn in (
        test_diff_claim_audit_rows,
        test_diff_active_claims_and_problem_mask,
        test_diff_counts_and_rates,
        test_diff_citation_rows_and_rate,
        test_diff_factual_rows,
        test_diff_exceptions,
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
