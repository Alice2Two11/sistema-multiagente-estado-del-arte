"""Pruebas de caracterización del Bloque 5B: auditoría de claims, citas y trazabilidad."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.claim_citation_audit import (
    CITATION_CHECK_COLUMNS,
    CLAIM_AUDIT_COLUMNS,
    build_claim_audit_rows,
    build_citation_rows,
    build_factual_metric_rows,
    build_valid_source_chunk_pairs,
    compute_citation_metrics,
    compute_claim_factual_metrics,
    count_removed_claims,
    first_non_empty,
    select_active_claims,
    validate_required_traceability_columns,
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


def _tr_row(claim_id, verdict, **kwargs):
    row = {"claim_id": claim_id, "verdict": verdict}
    row.update(kwargs)
    return row


def _chunks(rows):
    return [{"source_filename": s, "chunk_id": c} for s, c in rows]


VALID_PAIRS = build_valid_source_chunk_pairs(_chunks([("p.pdf", "c1"), ("p.pdf", "c2")]))


# ---------------------------------------------------------------------------
# Auditoría de claims: veredictos
# ---------------------------------------------------------------------------


@scenario("C01. Claim supported")
def test_claim_supported():
    rows = [_tr_row("1", "supported", claim="El modelo mejora los resultados.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows,
        generated_content_text="El modelo mejora los resultados.",
        valid_source_chunk_pairs=set(),
    )
    assert audit[0]["verdict"] == "supported"
    assert audit[0]["active_claim"] is True


@scenario("C02. Claim partially_supported")
def test_claim_partially_supported():
    rows = [_tr_row("1", "partially_supported", claim="Texto del claim.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="Texto del claim.", valid_source_chunk_pairs=set()
    )
    assert audit[0]["verdict"] == "partially_supported"


@scenario("C03. Claim unclear")
def test_claim_unclear():
    rows = [_tr_row("1", "unclear", claim="Texto del claim.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="Texto del claim.", valid_source_chunk_pairs=set()
    )
    assert audit[0]["verdict"] == "unclear"


@scenario("C04. Claim unsupported")
def test_claim_unsupported():
    rows = [_tr_row("1", "unsupported", claim="Texto del claim.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="Texto del claim.", valid_source_chunk_pairs=set()
    )
    assert audit[0]["verdict"] == "unsupported"


@scenario("C05. Claim removed -> active_claim=False y claim_in_final_text=True por defecto")
def test_claim_removed():
    rows = [_tr_row("1", "removed", claim="Texto eliminado.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="No contiene ese texto.", valid_source_chunk_pairs=set()
    )
    assert audit[0]["active_claim"] is False
    assert audit[0]["claim_in_final_text"] is True  # rama "else True" para claims no activos


# ---------------------------------------------------------------------------
# Riesgo de alucinación
# ---------------------------------------------------------------------------


@scenario("C06. Riesgo low: no cuenta como problem_claim por sí solo")
def test_risk_low():
    rows = [_tr_row("1", "supported", claim="X.", hallucination_risk="low")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set()
    )
    active = select_active_claims(audit)
    metrics = compute_claim_factual_metrics(active)
    assert metrics["problem_claims"] == 0


@scenario("C07. Riesgo medium: cuenta como problem_claim")
def test_risk_medium():
    rows = [_tr_row("1", "supported", claim="X.", hallucination_risk="medium")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set()
    )
    metrics = compute_claim_factual_metrics(select_active_claims(audit))
    assert metrics["problem_claims"] == 1


@scenario("C08. Riesgo high: cuenta como problem_claim")
def test_risk_high():
    rows = [_tr_row("1", "supported", claim="X.", hallucination_risk="high")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set()
    )
    metrics = compute_claim_factual_metrics(select_active_claims(audit))
    assert metrics["problem_claims"] == 1


# ---------------------------------------------------------------------------
# correction_needed
# ---------------------------------------------------------------------------


@scenario("C09. correction_needed=True hace que el claim cuente como problema aunque sea supported/riesgo low")
def test_correction_needed_true():
    rows = [_tr_row("1", "supported", claim="X.", hallucination_risk="low", correction_needed=True)]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set()
    )
    metrics = compute_claim_factual_metrics(select_active_claims(audit))
    assert metrics["problem_claims"] == 1
    assert audit[0]["correction_needed"] is True


@scenario("C10. correction_needed=False no penaliza")
def test_correction_needed_false():
    rows = [_tr_row("1", "supported", claim="X.", hallucination_risk="low", correction_needed=False)]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set()
    )
    metrics = compute_claim_factual_metrics(select_active_claims(audit))
    assert metrics["problem_claims"] == 0


# ---------------------------------------------------------------------------
# Agrupación / evidencia
# ---------------------------------------------------------------------------


@scenario("C11. Varias filas para el mismo claim: se agrupan en una sola fila de auditoría")
def test_multiple_rows_same_claim():
    rows = [
        _tr_row("1", "supported", claim="X.", source_filename="p.pdf", chunk_id="c1"),
        _tr_row("1", "supported", claim="X.", source_filename="p.pdf", chunk_id="c2"),
    ]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=VALID_PAIRS
    )
    assert len(audit) == 1
    assert audit[0]["evidence_pair_count"] == 2


@scenario("C12. Varias evidencias para un claim: evidence_pair_count las cuenta todas")
def test_multiple_evidences():
    rows = [
        _tr_row("1", "supported", claim="X.", source_filename="p.pdf", chunk_id="c1"),
        _tr_row("1", "supported", claim="X.", source_filename="q.pdf", chunk_id="c9"),
    ]
    audit = build_claim_audit_rows(
        traceability_rows=rows,
        generated_content_text="X.",
        valid_source_chunk_pairs=build_valid_source_chunk_pairs(_chunks([("p.pdf", "c1"), ("q.pdf", "c9")])),
    )
    assert audit[0]["evidence_pair_count"] == 2
    assert audit[0]["invalid_evidence_pair_count"] == 0


@scenario("C13. Evidencia duplicada: el par se cuenta una sola vez (es un set)")
def test_duplicate_evidence():
    rows = [
        _tr_row("1", "supported", claim="X.", source_filename="p.pdf", chunk_id="c1"),
        _tr_row("1", "supported", claim="X.", source_filename="p.pdf", chunk_id="c1"),
    ]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=VALID_PAIRS
    )
    assert audit[0]["evidence_pair_count"] == 1


@scenario("C14. Evidencia inválida: par que no está en valid_source_chunk_pairs")
def test_invalid_evidence():
    rows = [_tr_row("1", "supported", claim="X.", source_filename="desconocido.pdf", chunk_id="c99")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=VALID_PAIRS
    )
    assert audit[0]["invalid_evidence_pair_count"] == 1
    assert "desconocido.pdf" in audit[0]["invalid_evidence_pairs"]


# ---------------------------------------------------------------------------
# Localización del claim en el texto final
# ---------------------------------------------------------------------------


@scenario("C15. Claim presente en el texto final")
def test_claim_present_in_text():
    rows = [_tr_row("1", "supported", claim="mejora significativa")]
    audit = build_claim_audit_rows(
        traceability_rows=rows,
        generated_content_text="Se observó una mejora significativa en los resultados.",
        valid_source_chunk_pairs=set(),
    )
    assert audit[0]["claim_in_final_text"] is True


@scenario("C16. Claim ausente del texto final")
def test_claim_absent_from_text():
    rows = [_tr_row("1", "supported", claim="afirmación que no aparece")]
    audit = build_claim_audit_rows(
        traceability_rows=rows,
        generated_content_text="Texto completamente distinto.",
        valid_source_chunk_pairs=set(),
    )
    assert audit[0]["claim_in_final_text"] is False


# ---------------------------------------------------------------------------
# Claims/claim_id vacíos
# ---------------------------------------------------------------------------


@scenario("C17. Claim vacío (texto ''): active_claim=False aunque el veredicto no sea removed")
def test_empty_claim_text():
    rows = [_tr_row("1", "supported", claim="")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="Texto cualquiera.", valid_source_chunk_pairs=set()
    )
    assert audit[0]["active_claim"] is False  # normalize_claim_text("") es falsy


@scenario("C18. claim_id vacío (cadena vacía) se descarta del todo")
def test_empty_claim_id_discarded():
    rows = [_tr_row("", "supported", claim="X."), _tr_row("1", "supported", claim="Y.")]
    audit = build_claim_audit_rows(
        traceability_rows=rows, generated_content_text="X. Y.", valid_source_chunk_pairs=set()
    )
    assert len(audit) == 1
    assert audit[0]["claim_id"] == "1"


# ---------------------------------------------------------------------------
# Columnas obligatorias / cero claims
# ---------------------------------------------------------------------------


@scenario("C19. Columnas obligatorias ausentes lanza ValueError con mensaje real (07C literal)")
def test_missing_required_columns():
    try:
        validate_required_traceability_columns([{"claim_id": "1"}])  # falta "verdict"
    except ValueError as exc:
        assert "post_correction_traceability_matrix.csv" in str(exc)
        assert "verdict" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError")


@scenario("C20. Cero claims finales (todas las filas tienen claim_id vacío) lanza ValueError")
def test_zero_claims_raises():
    rows = [_tr_row("", "supported", claim="X."), _tr_row("", "unsupported", claim="Y.")]
    try:
        build_claim_audit_rows(traceability_rows=rows, generated_content_text="", valid_source_chunk_pairs=set())
    except ValueError as exc:
        assert "No se encontraron claims" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError")


@scenario("C21. Cero claims activos (todos removed) lanza ValueError en select_active_claims")
def test_zero_active_claims_raises():
    rows = [_tr_row("1", "removed", claim="X."), _tr_row("2", "removed", claim="Y.")]
    audit = build_claim_audit_rows(traceability_rows=rows, generated_content_text="", valid_source_chunk_pairs=set())
    try:
        select_active_claims(audit)
    except ValueError as exc:
        assert "No existen claims activos" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError")


# ---------------------------------------------------------------------------
# Auditoría de citas
# ---------------------------------------------------------------------------


@scenario("C22. Cita válida")
def test_valid_citation():
    section_text_by_id = build_section_text_by_id([{"section_id": "s1", "draft_text": "Texto con cita [p.pdf | c1]."}])
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert len(rows) == 1
    assert rows[0]["exists_in_clean_chunks"] is True


@scenario("C23. Cita inválida")
def test_invalid_citation():
    section_text_by_id = build_section_text_by_id(
        [{"section_id": "s1", "draft_text": "Texto con cita [desconocido.pdf | c99]."}]
    )
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert rows[0]["exists_in_clean_chunks"] is False
    metrics = compute_citation_metrics(rows)
    assert metrics["invalid_citations"] == 1
    assert metrics["citation_error_rate"] == 1.0


@scenario("C24. Múltiples citas: citation_index consecutivo por sección")
def test_multiple_citations_index():
    section_text_by_id = build_section_text_by_id(
        [{"section_id": "s1", "draft_text": "[p.pdf | c1] y también [p.pdf | c2]."}]
    )
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert [r["citation_index"] for r in rows] == [1, 2]


@scenario("C25. Cita duplicada: cada aparición genera su propia fila (no se deduplica)")
def test_duplicate_citation_not_deduplicated():
    section_text_by_id = build_section_text_by_id(
        [{"section_id": "s1", "draft_text": "[p.pdf | c1] y de nuevo [p.pdf | c1]."}]
    )
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert len(rows) == 2


@scenario("C26. Cero citas: total_citations=0, citation_error_rate=None (no 0.0)")
def test_zero_citations():
    rows = build_citation_rows(section_text_by_id={"s1": "Sin ninguna cita aquí."}, valid_source_chunk_pairs=set())
    metrics = compute_citation_metrics(rows)
    assert metrics["total_citations"] == 0
    assert metrics["citation_error_rate"] is None


@scenario("C27. Espacios alrededor de fuente y chunk en la cita se recortan (.strip())")
def test_citation_whitespace_trimmed():
    section_text_by_id = build_section_text_by_id(
        [{"section_id": "s1", "draft_text": "Cita con espacios [  p.pdf   |   c1  ]."}]
    )
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert rows[0]["source_filename"] == "p.pdf"
    assert rows[0]["chunk_id"] == "c1"
    assert rows[0]["exists_in_clean_chunks"] is True


# ---------------------------------------------------------------------------
# Tasas, conteos, esquema
# ---------------------------------------------------------------------------


@scenario("C28. Conteos y tasas factuales completos, con removed_claims incluido")
def test_full_counts_and_rates():
    rows = [
        _tr_row("1", "supported", claim="A."),
        _tr_row("2", "unsupported", claim="B."),
        _tr_row("3", "removed", claim="C."),
    ]
    audit = build_claim_audit_rows(traceability_rows=rows, generated_content_text="A. B.", valid_source_chunk_pairs=set())
    active = select_active_claims(audit)
    metrics = compute_claim_factual_metrics(active)
    removed = count_removed_claims(audit)
    assert metrics["total_active_claims"] == 2  # "3" (removed) excluido
    assert metrics["supported_claims"] == 1
    assert removed == 1


@scenario("C29. Esquema y orden exactos de la auditoría de claims")
def test_claim_audit_schema():
    rows = [_tr_row("1", "supported", claim="X.")]
    audit = build_claim_audit_rows(traceability_rows=rows, generated_content_text="X.", valid_source_chunk_pairs=set())
    assert list(audit[0].keys()) == CLAIM_AUDIT_COLUMNS


@scenario("C30. Esquema y orden exactos de la auditoría de citas")
def test_citation_audit_schema():
    section_text_by_id = build_section_text_by_id([{"section_id": "s1", "draft_text": "[p.pdf | c1]."}])
    rows = build_citation_rows(section_text_by_id=section_text_by_id, valid_source_chunk_pairs=VALID_PAIRS)
    assert list(rows[0].keys()) == CITATION_CHECK_COLUMNS


@scenario("C31. Orden exacto de las 12 filas factuales")
def test_factual_rows_order():
    rows = build_factual_metric_rows(
        total_active_claims=2, supported_claims=1, factual_precision=0.5, hallucination_rate=0.5,
        evidence_coverage=1.0, traceability_text_coverage=1.0, citation_error_rate=0.0,
        numeric_error_rate=None, invalid_traceability_pairs=0, removed_claims=1,
        total_final_citations=3, total_numeric_values_checked=0,
    )
    assert [r["metric"] for r in rows] == [
        "total_active_claims", "supported_claims", "factual_precision", "hallucination_rate",
        "evidence_coverage", "traceability_text_coverage", "citation_error_rate",
        "numeric_error_rate", "invalid_traceability_pairs", "removed_claims_after_correction",
        "total_final_citations", "total_numeric_values_checked",
    ]
    assert all("description" in r and "method" not in r for r in rows)


if __name__ == "__main__":
    for fn in (
        test_claim_supported, test_claim_partially_supported, test_claim_unclear,
        test_claim_unsupported, test_claim_removed, test_risk_low, test_risk_medium,
        test_risk_high, test_correction_needed_true, test_correction_needed_false,
        test_multiple_rows_same_claim, test_multiple_evidences, test_duplicate_evidence,
        test_invalid_evidence, test_claim_present_in_text, test_claim_absent_from_text,
        test_empty_claim_text, test_empty_claim_id_discarded, test_missing_required_columns,
        test_zero_claims_raises, test_zero_active_claims_raises, test_valid_citation,
        test_invalid_citation, test_multiple_citations_index, test_duplicate_citation_not_deduplicated,
        test_zero_citations, test_citation_whitespace_trimmed, test_full_counts_and_rates,
        test_claim_audit_schema, test_citation_audit_schema, test_factual_rows_order,
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
