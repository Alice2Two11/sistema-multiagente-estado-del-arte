"""Caracterización del comportamiento real de ``numeric_risk`` en el
precheck determinista de 07 (``deterministic_precheck``,
``src/tools/verification/validation.py``).

Tarea independiente registrada por instrucción explícita: NO se resuelve
ni se modifica el gate cuantitativo en esta ronda. Esta prueba documenta
el comportamiento ACTUAL, con dos condiciones reales distintas que a
menudo se confunden:

1. ``numeric_risk="HIGH"`` con presupuesto de RAG adicional disponible
   (``retrieval_possible=True``): el precheck agrega
   ``QUANTITATIVE_COVERAGE_INCOMPLETE`` pero NO bloquea el juicio
   científico — ``scientific_judgment_status`` sigue en ``"PENDING"``, el
   LLM SÍ se invoca normalmente.
2. ``numeric_risk="HIGH"`` SIN presupuesto de RAG adicional
   (``retrieval_possible=False``, ej. ``max_additional_retrieval_requests=0``):
   el precheck agrega ``UNSUPPORTED_NUMERIC_VALUE`` (terminal) y fuerza
   ``scientific_judgment_status="COMPLETED"`` con
   ``scientific_verdict="NOT_EVALUATED"`` de forma DETERMINISTA — el LLM
   NUNCA se invoca; ``allowed_verdicts_for_claim`` devuelve únicamente
   ``("NOT_EVALUATED",)``.

Condición exacta que activa el bloqueo terminal (caso 2), leída
literalmente de ``deterministic_precheck``:

    claim_type == "QUANTITATIVE"
    and not numeric_pairs_valid   # numeric_risk_status=="EVALUATED" y
                                   # numeric_risk en {HIGH,CRITICAL,
                                   # UNSUPPORTED,FAIL}
    and not retrieval_possible    # remaining_retrieval_requests<=0 o
                                   # terminal_technical_blocker=True

Campos requeridos para reproducir el bloqueo: ``numeric_risk``,
``numeric_risk_status`` (en el contexto crudo de 06→07,
``src/adapters/agent06_verification_handoff.py``, poblados desde
``numeric_hallucination_check.csv``, artefacto de la ruta histórica 07C
-- ver nota de compatibilidad abajo) y el presupuesto de RAG adicional
de la política (``max_additional_retrieval_requests``).

¿Regla científica intencional o incompatibilidad? Se documenta como
hallazgo, no como veredicto: la lógica en sí (bloquear un veredicto
automático cuando un valor numérico tiene riesgo alto Y ya no hay más
presupuesto de verificación adicional) es coherente con un principio
fail-closed deliberado -- evita que el LLM emita un veredicto sobre un
valor numérico de riesgo conocido sin haber agotado los medios
disponibles para corroborarlo. Lo que no se investigó en esta ronda es
si el ORIGEN del dato (``numeric_hallucination_check.csv``, un artefacto
de la ruta histórica 07C) es compatible sin fricción con la ruta activa
de 07 directo, y si el notebook 03B (que aparentemente calcula ese CSV)
sigue siendo la fuente real y vigente de ``numeric_risk`` en el flujo
activo -- eso queda para la tarea independiente pedida.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.claim_verification_context import (
    build_claim_verification_context_from_agent06_handoff,
    fingerprint_text,
)
from src.tools.verification.validation import allowed_verdicts_for_claim, deterministic_precheck

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


def _quantitative_ctx(*, numeric_risk, numeric_risk_status):
    claim_text = "El modelo alcanzó 91% de precisión en el conjunto de prueba."
    section_text = claim_text
    return {
        "claim_id": "S1_C1", "claim_id_origin": "inherited_agent06", "section_id": "S1",
        "section_title": "Resultados", "original_claim_text": claim_text, "section_text": section_text,
        "supporting_citations": ["[paper_a.pdf | a1]"], "source_free_organizational_section": False,
        "claim_span_in_section": {
            "coordinate_base": "SECTION_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint": fingerprint_text(section_text), "start": 0, "end": len(section_text),
            "text": section_text,
        },
        "claim_fingerprint": fingerprint_text(claim_text), "section_fingerprint": fingerprint_text(section_text),
        "eligible_evidence": (
            {"evidence_id": "a1", "source_filename": "paper_a.pdf", "chunk_id": "a1",
             "text": "El modelo alcanzó 95% de precisión.", "usage_role": "ELIGIBLE", "authorized_for_section": True},
        ),
        "authorized_source_filenames": ("paper_a.pdf",),
        "numeric_risk": numeric_risk, "numeric_risk_status": numeric_risk_status,
        "source_draft_fingerprint": "a" * 64,
    }


@scenario("Q01. claim cuantitativo, numeric_risk=HIGH, con presupuesto de RAG -> NO bloquea el juicio (PENDING)")
def test_quantitative_high_risk_with_budget_not_blocked():
    ctx = _quantitative_ctx(numeric_risk="HIGH", numeric_risk_status="EVALUATED")
    core_ctx = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={}, attempt_number=1)
    assert core_ctx["claim_type"] == "QUANTITATIVE"
    precheck = deterministic_precheck(core_ctx)
    assert precheck["scientific_judgment_status"] == "PENDING"
    # Nota: "scientific_verdict" en el precheck es solo un placeholder
    # ("NOT_EVALUATED" por defecto hasta que el LLM se ejecute) -- la
    # señal real de si el juicio está bloqueado o no es
    # scientific_judgment_status, no este campo.
    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" in precheck["deterministic_issue_codes"]
    allowed = allowed_verdicts_for_claim(core_ctx, precheck)
    assert "PARTIALLY_SUPPORTED" in allowed  # el LLM SÍ puede emitir un veredicto real aquí


@scenario("Q02. claim cuantitativo, numeric_risk=HIGH, SIN presupuesto de RAG -> bloqueo determinista terminal")
def test_quantitative_high_risk_without_budget_blocked():
    ctx = _quantitative_ctx(numeric_risk="HIGH", numeric_risk_status="EVALUATED")
    core_ctx = build_claim_verification_context_from_agent06_handoff(
        ctx, verification_policy={"max_additional_retrieval_requests": 0}, attempt_number=1
    )
    precheck = deterministic_precheck(core_ctx)
    assert precheck["scientific_judgment_status"] == "COMPLETED"
    assert precheck["scientific_verdict"] == "NOT_EVALUATED"
    assert "UNSUPPORTED_NUMERIC_VALUE" in precheck["deterministic_issue_codes"]
    allowed = allowed_verdicts_for_claim(core_ctx, precheck)
    assert allowed == ("NOT_EVALUATED",)  # el LLM NUNCA se invoca en este caso


@scenario("Q03. claim cuantitativo, numeric_risk_status=NOT_AVAILABLE -> numeric_pairs_valid=True, sin bloqueo")
def test_quantitative_risk_not_available_no_block():
    ctx = _quantitative_ctx(numeric_risk=None, numeric_risk_status="NOT_AVAILABLE")
    core_ctx = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={}, attempt_number=1)
    assert core_ctx["deterministic_validation"]["numeric_pairs_valid"] is True
    precheck = deterministic_precheck(core_ctx)
    assert precheck["scientific_judgment_status"] == "PENDING"
    assert "UNSUPPORTED_NUMERIC_VALUE" not in precheck["deterministic_issue_codes"]


@scenario("Q04. claim NO cuantitativo (metodológico) con numeric_risk=HIGH: el riesgo numérico no aplica en absoluto")
def test_non_quantitative_claim_ignores_numeric_risk():
    claim_text = "El método Alpha utiliza una arquitectura basada en transformadores."
    section_text = claim_text
    ctx = {
        "claim_id": "S1_C1", "claim_id_origin": "inherited_agent06", "section_id": "S1",
        "section_title": "Métodos", "original_claim_text": claim_text, "section_text": section_text,
        "supporting_citations": ["[paper_a.pdf | a1]"], "source_free_organizational_section": False,
        "claim_span_in_section": {
            "coordinate_base": "SECTION_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint": fingerprint_text(section_text), "start": 0, "end": len(section_text),
            "text": section_text,
        },
        "claim_fingerprint": fingerprint_text(claim_text), "section_fingerprint": fingerprint_text(section_text),
        "eligible_evidence": (
            {"evidence_id": "a1", "source_filename": "paper_a.pdf", "chunk_id": "a1",
             "text": "El método Alpha usa transformadores.", "usage_role": "ELIGIBLE", "authorized_for_section": True},
        ),
        "authorized_source_filenames": ("paper_a.pdf",),
        # numeric_risk=HIGH pero el claim NO es cuantitativo -> irrelevante.
        "numeric_risk": "HIGH", "numeric_risk_status": "EVALUATED",
        "source_draft_fingerprint": "a" * 64,
    }
    core_ctx = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={}, attempt_number=1)
    assert core_ctx["claim_type"] == "METHODOLOGICAL"
    assert core_ctx["deterministic_validation"]["numeric_pairs_valid"] is True  # el chequeo solo aplica a QUANTITATIVE
    precheck = deterministic_precheck(core_ctx)
    assert precheck["scientific_judgment_status"] == "PENDING"
    assert "UNSUPPORTED_NUMERIC_VALUE" not in precheck["deterministic_issue_codes"]


if __name__ == "__main__":
    for fn in (
        test_quantitative_high_risk_with_budget_not_blocked,
        test_quantitative_high_risk_without_budget_blocked,
        test_quantitative_risk_not_available_no_block,
        test_non_quantitative_claim_ignores_numeric_risk,
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
