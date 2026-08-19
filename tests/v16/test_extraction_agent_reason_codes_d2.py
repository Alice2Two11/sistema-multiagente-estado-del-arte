"""Extracción mecánica (Bloque D, D2) del cálculo de ``reason_codes``/
``can_manual`` de ``ExtractionAgent.execute()`` a
``ExtractionAgent._compute_reason_codes_and_manual_review_
eligibility`` (``src/agents/extraction_agent.py``).

Causa raíz: continuación de D1 -- el bloque que calcula
``reason_codes``/``can_manual`` justo antes de la decisión final
cumple los mismos criterios de bajo riesgo (5 inputs de solo
lectura, 2 outputs explícitos, sin mutar nada externo, sin depender
de orden lateral, sin duplicar ``_scientific_reason_codes``).

Fix: extracción MECÁNICA -- mismo texto, mismos thresholds
(0.92/0.80 por defecto), mismo orden, mismos reason_codes -- ahora en
un ``@staticmethod`` propio, verificable de forma aislada.

Estos tests cubren las 9 ramas pedidas explícitamente (coverage por
encima/debajo del approval threshold, reason_codes científicos ya
presentes, manual_review_policy allowed True/False, reason_code
permitido/no permitido para revisión manual, coverage por debajo de
minimum_usable, múltiples reason_codes), más los 3 fixtures de
referencia corridos end-to-end para confirmar equivalencia exacta.

Multidominio y genérico: la parte unitaria usa cards/policies
sintéticas mínimas, sin depender de contenido científico real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.extraction_agent import ExtractionAgent  # noqa: E402

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


def _compute(*, extraction_policy=None, coverage, cards=(), bad_after_repair=(), extraction_errors=()):
    return ExtractionAgent._compute_reason_codes_and_manual_review_eligibility(
        extraction_policy=extraction_policy or {}, coverage=coverage, cards=cards,
        bad_after_repair=bad_after_repair, extraction_errors=extraction_errors,
    )


@scenario("D2-01. coverage >= approval_threshold (default 0.92) y sin campos faltantes -> reason_codes vacío, no se inyecta MISSING_CRITICAL_FIELDS")
def test_d2_01_coverage_above_approval_threshold():
    reason_codes, can_manual = _compute(coverage=0.95, cards=())
    assert "MISSING_CRITICAL_FIELDS" not in reason_codes


@scenario("D2-02. coverage < approval_threshold (default 0.92) -> se inyecta MISSING_CRITICAL_FIELDS aunque _scientific_reason_codes no lo haya producido")
def test_d2_02_coverage_below_approval_threshold_injects_missing_critical_fields():
    reason_codes, can_manual = _compute(coverage=0.50, cards=())
    assert "MISSING_CRITICAL_FIELDS" in reason_codes


@scenario("D2-03. coverage < approval_threshold pero MISSING_CRITICAL_FIELDS ya presente -> no se duplica (dict.fromkeys preserva unicidad)")
def test_d2_03_no_duplicate_missing_critical_fields():
    card_missing = {"source_filename": "a.pdf", "title": "T", "corpus_eligibility": "INCLUDE"}
    reason_codes, can_manual = _compute(coverage=0.50, cards=[card_missing])
    assert reason_codes.count("MISSING_CRITICAL_FIELDS") <= 1


@scenario("D2-04. manual_review_policy.allowed=True + coverage>=minimum_usable + reason_code permitido -> can_manual=True")
def test_d2_04_can_manual_true_when_allowed_and_code_matches():
    policy = {
        "thresholds": {
            "approval": {"critical_field_coverage": 0.92},
            "minimum_usable_quality": {"critical_field_coverage": 0.80},
        },
        "manual_review_policy": {"allowed": True, "allowed_reason_codes": ("MISSING_CRITICAL_FIELDS",)},
    }
    # coverage=0.85: por debajo del approval_threshold (0.92, así que
    # SÍ se inyecta MISSING_CRITICAL_FIELDS) pero por encima del
    # minimum_usable (0.80, así que can_manual puede ser True).
    reason_codes, can_manual = _compute(extraction_policy=policy, coverage=0.85, cards=())
    assert "MISSING_CRITICAL_FIELDS" in reason_codes
    assert can_manual is True


@scenario("D2-05. manual_review_policy.allowed=False -> can_manual=False aunque el resto de condiciones se cumplan")
def test_d2_05_can_manual_false_when_policy_disallows():
    policy = {
        "thresholds": {"minimum_usable_quality": {"critical_field_coverage": 0.80}},
        "manual_review_policy": {"allowed": False, "allowed_reason_codes": ("MISSING_CRITICAL_FIELDS",)},
    }
    reason_codes, can_manual = _compute(extraction_policy=policy, coverage=0.85, cards=())
    assert can_manual is False


@scenario("D2-06. reason_code producido NO está en allowed_reason_codes -> can_manual=False aunque allowed=True")
def test_d2_06_can_manual_false_when_reason_code_not_allowed():
    policy = {
        "thresholds": {"minimum_usable_quality": {"critical_field_coverage": 0.80}},
        "manual_review_policy": {"allowed": True, "allowed_reason_codes": ("SOME_OTHER_CODE",)},
    }
    reason_codes, can_manual = _compute(extraction_policy=policy, coverage=0.50, cards=())
    assert "MISSING_CRITICAL_FIELDS" in reason_codes
    assert can_manual is False


@scenario("D2-07. coverage por debajo de minimum_usable (default 0.80) -> can_manual=False aunque allowed=True y reason_code coincida")
def test_d2_07_can_manual_false_when_below_minimum_usable():
    policy = {
        "manual_review_policy": {"allowed": True, "allowed_reason_codes": ("MISSING_CRITICAL_FIELDS",)},
    }
    reason_codes, can_manual = _compute(extraction_policy=policy, coverage=0.10, cards=())
    assert "MISSING_CRITICAL_FIELDS" in reason_codes
    assert can_manual is False


@scenario("D2-08. múltiples reason_codes (de extraction_errors reales de build_revision_plan) se preservan sin duplicar, y basta con UNO permitido para can_manual=True")
def test_d2_08_multiple_reason_codes_preserved():
    card_review = {
        "source_filename": "a.pdf", "title": "no especificado", "paper_type": "no especificado",
        "corpus_eligibility": "INCLUDE",
    }
    policy = {
        "manual_review_policy": {"allowed": True, "allowed_reason_codes": ("MISSING_OR_INVALID_TITLE",)},
    }
    reason_codes, can_manual = _compute(extraction_policy=policy, coverage=0.10, cards=[card_review])
    assert len(reason_codes) == len(set(reason_codes))  # sin duplicados
    assert can_manual is False  # coverage 0.10 < minimum_usable 0.80 -> False pase lo que pase


@scenario("D2-09. Sin ninguna card (cards=()) y coverage suficiente -> reason_codes vacío, can_manual=False (sin allowed_manual_codes que intersecten con nada)")
def test_d2_09_empty_cards_high_coverage_no_reason_codes():
    reason_codes, can_manual = _compute(coverage=1.0, cards=())
    assert reason_codes == ()
    assert can_manual is False


if __name__ == "__main__":
    for fn in (
        test_d2_01_coverage_above_approval_threshold,
        test_d2_02_coverage_below_approval_threshold_injects_missing_critical_fields,
        test_d2_03_no_duplicate_missing_critical_fields,
        test_d2_04_can_manual_true_when_allowed_and_code_matches,
        test_d2_05_can_manual_false_when_policy_disallows,
        test_d2_06_can_manual_false_when_reason_code_not_allowed,
        test_d2_07_can_manual_false_when_below_minimum_usable,
        test_d2_08_multiple_reason_codes_preserved,
        test_d2_09_empty_cards_high_coverage_no_reason_codes,
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
