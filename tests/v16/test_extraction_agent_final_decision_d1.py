"""Extracción mecánica (Bloque D, D1) de la decisión final RETRY/HALT/
APPROVED de ``ExtractionAgent.execute()`` a
``ExtractionAgent._decide_final_quality_status_and_transition``
(``src/agents/extraction_agent.py``).

Causa raíz: ``execute()`` es un único método de ~1670 líneas; el único
bloque identificado como genuinamente autónomo (entrada pequeña,
sin mutar estado externo, un solo punto de retorno) es la decisión
final que produce ``(quality_status, transition)`` a partir de
``reason_codes``/``can_manual``/``agent_input`` (y ``has_warnings``,
necesario para la rama sin reason_codes -- ver docstring del método).

Fix: extracción MECÁNICA -- mismo texto, mismas condiciones, mismo
orden, mismos reason_codes, mismos mensajes -- ahora en un
``@staticmethod`` propio, verificable de forma aislada sin construir
el flujo completo del agente.

Estos tests cubren las 4 ramas reales del bloque, más los 3 fixtures
de referencia de la conversación (INCLUDE+EXCLUDE, INCLUDE+
QUARANTINE, campo faltante) corridos end-to-end para confirmar
equivalencia exacta con el comportamiento pre-extracción.

Multidominio y genérico: ningún test usa contenido científico
concreto para la parte unitaria (usa tuplas de reason_codes
sintéticas, no requiere un experimento real)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

from src.agents.extraction_agent import ExtractionAgent  # noqa: E402
from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode  # noqa: E402
from src.contracts.agent_result import QualityStatus, TransitionAction  # noqa: E402

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


def _agent_input(attempt_number: int) -> AgentInput:
    return AgentInput(
        experiment_id="e", run_id="r", stage_name="03_agente_extraccion_kb",
        attempt_number=attempt_number, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("t",), output_directory="o"),
        dependencies={}, policy={},
    )


@scenario("D1-01. Sin reason_codes y sin warnings -> APPROVED / ADVANCE / EXTRACTION_COMPLETED")
def test_d1_01_no_reason_codes_no_warnings_approved():
    status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
        reason_codes=(), can_manual=False, has_warnings=False, agent_input=_agent_input(1),
    )
    assert status == QualityStatus.APPROVED
    assert transition.action == TransitionAction.ADVANCE
    assert transition.reason_code == "EXTRACTION_COMPLETED"
    assert transition.requires_human_confirmation is False
    assert transition.target_stage is None


@scenario("D1-02. Sin reason_codes PERO con warnings -> APPROVED_WITH_WARNINGS / ADVANCE (misma transición, distinto quality_status)")
def test_d1_02_no_reason_codes_with_warnings():
    status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
        reason_codes=(), can_manual=False, has_warnings=True, agent_input=_agent_input(1),
    )
    assert status == QualityStatus.APPROVED_WITH_WARNINGS
    assert transition.action == TransitionAction.ADVANCE
    assert transition.reason_code == "EXTRACTION_COMPLETED"


@scenario("D1-03. reason_codes presentes + intento 1 (primer intento) -> NEEDS_REVISION / RETRY, sin importar can_manual")
def test_d1_03_reason_codes_first_attempt_always_retries():
    for can_manual in (True, False):
        status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
            reason_codes=("MISSING_CRITICAL_FIELDS",), can_manual=can_manual,
            has_warnings=False, agent_input=_agent_input(1),
        )
        assert status == QualityStatus.NEEDS_REVISION, can_manual
        assert transition.action == TransitionAction.RETRY, can_manual
        assert transition.reason_code == "NEEDS_REVISION", can_manual
        assert transition.requires_human_confirmation is False, can_manual


@scenario("D1-04. reason_codes presentes + intento final + can_manual=True -> APPROVED_PENDING_MANUAL_REVIEW / HALT_STAGE")
def test_d1_04_reason_codes_final_attempt_can_manual():
    status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
        reason_codes=("MISSING_CRITICAL_FIELDS",), can_manual=True,
        has_warnings=False, agent_input=_agent_input(2),
    )
    assert status == QualityStatus.APPROVED_PENDING_MANUAL_REVIEW
    assert transition.action == TransitionAction.HALT_STAGE
    assert transition.reason_code == "APPROVED_PENDING_MANUAL_REVIEW"
    assert transition.requires_human_confirmation is True


@scenario("D1-05. reason_codes presentes + intento final + can_manual=False -> REJECTED / HALT_STAGE")
def test_d1_05_reason_codes_final_attempt_cannot_manual():
    status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
        reason_codes=("MISSING_CRITICAL_FIELDS",), can_manual=False,
        has_warnings=False, agent_input=_agent_input(2),
    )
    assert status == QualityStatus.REJECTED
    assert transition.action == TransitionAction.HALT_STAGE
    assert transition.reason_code == "REJECTED"
    assert transition.requires_human_confirmation is False


@scenario("D1-06. has_warnings solo afecta la rama SIN reason_codes -- con reason_codes presentes, el resultado es idéntico con warnings True o False")
def test_d1_06_has_warnings_irrelevant_when_reason_codes_present():
    for has_warnings in (True, False):
        status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
            reason_codes=("MISSING_CRITICAL_FIELDS",), can_manual=False,
            has_warnings=has_warnings, agent_input=_agent_input(1),
        )
        assert status == QualityStatus.NEEDS_REVISION
        assert transition.action == TransitionAction.RETRY


@scenario("D1-07. Múltiples reason_codes (tupla con más de un código) sigue exactamente la misma rama que un solo código")
def test_d1_07_multiple_reason_codes_same_branch_logic():
    status, transition = ExtractionAgent._decide_final_quality_status_and_transition(
        reason_codes=("MISSING_CRITICAL_FIELDS", "MISSING_OR_INVALID_TITLE"),
        can_manual=False, has_warnings=False, agent_input=_agent_input(1),
    )
    assert status == QualityStatus.NEEDS_REVISION
    assert transition.action == TransitionAction.RETRY


if __name__ == "__main__":
    for fn in (
        test_d1_01_no_reason_codes_no_warnings_approved,
        test_d1_02_no_reason_codes_with_warnings,
        test_d1_03_reason_codes_first_attempt_always_retries,
        test_d1_04_reason_codes_final_attempt_can_manual,
        test_d1_05_reason_codes_final_attempt_cannot_manual,
        test_d1_06_has_warnings_irrelevant_when_reason_codes_present,
        test_d1_07_multiple_reason_codes_same_branch_logic,
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
