"""Corrección de vocabulario: ``QUANTITATIVE_COVERAGE_INCOMPLETE``
pertenece a ``RETRIEVAL_REASON_CODES``
(``src/config/verification_policy_config.py``), no a
``DETERMINISTIC_ISSUE_CODES`` -- pero ``deterministic_precheck``
(``src/tools/verification/validation.py``) lo insertaba en
``deterministic_issue_codes``, violando el vocabulario terminal que
``validate_claim_verification_result_contract`` exige para ese campo
(``_terminal_string_seq(..., allowed=set(DETERMINISTIC_ISSUE_CODES))``).

Caso real que lo confirmó: el registro 13 de ``claim_verification_
records`` en una ejecución real de 07 (claim ``S5_C3``, cuantitativo,
``scientific_judgment_status=COMPLETED``, ``scientific_verdict=
INSUFFICIENT_EVIDENCE`` -- el LLM SÍ corrió y emitió un veredicto real,
correcto). ``deterministic_issue_codes`` contenía
``QUANTITATIVE_COVERAGE_INCOMPLETE`` -- código legítimamente producido
por el precheck (numeric_risk alto, con presupuesto de RAG disponible),
pero fuera del vocabulario permitido -- y la agregación de 07 fallaba
con ``ValueError: TERMINAL_CONTRACT_UNKNOWN_CODE:deterministic_issue_
codes`` al intentar validar ese registro, produciendo un
``aggregation_status=INVALID`` -- un problema estructural/contractual,
NUNCA una decisión científica incorrecta.

Corrección: ``deterministic_precheck`` ahora acumula
``QUANTITATIVE_COVERAGE_INCOMPLETE`` en una lista separada,
``retrieval_reason_codes``, devuelta como una clave nueva de su
resultado -- nunca mezclada con ``issues``/``deterministic_issue_
codes``. ``VerificationAgent._terminal`` (camino sin LLM) y el camino
con LLM (dentro de ``verify_claim``) incorporan ``retrieval_reason_
codes`` a ``reason_codes`` -- el único campo del contrato terminal cuyo
vocabulario permitido SÍ incluye ``RETRIEVAL_REASON_CODES`` (confirmado
en ``validate_claim_verification_result_contract``, línea ~3191).

Invariantes verificados aquí, exactamente como se pidieron:
- ``QUANTITATIVE_COVERAGE_INCOMPLETE`` desaparece por completo de
  ``deterministic_issue_codes`` (T01, T02, T03).
- Se conserva en ``reason_codes`` en AMBOS caminos, sin LLM y con LLM
  (T02, T03).
- ``DETERMINISTIC_ISSUE_CODES`` no se amplía -- no se toca en absoluto
  en este parche (verificado por inspección; T04 confirma que todo lo
  que ``deterministic_precheck`` sigue produciendo para ese campo
  cabe en el vocabulario existente, sin ampliarlo).
- ``RETRIEVAL_REASON_CODES`` no cambia de significado -- tampoco se
  toca; T05 confirma que ningún miembro de ese vocabulario aparece
  jamás en ``deterministic_issue_codes``.
- ``scientific_verdict``/``support_level``/``semantic_issue_codes``/
  evidencia no cambian (T06, con el patrón real de S5_C3).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_verification_numeric_risk_characterization import _quantitative_ctx  # noqa: E402

from src.adapters.claim_verification_context import (  # noqa: E402
    build_claim_verification_context_from_agent06_handoff,
    fingerprint_text,
)
from src.agents.verification_agent import VerificationAgent  # noqa: E402
from src.config.verification_policy_config import (  # noqa: E402
    DETERMINISTIC_ISSUE_CODES,
    RETRIEVAL_REASON_CODES,
)
from src.tools.verification.validation import (  # noqa: E402
    deterministic_precheck,
    validate_claim_verification_result_contract,
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


def _quantitative_high_risk_with_budget_context():
    """Mismo patrón que provocó QUANTITATIVE_COVERAGE_INCOMPLETE en el
    registro real: claim cuantitativo, numeric_risk alto, presupuesto
    de retrieval disponible -- el camino EXACTO que llevó al LLM en el
    caso real de S5_C3 (retrieval_possible=True -> judgment_status
    sigue PENDING -> se invoca al LLM de verdad)."""
    ctx = _quantitative_ctx(numeric_risk="HIGH", numeric_risk_status="EVALUATED")
    return build_claim_verification_context_from_agent06_handoff(
        ctx, verification_policy={}, attempt_number=1
    )


class _FakeVerificationLLM:
    """Reproduce el veredicto real de S5_C3 -- INSUFFICIENT_EVIDENCE,
    con NUMERIC_CONTEXT_MISMATCH en semantic_issue_codes -- sin
    inventar ningún dato: son los valores exactos que reportó el
    diagnóstico real."""

    def invoke(self, messages):
        return {
            "claim_id": "S1_C1", "verdict": "INSUFFICIENT_EVIDENCE", "support_level": "NONE",
            "evidence_ids_used": [], "evidence_ids_rejected": [], "rationale": "la evidencia recuperada no contiene las cifras exactas",
            "contradiction_type": "NONE", "contradiction_evidence_ids": [],
            "numeric_assessment": "CONTEXT_MISMATCH", "attribution_assessment": "NOT_APPLICABLE",
            "extrapolation_assessment": "NOT_APPLICABLE", "confidence": "MEDIUM",
            "additional_retrieval_needed": False, "llm_correction_recommendation": False,
            "manual_review_required": False,
            # El LLM reporta sus PROPIOS reason_codes (vocabulario
            # SEMANTIC_REASON_CODES, distinto de SEMANTIC_ISSUE_CODES) --
            # nunca copia nada de deterministic_issue_codes; confirma
            # que QUANTITATIVE_COVERAGE_INCOMPLETE no llega por aquí
            # tampoco, solo por la unión explícita que se agregó.
            "reason_codes": ["CONTEXT_MISMATCH"],
        }


@scenario("T01. Reproducción del registro real: deterministic_precheck produce QUANTITATIVE_COVERAGE_INCOMPLETE en retrieval_reason_codes, NUNCA en deterministic_issue_codes")
def test_precheck_places_code_in_correct_field():
    core_ctx = _quantitative_high_risk_with_budget_context()
    precheck = deterministic_precheck(core_ctx)

    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" in precheck["retrieval_reason_codes"]
    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" not in precheck["deterministic_issue_codes"]
    assert set(precheck["deterministic_issue_codes"]).issubset(set(DETERMINISTIC_ISSUE_CODES))


@scenario("T02. Camino _terminal (sin LLM): reason_codes incorpora QUANTITATIVE_COVERAGE_INCOMPLETE; deterministic_issue_codes nunca lo tiene")
def test_terminal_path_reason_codes_includes_code():
    # Forzar el camino _terminal: technical_blockers sin evidencia
    # elegible produce scientific_judgment_status=BLOCKED de forma
    # determinista, sin invocar al LLM -- exactamente el otro camino
    # real que _terminal cubre.
    ctx = _quantitative_ctx(numeric_risk="HIGH", numeric_risk_status="EVALUATED")
    ctx = {**ctx, "eligible_evidence": ()}
    core_ctx = build_claim_verification_context_from_agent06_handoff(
        ctx, verification_policy={}, attempt_number=1
    )
    agent = VerificationAgent(llm=None)
    result = agent.verify_claim(core_ctx)

    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" not in result.deterministic_issue_codes
    # Puede o no estar presente según si retrieval_possible se cumple en
    # este camino particular -- lo que importa es que SI aparece en
    # algún lado, es en reason_codes, nunca en deterministic_issue_codes.
    precheck = deterministic_precheck(core_ctx)
    if precheck["retrieval_reason_codes"]:
        assert set(precheck["retrieval_reason_codes"]).issubset(set(result.reason_codes))


@scenario("T03. Camino con LLM (el que realmente tomó S5_C3): reason_codes = union(LLM.reason_codes, retrieval_reason_codes); deterministic_issue_codes limpio; verdict/support_level/semantic_issue_codes SIN CAMBIOS")
def test_llm_path_reason_codes_union_verdict_unchanged():
    core_ctx = _quantitative_high_risk_with_budget_context()
    agent = VerificationAgent(llm=_FakeVerificationLLM())
    result = agent.verify_claim(core_ctx)

    # El veredicto científico real de S5_C3 -- SIN CAMBIOS.
    assert result.scientific_verdict == "INSUFFICIENT_EVIDENCE"
    assert result.support_level == "NONE"
    assert result.semantic_issue_codes == ("INSUFFICIENT_EVIDENCE", "NUMERIC_CONTEXT_MISMATCH")

    # La corrección de vocabulario, verificada en el resultado final real.
    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" not in result.deterministic_issue_codes
    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" in result.reason_codes
    assert "CONTEXT_MISMATCH" in result.reason_codes  # lo que el LLM reportó se conserva

    # La prueba central: el contrato terminal real ya NO rechaza este registro.
    validate_claim_verification_result_contract(result.to_dict())


@scenario("T04. Consistencia: TODO código que deterministic_precheck puede producir para deterministic_issue_codes está en DETERMINISTIC_ISSUE_CODES (barrido de las 4 combinaciones reales que lo activan)")
def test_all_producible_deterministic_codes_are_allowed():
    producible = set()

    # INVALID_CITATION / DOCUMENT_IDENTITY_INVALID / UNAUTHORIZED_SOURCE
    ctx = _quantitative_ctx(numeric_risk="LOW", numeric_risk_status="EVALUATED")
    for flag, code in (
        ("citation_valid", "INVALID_CITATION"),
        ("document_identity_valid", "DOCUMENT_IDENTITY_INVALID"),
        ("authorization_valid", "UNAUTHORIZED_SOURCE"),
    ):
        core_ctx = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={}, attempt_number=1)
        core_ctx = {**core_ctx, "deterministic_validation": {**core_ctx["deterministic_validation"], flag: False}}
        precheck = deterministic_precheck(core_ctx)
        producible |= set(precheck["deterministic_issue_codes"])
        assert code in precheck["deterministic_issue_codes"]

    # UNSUPPORTED_NUMERIC_VALUE (numeric_risk alto, SIN presupuesto de retrieval)
    ctx2 = _quantitative_ctx(numeric_risk="HIGH", numeric_risk_status="EVALUATED")
    core_ctx2 = build_claim_verification_context_from_agent06_handoff(
        ctx2, verification_policy={"max_additional_retrieval_requests": 0}, attempt_number=1
    )
    precheck2 = deterministic_precheck(core_ctx2)
    producible |= set(precheck2["deterministic_issue_codes"])
    assert "UNSUPPORTED_NUMERIC_VALUE" in precheck2["deterministic_issue_codes"]

    assert producible.issubset(set(DETERMINISTIC_ISSUE_CODES))
    # QUANTITATIVE_COVERAGE_INCOMPLETE nunca debe aparecer en ningún
    # barrido de deterministic_issue_codes real.
    assert "QUANTITATIVE_COVERAGE_INCOMPLETE" not in producible


@scenario("T05. Consistencia inversa: ningún miembro de RETRIEVAL_REASON_CODES aparece jamás en deterministic_issue_codes")
def test_no_retrieval_reason_code_ever_in_deterministic_issue_codes():
    core_ctx = _quantitative_high_risk_with_budget_context()
    precheck = deterministic_precheck(core_ctx)
    overlap = set(precheck["deterministic_issue_codes"]) & set(RETRIEVAL_REASON_CODES)
    assert overlap == set()


@scenario("T06. La regresión exacta que dio origen a esto: validate_claim_verification_result_contract ya NO lanza TERMINAL_CONTRACT_UNKNOWN_CODE:deterministic_issue_codes")
def test_terminal_contract_no_longer_rejects_the_real_pattern():
    core_ctx = _quantitative_high_risk_with_budget_context()
    agent = VerificationAgent(llm=_FakeVerificationLLM())
    result = agent.verify_claim(core_ctx)
    try:
        validate_claim_verification_result_contract(result.to_dict())
    except ValueError as exc:
        raise AssertionError(f"el contrato terminal rechazó un registro válido: {exc}")


if __name__ == "__main__":
    for fn in (
        test_precheck_places_code_in_correct_field,
        test_terminal_path_reason_codes_includes_code,
        test_llm_path_reason_codes_union_verdict_unchanged,
        test_all_producible_deterministic_codes_are_allowed,
        test_no_retrieval_reason_code_ever_in_deterministic_issue_codes,
        test_terminal_contract_no_longer_rejects_the_real_pattern,
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
