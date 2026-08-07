"""Pruebas de compatibilidad productiva del adaptador canónico
(``src/adapters/verification_claim_canonicalization.py``): confirman que
tanto la forma cruda (``claim_verification_records``) como la forma real
de ``build_provisional_verification_traceability_bundle``
(``claim_traceability_rows`` + colecciones asociadas) se convierten al
MISMO modelo canónico, y que la política de ``usage_role`` investigada
(``ELIGIBLE`` es fail-closed deliberado; ``SUPPORT`` solo lo asigna RAG
independiente de 07) se refleja correctamente sin inventar valores.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.verification_claim_canonicalization import (
    CANONICAL_CLAIM_FIELDS,
    canonicalize_claims_for_transition,
)
from src.tools.verification.writer_revision_cycle import classify_verification_transition

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


def _claim_traceability_row(**overrides):
    row = {
        "claim_id": "S1_C1",
        "section_id": "S1",
        "claim_type": "QUANTITATIVE",
        "original_claim_text": "El modelo alcanzó 91% de precisión.",
        "source_verdict": "PARTIALLY_SUPPORTED",
        "source_issue_codes": (),
        "source_hallucination_risk": "MEDIUM",
        "terminal_correction_recommendation": True,
        "has_correction_proposal": True,
        "correction_ids": ("C07_S1_C1_abc123",),
        "individual_proposal_decisions": (),
        "individual_accepted_correction_ids": (),
        "individual_rejected_correction_ids": (),
        "individual_deferred_correction_ids": (),
        "provisional_remaining_issue_codes": (),
        "manual_review_required": False,
        "correction_applied": False,
    }
    row.update(overrides)
    return row


def _evidence_row(**overrides):
    row = {
        "claim_id": "S1_C1", "section_id": "S1", "evidence_id": "E01",
        "source_filename": "paper_a.pdf", "chunk_id": "a1", "usage_role": "SUPPORT",
        "authorized_for_section": True, "used_in_original_verification": True,
    }
    row.update(overrides)
    return row


def _correction_row(**overrides):
    row = {
        "correction_id": "C07_S1_C1_abc123", "claim_id": "S1_C1", "section_id": "S1",
        "proposal_status": "ACCEPTED_FOR_REVERIFICATION", "replacement_text": "95%",
        "proposed_claim_text": "El modelo alcanzó 95% de precisión.",
        "is_scientific_correction_action": True,
    }
    row.update(overrides)
    return row


@scenario("N01. Bundle real con claim_traceability_rows: se extrae correctamente sin AGENT07_NO_CLAIMS")
def test_real_bundle_extraction():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(),),
        "claim_evidence_traceability_rows": (_evidence_row(),),
        "correction_traceability_rows": (_correction_row(),),
        "correction_evidence_traceability_rows": (),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert len(claims) == 1
    assert claims[0]["claim_id"] == "S1_C1"
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] != "HALT_STAGE" or decision["reason_code"] != "AGENT07_NO_CLAIMS"


@scenario("N02. Conversión al modelo canónico: los 10 campos exactos, en ambos formatos")
def test_canonical_field_set():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(),),
        "claim_evidence_traceability_rows": (_evidence_row(),),
        "correction_traceability_rows": (_correction_row(),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert set(claims[0].keys()) == set(CANONICAL_CLAIM_FIELDS)

    old_format_bundle = {
        "claim_verification_records": (
            {
                "section_id": "S1",
                "claim_verification_result": {
                    "claim_id": "S1_C1", "claim_text": "x", "scientific_verdict": "SUPPORTED",
                    "hallucination_risk": "LOW", "final_correction_eligibility": "NO_CORRECTION_NEEDED",
                    "evidence_used": (), "correction_proposal": None, "manual_review_required": False,
                    "llm_correction_recommendation": False,
                },
            },
        )
    }
    old_claims = canonicalize_claims_for_transition(old_format_bundle)
    assert set(old_claims[0].keys()) == set(CANONICAL_CLAIM_FIELDS)


@scenario("N03. Claim corregible con evidencia válida: elegibilidad AUTO_CORRECTION_ELIGIBLE derivada")
def test_correctable_claim_with_valid_evidence():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(manual_review_required=False, has_correction_proposal=True),),
        "claim_evidence_traceability_rows": (_evidence_row(usage_role="SUPPORT"),),
        "correction_traceability_rows": (_correction_row(proposal_status="ACCEPTED_FOR_REVERIFICATION"),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert claims[0]["final_correction_eligibility"] == "AUTO_CORRECTION_ELIGIBLE"
    assert claims[0]["correction_proposal"]["requested_change"] == "95%"


@scenario("N04. Claim sin rol determinable en su evidencia: no se inventa un rol")
def test_claim_no_determinable_role():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(),),
        "claim_evidence_traceability_rows": (_evidence_row(usage_role="ELIGIBLE"),),
        "correction_traceability_rows": (_correction_row(),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert claims[0]["evidence_used"][0]["usage_role"] == "ELIGIBLE"


@scenario("N05. usage_role=ELIGIBLE sin transformación válida: el claim no se declara corregible por eso solo")
def test_eligible_role_alone_does_not_force_correctable():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(has_correction_proposal=False, terminal_correction_recommendation=False),),
        "claim_evidence_traceability_rows": (_evidence_row(usage_role="ELIGIBLE"),),
        "correction_traceability_rows": (),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert claims[0]["final_correction_eligibility"] == "NO_CORRECTION_NEEDED"


@scenario("N06. Roles SUPPORT, NUMERIC y ATTRIBUTION: los tres se preservan tal cual en evidence_used")
def test_support_numeric_attribution_roles_preserved():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(),),
        "claim_evidence_traceability_rows": (
            _evidence_row(evidence_id="E01", usage_role="SUPPORT"),
            _evidence_row(evidence_id="E02", usage_role="NUMERIC"),
            _evidence_row(evidence_id="E03", usage_role="ATTRIBUTION"),
        ),
        "correction_traceability_rows": (_correction_row(),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    roles = {e["usage_role"] for e in claims[0]["evidence_used"]}
    assert roles == {"SUPPORT", "NUMERIC", "ATTRIBUTION"}


@scenario("N07. correction_proposal construido a partir de correction_traceability_rows real (propose_correction)")
def test_correction_proposal_from_real_row():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(),),
        "claim_evidence_traceability_rows": (_evidence_row(),),
        "correction_traceability_rows": (_correction_row(replacement_text="95%", proposed_claim_text="El modelo alcanzó 95%."),),
        "correction_evidence_traceability_rows": (
            {"claim_id": "S1_C1", "correction_id": "C07_S1_C1_abc123", "section_id": "S1",
             "evidence_id": "E01", "source_filename": "paper_a.pdf", "chunk_id": "a1",
             "usage_role": "SUPPORT", "authorized_for_section": True, "used_in_correction": True},
        ),
    }
    claims = canonicalize_claims_for_transition(bundle)
    proposal = claims[0]["correction_proposal"]
    assert proposal["requested_change"] == "95%"
    assert len(proposal["supporting_evidence"]) == 1


@scenario("N08. Elegibilidad conservada: manual_review_required en la fila tiene prioridad máxima")
def test_manual_review_takes_priority():
    bundle = {
        "claim_traceability_rows": (_claim_traceability_row(manual_review_required=True, has_correction_proposal=True),),
        "claim_evidence_traceability_rows": (_evidence_row(),),
        "correction_traceability_rows": (_correction_row(proposal_status="ACCEPTED_FOR_REVERIFICATION"),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert claims[0]["final_correction_eligibility"] == "MANUAL_REVIEW_REQUIRED"


@scenario("N09. Ausencia de claims no termina en aprobación accidental")
def test_no_claims_never_approves():
    decision = classify_verification_transition(claims=[], technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_NO_CLAIMS"

    empty_bundle_claims = canonicalize_claims_for_transition({"claim_traceability_rows": ()})
    assert empty_bundle_claims == []
    decision2 = classify_verification_transition(claims=empty_bundle_claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision2["action"] == "HALT_STAGE"


@scenario("N10. Formato antiguo (claim_verification_records) sigue soportado, mismo modelo canónico")
def test_old_format_still_supported():
    bundle = {
        "claim_verification_records": (
            {
                "section_id": "S1",
                "claim_verification_result": {
                    "claim_id": "S1_C1", "claim_text": "El modelo alcanzó 91%.",
                    "scientific_verdict": "PARTIALLY_SUPPORTED", "hallucination_risk": "MEDIUM",
                    "final_correction_eligibility": "AUTO_CORRECTION_ELIGIBLE",
                    "evidence_used": ({"source_filename": "paper_a.pdf", "chunk_id": "a1"},),
                    "correction_proposal": {"requested_change": "95%", "supporting_evidence": ()},
                    "manual_review_required": False, "llm_correction_recommendation": True,
                },
            },
        )
    }
    claims = canonicalize_claims_for_transition(bundle)
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "RETURN"
    assert "S1_C1" in decision["correctable_claim_ids"]


@scenario("N11. Fail-closed: elegibilidad no determinable (estado de propuesta científica no contemplado) -> None, HALT_STAGE aguas abajo")
def test_undetermined_eligibility_is_fail_closed():
    bundle = {
        # Corrección CIENTÍFICA real (is_scientific_correction_action=True)
        # pero con un proposal_status "PROPOSED" (propuesta hecha, aún sin
        # decisión de aceptación/rechazo) -- combinación no contemplada por
        # ninguna de las reglas conocidas.
        "claim_traceability_rows": (_claim_traceability_row(manual_review_required=False, has_correction_proposal=True),),
        "claim_evidence_traceability_rows": (_evidence_row(),),
        "correction_traceability_rows": (_correction_row(proposal_status="PROPOSED"),),
    }
    claims = canonicalize_claims_for_transition(bundle)
    assert claims[0]["final_correction_eligibility"] is None
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_UNKNOWN_ELIGIBILITY"


if __name__ == "__main__":
    for fn in (
        test_real_bundle_extraction,
        test_canonical_field_set,
        test_correctable_claim_with_valid_evidence,
        test_claim_no_determinable_role,
        test_eligible_role_alone_does_not_force_correctable,
        test_support_numeric_attribution_roles_preserved,
        test_correction_proposal_from_real_row,
        test_manual_review_takes_priority,
        test_no_claims_never_approves,
        test_old_format_still_supported,
        test_undetermined_eligibility_is_fail_closed,
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
