"""Corrección quirúrgica de Agente 07 -- caso real Exp04: 6 claims
bloqueantes con ``proposal_status=DEFERRED`` y ``reason_codes``/
``comparison_reason_codes``/``precheck_reason_codes`` vacíos, sin
ninguna causa auditable.

Diagnóstico confirmado (con los artefactos reales del Exp04):
- ``manual_review_required`` viene directamente de la propia respuesta
  del LLM de verificación (``validated["manual_review_required"]``,
  ``verification_agent.py``) -- ``determine_final_correction_
  eligibility`` (``validation.py``) lo trata como override
  incondicional a ``MANUAL_REVIEW_REQUIRED``, ANTES de evaluar si el
  claim tiene evidencia autorizada localmente corregible. No se
  encontró evidencia de que esto sea un bug -- es coherente con el
  resto del sistema (si el LLM de verificación pidió revisión humana,
  el sistema no lo sobreescribe). No se tocó esta regla.
- Bug real #1: ``_empty_proposal`` (``corrections.py``) tenía
  ``reason_codes`` HARDCODEADO a ``()`` en vez de ``tuple(issues)`` --
  mientras que ``validation_issue_codes`` sí recibía ``issues``
  correctamente.
- Bug real #2: ``CorrectionTraceabilityRow`` (``traceability.py``) no
  tenía NINGÚN campo para ``correction_decision``, ``final_proposal_
  status``, ``requires_manual_review``, ``accepted_for_
  reverification``, ``reason_codes``, ``validation_issue_codes``,
  ``decision_path``, ``retry_metrics`` ni ``raw_attempts`` -- aunque
  ``CorrectionProposal`` sí los captura todos. Extendidos con defaults
  (no rompe construcciones posicionales existentes) y poblados en el
  único sitio de construcción ACTIVO (confirmado explícitamente: hay
  tres definiciones de ``build_provisional_traceability_rows`` en
  ``validation.py``, la última gana) desde la ``CorrectionProposal``
  unida por ``correction_id``. ``resolution.py`` no se tocó."""

from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_qualitative_correction_keyerror_and_return as R  # noqa: E402
import test_verification_stagespec_integration as T  # noqa: E402

from src.tools.verification.corrections import fingerprint_text, propose_correction  # noqa: E402
from src.tools.verification.validation import build_provisional_traceability_rows  # noqa: E402

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


def _base_correction_context(*, claim_id="S1_C1", section_id="S1", claim_text="El modelo alcanza 91% de precisión.",
                              final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE", eligible_evidence=()):
    section_text = claim_text
    claim_fp = fingerprint_text(claim_text)
    section_fp = fingerprint_text(section_text)
    return {
        "claim_id": claim_id, "section_id": section_id, "original_claim_text": claim_text,
        "claim_fingerprint": claim_fp, "section_fingerprint": section_fp, "section_text": section_text,
        "claim_span_in_section": {
            "coordinate_base": "SECTION_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint": section_fp, "start": 0, "end": len(section_text), "text": section_text,
        },
        "final_correction_eligibility": final_correction_eligibility,
        "eligible_evidence": eligible_evidence,
        "policy": {},
    }


@scenario("CC01. DEFERRED legítimo (sin evidencia autorizada) conserva reason_codes y causa terminal -- no queda vacío como en el bug real")
def test_deferred_preserves_reason_codes_and_cause():
    context = _base_correction_context(eligible_evidence=())  # sin evidencia elegible -> DEFERRED
    proposal = propose_correction(context, llm=None)
    assert proposal.proposal_status == "DEFERRED"
    assert proposal.correction_decision == "DEFER_TO_MANUAL_REVIEW"
    # El bug real: reason_codes quedaba SIEMPRE vacío, sin importar la causa.
    assert proposal.reason_codes == ("AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE",)
    assert proposal.validation_issue_codes == ("AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE",)
    assert proposal.requires_manual_review is True
    assert proposal.final_proposal_status == "DEFERRED"


@scenario("CC02. PROPOSE_CHANGE aceptado SÍ continúa a precheck (flujo real completo, sin dobles del mecanismo)")
def test_accepted_propose_change_reaches_precheck():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = R._run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=R.CLAIM_TEXT_WRONG, evidence_text=R.CLAIM_TEXT_EVIDENCE,
        )
        row = executed.runtime_result.provisional_bundle["correction_traceability_rows"][0]
        assert row["correction_decision"] == "PROPOSE_CHANGE"
        assert row["proposal_status"] == "ACCEPTED_FOR_REVERIFICATION"
        assert row["precheck_stage_availability"] == "AVAILABLE"
        assert row["precheck_status"] == "PRECHECK_PASSED"
        assert row["accepted_for_reverification"] is True
        assert row["reason_codes"]  # no vacío
        assert row["decision_path"]  # no vacío
        assert row["retry_metrics"]


@scenario("CC03. DEFERRED nunca continúa a precheck: precheck_stage_availability=NOT_PRODUCED, con la causa real preservada")
def test_deferred_never_reaches_precheck():
    context = _base_correction_context(eligible_evidence=())
    proposal = propose_correction(context, llm=None)
    assert proposal.proposal_status == "DEFERRED"

    referential_result = {
        "referential_validation_status": "VALID",
        "joined_claim_records": (
            {
                "claim_verification_record": {
                    "claim_verification_result": {
                        "claim_id": "S1_C1", "claim_type": "SUBSTANTIVE_FACTUAL", "scientific_verdict": "PARTIALLY_SUPPORTED",
                        "deterministic_issue_codes": (), "semantic_issue_codes": ("PARTIAL_SUPPORT",),
                        "hallucination_risk": "MEDIUM", "llm_correction_recommendation": False,
                        "manual_review_required": False, "confidence": None,
                        "eligible_evidence": (), "evidence_used": (), "evidence_rejected": (),
                        "claim_uid": "",
                    },
                    "section_id": "S1",
                },
                "correction_ids": (proposal.correction_id,),
            },
        ),
        "joined_correction_records": (
            {"correction_id": proposal.correction_id, "claim_id": "S1_C1", "section_id": "S1", "proposal": proposal.to_dict()},
        ),
        "rejected_join_candidates": (),
        "referential_issue_codes": (),
        "referential_warnings": (),
        "orphan_records": (),
        "identity_conflicts": (),
        "aggregation_status": "VALID",
        "metrics_status": "NOT_COMPUTED",
        "result_contract_valid": True,
    }
    result = build_provisional_traceability_rows(referential_result)
    rows = result.to_dict()["correction_traceability_rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_status"] == "DEFERRED"
    assert row["precheck_stage_availability"] == "NOT_PRODUCED"
    assert row["reverification_stage_availability"] == "NOT_PRODUCED"
    # La causa terminal (requisito 5): reason_codes explica por qué
    # nunca hubo precheck -- ya no queda vacío como en el bug real.
    assert row["reason_codes"] == ("AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE",)
    assert row["correction_decision"] == "DEFER_TO_MANUAL_REVIEW"
    assert row["requires_manual_review"] is True
    assert row["accepted_for_reverification"] is False


@scenario("CC04. PARTIALLY_SUPPORTED auto-corregible con evidencia autorizada real NO se difiere indebidamente cuando manual_review_required=False")
def test_partially_supported_with_authorized_evidence_is_not_wrongly_deferred():
    # Reproduce exactamente el camino real (R02/CC02): un claim
    # PARTIALLY_SUPPORTED, con evidencia SUPPORT autorizada, y el LLM
    # de verificación NO pidiendo manual review -- debe llegar a
    # PROPOSE_CHANGE, nunca a DEFERRED por defecto.
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = R._run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=R.CLAIM_TEXT_WRONG, evidence_text=R.CLAIM_TEXT_EVIDENCE,
        )
        row = executed.runtime_result.provisional_bundle["correction_traceability_rows"][0]
        claim_row = executed.runtime_result.provisional_bundle["claim_traceability_rows"][0]
        assert claim_row["source_verdict"] == "PARTIALLY_SUPPORTED"
        assert row["proposal_status"] != "DEFERRED"
        assert row["correction_decision"] == "PROPOSE_CHANGE"


@scenario("CC05. Ausencia de evidencia sigue siendo fail-closed: sin evidencia elegible, DEFERRED explícito con causa -- nunca se inventa una corrección")
def test_missing_evidence_stays_fail_closed():
    context = _base_correction_context(eligible_evidence=())
    proposal = propose_correction(context, llm=None)
    assert proposal.proposal_status == "DEFERRED"
    assert proposal.action_type is None
    assert proposal.replacement_text == ""
    assert proposal.correction_applied is False
    assert "AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE" in proposal.reason_codes

    # También fail-closed cuando eligibility ya es NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE.
    context2 = _base_correction_context(final_correction_eligibility="NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE")
    proposal2 = propose_correction(context2, llm=None)
    assert proposal2.correction_decision == "NOT_CORRECTABLE"
    assert proposal2.proposal_status == "NOT_PROPOSED"
    assert proposal2.action_type is None


@scenario("CC06. Ground Truth isolation intacto: ni corrections.py ni el nuevo esquema de traceability referencian Ground Truth en ningún punto")
def test_ground_truth_isolation_unchanged():
    from src.tools.verification import corrections as corrections_module
    from src.tools.verification import traceability as traceability_module

    for module in (corrections_module, traceability_module):
        source = inspect.getsource(module)
        assert "ground_truth" not in source.lower(), module.__name__


if __name__ == "__main__":
    for fn in (
        test_deferred_preserves_reason_codes_and_cause,
        test_accepted_propose_change_reaches_precheck,
        test_deferred_never_reaches_precheck,
        test_partially_supported_with_authorized_evidence_is_not_wrongly_deferred,
        test_missing_evidence_stays_fail_closed,
        test_ground_truth_isolation_unchanged,
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
