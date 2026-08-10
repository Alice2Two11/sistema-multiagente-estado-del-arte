"""Pruebas del núcleo determinista (endurecido) del ciclo correctivo
07<->06: clasificación de transición (B1) y artefacto de feedback (B2).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.verification.writer_revision_cycle import (
    AUTO_CORRECTION_ELIGIBLE,
    MANUAL_REVIEW_REQUIRED,
    NO_CORRECTION_NEEDED,
    NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE,
    POTENTIALLY_AUTO_CORRECTABLE,
    build_writer_revision_request,
    classify_verification_transition,
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


_EVIDENCE = ({"source_filename": "p.pdf", "chunk_id": "c1", "text": "Evidencia real usada."},)


def _claim(claim_id, eligibility, **kwargs):
    base = {
        "claim_id": claim_id,
        "final_correction_eligibility": eligibility,
        "section_id": "s1",
        "claim_text": f"Texto del claim {claim_id}.",
        "scientific_verdict": "SUPPORTED",
        "hallucination_risk": "LOW",
        "evidence_used": (),
    }
    base.update(kwargs)
    return base


def _correctable_claim_with_evidence(claim_id, **kwargs):
    return _claim(claim_id, AUTO_CORRECTION_ELIGIBLE, evidence_used=_EVIDENCE, **kwargs)


# ---------------------------------------------------------------------------
# B1: política de transición
# ---------------------------------------------------------------------------


@scenario("R01. Todos NO_CORRECTION_NEEDED -> ADVANCE")
def test_all_approved_advances():
    claims = [_claim("c1", NO_CORRECTION_NEEDED), _claim("c2", NO_CORRECTION_NEEDED)]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "ADVANCE"
    assert result["reason_code"] == "AGENT07_ALL_CLAIMS_APPROVED"


@scenario("R02. Un claim AUTO_CORRECTION_ELIGIBLE CON evidencia -> RETURN")
def test_correctable_with_evidence_returns():
    claims = [_claim("c1", NO_CORRECTION_NEEDED), _correctable_claim_with_evidence("c2")]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "RETURN"
    assert result["correctable_claim_uids"] == ("c2",)


@scenario("R03. Un claim POTENTIALLY_AUTO_CORRECTABLE CON evidencia -> RETURN")
def test_potentially_correctable_with_evidence_returns():
    claims = [_claim("c1", POTENTIALLY_AUTO_CORRECTABLE, evidence_used=_EVIDENCE)]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "RETURN"


@scenario("R04. Un claim MANUAL_REVIEW_REQUIRED -> HALT_STAGE, incluso con otros corregibles con evidencia")
def test_manual_review_blocks_even_with_correctable():
    claims = [_correctable_claim_with_evidence("c1"), _claim("c2", MANUAL_REVIEW_REQUIRED)]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_NON_CORRECTABLE_ISSUE"
    assert result["blocking_claim_uids"] == ("c2",)


@scenario("R05. Un claim NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE -> HALT_STAGE")
def test_not_correctable_blocks():
    claims = [_claim("c1", NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE)]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_NON_CORRECTABLE_ISSUE"


@scenario("R06. Fallo técnico -> HALT_STAGE, tiene prioridad sobre todo lo demás")
def test_technical_failure_blocks():
    claims = [_claim("c1", NO_CORRECTION_NEEDED)]
    result = classify_verification_transition(
        claims=claims, technical_status="BLOCKED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_TECHNICAL_FAILURE"


@scenario("R07. Rondas agotadas -> HALT_STAGE aunque haya claims corregibles con evidencia")
def test_max_rounds_exhausted_blocks_even_if_correctable():
    claims = [_correctable_claim_with_evidence("c1")]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=3, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED"


@scenario("R08. Artefactos incompletos (sin claims) -> HALT_STAGE")
def test_no_claims_blocks():
    result = classify_verification_transition(
        claims=[], technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_NO_CLAIMS"


@scenario("R09. max_rounds sin default oculto: 0 rondas usadas, max_rounds=0 -> agotado de inmediato")
def test_max_rounds_zero_no_hidden_default():
    claims = [_correctable_claim_with_evidence("c1")]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=0
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED"


@scenario("H01. Claim corregible SIN evidencia ni propuesta -> HALT_STAGE, fail-closed (punto 1)")
def test_correctable_without_evidence_blocks():
    claims = [_claim("c1", AUTO_CORRECTION_ELIGIBLE, evidence_used=())]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT"
    assert result["blocking_claim_uids"] == ("c1",)


@scenario("H02. Claim corregible sin evidence_used pero con correction_proposal.supporting_evidence -> RETURN")
def test_correctable_with_proposal_support_returns():
    claims = [
        _claim(
            "c1", POTENTIALLY_AUTO_CORRECTABLE, evidence_used=(),
            correction_proposal={"supporting_evidence": ({"source_filename": "p.pdf", "chunk_id": "c9"},),
                                  "requested_change": "Cambiar X por Y según chunk c9."},
        )
    ]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "RETURN"


@scenario("H03. Elegibilidad desconocida -> HALT_STAGE, nunca ADVANCE por omisión (punto 2)")
def test_unknown_eligibility_blocks():
    claims = [_claim("c1", "SOME_NEW_VALUE_NOT_IN_THE_5")]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_UNKNOWN_ELIGIBILITY"


@scenario("H04. Elegibilidad ausente (None) -> HALT_STAGE")
def test_missing_eligibility_blocks():
    claims = [_claim("c1", None)]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_UNKNOWN_ELIGIBILITY"


@scenario("H05. claim_id ausente -> HALT_STAGE, AGENT07_MALFORMED_CLAIM")
def test_missing_claim_id_blocks():
    claim = _claim("c1", NO_CORRECTION_NEEDED)
    claim["claim_id"] = ""
    result = classify_verification_transition(
        claims=[claim], technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_MALFORMED_CLAIM"


@scenario("H06. Elegibilidad desconocida mezclada con claims aprobados -> igual bloquea el lote completo")
def test_unknown_eligibility_blocks_whole_batch():
    claims = [_claim("c1", NO_CORRECTION_NEEDED), _claim("c2", "WEIRD_VALUE")]
    result = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3
    )
    assert result["action"] == "HALT_STAGE"
    assert result["reason_code"] == "AGENT07_UNKNOWN_ELIGIBILITY"


# ---------------------------------------------------------------------------
# B2: artefacto de retroalimentación
# ---------------------------------------------------------------------------


@scenario("R10. writer_revision_request: solo incluye claims corregibles, no los aprobados")
def test_revision_request_only_correctable():
    claims = [_claim("c1", NO_CORRECTION_NEEDED), _correctable_claim_with_evidence("c2")]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp_draft", verification_fingerprint="fp_verif",
        claims=claims, correctable_claim_uids=("c2",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
    )
    assert len(request["issues"]) == 1
    assert request["issues"][0]["claim_id"] == "c2"


@scenario("R11. writer_revision_request: evidencia y cita se derivan de evidence_used real, no inventadas")
def test_revision_request_derives_evidence():
    claims = [
        _claim(
            "c1", AUTO_CORRECTION_ELIGIBLE,
            evidence_used=({"source_filename": "paper1.pdf", "chunk_id": "chunk3", "text": "El valor real fue 42."},),
        )
    ]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
    )
    issue = request["issues"][0]
    assert issue["source_filename"] == "paper1.pdf"
    assert issue["chunk_id"] == "chunk3"
    assert issue["evidence_text"] == "El valor real fue 42."
    assert issue["citation"] == "[paper1.pdf | chunk3]"


@scenario("R12. writer_revision_request: sin evidencia ni propuesta -> ValueError fail-closed (defensa en profundidad)")
def test_revision_request_no_evidence_raises():
    claims = [_claim("c1", POTENTIALLY_AUTO_CORRECTABLE, evidence_used=())]
    try:
        build_writer_revision_request(
            experiment_id="exp1", cycle_id="cyc1", round_number=1,
            source_draft_path="draft.json",
            source_draft_fingerprint="fp1", verification_fingerprint="fp2",
            claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
        )
    except ValueError as exc:
        assert "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError fail-closed")


@scenario("R13. writer_revision_request: sin duplicados por claim_id repetido")
def test_revision_request_no_duplicates():
    claims = [_correctable_claim_with_evidence("c1")]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
    )
    assert len(request["issues"]) == 1


@scenario("R14. writer_revision_request: campos mínimos del esquema presentes, incluido source_draft_path")
def test_revision_request_schema_fields():
    claims = [_correctable_claim_with_evidence("c1")]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=2,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
    )
    assert request["source_draft_path"] == "draft.json"
    for key in ("schema_version", "experiment_id", "cycle_id", "round_number",
                "source_draft_path", "source_draft_fingerprint", "verification_fingerprint",
                "created_at", "transition_reason", "summary", "issues"):
        assert key in request
    for key in ("issue_id", "claim_id", "section_id", "claim_text", "problem_type",
                "severity", "evidence_text", "citation", "requested_change",
                "requested_change_is_fallback", "constraints", "correctable"):
        assert key in request["issues"][0]


@scenario("R15. Múltiples claims en varias secciones se reflejan por section_id")
def test_multiple_sections():
    claims = [
        _correctable_claim_with_evidence("c1", section_id="intro"),
        _correctable_claim_with_evidence("c2", section_id="results"),
    ]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1", "c2"), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
    )
    sections = {issue["section_id"] for issue in request["issues"]}
    assert sections == {"intro", "results"}


@scenario("H07. source_draft_path vacío -> ValueError (punto: obligatorio)")
def test_missing_source_draft_path_raises():
    claims = [_correctable_claim_with_evidence("c1")]
    try:
        build_writer_revision_request(
            experiment_id="exp1", cycle_id="cyc1", round_number=1,
            source_draft_path="",
            source_draft_fingerprint="fp1", verification_fingerprint="fp2",
            claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="AGENT07_CORRECTABLE_ISSUES",
        )
    except ValueError as exc:
        assert "AGENT07_REVISION_REQUEST_MALFORMED" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError")


@scenario("H08. correctable_claim_ids inconsistente con claims recibidos -> ValueError")
def test_inconsistent_correctable_ids_raises():
    claims = [_correctable_claim_with_evidence("c1")]
    try:
        build_writer_revision_request(
            experiment_id="exp1", cycle_id="cyc1", round_number=1,
            source_draft_path="draft.json",
            source_draft_fingerprint="fp1", verification_fingerprint="fp2",
            claims=claims, correctable_claim_uids=("c1", "c_nonexistent"), claim_identity_contract_version="LEGACY",
            transition_reason="AGENT07_CORRECTABLE_ISSUES",
        )
    except ValueError as exc:
        assert "AGENT07_REVISION_REQUEST_MALFORMED" in str(exc)
        assert "c_nonexistent" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por inconsistencia")


@scenario("H09. issue_id estable: no depende del orden de iteración de correctable_claim_ids")
def test_issue_id_stable_regardless_of_order():
    claims = [_correctable_claim_with_evidence("c1"), _correctable_claim_with_evidence("c2")]
    request_a = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1", "c2"), claim_identity_contract_version="LEGACY", transition_reason="X",
    )
    request_b = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=list(reversed(claims)), correctable_claim_uids=("c2", "c1"), claim_identity_contract_version="LEGACY", transition_reason="X",
    )
    ids_a = {i["claim_id"]: i["issue_id"] for i in request_a["issues"]}
    ids_b = {i["claim_id"]: i["issue_id"] for i in request_b["issues"]}
    assert ids_a == ids_b  # mismos issue_id por claim_id, sin importar el orden


@scenario("H10. severity conserva el valor real de hallucination_risk (no colapsa a medium)")
def test_severity_preserves_real_value():
    claims = [
        _claim("c1", AUTO_CORRECTION_ELIGIBLE, evidence_used=_EVIDENCE, hallucination_risk="LOW"),
        _claim("c2", AUTO_CORRECTION_ELIGIBLE, evidence_used=_EVIDENCE, hallucination_risk="MEDIUM"),
        _claim("c3", AUTO_CORRECTION_ELIGIBLE, evidence_used=_EVIDENCE, hallucination_risk="HIGH"),
    ]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1", "c2", "c3"), claim_identity_contract_version="LEGACY", transition_reason="X",
    )
    severities = {i["claim_id"]: i["severity"] for i in request["issues"]}
    assert severities == {"c1": "low", "c2": "medium", "c3": "high"}


@scenario("H11. requested_change usa la propuesta correctiva real de 07 cuando existe")
def test_requested_change_uses_real_proposal():
    claims = [
        _claim(
            "c1", AUTO_CORRECTION_ELIGIBLE, evidence_used=_EVIDENCE,
            correction_proposal={"requested_change": "Reemplazar '50%' por '45%' según chunk c1."},
        )
    ]
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="X",
    )
    issue = request["issues"][0]
    assert issue["requested_change"] == "Reemplazar '50%' por '45%' según chunk c1."
    assert issue["requested_change_is_fallback"] is False


@scenario("H12. requested_change cae al fallback EXPLÍCITO marcado cuando no hay propuesta real")
def test_requested_change_fallback_marked_explicitly():
    claims = [_correctable_claim_with_evidence("c1")]  # sin correction_proposal
    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=1,
        source_draft_path="draft.json",
        source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=("c1",), claim_identity_contract_version="LEGACY", transition_reason="X",
    )
    issue = request["issues"][0]
    assert issue["requested_change_is_fallback"] is True
    assert "consistente exclusivamente con la evidencia citada" in issue["requested_change"]


@scenario("H13. issues vacío tras filtrar -> ValueError, nunca un artefacto vacío")
def test_empty_issues_raises():
    claims = [_claim("c1", NO_CORRECTION_NEEDED)]
    try:
        build_writer_revision_request(
            experiment_id="exp1", cycle_id="cyc1", round_number=1,
            source_draft_path="draft.json",
            source_draft_fingerprint="fp1", verification_fingerprint="fp2",
            claims=claims, correctable_claim_uids=(), claim_identity_contract_version="LEGACY", transition_reason="X",
        )
    except ValueError as exc:
        assert "AGENT07_REVISION_REQUEST_MALFORMED" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError por issues vacío")


if __name__ == "__main__":
    for fn in (
        test_all_approved_advances,
        test_correctable_with_evidence_returns,
        test_potentially_correctable_with_evidence_returns,
        test_manual_review_blocks_even_with_correctable,
        test_not_correctable_blocks,
        test_technical_failure_blocks,
        test_max_rounds_exhausted_blocks_even_if_correctable,
        test_no_claims_blocks,
        test_max_rounds_zero_no_hidden_default,
        test_correctable_without_evidence_blocks,
        test_correctable_with_proposal_support_returns,
        test_unknown_eligibility_blocks,
        test_missing_eligibility_blocks,
        test_missing_claim_id_blocks,
        test_unknown_eligibility_blocks_whole_batch,
        test_revision_request_only_correctable,
        test_revision_request_derives_evidence,
        test_revision_request_no_evidence_raises,
        test_revision_request_no_duplicates,
        test_revision_request_schema_fields,
        test_multiple_sections,
        test_missing_source_draft_path_raises,
        test_inconsistent_correctable_ids_raises,
        test_issue_id_stable_regardless_of_order,
        test_severity_preserves_real_value,
        test_requested_change_uses_real_proposal,
        test_requested_change_fallback_marked_explicitly,
        test_empty_issues_raises,
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
