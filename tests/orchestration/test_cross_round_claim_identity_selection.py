"""``classify_verification_transition``/``build_writer_revision_request``
resolvían la selección corregible por ``claim_id`` -- posicional, puede
referirse a un claim distinto entre rondas si la sección se regeneró
(ver ``src/tools/draft_writing/claim_identity.py``). Esto violaba el
requisito confirmado: "07 y todos los artefactos/decisiones cross-round
deben usar claim_uid como identidad primaria; claim_id queda únicamente
como etiqueta legible."

Corrección: ``_resolve_claim_identity_contract_version`` deriva el
contrato vigente EXCLUSIVAMENTE de los propios claims recibidos (todos
con ``claim_uid`` no vacío -> ``STABLE_UID_V1``; ninguno -> ``LEGACY``;
mezcla -> falla cerrado explícito, nunca elegido en silencio).
``_effective_correction_identity`` selecciona ``claim_uid`` bajo
``STABLE_UID_V1`` o ``claim_id`` bajo ``LEGACY`` -- ``correctable_
claim_uids``/``blocking_claim_uids`` (renombrados desde ``..._claim_
ids``) y ``build_writer_revision_request`` indexan y validan por esa
misma identidad efectiva, nunca por ``claim_id`` directamente.

Los 5 escenarios pedidos explícitamente."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.verification.writer_revision_cycle import (  # noqa: E402
    AUTO_CORRECTION_ELIGIBLE,
    NO_CORRECTION_NEEDED,
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


def _claim(claim_id, eligibility, *, claim_uid=None, **kwargs):
    base = {
        "claim_id": claim_id,
        "final_correction_eligibility": eligibility,
        "section_id": "s1",
        "claim_text": f"Texto del claim {claim_id}.",
        "scientific_verdict": "SUPPORTED",
        "hallucination_risk": "LOW",
        "evidence_used": (),
    }
    if claim_uid is not None:
        base["claim_uid"] = claim_uid
    base.update(kwargs)
    return base


def _correctable(claim_id, *, claim_uid=None, **kwargs):
    return _claim(claim_id, AUTO_CORRECTION_ELIGIBLE, claim_uid=claim_uid, evidence_used=_EVIDENCE, **kwargs)


def _approved(claim_id, *, claim_uid=None, **kwargs):
    return _claim(claim_id, NO_CORRECTION_NEEDED, claim_uid=claim_uid, **kwargs)


@scenario("U01. Dos rondas donde el claim cambia de texto/posición (claim_id distinto) pero el claim_uid es el mismo -> RETURN lo selecciona por claim_uid, no por claim_id")
def test_return_selects_by_uid_across_position_and_text_change():
    UID = "11111111-1111-1111-1111-111111111111"

    # Ronda 1: el claim vive en la posición "S5_C2", con un texto.
    claims_round1 = [
        _approved("S5_C1", claim_uid="uid-c1"),
        _correctable("S5_C2", claim_uid=UID, claim_text="Texto original con soporte parcial."),
    ]
    decision1 = classify_verification_transition(claims=claims_round1, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision1["action"] == "RETURN"
    assert decision1["claim_identity_contract_version"] == "STABLE_UID_V1"
    assert decision1["correctable_claim_uids"] == (UID,)  # identidad real, no "S5_C2"

    # Ronda 2: la sección se regeneró -- el MISMO claim_uid ahora vive en
    # otra posición ("S5_C4") con OTRO texto (una reescritura real).
    claims_round2 = [
        _approved("S5_C1", claim_uid="uid-nuevo-1"),
        _approved("S5_C3", claim_uid="uid-nuevo-2"),
        _correctable("S5_C4", claim_uid=UID, claim_text="Texto reescrito, todavía con un problema distinto."),
    ]
    decision2 = classify_verification_transition(claims=claims_round2, technical_status="COMPLETED", rounds_used=1, max_rounds=3)
    assert decision2["action"] == "RETURN"
    # La selección sigue apuntando al MISMO claim_uid, pese al cambio
    # completo de claim_id/posición/texto entre rondas.
    assert decision2["correctable_claim_uids"] == (UID,)


@scenario("U02. Dos claims intercambian claim_id posicional sin intercambiar identidad -- RETURN selecciona el UID correcto, no el que 'heredó' la etiqueta")
def test_swapped_claim_ids_do_not_swap_identity():
    UID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    UID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    # Ronda 1: A en la posición C1 (aprobado), B en la posición C2 (corregible).
    claims_round1 = [
        _approved("S1_C1", claim_uid=UID_A),
        _correctable("S1_C2", claim_uid=UID_B),
    ]
    decision1 = classify_verification_transition(claims=claims_round1, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision1["correctable_claim_uids"] == (UID_B,)

    # Ronda 2: las posiciones se INTERCAMBIARON (A ahora es C2, B ahora es
    # C1) -- pero la elegibilidad real sigue perteneciendo a B (UID_B),
    # que ahora ocupa "S1_C1"; A (UID_A) ocupa "S1_C2" y sigue aprobado.
    claims_round2 = [
        _correctable("S1_C1", claim_uid=UID_B),  # B, ahora en la posición C1
        _approved("S1_C2", claim_uid=UID_A),      # A, ahora en la posición C2
    ]
    decision2 = classify_verification_transition(claims=claims_round2, technical_status="COMPLETED", rounds_used=1, max_rounds=3)
    # La selección sigue siendo UID_B -- nunca "S1_C1" (que en la ronda 1
    # habría señalado a A, el claim equivocado, si se hubiera comparado
    # por claim_id entre rondas).
    assert decision2["correctable_claim_uids"] == (UID_B,)
    assert UID_A not in decision2["correctable_claim_uids"]


@scenario("U03. Claim bajo contrato STABLE_UID_V1 (al menos uno de la lista trae claim_uid) sin UID en otro claim -> falla cerrado, nunca se elige un contrato en silencio")
def test_missing_uid_under_stable_contract_fails_closed():
    claims = [
        _approved("S1_C1", claim_uid="uid-real"),
        _correctable("S1_C2"),  # sin claim_uid -- mezcla con el anterior
    ]
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_MIXED_CLAIM_IDENTITY_CONTRACT"
    assert decision["correctable_claim_uids"] == ()
    assert decision["claim_identity_contract_version"] is None  # nunca resuelto en un lote ambiguo

    # Caso puro STABLE_UID_V1 (todos con claim_uid) pero UNO llega vacío
    # explícitamente (string vacío, no ausente) -- también debe fallar,
    # vía _validate_claims_shape, con un reason_code distinto y específico.
    claims_empty_uid = [
        _approved("S1_C1", claim_uid="uid-real"),
        _correctable("S1_C2", claim_uid=""),
    ]
    decision2 = classify_verification_transition(claims=claims_empty_uid, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision2["action"] == "HALT_STAGE"
    assert decision2["reason_code"] == "AGENT07_MIXED_CLAIM_IDENTITY_CONTRACT"


@scenario("U04. Legacy (ningún claim trae claim_uid) -> ruta posicional explícita, contrato LEGACY reconocido, nunca tratado como error")
def test_legacy_no_uid_uses_explicit_positional_route():
    claims = [
        _approved("S1_C1"),
        _correctable("S1_C2"),
    ]
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "RETURN"
    assert decision["claim_identity_contract_version"] == "LEGACY"
    assert decision["correctable_claim_uids"] == ("S1_C2",)  # claim_id como identidad efectiva, explícito


@scenario("U05. writer_revision_request selecciona el claim correcto por claim_uid aunque claim_id haya cambiado, y conserva ambos campos en el issue")
def test_writer_revision_request_selects_by_uid_despite_claim_id_change():
    UID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    claims = [
        _approved("S3_C1", claim_uid="uid-otro"),
        _correctable("S3_C5", claim_uid=UID, claim_text="Reescrito y reubicado tras regenerar la sección."),
    ]
    decision = classify_verification_transition(claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3)
    assert decision["action"] == "RETURN"

    request = build_writer_revision_request(
        experiment_id="exp1", cycle_id="cyc1", round_number=2,
        source_draft_path="draft.json", source_draft_fingerprint="fp1", verification_fingerprint="fp2",
        claims=claims, correctable_claim_uids=decision["correctable_claim_uids"],
        claim_identity_contract_version=decision["claim_identity_contract_version"],
        transition_reason=decision["reason_code"],
    )
    assert len(request["issues"]) == 1
    issue = request["issues"][0]
    # Identidad primaria correcta (el claim_uid, no confundido con
    # ningún claim_id de otra posición) + etiqueta legible conservada.
    assert issue["claim_uid"] == UID
    assert issue["claim_id"] == "S3_C5"
    assert request["claim_identity_contract_version"] == "STABLE_UID_V1"


if __name__ == "__main__":
    for fn in (
        test_return_selects_by_uid_across_position_and_text_change,
        test_swapped_claim_ids_do_not_swap_identity,
        test_missing_uid_under_stable_contract_fails_closed,
        test_legacy_no_uid_uses_explicit_positional_route,
        test_writer_revision_request_selects_by_uid_despite_claim_id_change,
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
