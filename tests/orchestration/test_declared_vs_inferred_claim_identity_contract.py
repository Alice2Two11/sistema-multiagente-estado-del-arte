"""Resolución HÍBRIDA del contrato de identidad (``_resolve_claim_
identity_contract_version``, ``classify_verification_transition``).

La resolución puramente "inferida de los claims" (parche anterior)
permitía un DOWNGRADE (o UPGRADE) silencioso: un experimento ya
``STABLE_UID_V1`` que por un bug de serialización/propagación pierde
todos sus ``claim_uid`` se reinterpretaría como ``LEGACY``, volviendo a
seleccionar por ``claim_id`` posicional sin que nadie lo notara.

Corrección: si el ciclo ya tiene un contrato DECLARADO
(``CycleState.claim_identity_contract_version``), ese es normativo --
los claims se validan CONTRA él, nunca se re-infiere. Solo cuando NO
hay contrato declarado todavía (``declared_contract_version=None`` --
la primera frontera real) se infiere una vez de los propios claims, y
esa inferencia se marca (``claim_identity_contract_version_newly_
inferred=True``) para que el llamador la persista."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.verification.writer_revision_cycle import (  # noqa: E402
    AUTO_CORRECTION_ELIGIBLE,
    NO_CORRECTION_NEEDED,
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


@scenario("V01. Ciclo declarado STABLE_UID_V1 + TODOS los claim_uid perdidos -> HALT (AGENT07_CLAIM_UID_CONTRACT_VIOLATION), NUNCA reinterpretado como LEGACY")
def test_declared_stable_all_uids_lost_halts_never_downgrades():
    # Ningún claim trae claim_uid -- si se infiriera de los datos,
    # esto pasaría silenciosamente como LEGACY. Con un contrato
    # declarado STABLE_UID_V1, debe fallar cerrado en su lugar.
    claims = [
        _approved("S1_C1"),
        _correctable("S1_C2"),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version="STABLE_UID_V1",
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"
    assert decision["claim_identity_contract_version"] is None
    assert decision["claim_identity_contract_version_newly_inferred"] is False
    # Nunca se comportó como LEGACY -- correctable_claim_uids vacío, no ("S1_C2",).
    assert decision["correctable_claim_uids"] == ()


@scenario("V02. Ciclo declarado STABLE_UID_V1 + UN SOLO claim_uid perdido (el resto sí lo tiene) -> HALT igual, no solo cuando faltan todos")
def test_declared_stable_one_uid_missing_halts():
    claims = [
        _approved("S1_C1", claim_uid="uid-real-1"),
        _correctable("S1_C2"),  # este perdió su claim_uid, los demás no
        _approved("S1_C3", claim_uid="uid-real-3"),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version="STABLE_UID_V1",
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"


@scenario("V03. Ciclo declarado LEGACY + aparición inesperada de claim_uid -> NO se migra silenciosamente, falla cerrado exigiendo una frontera explícita")
def test_declared_legacy_unexpected_uid_does_not_silently_upgrade():
    claims = [
        _approved("S1_C1"),
        _correctable("S1_C2", claim_uid="uid-inesperado"),  # apareció sin que nadie declarara la migración
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version="LEGACY",
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"
    # Nunca se comportó como si ya fuera STABLE_UID_V1.
    assert decision["claim_identity_contract_version"] is None


@scenario("V04. Primera ronda SIN contrato declarado + TODOS con claim_uid -> infiere STABLE_UID_V1 y lo marca para persistir")
def test_first_round_no_declared_contract_all_uids_infers_stable():
    UID_A, UID_B = "uid-aaaa", "uid-bbbb"
    claims = [
        _approved("S1_C1", claim_uid=UID_A),
        _correctable("S1_C2", claim_uid=UID_B),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version=None,  # ciclo aún no existe -- frontera real
    )
    assert decision["action"] == "RETURN"
    assert decision["claim_identity_contract_version"] == "STABLE_UID_V1"
    assert decision["claim_identity_contract_version_newly_inferred"] is True
    assert decision["correctable_claim_uids"] == (UID_B,)


@scenario("V05. Primera ronda SIN contrato declarado + NINGUNO con claim_uid -> infiere LEGACY y lo marca para persistir")
def test_first_round_no_declared_contract_no_uids_infers_legacy():
    claims = [
        _approved("S1_C1"),
        _correctable("S1_C2"),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version=None,
    )
    assert decision["action"] == "RETURN"
    assert decision["claim_identity_contract_version"] == "LEGACY"
    assert decision["claim_identity_contract_version_newly_inferred"] is True
    assert decision["correctable_claim_uids"] == ("S1_C2",)


if __name__ == "__main__":
    for fn in (
        test_declared_stable_all_uids_lost_halts_never_downgrades,
        test_declared_stable_one_uid_missing_halts,
        test_declared_legacy_unexpected_uid_does_not_silently_upgrade,
        test_first_round_no_declared_contract_all_uids_infers_stable,
        test_first_round_no_declared_contract_no_uids_infers_legacy,
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
