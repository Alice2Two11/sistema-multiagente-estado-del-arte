"""Dos huecos finales sobre el parche 16:

A. Cero claims sin contrato declarado -> NUNCA inferir ni persistir
   LEGACY. El contrato queda indeterminado y AGENT07_NO_CLAIMS se
   maneja sin crear una frontera de identidad a partir de un lote
   vacío.

B. Migración explícita LEGACY -> STABLE_UID_V1: una frontera única y
   auditable, disparada SOLO por una señal explícita de 06
   (``migration_signal`` -- nunca inferida de la mera presencia de
   ``claim_uid``). Se persiste el registro completo (``from``, ``to``,
   ``round``, ``decision_id``, ``migration_mode``) en ``CycleState``.
   Después de migrar, cualquier ``claim_uid`` faltante falla cerrado
   igual que cualquier otra violación de ``STABLE_UID_V1``."""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.pipeline_orchestrator import ensure_pipeline_state  # noqa: E402
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402
from src.state.pipeline_state import CycleState  # noqa: E402
from src.tools.verification.writer_revision_cycle import (  # noqa: E402
    AUTO_CORRECTION_ELIGIBLE,
    NO_CORRECTION_NEEDED,
    _resolve_claim_identity_contract_version,
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


@scenario("W01. Sin contrato declarado + claims=[] -> NO se infiere ni se persiste LEGACY; contrato queda indeterminado (None, None)")
def test_empty_claims_no_declared_contract_never_infers():
    contract, reason, newly_inferred, migrated = _resolve_claim_identity_contract_version(
        [], declared_contract_version=None,
    )
    assert contract is None
    assert reason is None
    assert newly_inferred is False
    assert migrated is False

    # A través de classify_verification_transition: AGENT07_NO_CLAIMS,
    # sin ninguna frontera de identidad creada.
    decision = classify_verification_transition(
        claims=[], technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version=None,
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_NO_CLAIMS"
    assert decision["claim_identity_contract_version"] is None
    assert decision["claim_identity_contract_version_newly_inferred"] is False
    assert decision["claim_identity_contract_migrated"] is False


@scenario("W02. LEGACY declarado + aparecen claim_uid SIN señal de migración -> HALT, no migra")
def test_legacy_uids_without_signal_halts():
    claims = [
        _approved("S1_C1"),
        _correctable("S1_C2", claim_uid="uid-aparecido"),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version="LEGACY", migration_signal=False,
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"
    assert decision["claim_identity_contract_migrated"] is False


@scenario("W03. LEGACY declarado + claim_uid en TODOS + señal explícita válida -> migra a STABLE_UID_V1")
def test_legacy_with_signal_and_full_uids_migrates():
    UID_A, UID_B = "uid-migrado-a", "uid-migrado-b"
    claims = [
        _approved("S1_C1", claim_uid=UID_A),
        _correctable("S1_C2", claim_uid=UID_B),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=0, max_rounds=3,
        declared_contract_version="LEGACY", migration_signal=True,
    )
    assert decision["action"] == "RETURN"
    assert decision["claim_identity_contract_version"] == "STABLE_UID_V1"
    assert decision["claim_identity_contract_migrated"] is True
    assert decision["claim_identity_contract_version_newly_inferred"] is False  # no es inferencia, es migración
    assert decision["correctable_claim_uids"] == (UID_B,)


@scenario("W04. Después de migrar (contrato ya STABLE_UID_V1), un claim_uid faltante -> HALT, igual que cualquier violación")
def test_after_migration_missing_uid_fails_closed():
    claims = [
        _approved("S1_C1", claim_uid="uid-real"),
        _correctable("S1_C2"),  # perdió su claim_uid en una ronda posterior a la migración
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=1, max_rounds=3,
        declared_contract_version="STABLE_UID_V1",  # ya migrado en una ronda anterior
        migration_signal=False,
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"


@scenario("W05. STABLE_UID_V1 -> LEGACY prohibido, incluso con migration_signal presente (una migración no autoriza retroceder)")
def test_stable_to_legacy_forbidden_even_with_signal():
    claims = [
        _approved("S1_C1"),  # sin claim_uid -- como si "retrocediera"
        _correctable("S1_C2"),
    ]
    decision = classify_verification_transition(
        claims=claims, technical_status="COMPLETED", rounds_used=1, max_rounds=3,
        declared_contract_version="STABLE_UID_V1", migration_signal=True,
    )
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"
    assert decision["claim_identity_contract_version"] is None
    assert decision["claim_identity_contract_migrated"] is False


def _generic_fp(stage_key, attempt=1):
    return build_stage_fingerprints(
        input_data={"stage_name": stage_key, "attempt_number": attempt}, config_data={}, dependencies_data={},
    )


@scenario("W06. Integración: LEGACY histórico -> 06 revision con señal de migración -> migration boundary persistido en CycleState -> 07 queda STABLE_UID_V1")
def test_integration_legacy_to_stable_migration_boundary_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        import json

        (project_dir / "active_experiment.json").write_text(
            json.dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
        )
        store = ensure_pipeline_state(project_dir)

        # Estado previo: ciclo LEGACY histórico, ya con 1 ronda usada
        # (round_01 real, sin claim_uid -- comportamiento posicional
        # histórico, nunca reconstruido).
        state = store.load()
        from dataclasses import replace

        legacy_cycle = CycleState(
            rounds_used=1, max_rounds=3, status="ACTIVE",
            claim_identity_contract_version="LEGACY",
        )
        state = replace(state, cycles={**state.cycles, "writer_verifier": legacy_cycle})
        store.save(state)

        # 06 produce, en la ronda 2, claims YA con claim_uid -- con la
        # señal explícita de migración (no se infiere de los UIDs solos).
        UID_A, UID_B = "uid-boundary-a", "uid-boundary-b"
        claims = [
            _approved("S1_C1", claim_uid=UID_A),
            _correctable("S1_C2", claim_uid=UID_B),
        ]
        loaded = store.load()
        cycle = loaded.cycles["writer_verifier"]
        decision = classify_verification_transition(
            claims=claims, technical_status="COMPLETED",
            rounds_used=cycle.rounds_used, max_rounds=cycle.max_rounds,
            declared_contract_version=cycle.claim_identity_contract_version,
            migration_signal=True,
        )
        assert decision["action"] == "RETURN"
        assert decision["claim_identity_contract_migrated"] is True

        # Persistir el migration boundary tal como lo haría
        # execute_prepared_agent07 -- registro completo, ronda anterior
        # NUNCA reescrita (rounds_used se conserva hasta que el RETURN
        # real lo incremente por su cuenta, fuera de este test).
        fake_decision_id = "decision-uuid-real"
        migrated_cycle = replace(
            cycle,
            claim_identity_contract_version="STABLE_UID_V1",
            claim_identity_migration_from="LEGACY",
            claim_identity_migration_to="STABLE_UID_V1",
            claim_identity_migration_round=cycle.rounds_used + 1,
            claim_identity_migration_decision_id=fake_decision_id,
            claim_identity_migration_mode="EXPLICIT_SIGNAL_FROM_06",
        )
        final_state = store.load()
        final_state = replace(final_state, cycles={**final_state.cycles, "writer_verifier": migrated_cycle})
        store.save(final_state)

        persisted = store.load().cycles["writer_verifier"]
        assert persisted.claim_identity_contract_version == "STABLE_UID_V1"
        assert persisted.claim_identity_migration_from == "LEGACY"
        assert persisted.claim_identity_migration_to == "STABLE_UID_V1"
        assert persisted.claim_identity_migration_round == 2
        assert persisted.claim_identity_migration_decision_id == fake_decision_id
        assert persisted.claim_identity_migration_mode == "EXPLICIT_SIGNAL_FROM_06"
        assert persisted.rounds_used == 1  # la ronda anterior NUNCA se reescribió

        # Ronda 3 (posterior a la migración): 07 ya exige STABLE_UID_V1
        # estrictamente -- un claim sin uid ahora falla cerrado.
        claims_round3 = [_approved("S1_C1"), _correctable("S1_C2", claim_uid="uid-nuevo")]
        decision3 = classify_verification_transition(
            claims=claims_round3, technical_status="COMPLETED", rounds_used=2, max_rounds=3,
            declared_contract_version=persisted.claim_identity_contract_version,
        )
        assert decision3["action"] == "HALT_STAGE"
        assert decision3["reason_code"] == "AGENT07_CLAIM_UID_CONTRACT_VIOLATION"


if __name__ == "__main__":
    for fn in (
        test_empty_claims_no_declared_contract_never_infers,
        test_legacy_uids_without_signal_halts,
        test_legacy_with_signal_and_full_uids_migrates,
        test_after_migration_missing_uid_fails_closed,
        test_stable_to_legacy_forbidden_even_with_signal,
        test_integration_legacy_to_stable_migration_boundary_persisted,
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
