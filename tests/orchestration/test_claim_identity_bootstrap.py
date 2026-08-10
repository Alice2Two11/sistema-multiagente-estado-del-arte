"""``bootstrap_legacy_claim_identity_for_exhausted_cycle`` -- caso
operativo único: un experimento LEGACY cuyo ciclo ``writer_verifier``
ya agotó todas sus rondas científicas (``rounds_used >= max_rounds``)
y todavía no tiene contrato de identidad declarado como
``STABLE_UID_V1``. Los parches 14-18 solo migran DENTRO de una ronda
real de revisión -- este experimento ya no puede consumir otra.

Migración técnica administrativa, nunca una ronda científica:
``migration_mode="EXPLICIT_FRESH_UID_MINT"``, ``rounds_used``/
``max_rounds`` sin tocar, ningún archivo ni entrada de ``decision_log``
histórica modificada, contenido científico byte-idéntico -- solo se
añaden campos de identidad nuevos a cada claim en una publicación
nueva y separada."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.agent06_verification_handoff import resolve_committed_agent06_artifacts  # noqa: E402
from src.state.pipeline_state import CycleState  # noqa: E402
from src.tools.draft_writing.claim_identity_bootstrap import (  # noqa: E402
    bootstrap_legacy_claim_identity_for_exhausted_cycle,
)
from src.tools.verification.writer_revision_cycle import (  # noqa: E402
    AUTO_CORRECTION_ELIGIBLE,
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


def _declare_exhausted_legacy_cycle(store):
    state = store.load()
    cycle = CycleState(rounds_used=3, max_rounds=3, status="EXHAUSTED", claim_identity_contract_version="LEGACY")
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))


def _declare_exhausted_stable_cycle(store):
    state = store.load()
    cycle = CycleState(rounds_used=3, max_rounds=3, status="EXHAUSTED", claim_identity_contract_version="STABLE_UID_V1")
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))


def _original_claim_texts(project_dir, mapping_path, store):
    _, result, paths, _ = resolve_committed_agent06_artifacts(
        store=store, stage_name=T.DRAFT,
    )
    draft = json.loads(paths["state_of_art_draft.json"].read_text(encoding="utf-8"))
    texts = []
    for section in draft.get("sections", []):
        for claim in section.get("claims", []):
            texts.append(claim.get("claim") or claim.get("claim_text"))
    return texts


@scenario("Y01. Legacy agotado (3/3) sin claim_uid -> bootstrap migra a STABLE_UID_V1; rounds_used/max_rounds NO cambian")
def test_exhausted_legacy_bootstraps_preserving_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)

        result = bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        assert result.bootstrapped is True
        assert result.already_stable is False
        assert result.claims_migrated > 0

        cycle = store.load().cycles["writer_verifier"]
        assert cycle.rounds_used == 3
        assert cycle.max_rounds == 3
        assert cycle.claim_identity_contract_version == "STABLE_UID_V1"
        assert cycle.claim_identity_migration_from == "LEGACY"
        assert cycle.claim_identity_migration_to == "STABLE_UID_V1"
        assert cycle.claim_identity_migration_mode == "EXPLICIT_FRESH_UID_MINT"
        assert cycle.claim_identity_migration_round == 3
        assert cycle.claim_identity_migration_decision_id == result.decision_id


@scenario("Y02. Contenido científico antes/después idéntico -- ningún texto de claim se reescribe")
def test_scientific_content_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)
        before = _original_claim_texts(project_dir, mapping_path, store)

        bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)

        after = _original_claim_texts(project_dir, mapping_path, store)
        assert before == after
        assert len(after) > 0


@scenario("Y03. Todos los claims reciben claim_uid (UUID4 no vacío) y claim_version=1, parent_claim_uids=[]")
def test_all_claims_receive_identity():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)

        bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)

        _, result, paths, _ = resolve_committed_agent06_artifacts(store=store, stage_name=T.DRAFT)
        draft = json.loads(paths["state_of_art_draft.json"].read_text(encoding="utf-8"))
        seen_uids = set()
        checked = 0
        for section in draft.get("sections", []):
            for claim in section.get("claims", []):
                checked += 1
                assert claim.get("claim_uid")
                assert claim["claim_uid"] not in seen_uids  # cada uno genuinamente nuevo, sin colisión
                seen_uids.add(claim["claim_uid"])
                assert claim.get("claim_version") == 1
                assert claim.get("parent_claim_uids") == []
                assert claim.get("claim_text_fingerprint")
        assert checked > 0


@scenario("Y04. El historial (decision_log) no cambia -- la entrada original de 06 sigue intacta, se agrega una nueva sin reemplazarla")
def test_history_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)
        before_log = store.load().decision_log
        original_entry = before_log[0]

        bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)

        after_log = store.load().decision_log
        assert len(after_log) == len(before_log) + 1  # se agregó, nunca se reemplazó
        assert after_log[0] == original_entry  # la entrada original, byte a byte, intacta
        new_entry = after_log[-1]
        assert new_entry.decision["code"] == "AGENT06_CLAIM_IDENTITY_BOOTSTRAP"
        assert new_entry.requested_transition.action.value == "ADVANCE"  # nunca RETURN


@scenario("Y05. Segunda ejecución del bootstrap (ya migrado) -- no vuelve a mintar UIDs, no crea otro commit")
def test_second_run_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)

        first = bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        assert first.bootstrapped is True
        log_len_after_first = len(store.load().decision_log)

        second = bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        assert second.bootstrapped is False
        assert second.already_stable is True
        assert second.claims_migrated == 0
        assert len(store.load().decision_log) == log_len_after_first  # ningún commit nuevo

        cycle_after_second = store.load().cycles["writer_verifier"]
        assert cycle_after_second.claim_identity_migration_decision_id == first.decision_id  # sin cambios


@scenario("Y06. Ciclo ya STABLE_UID_V1 desde el principio -> bootstrap es no-op inmediato")
def test_already_stable_cycle_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_stable_cycle(store)
        log_len_before = len(store.load().decision_log)

        result = bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        assert result.bootstrapped is False
        assert result.already_stable is True
        assert len(store.load().decision_log) == log_len_before


@scenario("Y07. 07 posterior lee identidad estable pero sigue respetando max_rounds: con issues corregibles y 3/3 usadas, mantiene el HALT humano, nunca RETURN")
def test_post_bootstrap_still_respects_max_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)

        bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        cycle = store.load().cycles["writer_verifier"]

        claims = [
            {"claim_id": "S1_C1", "claim_uid": "uid-post-1", "final_correction_eligibility": AUTO_CORRECTION_ELIGIBLE,
             "evidence_used": ({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x"},)},
        ]
        decision = classify_verification_transition(
            claims=claims, technical_status="COMPLETED",
            rounds_used=cycle.rounds_used, max_rounds=cycle.max_rounds,
            declared_contract_version=cycle.claim_identity_contract_version,
        )
        assert decision["action"] == "HALT_STAGE"
        assert decision["reason_code"] == "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED"
        assert decision["claim_identity_contract_version"] == "STABLE_UID_V1"  # identidad SÍ se lee bien


if __name__ == "__main__":
    for fn in (
        test_exhausted_legacy_bootstraps_preserving_rounds,
        test_scientific_content_unchanged,
        test_all_claims_receive_identity,
        test_history_unchanged,
        test_second_run_is_noop,
        test_already_stable_cycle_is_noop,
        test_post_bootstrap_still_respects_max_rounds,
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
