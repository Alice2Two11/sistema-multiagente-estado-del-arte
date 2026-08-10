"""Identidad estable de claims entre rondas -- ``src/tools/draft_writing/
claim_identity.py``.

``claim_id`` (``f"{section_id}_C{idx}"``) es posicional y se recalcula
cada vez que 06 regenera una sección completa en modo REVISION (docstring
real de ``DraftWritingAgent._execute_revision``: "Regenera SOLO las
secciones CON issues" -- toda la sección, no solo el claim señalado).
Esto rompe cualquier rastreo entre rondas basado en ``claim_id``.

Estos 10 escenarios prueban ``resolve_claim_identity``/
``check_no_claim_uid_collisions`` de forma aislada, con datos de sección
construidos a mano -- sin depender de ningún texto real de ningún
experimento (generalización exigida explícitamente)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.draft_writing.claim_identity import (  # noqa: E402
    ClaimIdentityDeclaration,
    ClaimIdentityRecord,
    check_no_claim_uid_collisions,
    resolve_claim_identity,
)
from src.tools.verification.corrections import fingerprint_text  # noqa: E402

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


def _counter_mint(prefix="uid"):
    state = {"n": 0}

    def mint():
        state["n"] += 1
        return f"{prefix}-{state['n']}"

    return mint


def _existing_record(uid, *, version=1, claim_id="S1_C1", created_round=1, updated_round=1, text="texto original"):
    return ClaimIdentityRecord(
        claim_uid=uid, claim_version=version, claim_id=claim_id, parent_claim_uids=(),
        claim_text_fingerprint=fingerprint_text(text), created_round=created_round, updated_round=updated_round,
    )


@scenario("O01. Cambio de posición: mismo claim_uid continuado, mismo texto -> claim_version NO se incrementa por posición (solo el texto importa para el fingerprint, pero la versión SÍ avanza porque hubo una nueva ronda de CONTINUE)")
def test_position_change_preserves_uid():
    parent = _existing_record("uid-A", claim_id="S1_C2")
    declaration = ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-A",))
    record = resolve_claim_identity(
        declaration=declaration, claim_text="texto original", claim_id="S1_C4",  # posición nueva -> claim_id distinto
        previous_claims_by_uid={"uid-A": parent}, forced_parent_uid=None, round_number=2,
        text_fingerprint=fingerprint_text,
    )
    assert record.claim_uid == "uid-A"  # el uid sobrevive el cambio de posición
    assert record.claim_id == "S1_C4"  # la etiqueta sí cambia, es solo posicional
    assert record.claim_version == 2


@scenario("O02. Inserción de un claim anterior: un claim nuevo se inserta antes; el claim existente conserva su claim_uid aunque su claim_id posicional cambie")
def test_insertion_before_preserves_existing_uid():
    parent = _existing_record("uid-B", claim_id="S1_C1")
    new_declaration = ClaimIdentityDeclaration(action="NEW")
    inserted = resolve_claim_identity(
        declaration=new_declaration, claim_text="afirmación nueva insertada", claim_id="S1_C1",
        previous_claims_by_uid={"uid-B": parent}, forced_parent_uid=None, round_number=2,
        text_fingerprint=fingerprint_text, mint_uid=_counter_mint("new"),
    )
    continued_declaration = ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-B",))
    existing = resolve_claim_identity(
        declaration=continued_declaration, claim_text="texto original", claim_id="S1_C2",  # ahora en la posicion 2, no 1
        previous_claims_by_uid={"uid-B": parent}, forced_parent_uid=None, round_number=2,
        text_fingerprint=fingerprint_text,
    )
    assert inserted.claim_uid != "uid-B"
    assert existing.claim_uid == "uid-B"
    assert existing.claim_id == "S1_C2"


@scenario("O03. Reescritura menor: CONTINUE con texto ligeramente distinto -> mismo claim_uid, claim_version+1, fingerprint distinto, parent_claim_uids=(uid_previo,)")
def test_minor_rewrite_continues_uid_new_fingerprint():
    parent = _existing_record("uid-C", version=1, text="El modelo mejora el rendimiento.")
    declaration = ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-C",))
    record = resolve_claim_identity(
        declaration=declaration, claim_text="El modelo mejora el rendimiento de forma consistente.", claim_id="S1_C1",
        previous_claims_by_uid={"uid-C": parent}, forced_parent_uid=None, round_number=2,
        text_fingerprint=fingerprint_text,
    )
    assert record.claim_uid == "uid-C"
    assert record.claim_version == 2
    assert record.parent_claim_uids == ("uid-C",)
    assert record.claim_text_fingerprint != parent.claim_text_fingerprint


@scenario("O04. Eliminación: un claim_uid previo que ningún claim nuevo continúa simplemente no aparece en el conjunto resuelto de la ronda -- no se reasigna a otro claim, no queda huérfano activo")
def test_deletion_leaves_no_active_trace():
    parent_kept = _existing_record("uid-D1", claim_id="S1_C1")
    parent_removed = _existing_record("uid-D2", claim_id="S1_C2")
    previous = {"uid-D1": parent_kept, "uid-D2": parent_removed}
    # Solo se resuelve un claim en la ronda nueva -- el otro (uid-D2) se eliminó.
    kept = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-D1",)),
        claim_text="texto original", claim_id="S1_C1", previous_claims_by_uid=previous,
        forced_parent_uid=None, round_number=2, text_fingerprint=fingerprint_text,
    )
    active_uids_this_round = {kept.claim_uid}
    assert "uid-D2" not in active_uids_this_round
    assert "uid-D2" not in {kept.claim_uid}  # no reaparece bajo ningún otro claim


@scenario("O05. Claim nuevo: NEW -> 0 padres, claim_uid nuevo, claim_version=1, created_round=updated_round=ronda actual")
def test_new_claim():
    record = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="NEW"), claim_text="afirmación genuinamente nueva", claim_id="S1_C3",
        previous_claims_by_uid={}, forced_parent_uid=None, round_number=3, text_fingerprint=fingerprint_text,
    )
    assert record.parent_claim_uids == ()
    assert record.claim_version == 1
    assert record.created_round == 3 and record.updated_round == 3
    assert record.claim_uid  # no vacío


@scenario("O06. UID inexistente: continues_claim_uid referencia un uid que no existe -> falla cerrado, nunca se asume 'debe ser nuevo'")
def test_nonexistent_parent_uid_fails_closed():
    try:
        resolve_claim_identity(
            declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-que-no-existe",)),
            claim_text="x", claim_id="S1_C1", previous_claims_by_uid={}, forced_parent_uid=None,
            round_number=2, text_fingerprint=fingerprint_text,
        )
    except ValueError as exc:
        assert "CLAIM_IDENTITY_PARENT_NOT_FOUND" in str(exc)
    else:
        raise AssertionError("debía fallar cerrado ante un parent_claim_uid inexistente")


@scenario("O07. Split: un claim previo se divide en dos -- ambos hijos con parent_claim_uids=(uid_padre,), cada uno con un claim_uid NUEVO propio, ninguno reutiliza el del padre")
def test_split_produces_distinct_new_uids_for_each_child():
    parent = _existing_record("uid-E", claim_id="S1_C1", text="El modelo es rápido y preciso.")
    previous = {"uid-E": parent}
    mint = _counter_mint("split")
    child_a = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="SPLIT_CHILD", parent_claim_uids=("uid-E",)),
        claim_text="El modelo es rápido.", claim_id="S1_C1", previous_claims_by_uid=previous,
        forced_parent_uid=None, round_number=2, text_fingerprint=fingerprint_text, mint_uid=mint,
    )
    child_b = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="SPLIT_CHILD", parent_claim_uids=("uid-E",)),
        claim_text="El modelo es preciso.", claim_id="S1_C2", previous_claims_by_uid=previous,
        forced_parent_uid=None, round_number=2, text_fingerprint=fingerprint_text, mint_uid=mint,
    )
    assert child_a.claim_uid != child_b.claim_uid
    assert child_a.claim_uid != "uid-E" and child_b.claim_uid != "uid-E"
    assert child_a.parent_claim_uids == ("uid-E",) and child_b.parent_claim_uids == ("uid-E",)
    check_no_claim_uid_collisions([child_a, child_b])  # no debe fallar -- son distintos


@scenario("O08. Merge: dos claims previos se fusionan en uno -- parent_claim_uids=(uid1,uid2), claim_uid nuevo, ninguno de los padres reaparece como identidad activa por separado")
def test_merge_produces_new_uid_with_both_parents():
    parent1 = _existing_record("uid-F1", claim_id="S1_C1")
    parent2 = _existing_record("uid-F2", claim_id="S1_C2")
    previous = {"uid-F1": parent1, "uid-F2": parent2}
    merged = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="MERGE", parent_claim_uids=("uid-F1", "uid-F2")),
        claim_text="afirmación fusionada", claim_id="S1_C1", previous_claims_by_uid=previous,
        forced_parent_uid=None, round_number=2, text_fingerprint=fingerprint_text,
    )
    assert set(merged.parent_claim_uids) == {"uid-F1", "uid-F2"}
    assert merged.claim_uid not in {"uid-F1", "uid-F2"}
    active_uids_this_round = {merged.claim_uid}
    assert "uid-F1" not in active_uids_this_round and "uid-F2" not in active_uids_this_round


@scenario("O09. Claim explícitamente corregido por 07 (forced_parent_uid): si el LLM declara el MISMO continues_claim_uid, se preserva; si declara otro uid o NEW, falla cerrado -- nunca se sobrescribe en silencio")
def test_forced_uid_from_issue_must_match_llm_declaration():
    parent = _existing_record("uid-G", claim_id="S5_C2", text="texto con soporte parcial")
    previous = {"uid-G": parent}

    # Caso correcto: el LLM declara CONTINUE con el mismo uid que el issue señalaba.
    record = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-G",)),
        claim_text="texto reescrito dentro del alcance soportado", claim_id="S5_C1",
        previous_claims_by_uid=previous, forced_parent_uid="uid-G", round_number=2,
        text_fingerprint=fingerprint_text,
    )
    assert record.claim_uid == "uid-G"

    # Caso incorrecto 1: el LLM declara un uid DISTINTO -- falla cerrado, no se sobrescribe.
    try:
        resolve_claim_identity(
            declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-OTRO",)),
            claim_text="x", claim_id="S5_C1", previous_claims_by_uid={**previous, "uid-OTRO": _existing_record("uid-OTRO")},
            forced_parent_uid="uid-G", round_number=2, text_fingerprint=fingerprint_text,
        )
    except ValueError as exc:
        assert "CLAIM_IDENTITY_FORCED_UID_MISMATCH" in str(exc)
    else:
        raise AssertionError("debía fallar cerrado -- el LLM declaró un uid distinto al forzado")

    # Caso incorrecto 2: el LLM declara NEW en vez de continuar el uid forzado -- también falla cerrado.
    try:
        resolve_claim_identity(
            declaration=ClaimIdentityDeclaration(action="NEW"),
            claim_text="x", claim_id="S5_C1", previous_claims_by_uid=previous,
            forced_parent_uid="uid-G", round_number=2, text_fingerprint=fingerprint_text,
        )
    except ValueError as exc:
        assert "CLAIM_IDENTITY_FORCED_UID_MISMATCH" in str(exc)
    else:
        raise AssertionError("debía fallar cerrado -- el LLM declaró NEW en vez de continuar el uid forzado")


@scenario("O10. Colisión: dos claims activos resueltos en la misma ronda no pueden compartir claim_uid -- check_no_claim_uid_collisions falla cerrado")
def test_collision_detection_fails_closed():
    parent = _existing_record("uid-H", claim_id="S1_C1")
    # Construir artificialmente una colisión: dos records CONTINUE del MISMO padre en la misma ronda
    # (nunca debería producirse por resolve_claim_identity en un flujo normal -- CONTINUE consume su
    # padre una sola vez -- pero el chequeo de colisión debe detectarlo igual si ocurriera).
    record_a = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-H",)),
        claim_text="version A", claim_id="S1_C1", previous_claims_by_uid={"uid-H": parent},
        forced_parent_uid=None, round_number=2, text_fingerprint=fingerprint_text,
    )
    record_b = resolve_claim_identity(
        declaration=ClaimIdentityDeclaration(action="CONTINUE", parent_claim_uids=("uid-H",)),
        claim_text="version B (distinta, pero mismo padre por error)", claim_id="S1_C5",
        previous_claims_by_uid={"uid-H": parent}, forced_parent_uid=None, round_number=2,
        text_fingerprint=fingerprint_text,
    )
    assert record_a.claim_uid == record_b.claim_uid == "uid-H"  # la colisión real
    try:
        check_no_claim_uid_collisions([record_a, record_b])
    except ValueError as exc:
        assert "CLAIM_IDENTITY_UID_COLLISION" in str(exc)
    else:
        raise AssertionError("debía detectar la colisión de claim_uid entre dos claims activos distintos")


if __name__ == "__main__":
    for fn in (
        test_position_change_preserves_uid,
        test_insertion_before_preserves_existing_uid,
        test_minor_rewrite_continues_uid_new_fingerprint,
        test_deletion_leaves_no_active_trace,
        test_new_claim,
        test_nonexistent_parent_uid_fails_closed,
        test_split_produces_distinct_new_uids_for_each_child,
        test_merge_produces_new_uid_with_both_parents,
        test_forced_uid_from_issue_must_match_llm_declaration,
        test_collision_detection_fails_closed,
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
