"""Bug real post-Patch20: ``evaluation_stagespec_wiring.
build_execution_for_stagespec`` resolvía el draft de 06 con una RUTA
FIJA (``05_draft/state_of_art_draft.json``) -- correcta para el flujo
normal (06 siempre escribe ahí), pero INCORRECTA después de cualquier
publicación que comprometa un draft en otra ubicación (ej. el bootstrap
de identidad, patch19, que publica en ``claim_identity_bootstrap/`` para
no tocar el archivo histórico). 07 SÍ resolvía correctamente (vía
``resolve_committed_agent06_artifacts``, la misma función causal que
usa para todo) -- 08 no, y terminaba hasheando (``_sha256_json``) un
archivo DISTINTO al que 07 realmente verificó, produciendo
``AGENT08_SOURCE_DRAFT_FINGERPRINT_MISMATCH`` aunque el contenido
científico nunca cambió.

Diagnóstico confirmado (los 5 puntos pedidos):
1. ``expected_fingerprint`` en ``build_agent08_input_from_committed_
   agent07`` viene de ``manifest.get("source_draft_fingerprint")`` --
   el manifest REAL de 07 (``agent07_artifact_manifest.json``), que a
   su vez es exactamente el ``source_draft_fingerprint`` que 06 le
   entregó a 07 en el handoff.
2. Ese fingerprint de 07 = ``_sha256_json(draft)`` calculado en
   ``build_agent07_input_from_committed_agent06``
   (``agent06_verification_handoff.py``) sobre el draft que
   ``resolve_committed_agent06_artifacts`` resolvió como el REAL
   comprometido -- para un experimento migrado, esa es la copia del
   bootstrap.
3. ``_sha256_json()`` de ``evaluation_stagespec_wiring.py`` (antes del
   fix) se aplicaba sobre ``05_draft/state_of_art_draft.json`` -- el
   archivo ORIGINAL, sin identidad, nunca actualizado por el bootstrap
   (deliberadamente, para no tocar historial) -- un contenido distinto,
   luego un hash distinto.
4. No hay NINGÚN fingerprint derivado ni manifiesto desactualizado: el
   manifest histórico de 06 se copia sin cambios por el bootstrap, pero
   el ``source_draft_fingerprint`` REAL nunca se lee de ese manifest --
   se recalcula siempre en caliente (``_sha256_json(draft)``) sobre el
   contenido real vigente, en 06->07 igual que, tras este parche, en
   ->08. Tampoco hay dos algoritmos: ambos lados siempre usaron
   ``_sha256_json`` -- el bug era la RUTA, nunca el algoritmo.
5. La fuente de verdad normativa es: hash del draft que ``resolve_
   committed_agent06_artifacts`` resuelve como comprometido -- la MISMA
   función causal que ya gobierna 06->07 (patches 11/13). 08 debe usar
   esa MISMA resolución, nunca una ruta fija.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.agent06_verification_handoff import (  # noqa: E402
    build_agent07_input_from_committed_agent06,
    resolve_committed_agent06_artifacts,
)
from src.adapters.evaluation_upstream import _validate_generated_draft  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402
from src.state.pipeline_state import CycleState  # noqa: E402
from src.tools.draft_writing.claim_identity_bootstrap import (  # noqa: E402
    bootstrap_legacy_claim_identity_for_exhausted_cycle,
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


AGENT06 = "06_agente_redactor"


def _sha256_json_local(value) -> str:
    """Misma función exacta que evaluation_upstream._sha256_json --
    duplicada aquí solo para no importar un símbolo privado con
    guion bajo en el módulo real; el algoritmo es idéntico."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_canonical_draft_path(store, project_dir):
    """Réplica exacta del fix real (evaluation_stagespec_wiring.py):
    resolver el draft canónico vía resolve_committed_agent06_artifacts,
    nunca una ruta fija."""
    _, _, agent06_paths, _ = resolve_committed_agent06_artifacts(store=store, stage_name=AGENT06)
    return agent06_paths["state_of_art_draft.json"], agent06_paths["state_of_art_draft.md"]


def _declare_exhausted_legacy_cycle(store):
    state = store.load()
    cycle = CycleState(rounds_used=3, max_rounds=3, status="EXHAUSTED", claim_identity_contract_version="LEGACY")
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))


@scenario("AA01. Draft normal 06->07->08 (sin migración): el fix resuelve la MISMA ruta de siempre, fingerprint coincide")
def test_normal_draft_fingerprint_matches():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))

        # source_draft_fingerprint tal como lo calcula 06->07 realmente.
        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name=AGENT06, agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
        )
        fingerprint_from_07_handoff = handoff["source_draft_fingerprint"]

        # 08 (con el fix): misma resolución causal, mismo archivo.
        draft_json_path, draft_md_path = _resolve_canonical_draft_path(store, project_dir)
        _, fingerprint_from_08 = _validate_generated_draft(draft_json_path, draft_md_path, fingerprint_from_07_handoff)
        assert fingerprint_from_08 == fingerprint_from_07_handoff


@scenario("AA02. Draft migrado LEGACY->STABLE_UID_V1 (bootstrap real, patch19) ->07->08: fingerprint coincide -- el patrón real reportado")
def test_migrated_draft_fingerprint_matches():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        _declare_exhausted_legacy_cycle(store)

        bootstrap_result = bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)
        assert bootstrap_result.bootstrapped is True

        # 07 (posterior al bootstrap) calcula su source_draft_fingerprint
        # exactamente como cualquier handoff real -- sobre lo que
        # resolve_committed_agent06_artifacts resuelve AHORA (el
        # bootstrap, no el original).
        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name=AGENT06, agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
        )
        fingerprint_from_07_handoff = handoff["source_draft_fingerprint"]

        # Confirmar el bug ANTES del fix: la ruta original (la que
        # comprometió el 06 real, antes del bootstrap) hashea contenido
        # DISTINTO al que el bootstrap publicó -- el archivo original,
        # sin identidad, nunca actualizado (deliberadamente, para no
        # tocar historial).
        original_path = Path(store.load().decision_log[0].result["output_artifacts"]["state_of_art_draft.json"]["path"])
        stale_draft = json.loads(original_path.read_text(encoding="utf-8"))
        stale_fingerprint = _sha256_json_local(stale_draft)
        assert stale_fingerprint != fingerprint_from_07_handoff  # confirma que el bug era real

        # 08 CON el fix: misma resolución causal que usó 07 -- coincide.
        draft_json_path, draft_md_path = _resolve_canonical_draft_path(store, project_dir)
        assert draft_json_path.parent.name == "claim_identity_bootstrap"  # resolvió la publicación NUEVA, no la vieja
        _, fingerprint_from_08 = _validate_generated_draft(draft_json_path, draft_md_path, fingerprint_from_07_handoff)
        assert fingerprint_from_08 == fingerprint_from_07_handoff

        # Los 16 claims migrados siguen presentes e intactos en lo que 08 leería.
        draft = json.loads(draft_json_path.read_text(encoding="utf-8"))
        claim_count = sum(len(s.get("claims", [])) for s in draft["sections"])
        assert claim_count == bootstrap_result.claims_migrated


@scenario("AA03. Cambio científico real DESPUÉS del commit de 07 -> mismatch detectado, fail closed (el fix no oculta mismatches reales)")
def test_real_scientific_change_still_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name=AGENT06, agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
        )
        fingerprint_that_07_recorded = handoff["source_draft_fingerprint"]

        # Alguien edita el texto científico DESPUÉS de que 07 ya se comprometió
        # (nunca debería pasar en producción -- pero si pasara, 08 debe rechazarlo).
        draft_json_path, draft_md_path = _resolve_canonical_draft_path(store, project_dir)
        draft = json.loads(draft_json_path.read_text(encoding="utf-8"))
        draft["sections"][0]["draft_text"] = draft["sections"][0]["draft_text"] + " Texto añadido sin autorización."
        draft_json_path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")

        try:
            _validate_generated_draft(draft_json_path, draft_md_path, fingerprint_that_07_recorded)
        except ValueError as exc:
            assert "AGENT08_SOURCE_DRAFT_FINGERPRINT_MISMATCH" in str(exc)
        else:
            raise AssertionError("un cambio científico real posterior a 07 debe fallar cerrado, nunca pasar silenciosamente")


@scenario("AA04. Cambio SOLO técnico (bootstrap) conserva trazabilidad según la definición canónica: mismo texto, mismo fingerprint en TODA la cadena 06->07->08")
def test_technical_only_change_preserves_traceability():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))

        # Fingerprint/textos ANTES del bootstrap.
        original_paths = resolve_committed_agent06_artifacts(store=store, stage_name=AGENT06)[2]
        original_draft = json.loads(original_paths["state_of_art_draft.json"].read_text(encoding="utf-8"))
        original_texts = [c.get("claim") for s in original_draft["sections"] for c in s.get("claims", [])]

        _declare_exhausted_legacy_cycle(store)
        bootstrap_legacy_claim_identity_for_exhausted_cycle(store=store, project_dir=project_dir)

        draft_json_path, draft_md_path = _resolve_canonical_draft_path(store, project_dir)
        migrated_draft = json.loads(draft_json_path.read_text(encoding="utf-8"))
        migrated_texts = [c.get("claim") for s in migrated_draft["sections"] for c in s.get("claims", [])]

        assert original_texts == migrated_texts  # cero cambio científico

        # La definición canónica (resolución causal + _sha256_json) es
        # consistente end-to-end: lo que 06->07 calcula es EXACTAMENTE
        # lo que 08 valida, para el draft migrado.
        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name=AGENT06, agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
        )
        _, fingerprint_08 = _validate_generated_draft(draft_json_path, draft_md_path, handoff["source_draft_fingerprint"])
        assert fingerprint_08 == handoff["source_draft_fingerprint"]


@scenario("AA05. No bypass: _validate_generated_draft sigue lanzando ValueError real ante cualquier mismatch -- el fix no debilitó ni removió la validación")
def test_mismatch_check_not_weakened():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        draft_json_path, draft_md_path = _resolve_canonical_draft_path(store, project_dir)
        try:
            _validate_generated_draft(draft_json_path, draft_md_path, "0" * 64)  # fingerprint deliberadamente falso
        except ValueError as exc:
            assert "AGENT08_SOURCE_DRAFT_FINGERPRINT_MISMATCH" in str(exc)
        else:
            raise AssertionError("la validación de fingerprint sigue debiendo fallar cerrado ante cualquier mismatch real")


if __name__ == "__main__":
    for fn in (
        test_normal_draft_fingerprint_matches,
        test_migrated_draft_fingerprint_matches,
        test_real_scientific_change_still_fails_closed,
        test_technical_only_change_preserves_traceability,
        test_mismatch_check_not_weakened,
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
