"""Unificación de ``LEGACY_VERSIONS``/``HYBRID_VERSIONS`` (Stage 06).

Causa raíz: ``draft_writing_agent.py`` y ``draft_writing_runtime.py``
definían cada uno su propia copia de ``LEGACY_VERSIONS``/``HYBRID_
VERSIONS`` (bajo los nombres ``LEGACY_RUNTIME_VERSIONS``/``HYBRID_
RUNTIME_VERSIONS`` en el runtime) -- dos pares de diccionarios
idénticos mantenidos a mano en paralelo, documentado como deuda
preexistente en el propio código antes de esta unificación.

Fix: ``draft_writing_agent.py`` es ahora la fuente canónica única.
``draft_writing_runtime.py`` importa esos mismos dicts con alias
(``LEGACY_VERSIONS as LEGACY_RUNTIME_VERSIONS``, ``HYBRID_VERSIONS as
HYBRID_RUNTIME_VERSIONS``) en vez de definir su propia copia -- mismos
nombres públicos, mismos valores, mismo comportamiento; la única
diferencia es que ahora es un solo objeto en memoria, no dos.

Verificado antes de esta unificación (ver auditoría previa a este
cambio): ambos pares de diccionarios eran, valor por valor y clave por
clave, exactamente idénticos -- esta unificación no cambia ningún
valor de versión ni ninguna semántica de contrato."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.draft_writing_agent import (  # noqa: E402
    HYBRID_VERSIONS,
    LEGACY_VERSIONS,
)
from src.adapters.draft_writing_runtime import (  # noqa: E402
    HYBRID_RUNTIME_VERSIONS,
    LEGACY_RUNTIME_VERSIONS,
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


@scenario("VERSIONS-01. LEGACY_VERSIONS (agent) y LEGACY_RUNTIME_VERSIONS (runtime) son el MISMO objeto en memoria, no solo iguales por valor")
def test_versions_01_legacy_same_object_identity():
    assert LEGACY_VERSIONS is LEGACY_RUNTIME_VERSIONS


@scenario("VERSIONS-02. HYBRID_VERSIONS (agent) y HYBRID_RUNTIME_VERSIONS (runtime) son el MISMO objeto en memoria")
def test_versions_02_hybrid_same_object_identity():
    assert HYBRID_VERSIONS is HYBRID_RUNTIME_VERSIONS


@scenario("VERSIONS-03. Los valores de LEGACY_VERSIONS son exactamente los documentados -- ningún valor cambió al unificar")
def test_versions_03_legacy_values_unchanged():
    assert LEGACY_VERSIONS == {
        "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
        "rag_version": "legacy_chroma_then_csv_restricted_v1",
        "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
        "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
    }


@scenario("VERSIONS-04. Los valores de HYBRID_VERSIONS son exactamente los documentados -- ningún valor cambió al unificar")
def test_versions_04_hybrid_values_unchanged():
    assert HYBRID_VERSIONS == {
        "stage_version": "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE",
        "rag_version": "hybrid_chroma_csv_rrf_balanced_v1",
        "quantitative_selection_version": "confirmed_literal_greedy_coverage_v1",
        "budget_version": "source_aware_exact_total_v1",
        "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
        "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
    }


@scenario("VERSIONS-05. draft_writing_runtime.py ya no contiene una definición propia de estos diccionarios -- solo el import con alias")
def test_versions_05_runtime_has_no_own_definition():
    source = (REPO_ROOT / "src" / "adapters" / "draft_writing_runtime.py").read_text(encoding="utf-8")
    assert "LEGACY_RUNTIME_VERSIONS = {" not in source
    assert "HYBRID_RUNTIME_VERSIONS = {" not in source
    assert "HYBRID_VERSIONS as HYBRID_RUNTIME_VERSIONS" in source
    assert "LEGACY_VERSIONS as LEGACY_RUNTIME_VERSIONS" in source


@scenario("VERSIONS-06. Mutar el dict vía el nombre del agent es visible vía el nombre del runtime, y viceversa (mismo objeto, no una copia sincronizada)")
def test_versions_06_mutation_visible_across_both_names():
    marker_key = "_identity_probe_marker"
    assert marker_key not in LEGACY_VERSIONS
    try:
        LEGACY_VERSIONS[marker_key] = "probe"
        assert LEGACY_RUNTIME_VERSIONS.get(marker_key) == "probe"
    finally:
        LEGACY_VERSIONS.pop(marker_key, None)
    assert marker_key not in LEGACY_VERSIONS
    assert marker_key not in LEGACY_RUNTIME_VERSIONS


if __name__ == "__main__":
    for fn in (
        test_versions_01_legacy_same_object_identity,
        test_versions_02_hybrid_same_object_identity,
        test_versions_03_legacy_values_unchanged,
        test_versions_04_hybrid_values_unchanged,
        test_versions_05_runtime_has_no_own_definition,
        test_versions_06_mutation_visible_across_both_names,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    raise SystemExit(1 if failed else 0)
