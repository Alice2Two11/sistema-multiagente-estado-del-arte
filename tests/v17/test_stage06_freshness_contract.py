"""Contrato de freshness de Stage 06 -- el fingerprint (``_draft_
signature``, ``src/adapters/draft_writing_runtime.py``) debe
invalidarse cuando cambia el contrato de validación/redacción, no solo
cuando cambian los inputs de datos.

Causa raíz cerrada: el fingerprint YA incluía ``validation_version``
(parte de ``LEGACY_RUNTIME_VERSIONS``/``HYBRID_RUNTIME_VERSIONS``,
inyectadas vía ``build_runtime_draft_policy`` -> ``_runtime_versions``,
consumidas por ``_draft_signature``) -- pero esa constante nunca se
incrementó al aplicar el parche de longitud dura (gate ``configured_
min_total_words``/``configured_max_total_words``, reason codes
específicos, ``length_repair.py``), así que el freshness nunca detectó
el cambio de comportamiento y el pipeline reportó ``[SKIPPED_FRESH]``
para 06 y 07 con código YA desactualizado.

No se inventó ningún mecanismo paralelo: se reutilizó ``validation_
version``, la misma abstracción de versión estable ya existente y ya
consumida por ``_draft_signature`` -- incrementada en las 4 copias
sincronizadas (``LEGACY_VERSIONS``/``HYBRID_VERSIONS`` en
``draft_writing_agent.py``, ``LEGACY_RUNTIME_VERSIONS``/``HYBRID_
RUNTIME_VERSIONS`` en ``draft_writing_runtime.py`` -- duplicación
preexistente, ya documentada como debiendo mantenerse sincronizada).

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.draft_writing_protocol import build_draft_fingerprints  # noqa: E402
from src.orchestration.decision_engine import fingerprints_match  # noqa: E402
from src.contracts.agent_input import AgentInput, AgentContext, ExecutionMode  # noqa: E402
from src.adapters.draft_writing_runtime import (  # noqa: E402
    LEGACY_RUNTIME_VERSIONS,
    HYBRID_RUNTIME_VERSIONS,
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


CURRENT_VALIDATION_VERSION = LEGACY_RUNTIME_VERSIONS["validation_version"]
OLD_VALIDATION_VERSION = "legacy_notebook06_validation_v1"


def _base_policy(**overrides):
    policy = {
        "stage_version": LEGACY_RUNTIME_VERSIONS["stage_version"],
        "prompt_version": "P1",
        "rag_version": LEGACY_RUNTIME_VERSIONS["rag_version"],
        "validation_version": CURRENT_VALIDATION_VERSION,
        "normalization_version": "N1",
        "min_total_words": 1300,
        "target_total_words": 1600,
        "max_total_words": 2200,
    }
    policy.update(overrides)
    return policy


def _agent_input(policy, stage_name="06_agente_redactor"):
    return AgentInput(
        "exp_synthetic", "run_synthetic", stage_name, 1, ExecutionMode.FULL_RUN,
        AgentContext(("llm",), "/tmp/out", {}), {}, policy, None,
    )


@scenario("FRESH06-01. Mismo input + misma contract version -> fingerprints coinciden (SKIPPED_FRESH esperado)")
def test_fresh06_01_same_input_same_version_matches():
    fp_a = build_draft_fingerprints(_agent_input(_base_policy()))
    fp_b = build_draft_fingerprints(_agent_input(_base_policy()))
    assert fingerprints_match(fp_a, fp_b) is True


@scenario("FRESH06-02. Cambia validation_version (contract version) -> Stage 06 deja de estar fresh")
def test_fresh06_02_contract_version_change_breaks_freshness():
    fp_old = build_draft_fingerprints(_agent_input(_base_policy(validation_version=OLD_VALIDATION_VERSION)))
    fp_new = build_draft_fingerprints(_agent_input(_base_policy(validation_version=CURRENT_VALIDATION_VERSION)))
    assert fingerprints_match(fp_old, fp_new) is False

    # Reproduce el escenario real reportado: el fingerprint de una
    # corrida hecha ANTES del parche de longitud (validation_version
    # vieja) ya no coincide con el fingerprint que produce el código
    # actual -- el pipeline ya no reportaría SKIPPED_FRESH.
    assert CURRENT_VALIDATION_VERSION != OLD_VALIDATION_VERSION


@scenario("FRESH06-03. Cambio en min_total_words/target_total_words/max_total_words -> Stage 06 deja de estar fresh")
def test_fresh06_03_word_range_change_breaks_freshness():
    fp_a = build_draft_fingerprints(_agent_input(_base_policy(min_total_words=1300, target_total_words=1600, max_total_words=2200)))
    fp_b = build_draft_fingerprints(_agent_input(_base_policy(min_total_words=1500, target_total_words=1600, max_total_words=2200)))
    fp_c = build_draft_fingerprints(_agent_input(_base_policy(min_total_words=1300, target_total_words=1800, max_total_words=2200)))
    fp_d = build_draft_fingerprints(_agent_input(_base_policy(min_total_words=1300, target_total_words=1600, max_total_words=2500)))
    assert fingerprints_match(fp_a, fp_b) is False
    assert fingerprints_match(fp_a, fp_c) is False
    assert fingerprints_match(fp_a, fp_d) is False


@scenario("FRESH06-04. Cambio de fingerprint de Stage 06 invalida el escenario que 07/08 dependen de -- el fingerprint COMPUESTO ya no coincide")
def test_fresh06_04_stage06_change_propagates_to_dependents():
    # No se toca 07/08: se confirma que el MECANISMO por el que 07/08
    # detectan cambios (comparación de fingerprints/hashes de los
    # artefactos de 06 de los que dependen) recibe una entrada distinta
    # cuando 06 cambia -- el fingerprint de 06 antes/después del parche
    # es diferente, así que cualquier verificación de dependencia aguas
    # abajo que lo incluya (directamente, o vía el hash de los
    # artefactos que 06 vuelve a escribir al re-ejecutarse) deja de
    # coincidir. Se confirma a nivel de la pieza real que cambia.
    fp_before = build_draft_fingerprints(_agent_input(_base_policy(validation_version=OLD_VALIDATION_VERSION)))
    fp_after = build_draft_fingerprints(_agent_input(_base_policy(validation_version=CURRENT_VALIDATION_VERSION)))
    assert fp_before.composite != fp_after.composite


@scenario("FRESH06-05. Stage 03/03B/04/05 permanecen fresh -- ningún archivo de esas etapas importa nada de draft_writing_runtime.py/draft_writing_agent.py")
def test_fresh06_05_upstream_stages_unaffected():
    import inspect

    from src.agents import extraction_agent, thematic_analysis_agent, outline_generation_agent
    from src.adapters import extraction_runtime, thematic_analysis_runtime, outline_generation_runtime, quantitative_extraction_runtime

    for module in (
        extraction_agent, thematic_analysis_agent, outline_generation_agent,
        extraction_runtime, thematic_analysis_runtime, outline_generation_runtime,
        quantitative_extraction_runtime,
    ):
        source = inspect.getsource(module)
        assert "draft_writing_runtime" not in source
        assert "draft_writing_agent" not in source


@scenario("FRESH06-06. Ground Truth no participa en el fingerprint de 06 -- ni en la firma ni en la validación de longitud")
def test_fresh06_06_ground_truth_never_in_fingerprint():
    import inspect

    from src.adapters import draft_writing_runtime

    signature_source = inspect.getsource(draft_writing_runtime._draft_signature)
    assert "ground_truth" not in signature_source.lower()

    fp = build_draft_fingerprints(_agent_input(_base_policy()))
    import json
    fp_dict = fp.__dict__ if hasattr(fp, "__dict__") else {}
    assert "ground_truth" not in json.dumps(fp_dict, default=str).lower()


if __name__ == "__main__":
    for fn in (
        test_fresh06_01_same_input_same_version_matches,
        test_fresh06_02_contract_version_change_breaks_freshness,
        test_fresh06_03_word_range_change_breaks_freshness,
        test_fresh06_04_stage06_change_propagates_to_dependents,
        test_fresh06_05_upstream_stages_unaffected,
        test_fresh06_06_ground_truth_never_in_fingerprint,
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
