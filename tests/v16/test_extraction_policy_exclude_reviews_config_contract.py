"""Contrato de configuración de ``extraction_policy.exclude_reviews``
(``src/config/generation_policy_config.py``). Complementa el parche
anterior de exclusión determinista de reviews en Stage 03
(``review_exclusion.py``): esa lógica ya funcionaba, pero la clave
``exclude_reviews`` nunca fue añadida al contrato de configuración, así
que cualquier ``active_experiment.json`` que la incluyera fallaba antes
de llegar a ejecutar Stage 03 con ``ValueError: extraction_policy
contains unsupported keys: ['exclude_reviews']``.

Este módulo (``generation_policy_config.py``) exige un
``active_experiment.json`` real en disco al importarse -- cada test
construye su propio fixture sintético (multidominio, sin ningún
nombre de experimento real) en un directorio temporal, fija
``THESIS_PROJECT_DIR`` a esa ruta, y recarga el módulo explícitamente
para que la carga de configuración use ese fixture y no uno cacheado
de una ejecución anterior."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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


def _write_active_experiment(project_dir: Path, extraction_policy_overrides: dict) -> None:
    active_experiment = {
        "active_experiment_id": "exp_synthetic",
        "experiment_dir": "exp_synthetic",
        "openai_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "chroma_collection_name": "reference_papers_chunks",
        "rag_policy": {
            "retrieval_profiles": {"strict": {"top_k": 10, "fetch_k": 40, "max_per_source": 2}},
            "use_ground_truth_for_generation": False,
            "use_ground_truth_for_rag": False,
            "use_ground_truth_for_verification": False,
        },
        "generation_profile": {"temperature": 0.1},
        "topic_profile": {"topic": "tema sintetico multidominio"},
        "extraction_policy": extraction_policy_overrides,
        "quantitative_extraction_policy": {},
        "thematic_analysis_policy": {},
        "ingestion_policy": {},
        "outline_generation_policy": {},
        "draft_generation_policy": {},
        "verification_policy": {},
    }
    (project_dir / "active_experiment.json").write_text(
        json.dumps(active_experiment, indent=2), encoding="utf-8"
    )


def _load_module_with_extraction_policy(extraction_policy_overrides: dict):
    """Construye un active_experiment.json sintético, fija THESIS_
    PROJECT_DIR, y (re)carga generation_policy_config.py desde cero --
    devuelve el módulo cargado."""

    tmp = tempfile.TemporaryDirectory()
    project_dir = Path(tmp.name)
    _write_active_experiment(project_dir, extraction_policy_overrides)

    previous_env = os.environ.get("THESIS_PROJECT_DIR")
    os.environ["THESIS_PROJECT_DIR"] = str(project_dir)
    try:
        for mod_name in ("src.config.generation_policy_config", "src.config.common_config"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        module = importlib.import_module("src.config.generation_policy_config")
    finally:
        if previous_env is None:
            os.environ.pop("THESIS_PROJECT_DIR", None)
        else:
            os.environ["THESIS_PROJECT_DIR"] = previous_env
    module._test_tmp_dir_ref = tmp  # mantiene vivo el TemporaryDirectory mientras el módulo esté en uso
    return module


@scenario("C1. exclude_reviews=True es aceptado y llega como True (bool real) en la policy validada")
def test_c1_exclude_reviews_true_accepted():
    module = _load_module_with_extraction_policy({"exclude_reviews": True})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is True


@scenario("C2. exclude_reviews=False es aceptado")
def test_c2_exclude_reviews_false_accepted():
    module = _load_module_with_extraction_policy({"exclude_reviews": False})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is False


@scenario("C3. Ausencia de la clave conserva el comportamiento CANÓNICO: exclude_reviews=True (todo experimento nuevo excluye reviews por defecto)")
def test_c3_absent_key_defaults_to_true():
    module = _load_module_with_extraction_policy({})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is True
    assert "exclude_reviews" not in module._RAW_POLICY


@scenario("C4. \"true\" (string), 1 (int) y None son rechazados -- solo bool real es válido")
def test_c4_non_bool_values_rejected():
    for bad_value in ("true", 1, None):
        try:
            _load_module_with_extraction_policy({"exclude_reviews": bad_value})
        except TypeError as exc:
            assert "exclude_reviews" in str(exc)
            assert "must be bool" in str(exc)
        else:
            raise AssertionError(f"{bad_value!r} debió ser rechazado con TypeError")


@scenario("C5. Una clave realmente desconocida sigue siendo rechazada (la whitelist no se abrió de más)")
def test_c5_truly_unknown_key_still_rejected():
    try:
        _load_module_with_extraction_policy({"totally_unknown_extraction_key": 1})
    except ValueError as exc:
        assert "totally_unknown_extraction_key" in str(exc)
        assert "unsupported keys" in str(exc)
    else:
        raise AssertionError("una clave desconocida debió ser rechazada con ValueError")


@scenario("C6. exclude_reviews participa en el fingerprint/snapshot ya derivado de la policy validada completa -- sin mecanismo paralelo")
def test_c6_exclude_reviews_participates_in_fingerprint():
    module = _load_module_with_extraction_policy({"exclude_reviews": True})
    fingerprint_config = module.get_extraction_fingerprint_config()
    assert fingerprint_config["policy"]["exclude_reviews"] is True

    snapshot = module.generation_config_snapshot()
    assert snapshot["extraction_policy"]["exclude_reviews"] is True

    # Confirmar también que el fingerprint DIFIERE entre True y False --
    # participa realmente en la firma, no es un campo decorativo.
    module_false = _load_module_with_extraction_policy({"exclude_reviews": False})
    fp_true = json.dumps(module.get_extraction_fingerprint_config()["policy"], sort_keys=True, default=str)
    fp_false = json.dumps(module_false.get_extraction_fingerprint_config()["policy"], sort_keys=True, default=str)
    assert fp_true != fp_false


@scenario("C7. El escenario real reportado (active_experiment.json con extraction_policy.exclude_reviews=true) ya no falla, y la policy validada llega con la clave")
def test_c7_reported_scenario_reproduced_and_fixed():
    module = _load_module_with_extraction_policy({"exclude_reviews": True})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is True
    # Confirmar que sigue conteniendo el resto del contrato histórico intacto.
    assert policy["auto_rebuild"] is True
    assert policy["force_rebuild"] is False


if __name__ == "__main__":
    for fn in (
        test_c1_exclude_reviews_true_accepted,
        test_c2_exclude_reviews_false_accepted,
        test_c3_absent_key_defaults_to_true,
        test_c4_non_bool_values_rejected,
        test_c5_truly_unknown_key_still_rejected,
        test_c6_exclude_reviews_participates_in_fingerprint,
        test_c7_reported_scenario_reproduced_and_fixed,
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
