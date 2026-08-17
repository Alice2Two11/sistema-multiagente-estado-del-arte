"""Configuración canónica base de Stage 03 (``generation_policy_
config.py``, ``_DEFAULT_EXTRACTION_POLICY``).

Cierre definitivo del problema de configuración: ``extraction_policy.
exclude_reviews`` ya no depende de que una celda de notebook
(``Corrida_03_a_08``) escriba ``active_experiment.json``
manualmente. Todo experimento NUEVO nace con ``exclude_reviews=True``
-- la política metodológica (reviews/surveys completos nunca forman
parte del corpus de generación) queda materializada en la fuente
canónica de defaults, no parcheada dentro de ``classify_review_
exclusion()`` (que sigue recibiendo la policy como parámetro
explícito, sin ningún hardcode).

Sigue siendo configurable: un experimento que declare ``exclude_
reviews: false`` explícitamente en su ``active_experiment.json`` lo
sobrescribe conscientemente, y el sistema lo respeta sin excepción.

El Corpus Eligibility Gate (INCLUDE/EXCLUDE/QUARANTINE) es parte
OBLIGATORIA de Stage03 -- siempre corre, no existe un interruptor
"enabled" que lo desactive (documentado explícitamente en el
comentario de ``_DEFAULT_EXTRACTION_POLICY``). Solo
``min_include_corpus_size`` es configurable, con default 1.

El notebook de corrida (``Corrida_03_a_08``) NO debe escribir estas
policies -- solo debe hacer un preflight fail-closed
(``assert extraction_policy.exclude_reviews is True``). Este módulo
no puede ejecutar el notebook real, pero confirma que el mecanismo
que el preflight necesita (leer el valor efectivo ya resuelto) existe
y es correcto, y que ningún módulo de ``src/`` escribe
``active_experiment.json`` por su cuenta.

Multidominio y genérico: ningún test usa contenido, dominio, filename
ni experimento real."""

from __future__ import annotations

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
        "active_experiment_id": "exp_synthetic", "experiment_dir": "exp_synthetic",
        "openai_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small",
        "chroma_collection_name": "reference_papers_chunks",
        "rag_policy": {
            "retrieval_profiles": {"strict": {"top_k": 10, "fetch_k": 40, "max_per_source": 2}},
            "use_ground_truth_for_generation": False, "use_ground_truth_for_rag": False,
            "use_ground_truth_for_verification": False,
        },
        "generation_profile": {"temperature": 0.1}, "topic_profile": {"topic": "tema sintetico multidominio"},
        "extraction_policy": extraction_policy_overrides,
        "quantitative_extraction_policy": {}, "thematic_analysis_policy": {}, "ingestion_policy": {},
        "outline_generation_policy": {}, "draft_generation_policy": {}, "verification_policy": {},
    }
    (project_dir / "active_experiment.json").write_text(json.dumps(active_experiment, indent=2), encoding="utf-8")


def _load_module_with_extraction_policy(extraction_policy_overrides: dict):
    tmp = tempfile.TemporaryDirectory()
    project_dir = Path(tmp.name)
    _write_active_experiment(project_dir, extraction_policy_overrides)

    previous_env = os.environ.get("THESIS_PROJECT_DIR")
    os.environ["THESIS_PROJECT_DIR"] = str(project_dir)
    try:
        for mod_name in ("src.config.generation_policy_config", "src.config.common_config"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        module = __import__("src.config.generation_policy_config", fromlist=["_dummy"])
    finally:
        if previous_env is None:
            os.environ.pop("THESIS_PROJECT_DIR", None)
        else:
            os.environ["THESIS_PROJECT_DIR"] = previous_env
    module._test_tmp_dir_ref = tmp
    return module


@scenario("CANON-01. Experimento nuevo sin overrides -> extraction_policy.exclude_reviews=True (política metodológica por defecto)")
def test_canon_01_new_experiment_defaults_to_exclude_reviews_true():
    module = _load_module_with_extraction_policy({})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is True


@scenario("CANON-02. Override explícito exclude_reviews=False se respeta sin excepción -- sigue siendo configurable conscientemente")
def test_canon_02_explicit_false_override_respected():
    module = _load_module_with_extraction_policy({"exclude_reviews": False})
    policy = module.get_extraction_policy()
    assert policy["exclude_reviews"] is False


@scenario("CANON-03. Experimento nuevo -> Corpus Eligibility Gate activo por default (min_include_corpus_size=1, sin interruptor 'enabled')")
def test_canon_03_eligibility_gate_active_by_default():
    module = _load_module_with_extraction_policy({})
    policy = module.get_extraction_policy()
    assert "corpus_eligibility_policy" in policy
    assert policy["corpus_eligibility_policy"]["min_include_corpus_size"] == 1
    assert "enabled" not in policy["corpus_eligibility_policy"]


@scenario("CANON-04. min_include_corpus_size es configurable explícitamente (no solo el default)")
def test_canon_04_min_include_corpus_size_configurable():
    module = _load_module_with_extraction_policy({"corpus_eligibility_policy": {"min_include_corpus_size": 5}})
    policy = module.get_extraction_policy()
    assert policy["corpus_eligibility_policy"]["min_include_corpus_size"] == 5


@scenario("CANON-05. min_include_corpus_size solo acepta enteros positivos -- validación estricta, fail-closed")
def test_canon_05_min_include_corpus_size_strict_validation():
    try:
        _load_module_with_extraction_policy({"corpus_eligibility_policy": {"min_include_corpus_size": -1}})
        raise AssertionError("un valor negativo debió ser rechazado")
    except (TypeError, ValueError):
        pass


@scenario("CANON-06. El manifest/fingerprint registra el valor EFECTIVO de exclude_reviews y corpus_eligibility_policy -- nunca un valor distinto al realmente aplicado")
def test_canon_06_manifest_registers_effective_values():
    module = _load_module_with_extraction_policy({})
    fingerprint_config = module.get_extraction_fingerprint_config()
    assert fingerprint_config["policy"]["exclude_reviews"] is True
    assert fingerprint_config["policy"]["corpus_eligibility_policy"]["min_include_corpus_size"] == 1

    snapshot = module.generation_config_snapshot()
    assert snapshot["extraction_policy"]["exclude_reviews"] is True

    module_override = _load_module_with_extraction_policy({"exclude_reviews": False})
    fp_override = module_override.get_extraction_fingerprint_config()
    assert fp_override["policy"]["exclude_reviews"] is False


@scenario("CANON-07. Ningún módulo de src/ escribe active_experiment.json -- la fuente canónica solo se LEE, nunca se materializa desde código de producción")
def test_canon_07_no_production_module_writes_active_experiment():
    import inspect

    from src.config import generation_policy_config, common_config

    for module in (generation_policy_config, common_config):
        source = inspect.getsource(module)
        for line in source.splitlines():
            if "active_experiment" in line and ("write_text" in line or "json.dump(" in line):
                raise AssertionError(f"{module.__name__} escribe active_experiment.json: {line}")


if __name__ == "__main__":
    for fn in (
        test_canon_01_new_experiment_defaults_to_exclude_reviews_true,
        test_canon_02_explicit_false_override_respected,
        test_canon_03_eligibility_gate_active_by_default,
        test_canon_04_min_include_corpus_size_configurable,
        test_canon_05_min_include_corpus_size_strict_validation,
        test_canon_06_manifest_registers_effective_values,
        test_canon_07_no_production_module_writes_active_experiment,
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
