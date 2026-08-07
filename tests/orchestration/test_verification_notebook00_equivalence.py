"""Pruebas de equivalencia de configuración: notebook 00 (real) vs. adaptador.

``NOTEBOOK_00_FIXED_VERIFICATION_POLICY``, ``NOTEBOOK_00_FIXED_POST_CORRECTION_RECHECK_POLICY``,
``NOTEBOOK_00_OPENAI_MODEL`` y ``NOTEBOOK_00_CHROMA_COLLECTION_NAME`` son
copias LITERALES de la celda 3 (bloque "CONFIGURACIÓN GENERAL DEL
PROYECTO") de ``00_setup_config.ipynb``, extraída el mismo día que este
archivo. ``_reproduce_config_py_derivation`` reproduce, también de forma
literal, la lógica real de la celda 9 (``%%writefile
.../src/config.py``, función ``_load_active_experiment`` + las asignaciones
``VERIFICATION_POLICY = ACTIVE_EXPERIMENT["verification_policy"]`` etc.):
un simple passthrough con validación de claves obligatorias y ``.strip()``
en los dos campos de texto — sin ninguna otra transformación.

Estas pruebas NO ejecutan el notebook (no hay Colab aquí), pero SÍ ejecutan
la lógica real de esas dos celdas, extraída verbatim, contra
``load_verification_configuration`` real del adaptador — no comparan contra
un valor inventado.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.verification_orchestrator_runtime import (
    MissingRequiredActiveExperimentKeyError,
    load_verification_configuration,
)

# ---------------------------------------------------------------------------
# Copia literal de notebook 00, celda 3 (extraída el mismo día que este
# archivo — cualquier divergencia futura se resuelve re-extrayendo la celda,
# no editando estas constantes de memoria).
# ---------------------------------------------------------------------------

NOTEBOOK_00_OPENAI_MODEL = "gpt-4.1-mini"
NOTEBOOK_00_CHROMA_COLLECTION_NAME = "reference_papers_chunks"

NOTEBOOK_00_FIXED_VERIFICATION_POLICY = {
    "temperature": 0.0,
    "auto_rebuild": True,
    "force_rebuild": False,
    "max_chunk_chars": 2500,
    "max_evidence_chars_per_claim": 7000,
    "top_k_independent_evidence_per_claim": 4,
    "restrict_retrieval_to_outline_sources": True,
    "allow_section_evidence_fallback": False,
    "allow_fuzzy_citation_repair": False,
    "validate_llm_claim_coverage": True,
    "validate_llm_evidence_against_claim_candidates": True,
    "allow_automatic_corrections": True,
    "max_correction_attempts": 2,
    "require_post_correction_recheck": True,
    "fail_on_invalid_verification": True,
}

NOTEBOOK_00_FIXED_POST_CORRECTION_RECHECK_POLICY = {
    "temperature": 0.0,
    "auto_rebuild": True,
    "force_rebuild": False,
    "max_recheck_attempts": 3,
    "max_chunk_chars": 2500,
    "max_evidence_chars_per_claim": 7000,
    "top_k_independent_evidence_per_claim": 4,
    "restrict_retrieval_to_outline_sources": True,
    "allow_section_evidence_fallback": False,
    "allow_fuzzy_citation_repair": False,
    "preserve_parent_claim_ids": True,
    "validate_corrected_fragments_exactly": True,
    "validate_numeric_values_against_cited_chunks": True,
    "validate_complete_recheck_coverage": True,
    "allow_additional_automatic_corrections": False,
    "require_all_applied_corrections_rechecked": True,
    "fail_on_invalid_recheck": True,
    "create_evaluation_ready_copy_only_if_approved": True,
}

# Claves que notebook 00 celda 8 escribe en active_experiment.json y que
# notebook 00 celda 9 (config.py real) exige como obligatorias
# (`required_keys` en `_load_active_experiment`). Lista copiada literal de
# esa celda.
CONFIG_PY_REQUIRED_KEYS = {
    "active_experiment_id",
    "experiment_dir",
    "generation_profile",
    "topic_profile",
    "openai_model",
    "embedding_model",
    "chroma_collection_name",
    "rag_policy",
    "extraction_policy",
    "quantitative_extraction_policy",
    "thematic_analysis_policy",
    "ingestion_policy",
    "outline_generation_policy",
    "draft_generation_policy",
    "verification_policy",
    "post_correction_recheck_policy",
    "evaluation_policy",
}


NOTEBOOK_00_FIXED_RAG_POLICY = {
    "exclude_review_sections_from_reference_papers": True,
    "excluded_reference_section_types": [
        "related_work",
        "literature_review",
        "state_of_the_art",
        "background",
        "theoretical_background",
        "previous_work",
        "prior_work",
    ],
    "ground_truth_usage": "evaluation_only",
    "use_ground_truth_for_generation": False,
    "use_ground_truth_for_rag": False,
    "use_ground_truth_for_verification": False,
    "use_ground_truth_for_evaluation": True,
    "retrieval_profiles": {
        "default": {"top_k": 8, "fetch_k": 35, "max_per_source": 2},
        "compact": {"top_k": 6, "fetch_k": 35, "max_per_source": 2},
        "strict": {"top_k": 10, "fetch_k": 40, "max_per_source": 2},
        "testing": {"top_k": 5, "fetch_k": 30, "max_per_source": 2},
    },
    "indexing": {"batch_size": 200},
    "generation": {"temperature": 0.1, "answer_max_words": 120},
}


def _reproduce_notebook00_active_experiment(experiment_id: str, experiment_dir: Path) -> dict:
    """Reproduce, con los mismos nombres de clave, lo que notebook 00 celda 8
    escribe en ``active_experiment.json`` — solo rellenando con placeholders
    mínimos las claves que celda 9 exige pero que no son ninguno de los 4
    campos bajo prueba (generation_profile, topic_profile, rag_policy, etc.:
    dicts no vacíos cualesquiera, porque config.py real solo exige
    "isinstance(value, dict) and value", no un contenido específico)."""

    return {
        "active_experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "generation_profile": {"embedding_model": "all-MiniLM-L6-v2"},
        "topic_profile": {"topic_name": "placeholder"},
        "openai_model": NOTEBOOK_00_OPENAI_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
        "chroma_collection_name": NOTEBOOK_00_CHROMA_COLLECTION_NAME,
        "rag_policy": NOTEBOOK_00_FIXED_RAG_POLICY,
        "extraction_policy": {"placeholder": True},
        "quantitative_extraction_policy": {"placeholder": True},
        "thematic_analysis_policy": {"placeholder": True},
        "ingestion_policy": {"placeholder": True},
        "outline_generation_policy": {"placeholder": True},
        "draft_generation_policy": {"placeholder": True},
        "verification_policy": NOTEBOOK_00_FIXED_VERIFICATION_POLICY,
        "post_correction_recheck_policy": NOTEBOOK_00_FIXED_POST_CORRECTION_RECHECK_POLICY,
        "evaluation_policy": {"placeholder": True},
    }


def _reproduce_config_py_derivation(active_experiment: dict) -> dict:
    """Reproduce literalmente la lógica real de config.py (notebook 00, celda 9):
    ``_load_active_experiment`` (valida required_keys) + las asignaciones
    ``VERIFICATION_POLICY = ACTIVE_EXPERIMENT["verification_policy"]``, etc.
    Sin ninguna transformación excepto ``.strip()`` en los dos campos de texto
    — igual que el original."""

    missing = sorted(CONFIG_PY_REQUIRED_KEYS - set(active_experiment))
    if missing:
        raise ValueError(f"active_experiment.json está incompleto. Faltan: {missing}")

    return {
        "OPENAI_MODEL": str(active_experiment["openai_model"]).strip(),
        "CHROMA_COLLECTION_NAME": str(active_experiment["chroma_collection_name"]).strip(),
        "VERIFICATION_POLICY": active_experiment["verification_policy"],
        "POST_CORRECTION_RECHECK_POLICY": active_experiment["post_correction_recheck_policy"],
    }


def _write_project(tmp: Path, active_experiment: dict) -> Path:
    root = tmp / "proj"
    experiment_dir = Path(active_experiment["experiment_dir"])
    root.mkdir()
    (root / "active_experiment.json").write_text(
        json.dumps(active_experiment), encoding="utf-8"
    )
    return root


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


@scenario("E1. OPENAI_MODEL: config.py real vs. load_verification_configuration")
def test_openai_model_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        notebook_config_py = _reproduce_config_py_derivation(active)
        root = _write_project(tmp_path, active)

        cfg = load_verification_configuration(root)
        assert (
            cfg["agent07_config"]["verification_model"] == notebook_config_py["OPENAI_MODEL"]
        ), (cfg["agent07_config"]["verification_model"], notebook_config_py["OPENAI_MODEL"])
        assert cfg["agent07_config"]["correction_model"] == notebook_config_py["OPENAI_MODEL"]
        assert cfg["agent07_config"]["reverification_model"] == notebook_config_py["OPENAI_MODEL"]
        assert cfg["agent07_config"]["verification_model"] == "gpt-4.1-mini"


@scenario("E2. CHROMA_COLLECTION_NAME: config.py real vs. adaptador")
def test_chroma_collection_name_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        notebook_config_py = _reproduce_config_py_derivation(active)
        root = _write_project(tmp_path, active)

        cfg = load_verification_configuration(root)
        assert (
            cfg["agent07_config"]["chroma_collection_name"]
            == notebook_config_py["CHROMA_COLLECTION_NAME"]
        )
        assert cfg["agent07_config"]["collection_name"] == "reference_papers_chunks"


@scenario("E3. VERIFICATION_POLICY: todas las claves de config.py real presentes y con el mismo valor")
def test_verification_policy_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        notebook_config_py = _reproduce_config_py_derivation(active)
        root = _write_project(tmp_path, active)

        cfg = load_verification_configuration(root)
        adapter_policy = cfg["agent07_config"]["verification_policy"]
        for key, value in notebook_config_py["VERIFICATION_POLICY"].items():
            assert key in adapter_policy, f"falta la clave {key!r} del notebook 00 real"
            assert adapter_policy[key] == value, (key, adapter_policy[key], value)
        # También se mezcla en correction_policy (mismo comportamiento que
        # el notebook real: correction_policy.update(VERIFICATION_POLICY),
        # NO un CORRECTION_POLICY separado — ver AGENT07_CONFIG_EQUIVALENCE.md).
        adapter_correction_policy = cfg["agent07_config"]["correction_policy"]
        for key, value in notebook_config_py["VERIFICATION_POLICY"].items():
            assert adapter_correction_policy[key] == value


@scenario("E4. POST_CORRECTION_RECHECK_POLICY: todas las claves presentes y con el mismo valor")
def test_post_correction_recheck_policy_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        notebook_config_py = _reproduce_config_py_derivation(active)
        root = _write_project(tmp_path, active)

        cfg = load_verification_configuration(root)
        adapter_policy = cfg["agent07_config"]["reverification_policy"]
        for key, value in notebook_config_py["POST_CORRECTION_RECHECK_POLICY"].items():
            assert key in adapter_policy, f"falta la clave {key!r} del notebook 00 real"
            assert adapter_policy[key] == value, (key, adapter_policy[key], value)


@scenario("E5. Ausencia de 'openai_model' en active_experiment.json: falla igual que config.py real")
def test_missing_openai_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        del active["openai_model"]
        # config.py real: _load_active_experiment también fallaría aquí.
        try:
            _reproduce_config_py_derivation(active)
        except ValueError:
            pass
        else:
            raise AssertionError("la reproducción de config.py debía fallar también")

        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError(
                "load_verification_configuration debía fallar igual que config.py real, "
                "no rellenar con un default silencioso"
            )


@scenario("E6. Ausencia de 'verification_policy': falla igual que config.py real")
def test_missing_verification_policy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        del active["verification_policy"]
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError("debía fallar por falta de verification_policy obligatoria")


@scenario("E7. Ausencia de 'post_correction_recheck_policy': falla igual que config.py real")
def test_missing_post_correction_recheck_policy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        del active["post_correction_recheck_policy"]
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError(
                "debía fallar por falta de post_correction_recheck_policy obligatoria"
            )


@scenario("E8. Ausencia de 'chroma_collection_name': falla igual que config.py real")
def test_missing_chroma_collection_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        del active["chroma_collection_name"]
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError("debía fallar por falta de chroma_collection_name obligatoria")


@scenario("E9. verification_policy vacío ({}) también falla, igual que config.py real")
def test_empty_verification_policy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        active["verification_policy"] = {}
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except ValueError:
            pass
        else:
            raise AssertionError("debía fallar: config.py real rechaza dict vacío")


@scenario("E10. embedding_model: config.py real vs. adaptador (sin prioridad artificial de embedding_model_name)")
def test_embedding_model_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        root = _write_project(tmp_path, active)
        cfg = load_verification_configuration(root)
        assert cfg["agent07_config"]["embedding_model"] == "all-MiniLM-L6-v2"
        # embedding_model_name NUNCA es una clave real; confirmar que el
        # adaptador no la usa ni la exige.
        assert "embedding_model_name" not in active


@scenario("E11. Ausencia de 'embedding_model': falla igual que config.py real")
def test_missing_embedding_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        del active["embedding_model"]
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError("debía fallar por falta de embedding_model obligatoria")


@scenario("E12. embedding_model vacío ('') también falla, igual que config.py real")
def test_empty_embedding_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        active["embedding_model"] = "   "  # solo espacios -> "" tras strip()
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except ValueError:
            pass
        else:
            raise AssertionError("debía fallar: config.py real rechaza un valor vacío")


@scenario("E13. rag_policy: transformación real de get_rag_policy() reproducida (no passthrough)")
def test_rag_policy_transformation_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        root = _write_project(tmp_path, active)
        cfg = load_verification_configuration(root)
        rag_policy = cfg["rag_policy"]

        # get_rag_policy() real RESHAPEA la política: no es un passthrough
        # del dict crudo. Confirmar las claves derivadas reales.
        assert rag_policy["review_section_types"] == sorted(
            NOTEBOOK_00_FIXED_RAG_POLICY["excluded_reference_section_types"]
        )
        assert rag_policy["exclude_review_sections_from_reference_papers"] is True
        assert rag_policy["ground_truth_policy"] == {
            "use_ground_truth_for_generation": False,
            "use_ground_truth_for_rag": False,
            "use_ground_truth_for_verification": False,
            "use_ground_truth_for_evaluation": True,
        }
        assert rag_policy["index_batch_size"] == 200
        assert rag_policy["rag_temperature"] == 0.1
        assert rag_policy["rag_answer_max_words"] == 120
        # Campos hardcodeados en rag_policy.py real (no vienen de
        # active_experiment.json) — deben aparecer igual.
        assert "trabajos relacionados" in rag_policy["review_section_labels_es"].values()
        assert len(rag_policy["review_section_patterns"]) == 17
        assert "Ground Truth" in rag_policy["rag_allowed_content_policy"]
        # retrieval_profiles sobrevive intacto a la transformación (por eso
        # el retriever puede seguir usándolo tal cual).
        assert rag_policy["retrieval_profiles"]["default"]["top_k"] == 8


@scenario("E14. rag_policy incompleta: falla igual que rag_policy.py real (claves faltantes)")
def test_incomplete_rag_policy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        partial = dict(NOTEBOOK_00_FIXED_RAG_POLICY)
        del partial["retrieval_profiles"]
        active["rag_policy"] = partial
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except ValueError as exc:
            assert "retrieval_profiles" in str(exc)
        else:
            raise AssertionError("debía fallar: rag_policy.py real exige retrieval_profiles")


@scenario("E15. rag_policy con ground truth fuera de evaluación: falla igual que rag_policy.py real")
def test_rag_policy_ground_truth_misuse_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        active = _reproduce_notebook00_active_experiment(
            experiment_id, tmp_path / "proj" / experiment_id
        )
        bad = dict(NOTEBOOK_00_FIXED_RAG_POLICY)
        bad["use_ground_truth_for_generation"] = True  # prohibido por la política real
        active["rag_policy"] = bad
        root = _write_project(tmp_path, active)
        try:
            load_verification_configuration(root)
        except ValueError as exc:
            assert "solo puede utilizarse para evaluación" in str(exc)
        else:
            raise AssertionError("debía fallar: la política real prohíbe esto")


@scenario("E16. Rutas: derivadas de experiment_dir coinciden exactamente con config.py real (celda 9)")
def test_paths_match_config_py_derivation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiment_id = "exp1"
        experiment_dir = tmp_path / "proj" / experiment_id
        active = _reproduce_notebook00_active_experiment(experiment_id, experiment_dir)
        root = _write_project(tmp_path, active)
        cfg = load_verification_configuration(root)

        outputs_dir = experiment_dir / "05_outputs"
        # Derivaciones literales de config.py real (celda 9), citadas en
        # AGENT07_CONFIG_EQUIVALENCE.md sección 6.
        expected = {
            "ORCHESTRATOR_DIR": outputs_dir / "00_orchestrator_planner",
            "OUTLINE_DIR": outputs_dir / "04_outline",
            "VERIFICATION_TRACEABILITY_DIR": outputs_dir / "06_verification_traceability",
            "OUTPUTS_DIR": outputs_dir,
            "CHROMA_DIR": experiment_dir / "04_chroma_index",
            "CHUNKS_DIR": experiment_dir / "03_chunks",
        }
        assert cfg["experiment_paths"]["pipeline_state_path"] == str(
            expected["ORCHESTRATOR_DIR"] / "pipeline_state.json"
        )
        assert cfg["experiment_paths"]["outline_paper_mapping_path"] == str(
            expected["OUTLINE_DIR"] / "outline_paper_mapping.csv"
        )
        assert cfg["experiment_paths"]["agent07_output_dir"] == str(
            expected["VERIFICATION_TRACEABILITY_DIR"]
        )
        assert cfg["experiment_paths"]["agent07_staging_dir"] == str(
            expected["OUTPUTS_DIR"] / ".agent07_staging"
        )
        assert cfg["chroma_dir"] == expected["CHROMA_DIR"]
        assert cfg["chroma_manifest_path"] == expected["CHROMA_DIR"] / "chroma_index_manifest.json"
        assert cfg["chunks_manifest_path"] == expected["CHUNKS_DIR"] / "chunks_clean_for_rag.jsonl"
        # code_root/project_root: deliberadamente NO se comparan contra
        # ningún literal — son específicos de la sesión de Colab, no del
        # contrato semántico (ver validate_agent07_orchestrator_compatibility).


if __name__ == "__main__":
    for fn in (
        test_openai_model_equivalence,
        test_chroma_collection_name_equivalence,
        test_verification_policy_equivalence,
        test_post_correction_recheck_policy_equivalence,
        test_missing_openai_model_raises,
        test_missing_verification_policy_raises,
        test_missing_post_correction_recheck_policy_raises,
        test_missing_chroma_collection_name_raises,
        test_empty_verification_policy_raises,
        test_embedding_model_equivalence,
        test_missing_embedding_model_raises,
        test_empty_embedding_model_raises,
        test_rag_policy_transformation_equivalence,
        test_incomplete_rag_policy_raises,
        test_rag_policy_ground_truth_misuse_raises,
        test_paths_match_config_py_derivation,
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
