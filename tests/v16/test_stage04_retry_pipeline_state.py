"""Contrato de retry/resume de Stage 04 (Análisis temático).

Causa raíz: ``load_thematic_configuration`` (``thematic_analysis_
runtime.py``) construía ``state_path`` con una ruta HARDCODEADA
(``experiment_dir / "pipeline_state.json"``) que nunca coincide con la
ruta CANÓNICA real que escribe el orquestador (``ensure_pipeline_
state``/``resolve_state_path``, ``pipeline_orchestrator.py``:
``experiment_dir / "05_outputs" / "00_orchestrator_planner" /
"pipeline_state.json"``). El commit del intento 1 SIEMPRE escribía
correctamente en la ruta canónica -- el bug estaba exclusivamente en
dónde el intento 2 buscaba LEER ese estado, nunca en la persistencia
en sí.

El fix agrega ``resolve_thematic_pipeline_state_path`` -- réplica
exacta, en Stage 04, del mismo patrón que ``resolve_pipeline_state_
path`` (Stage 06, ``draft_writing_runtime.py``): ruta canónica primero,
``rglob`` como fallback para esquemas de directorio legacy, error
explícito si es ambiguo -- sin crear ni sobrescribir nada. A diferencia
de Stage 06, nunca levanta si no existe ningún candidato (necesario
porque esta función se usa para construir la configuración de
CUALQUIER intento, y en el intento 1 es legítimo que el archivo
todavía no exista).

``STRUCTURE_TOO_LONG`` en sí NO se toca -- sigue siendo un reason code
legítimo que produce NEEDS_REVISION/RETRY exactamente igual que antes.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.pipeline_orchestrator import ensure_pipeline_state  # noqa: E402
from src.runtime.thematic_analysis_protocol import execute_thematic_runtime_transaction  # noqa: E402
from src.adapters.thematic_analysis_runtime import (  # noqa: E402
    ThematicRuntimeDependencies,
    build_thematic_agent_input,
    load_thematic_configuration,
    parse_json,
    resolve_thematic_pipeline_state_path,
)
from src.agents.thematic_analysis_agent import ThematicAnalysisAgent  # noqa: E402

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


VALID_PAYLOAD = {
    "corpus_summary": {},
    "themes": [{
        "theme_id": "T1", "theme_name": "Modelos", "description": "ANN y SVM",
        "representative_papers": [
            {"source_filename": "a.pdf", "title": "Alpha"},
            {"source_filename": "b.pdf", "title": "Beta"},
        ],
    }],
    "research_gaps": [{
        "gap_id": "G1", "description": "Falta mas evidencia", "basis": "limitations",
        "supporting_sources": ["a.pdf"],
    }],
    "suggested_state_of_art_structure": [
        {"section_id": "S1", "section_title": "Modelos", "recommended_sources": ["a.pdf", "b.pdf"]},
    ],
    "comparative_dimensions": [{
        "dimension": "Metodo", "description": "Compara modelos", "relevant_sources": ["a.pdf", "b.pdf"],
    }],
}


def _build_synthetic_project(tmp_path: Path, *, max_sections=None, experiment_id="exp_synthetic"):
    project_dir = tmp_path
    project_dir.mkdir(parents=True, exist_ok=True)
    active_experiment = {
        "active_experiment_id": experiment_id,
        "experiment_dir": experiment_id,
        "openai_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "chroma_collection_name": "reference_papers_chunks",
        "run_id": experiment_id,
        "rag_policy": {"retrieval_profiles": {"strict": {"top_k": 10, "fetch_k": 40, "max_per_source": 2}}},
        "generation_profile": {"min_sections": 1, "max_sections": max_sections},
        "topic_profile": {"topic": "tema sintetico multidominio"},
        "thematic_analysis_policy": {},
    }
    (project_dir / "active_experiment.json").write_text(json.dumps(active_experiment), encoding="utf-8")

    experiment_dir = project_dir / experiment_id
    kb_dir = experiment_dir / "05_outputs" / "02_scientific_knowledge_base"
    kb_dir.mkdir(parents=True)
    (kb_dir / "scientific_knowledge_base.csv").write_text(
        "source_filename,title,include_in_state_of_art,relevance_level,methods_or_models,limitations_or_gaps\n"
        "a.pdf,Alpha,True,alta,ANN,Need more data\n"
        "b.pdf,Beta,True,alta,SVM,Limited sites\n",
        encoding="utf-8",
    )
    (kb_dir / "scientific_knowledge_base.jsonl").write_text("{}\n", encoding="utf-8")

    extraction_dir = experiment_dir / "05_outputs" / "01_scientific_extraction"
    extraction_dir.mkdir(parents=True)
    (extraction_dir / "scientific_extraction_manifest.json").write_text(
        json.dumps({"experiment_id": experiment_id, "stage": "03_agente_extraccion_kb", "safety_policy": {"uses_ground_truth": False}}),
        encoding="utf-8",
    )
    return project_dir, experiment_id


def _build_execution(project_dir, attempt_number, payload):
    def _build():
        configuration = load_thematic_configuration(str(project_dir), attempt_number)
        deps = ThematicRuntimeDependencies(lambda prompt: json.dumps(payload), parse_json)
        return ThematicAnalysisAgent(deps), build_thematic_agent_input(configuration)
    return _build


@scenario("T04-01. Intento 1 APPROVED -> no retry (camino normal sin cambios)")
def test_t04_01_approved_no_retry():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=5)
        store = ensure_pipeline_state(str(project_dir))
        result = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 1, VALID_PAYLOAD), attempt_number=1,
        )
        assert result.agent_result.execution_status.value == "COMPLETED"
        assert result.agent_result.quality_status.value == "APPROVED"
        assert result.agent_result.requested_transition.action.value == "ADVANCE"


@scenario("T04-02. Intento 1 NEEDS_REVISION + STRUCTURE_TOO_LONG -> intento 2 recibe pipeline_state real, sin RuntimeError")
def test_t04_02_attempt2_receives_pipeline_state():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=0)
        store = ensure_pipeline_state(str(project_dir))

        result1 = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 1, VALID_PAYLOAD), attempt_number=1,
        )
        assert result1.agent_result.execution_status.value == "COMPLETED"
        assert result1.agent_result.quality_status.value == "NEEDS_REVISION"
        assert "STRUCTURE_TOO_LONG" in result1.agent_result.failure_reason_codes
        assert result1.agent_result.requested_transition.action.value == "RETRY"

        result2 = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 2, VALID_PAYLOAD), attempt_number=2,
        )
        # El bug reportado producía exactamente esto:
        assert result2.agent_result.execution_status.value == "COMPLETED"
        assert result2.agent_result.error is None


@scenario("T04-03. pipeline_state del intento 1 se conserva físicamente después del COMMIT, en la ruta canónica real")
def test_t04_03_pipeline_state_persists_after_commit():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=0)
        store = ensure_pipeline_state(str(project_dir))
        execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 1, VALID_PAYLOAD), attempt_number=1,
        )
        canonical = project_dir / experiment_id / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
        assert canonical.is_file()
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        stage = payload["stages"]["04_agente_analisis_tematico"]
        assert stage["requested_transition"]["action"] == "RETRY"
        assert "STRUCTURE_TOO_LONG" in stage["failure_reason_codes"]


@scenario("T04-04. Intento 2 sin pipeline_state válido sigue fallando fail-closed -- la protección se mantiene, no se elimina")
def test_t04_04_attempt2_without_state_still_fails_closed():
    # Caso A: ningún pipeline_state.json existe en absoluto (verificado
    # directamente sobre _previous_attempt_from_state, sin pasar por
    # ensure_pipeline_state -- que crearía el archivo al abrir el store).
    from src.adapters.thematic_analysis_runtime import _previous_attempt_from_state
    with tempfile.TemporaryDirectory() as tmp:
        missing_state_path = Path(tmp) / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
        try:
            _previous_attempt_from_state({"attempt_number": 2, "state_path": missing_state_path})
            raise AssertionError("debía fallar sin ningún pipeline_state.json")
        except RuntimeError as exc:
            assert "El intento 2 requiere pipeline_state del intento 1" in str(exc)

    # Caso B: pipeline_state.json existe (el store ya fue inicializado)
    # pero SIN una transición RETRY persistida para Stage 04 -- fail-closed
    # con un mensaje distinto, igualmente protector.
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=5)
        # NUNCA se ejecuta el intento 1 -- solo se inicializa el store.
        store = ensure_pipeline_state(str(project_dir))
        result2 = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 2, VALID_PAYLOAD), attempt_number=2,
        )
        assert result2.agent_result.execution_status.value == "FAILED"
        assert result2.agent_result.error["type"] == "RuntimeError"
        assert "El intento 2 requiere" in result2.agent_result.error["message"]


@scenario("T04-05. El estado leído pertenece al mismo experiment_id -- nunca se reutiliza el pipeline_state de otro experimento")
def test_t04_05_state_belongs_to_same_experiment():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_a, exp_a = _build_synthetic_project(tmp_path / "proj_a", max_sections=0, experiment_id="exp_a")
        project_b, exp_b = _build_synthetic_project(tmp_path / "proj_b", max_sections=0, experiment_id="exp_b")

        store_a = ensure_pipeline_state(str(project_a))
        execute_thematic_runtime_transaction(
            store=store_a, build_execution=_build_execution(project_a, 1, VALID_PAYLOAD), attempt_number=1,
        )
        # El experimento B nunca ejecutó nada -- su propia resolución de
        # estado NUNCA debe encontrar el pipeline_state de A (proyectos
        # distintos, directorios distintos).
        resolved_b = resolve_thematic_pipeline_state_path(project_b, exp_b)
        assert not resolved_b.is_file()
        resolved_a = resolve_thematic_pipeline_state_path(project_a, exp_a)
        assert resolved_a.is_file()
        assert resolved_a != resolved_b
        assert exp_a in str(resolved_a)
        assert exp_b not in str(resolved_a)


@scenario("T04-06. El retry de Stage 04 no reinicia Stage 03/03B -- solo lee sus artefactos ya existentes, nunca los reconstruye")
def test_t04_06_retry_never_touches_stage03_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=0)
        kb_csv = project_dir / experiment_id / "05_outputs" / "02_scientific_knowledge_base" / "scientific_knowledge_base.csv"
        manifest = project_dir / experiment_id / "05_outputs" / "01_scientific_extraction" / "scientific_extraction_manifest.json"
        mtime_kb_before = kb_csv.stat().st_mtime_ns
        mtime_manifest_before = manifest.stat().st_mtime_ns

        store = ensure_pipeline_state(str(project_dir))
        execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 1, VALID_PAYLOAD), attempt_number=1,
        )
        execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 2, VALID_PAYLOAD), attempt_number=2,
        )

        assert kb_csv.stat().st_mtime_ns == mtime_kb_before
        assert manifest.stat().st_mtime_ns == mtime_manifest_before


@scenario("T04-07. Regresión de Stage 04 'limpia': secuencia real de retry con feedback STRUCTURE_TOO_LONG usado, freshness/fingerprint estable dentro del mismo intento")
def test_t04_07_clean_regression_full_flow():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, experiment_id = _build_synthetic_project(Path(tmp), max_sections=0)
        store = ensure_pipeline_state(str(project_dir))

        result1 = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 1, VALID_PAYLOAD), attempt_number=1,
        )
        assert result1.agent_result.attempt_number == 1

        configuration2 = load_thematic_configuration(str(project_dir), 2)
        agent_input2 = build_thematic_agent_input(configuration2)
        # El intento 2 recibe el feedback real del intento 1 (no repite desde cero, sabe por qué falló).
        assert agent_input2.previous_attempt is not None
        assert "STRUCTURE_TOO_LONG" in agent_input2.previous_attempt.failure_reason_codes
        assert agent_input2.previous_attempt.quality_status == "NEEDS_REVISION"

        result2 = execute_thematic_runtime_transaction(
            store=store, build_execution=_build_execution(project_dir, 2, VALID_PAYLOAD), attempt_number=2,
        )
        assert result2.agent_result.execution_status.value == "COMPLETED"
        assert result2.agent_result.attempt_number == 2


if __name__ == "__main__":
    for fn in (
        test_t04_01_approved_no_retry,
        test_t04_02_attempt2_receives_pipeline_state,
        test_t04_03_pipeline_state_persists_after_commit,
        test_t04_04_attempt2_without_state_still_fails_closed,
        test_t04_05_state_belongs_to_same_experiment,
        test_t04_06_retry_never_touches_stage03_artifacts,
        test_t04_07_clean_regression_full_flow,
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
