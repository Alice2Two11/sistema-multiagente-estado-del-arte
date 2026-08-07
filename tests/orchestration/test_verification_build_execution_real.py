"""Prueba de `build_experimental_verification_execution` atravesando el constructor real.

A diferencia de test_verification_stagespec_integration.py (que arma
``dependencies``/``runtime_input`` manualmente para probar el StageSpec), acá
se llama al constructor real completo con factories deterministas inyectadas
para LLM, embeddings y colección Chroma — así se prueba la construcción en
sí (carga de active_experiment.json, resolución del experimento 06,
construcción de agent07_config, del retriever, de Agent07RuntimeInput, de la
compatibilidad, de los fingerprints), no un runtime armado a mano.

No llama a OpenAI ni a Chroma real.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.agent06_verification_handoff import Agent07RetrieverBinding
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever
from src.adapters.verification_orchestrator_runtime import (
    build_experimental_verification_execution,
)
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.state.fingerprints import sha256_file
from src.state.pipeline_state import (
    ArtifactState,
    DecisionLogEntry,
    PipelineIdentity,
    PipelineState,
    StageState,
)
from src.state.state_store import StateStore

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "agent06_v17_e2e_snapshot"
DRAFT = "06_agente_redactor"
VERIFY = "07_agente_verificador"

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


class FakeLLM:
    """Doble determinista de ChatOpenAI: no llama a ninguna red."""

    def __init__(self, *, model, temperature):
        self.model = model
        self.temperature = temperature

    def invoke(self, *args, **kwargs):
        raise AssertionError("no debería invocarse en esta prueba (solo se prueba la construcción)")


class FakeChromaCollection:
    def query(self, *, query_texts, n_results):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class FakeChromaClient:
    def __init__(self, *, path):
        self.path = path

    def get_collection(self, *, name, embedding_function):
        return FakeChromaCollection()


def _fake_embedding_function_factory(*, model_name):
    return {"model_name": model_name}


def _seed_project(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    experiment_id = "exp_build_execution"
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outputs.mkdir(parents=True)
    state_path = outputs / "00_orchestrator_planner" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True)

    active_experiment = {
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
    (root / "active_experiment.json").write_text(json.dumps(active_experiment), encoding="utf-8")

    # Artefactos mínimos reales de 06 (mismo fixture que las otras suites).
    names = (
        "state_of_art_draft.json",
        "state_of_art_draft.md",
        "draft_sections.csv",
        "draft_rag_evidence.csv",
        "draft_claim_evidence.csv",
        "numeric_hallucination_check.csv",
        "draft_validation_report.json",
        "draft_generation_manifest.json",
    )
    artifacts_dir = tmp_path / "agent06_artifacts"
    artifacts_dir.mkdir()
    refs = {}
    for name in names:
        target = artifacts_dir / name
        target.write_bytes((FIXTURE_DIR / name).read_bytes())
        refs[name] = ArtifactReference(str(target), sha256_file(target))

    now = "2026-01-01T00:00:00+00:00"
    result = AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo("OK", "ok"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(TransitionAction.ADVANCE, VERIFY, "OK", False),
        output_artifacts=refs,
        tool_usage=ToolUsage(),
        attempt_number=1,
        started_at=now,
        completed_at=now,
    )
    log = DecisionLogEntry(
        "d06", now, DRAFT, DRAFT, 1, {}, {"code": "OK"}, (), None, result.to_dict()
    )
    state = PipelineState(
        identity=PipelineIdentity(experiment_id, "run_build_execution", now, now, "v1"),
        stages={DRAFT: StageState(execution_status=ExecutionStatus.COMPLETED)},
        artifacts={name: ArtifactState(ref, now) for name, ref in refs.items()},
        decision_log=(log,),
    )
    StateStore(state_path).initialize(state)

    outline_dir = outputs / "04_outline"
    outline_dir.mkdir(parents=True)
    (outline_dir / "outline_paper_mapping.csv").write_bytes(
        (FIXTURE_DIR / "outline_paper_mapping.csv").read_bytes()
    )

    # Manifest y chunks mínimos de Chroma para el retriever (rutas por
    # defecto de load_verification_configuration).
    chroma_dir = experiment_dir / "04_chroma_index"
    chroma_dir.mkdir(parents=True)
    chroma_manifest_path = chroma_dir / "chroma_index_manifest.json"
    chroma_manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "collection_name": NOTEBOOK_00_CHROMA_COLLECTION_NAME,
                "embedding_model": "all-MiniLM-L6-v2",
            }
        ),
        encoding="utf-8",
    )
    chunks_dir = experiment_dir / "03_chunks"
    chunks_dir.mkdir(parents=True)
    chunks_manifest_path = chunks_dir / "chunks_clean_for_rag.jsonl"
    chunks_manifest_path.write_text("{}\n", encoding="utf-8")

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


@scenario("B1. build_experimental_verification_execution real, sin red, produce dependencies/runtime_input válidos")
def test_build_execution_real_with_deterministic_factories():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = _seed_project(Path(tmp))

        dependencies, runtime_input = build_experimental_verification_execution(
            project_dir,
            attempt_number=1,
            llm_factory=FakeLLM,
            chroma_client_factory=FakeChromaClient,
            embedding_function_factory=_fake_embedding_function_factory,
        )

        # --- carga de active_experiment.json + resolución del experimento ---
        assert runtime_input.experiment_paths["root"].endswith("exp_build_execution")
        assert runtime_input.committed_agent06_output["experiment_id"] == "exp_build_execution"

        # --- construcción de agent07_config (los 4 campos bajo prueba) ---
        assert runtime_input.agent07_config["verification_model"] == "gpt-4.1-mini"
        assert runtime_input.agent07_config["correction_model"] == "gpt-4.1-mini"
        assert runtime_input.agent07_config["reverification_model"] == "gpt-4.1-mini"
        assert runtime_input.agent07_config["chroma_collection_name"] == "reference_papers_chunks"
        for key, value in NOTEBOOK_00_FIXED_VERIFICATION_POLICY.items():
            assert runtime_input.agent07_config["verification_policy"][key] == value
        for key, value in NOTEBOOK_00_FIXED_POST_CORRECTION_RECHECK_POLICY.items():
            assert runtime_input.agent07_config["reverification_policy"][key] == value

        # --- construcción del retriever incremental (real, conectado) ---
        assert dependencies.retrieval_tool is not None
        assert isinstance(dependencies.retrieval_tool, Agent07ChromaRetriever)
        assert dependencies.retrieval_tool.collection_name == "reference_papers_chunks"
        assert dependencies.retrieval_tool.embedding_model == "all-MiniLM-L6-v2"
        assert isinstance(dependencies.retrieval_tool.collection, FakeChromaCollection)
        assert dependencies.retriever_binding is not None

        # --- construcción de Agent07RuntimeInput ---
        assert runtime_input.policy_versions
        assert runtime_input.schema_versions

        # --- configuración del VerificationAgent (LLM inyectado, no red real) ---
        assert isinstance(dependencies.verification_llm, FakeLLM)
        assert dependencies.verification_llm.model == "gpt-4.1-mini"
        assert isinstance(dependencies.correction_llm, FakeLLM)
        assert isinstance(dependencies.reverification_llm, FakeLLM)

        # --- fingerprints: se pueden calcular sin error a partir del runtime_input real ---
        from src.adapters.verification_notebook import _stage_fingerprints

        fp = _stage_fingerprints(runtime_input)
        assert set(fp.to_dict()) == {"input", "config", "dependencies", "composite"}
        assert all(fp.to_dict().values())

        # --- rutas y artefactos: experiment_paths completo y consistente ---
        for key in (
            "code_root",
            "project_root",
            "experiment_root",
            "root",
            "pipeline_state_path",
            "outline_paper_mapping_path",
            "agent07_output_dir",
            "agent07_staging_dir",
            "chroma_dir",
            "chunks_dir",
        ):
            assert runtime_input.experiment_paths.get(key), f"falta {key} en experiment_paths"

        # --- compatibilidad del experimento: si esta línea se alcanzó,
        # validate_agent07_orchestrator_compatibility ya se ejecutó sin
        # lanzar dentro del constructor real ---


@scenario("B2. build_experimental_verification_execution real: falla igual que config.py real si falta una clave obligatoria")
def test_build_execution_real_fails_on_missing_required_key():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = _seed_project(Path(tmp))
        active_path = project_dir / "active_experiment.json"
        active = json.loads(active_path.read_text(encoding="utf-8"))
        del active["verification_policy"]
        active_path.write_text(json.dumps(active), encoding="utf-8")

        from src.adapters.verification_orchestrator_runtime import (
            MissingRequiredActiveExperimentKeyError,
        )

        try:
            build_experimental_verification_execution(
                project_dir,
                attempt_number=1,
                llm_factory=FakeLLM,
                chroma_client_factory=FakeChromaClient,
                embedding_function_factory=_fake_embedding_function_factory,
            )
        except MissingRequiredActiveExperimentKeyError:
            pass
        else:
            raise AssertionError("debía fallar por falta de verification_policy obligatoria")


if __name__ == "__main__":
    for fn in (
        test_build_execution_real_with_deterministic_factories,
        test_build_execution_real_fails_on_missing_required_key,
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
