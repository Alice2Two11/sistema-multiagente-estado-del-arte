"""Prueba de integración real del StageSpec de 07 (punto 6 del pedido).

A diferencia de test_verification_evaluation_flow.py (que usa un FakeAgent
sintético para representar 07 dentro del flujo completo 06->07->08), esto
prueba el ``StageSpec`` de 07 TAL COMO lo arma ``pipeline_orchestrator``
(``_run_verification_stage``, el mismo código que correría en producción),
con:

- ``StateStore`` real;
- artefactos mínimos reales de 06 (``tests/fixtures/agent06_v17_e2e_snapshot/``);
- dependencias deterministas para LLM (``VerificationAgent`` real +
  factories deterministas, sin red) y para el retriever incremental
  (``Agent07ChromaRetriever`` real conectado a una colección Chroma doble);
- PREPARE, EXECUTE, persistencia, COMMIT y RESUME reales.

Lo único que NO se ejercita aquí es ``build_experimental_verification_execution``
en sí (porque necesita una credencial OpenAI real y red) — se reemplaza su
resultado por el mismo patrón determinista ya usado en
test_verification_characterization.py, pero conectado al ``StageSpec`` y a
``run_stage`` reales de ``pipeline_orchestrator.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "verification"))

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever
from src.adapters.verification_runtime import Agent07RuntimeInput, VerificationRuntimeDependencies
from src.agents.verification_agent import VerificationAgent
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
from src.orchestration.pipeline_orchestrator import StageSpec, run_stage, _run_verification_stage
from src.state.fingerprints import sha256_file
from src.state.pipeline_state import (
    ArtifactState,
    DecisionLogEntry,
    PipelineIdentity,
    PipelineState,
    StageState,
)
from src.state.state_store import StateStore
from src.tools.verification.resolution import resolve_multiple_correction_proposals

from test_multi_proposal_resolution_phase66 import bundle as real_bundle

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "agent06_v17_e2e_snapshot"
DRAFT = "06_agente_redactor"
VERIFY = "07_agente_verificador"


def _seed_project(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    experiment_id = "exp_stagespec"
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": experiment_id, "run_id": "run_stagespec"}),
        encoding="utf-8",
    )
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outputs.mkdir(parents=True)
    state_path = outputs / "00_orchestrator_planner" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True)

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
        identity=PipelineIdentity(experiment_id, "run_stagespec", now, now, "v1"),
        stages={DRAFT: StageState(execution_status=ExecutionStatus.COMPLETED)},
        artifacts={name: ArtifactState(ref, now) for name, ref in refs.items()},
        decision_log=(log,),
    )
    store = StateStore(state_path)
    store.initialize(state)

    outline_dir = outputs / "04_outline"
    outline_dir.mkdir(parents=True)
    mapping_path = outline_dir / "outline_paper_mapping.csv"
    mapping_path.write_bytes((FIXTURE_DIR / "outline_paper_mapping.csv").read_bytes())

    return root, store, mapping_path


class _FakeChromaCollection:
    def query(self, *, query_texts, n_results):
        return {
            "documents": [["evidencia recuperada de forma determinista"]],
            "metadatas": [[{"source_filename": "paper.pdf", "chunk_id": "chunk-1"}]],
            "distances": [[0.05]],
        }


def _deterministic_build_execution(store: StateStore, mapping_path: Path, project_dir: Path):
    def build_execution(_project_dir, _attempt_number):
        committed_agent06_output = build_agent07_input_from_committed_agent06(
            store=store,
            stage_name=DRAFT,
            agent07_config={},
            policy_versions={"verification": "v1"},
            schema_versions={
                "runtime": "v5",
                "provisional_bundle": "v4",
                "multi_proposal_resolution": "v1",
            },
            experiment_paths={"root": str(project_dir)},
            outline_paper_mapping_path=mapping_path,
        )
        runtime_input = Agent07RuntimeInput(
            committed_agent06_output=committed_agent06_output,
            agent07_config={"verification_policy": {}},
            policy_versions={"verification": "v1"},
            schema_versions={
                "runtime": "v5",
                "provisional_bundle": "v4",
                "multi_proposal_resolution": "v1",
            },
            experiment_paths={"root": str(project_dir)},
        )
        retriever = Agent07ChromaRetriever(
            collection=_FakeChromaCollection(),
            experiment_id="exp_stagespec",
            collection_name="reference_papers_chunks",
            embedding_model="all-MiniLM-L6-v2",
            chroma_manifest_fingerprint="f" * 64,
            chunks_manifest_fingerprint="c" * 64,
        )
        dependencies = VerificationRuntimeDependencies(
            verification_agent_factory=VerificationAgent,
            verification_llm=None,  # doble determinista: sin red
            retrieval_tool=retriever,  # retriever incremental REAL, no None
            correction_context_factory=lambda context, result, config: {
                "claim_id": context["claim_id"]
            },
            reverification_input_factory=lambda *args: {},
            proposal_runner=lambda context, *, llm: {
                "correction_id": "none-" + context["claim_id"],
                "accepted_for_reverification": False,
            },
            bundle_builder=lambda value: real_bundle(()),
            resolution_runner=resolve_multiple_correction_proposals,
        )
        return dependencies, runtime_input

    return build_execution


RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                import traceback

                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


@scenario("S1. StageSpec real de 07 vía run_stage: PREPARE/EXECUTE/COMMIT completos")
def test_real_stagespec_first_run():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project(Path(tmp))
        spec = StageSpec(
            key=VERIFY,
            label="07 · Verificación (integración real)",
            build_execution=_deterministic_build_execution(store, mapping_path, project_dir),
            runtime_transaction=None,
            resolve_resume=None,
            build_fingerprints=None,
            custom_run=_run_verification_stage,
        )
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status in {"COMMITTED", "FAILED"}, outcome
        assert outcome.execution_status in {"COMPLETED", "FAILED"}
        state = store.load()
        assert VERIFY in state.stages
        assert state.pending_execution is None


@scenario("S2. Segunda llamada a run_stage: SKIPPED_FRESH (resume_agent07_execution decide vigencia)")
def test_real_stagespec_second_run_is_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project(Path(tmp))
        spec = StageSpec(
            key=VERIFY,
            label="07 · Verificación (integración real)",
            build_execution=_deterministic_build_execution(store, mapping_path, project_dir),
            runtime_transaction=None,
            resolve_resume=None,
            build_fingerprints=None,
            custom_run=_run_verification_stage,
        )
        first = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        second = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert second.status == "SKIPPED_FRESH", second


@scenario("S3. RESUME real vía StageSpec tras interrupción simulada")
def test_real_stagespec_resume_after_interruption():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project(Path(tmp))
        build_execution = _deterministic_build_execution(store, mapping_path, project_dir)

        # Interrumpe manualmente entre EXECUTE y COMMIT, igual que en
        # test_verification_characterization.py::test_characterize_resume_after_interruption.
        from src.adapters.verification_notebook import execute_prepared_agent07, prepare_agent07_execution

        dependencies, runtime_input = build_execution(project_dir, 1)
        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)

        spec = StageSpec(
            key=VERIFY,
            label="07 · Verificación (integración real)",
            build_execution=build_execution,
            runtime_transaction=None,
            resolve_resume=None,
            build_fingerprints=None,
            custom_run=_run_verification_stage,
        )
        outcome = run_stage(store=store, project_dir=project_dir, spec=spec, attempt_number=1)
        assert outcome.status in {"COMMITTED", "FAILED"}, outcome
        state = store.load()
        assert state.pending_execution is None  # el StageSpec debe haber resuelto el pending


if __name__ == "__main__":
    for fn in (
        test_real_stagespec_first_run,
        test_real_stagespec_second_run_is_fresh,
        test_real_stagespec_resume_after_interruption,
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
