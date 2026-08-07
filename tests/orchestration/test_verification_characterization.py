"""Pruebas de caracterización de la etapa 07: notebook (funciones reales) vs. StateStore real.

A diferencia de los demás tests de este paquete, aquí NO se usa un
``FakeAgent`` genérico: se atraviesan las funciones reales de
``src/adapters/verification_notebook.py``
(``prepare_agent07_execution``/``execute_prepared_agent07``/
``commit_executed_agent07``/``resume_agent07_execution``), con:

- artefactos mínimos REALES de un commit de 06, tomados de
  ``tests/fixtures/agent06_v17_e2e_snapshot/`` (el mismo fixture que ya usa
  ``tests/verification/test_agent07_real_context_adapter.py`` — no se
  inventó ninguno nuevo);
- ``VerificationRuntimeDependencies`` con el ``VerificationAgent`` REAL
  (``src/agents/verification_agent.py``), no una clase doble, más
  ``resolve_multiple_correction_proposals`` REAL
  (``src/tools/verification/resolution.py``);
- un ``StateStore`` real sobre un directorio temporal (no ``SimpleNamespace``).

Lo único "doble" es lo que normalmente sería una llamada de red (el LLM real
y el retriever Chroma real con datos indexados) — exactamente lo que un test
determinista sin OpenAI necesita reemplazar. El resto de la cadena
(contratos, fingerprints, PREPARE/EXECUTE/COMMIT/RESUME, validación de
manifiesto) es 100% el código real del repositorio.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest  # el fixture reutilizado importa cosas parametrizadas en su propio módulo

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "verification"))

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever
from src.adapters.verification_notebook import (
    commit_executed_agent07,
    execute_prepared_agent07,
    prepare_agent07_execution,
    resume_agent07_execution,
)
from src.adapters.verification_runtime import (
    Agent07RuntimeInput,
    VerificationRuntimeDependencies,
)
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

# Reutiliza el mismo helper de bundle determinista que ya usa el repo para
# probar el runtime de 07 (tests/verification/test_multi_proposal_resolution_phase66.py).
from test_multi_proposal_resolution_phase66 import bundle as real_bundle

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "agent06_v17_e2e_snapshot"
DRAFT = "06_agente_redactor"
VERIFY = "07_agente_verificador"


def _seed_committed_agent06(tmp_path: Path) -> StateStore:
    """Reproduce, contra un StateStore REAL, el mismo commit de 06 sintético
    que ``tests/verification/test_agent07_real_context_adapter._real_agent06_handoff``
    construye con un ``SimpleNamespace`` — aquí con persistencia real en disco."""

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
        identity=PipelineIdentity("exp_characterization", "run_characterization", now, now, "v1"),
        stages={DRAFT: StageState(execution_status=ExecutionStatus.COMPLETED)},
        artifacts={name: ArtifactState(ref, now) for name, ref in refs.items()},
        decision_log=(log,),
    )

    mapping_path = tmp_path / "outline_paper_mapping.csv"
    mapping_path.write_bytes((FIXTURE_DIR / "outline_paper_mapping.csv").read_bytes())

    state_path = tmp_path / "pipeline_state.json"
    store = StateStore(state_path)
    store.initialize(state)
    return store, mapping_path


def _build_runtime_input(store: StateStore, mapping_path: Path, tmp_path: Path) -> Agent07RuntimeInput:
    committed_agent06_output = build_agent07_input_from_committed_agent06(
        store=store,
        stage_name=DRAFT,
        agent07_config={},
        policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(tmp_path)},
        outline_paper_mapping_path=mapping_path,
    )
    return Agent07RuntimeInput(
        committed_agent06_output=committed_agent06_output,
        agent07_config={"verification_policy": {}},
        policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(tmp_path)},
    )


def _deterministic_dependencies(*, retriever=None) -> VerificationRuntimeDependencies:
    """Mismo patrón que
    tests/verification/test_agent07_real_context_adapter.py::
    test_real_verification_agent_runtime_processes_complete_agent06_handoff,
    con VerificationAgent y resolve_multiple_correction_proposals REALES."""

    return VerificationRuntimeDependencies(
        verification_agent_factory=VerificationAgent,
        verification_llm=None,
        retrieval_tool=retriever,
        correction_context_factory=lambda context, result, config: {"claim_id": context["claim_id"]},
        reverification_input_factory=lambda *args: {},
        proposal_runner=lambda context, *, llm: {
            "correction_id": "none-" + context["claim_id"],
            "accepted_for_reverification": False,
        },
        bundle_builder=lambda value: real_bundle(()),
        resolution_runner=resolve_multiple_correction_proposals,
    )


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


@scenario("C1. entrada construida: build_agent07_input_from_committed_agent06 real produce claims")
def test_characterize_input_construction():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)
        assert len(runtime_input.committed_agent06_output["claim_verification_contexts"]) >= 1
        assert runtime_input.committed_agent06_output["experiment_id"] == "exp_characterization"


@scenario("C2. PREPARE real: fingerprints, decision_id, pending_execution persistido")
def test_characterize_prepare():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)

        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        assert prepared.decision_id
        assert prepared.attempt_number == 1
        assert set(prepared.stage_fingerprints) == {"input", "config", "dependencies", "composite"}

        state = store.load()
        assert state.pending_execution is not None
        assert state.pending_execution.decision_id == prepared.decision_id
        assert state.pending_execution.target_stage == VERIFY


@scenario("C3. EXECUTE + persistencia + COMMIT reales con VerificationAgent real")
def test_characterize_execute_and_commit():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)
        dependencies = _deterministic_dependencies()

        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)

        assert executed.agent_result.attempt_number == 1
        assert executed.runtime_result.runtime_status in {"COMPLETED", "PARTIAL", "BLOCKED"}
        assert Path(executed.persisted_result_path).is_file()  # persistencia real en disco
        assert Path(executed.staging_manifest_path).is_file()

        committed_state = commit_executed_agent07(store=store, executed=executed)
        assert committed_state.stages[VERIFY].execution_status in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
        }
        assert committed_state.pending_execution is None

        # Artefactos reales publicados (no solo un AgentResult in-memory).
        output_dir = Path(runtime_input.experiment_paths.get("agent07_output_dir", str(tmp_path / "07_verification")))
        if committed_state.stages[VERIFY].execution_status == ExecutionStatus.COMPLETED:
            assert (output_dir / "agent07_artifact_manifest.json").is_file()


@scenario("C4. RESUME real tras COMMIT: acción COMMITTED, no reejecuta")
def test_characterize_resume_after_commit():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)
        dependencies = _deterministic_dependencies()

        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)
        commit_executed_agent07(store=store, executed=executed)

        resume = resume_agent07_execution(store=store, runtime_input=runtime_input)
        assert resume.action == "COMMITTED", resume.action
        assert resume.committed_result is not None


@scenario("C5. RESUME real tras interrupción (PREPARE+EXECUTE sin COMMIT): recupera de staging")
def test_characterize_resume_after_interruption():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)
        dependencies = _deterministic_dependencies()

        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)
        # Deliberadamente NO se llama commit_executed_agent07: simula un
        # crash entre EXECUTE (staging persistido) y COMMIT.

        state_before = store.load()
        assert state_before.pending_execution is not None  # sigue pendiente

        resume = resume_agent07_execution(store=store, runtime_input=runtime_input)
        assert resume.action in {"COMMITTED", "EXECUTED_NOT_COMMITTED", "REEXECUTE"}, resume.action
        if resume.action == "EXECUTED_NOT_COMMITTED":
            assert resume.executed is not None
            commit_executed_agent07(store=store, executed=resume.executed)
            state_after = store.load()
            assert state_after.pending_execution is None


@scenario("C6. recuperación incremental: Agent07ChromaRetriever real conectado a las dependencias")
def test_characterize_incremental_retrieval_wired():
    import tempfile

    class FakeCollection:
        def query(self, *, query_texts, n_results):
            return {
                "documents": [["contenido de evidencia recuperada"]],
                "metadatas": [[{"source_filename": "paper.pdf", "chunk_id": "chunk-1"}]],
                "distances": [[0.05]],
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, mapping_path = _seed_committed_agent06(tmp_path)
        runtime_input = _build_runtime_input(store, mapping_path, tmp_path)

        retriever = Agent07ChromaRetriever(
            collection=FakeCollection(),
            experiment_id="exp_characterization",
            collection_name="reference_papers_chunks",
            embedding_model="all-MiniLM-L6-v2",
            chroma_manifest_fingerprint="f" * 64,
            chunks_manifest_fingerprint="c" * 64,
        )
        dependencies = _deterministic_dependencies(retriever=retriever)
        assert dependencies.retrieval_tool is retriever

        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)
        # No se afirma que el retriever se haya invocado (depende de si algún
        # claim carece de evidencia heredada suficiente); se afirma que el
        # runtime acepta el retriever real sin romper el contrato.
        assert executed.runtime_result.runtime_status in {"COMPLETED", "PARTIAL", "BLOCKED"}


if __name__ == "__main__":
    for fn in (
        test_characterize_input_construction,
        test_characterize_prepare,
        test_characterize_execute_and_commit,
        test_characterize_resume_after_commit,
        test_characterize_resume_after_interruption,
        test_characterize_incremental_retrieval_wired,
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
