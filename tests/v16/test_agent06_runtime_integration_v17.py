from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.adapters.draft_writing_runtime import (
    HYBRID_RUNTIME_VERSIONS,
    LEGACY_RUNTIME_VERSIONS,
    REQUIRED_DRAFT_ARTIFACTS,
    build_draft_agent_input,
    build_real_draft_execution,
    build_runtime_draft_policy,
    commit_executed_draft,
    execute_prepared_draft,
    load_draft_configuration,
    prepare_draft_execution,
    resume_draft_execution,
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
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore


class _Collection:
    def query(self, **kwargs):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class _Runtime:
    def __init__(self, collection):
        self.collection = collection

    def invoke(self, prompt):
        raise AssertionError("LLM real must not be called")

    def parse(self, raw):
        return raw


def _write_project(root: Path, *, quantitative: bool = True) -> tuple[Path, StateStore]:
    experiment_id = "exp"
    experiment = root / experiment_id
    outputs = experiment / "05_outputs"
    outline = outputs / "04_outline"
    thematic = outputs / "03_thematic_analysis"
    rag = outputs / "01_rag"
    state_dir = outputs / "00_orchestrator_planner"
    chunks = experiment / "03_chunks"
    chroma = experiment / "04_chroma_index"
    for path in (outline, thematic, rag, state_dir, chunks, chroma):
        path.mkdir(parents=True, exist_ok=True)
    (chroma / "chroma.sqlite3").write_bytes(b"fake")

    active = {
        "active_experiment_id": experiment_id,
        "run_id": "run",
        "openai_model": "stub",
        "embedding_model_name": "emb",
        "chroma_collection_name": "col",
        "generation_profile": {
            "output_language": "español",
            "writing_mode": "síntesis",
            "focus_mode": "comparativo",
            "citation_style": "trazable",
            "target_total_words": 1000,
            "min_total_words": 650,
            "max_total_words": 1400,
        },
        "draft_generation_policy": {},
    }
    (root / "active_experiment.json").write_text(json.dumps(active), encoding="utf-8")

    outline_payload = {
        "title": "Draft",
        "sections": [
            {
                "section_id": "S1",
                "section_title": "Introduction",
                "section_type": "organizational",
                "requires_sources": False,
                "papers_to_use": [],
            },
            {
                "section_id": "S2",
                "section_title": "Methods",
                "section_type": "substantive",
                "requires_sources": True,
                "papers_to_use": [{"source_filename": "a.pdf"}],
            },
        ],
    }
    (outline / "state_of_art_outline.json").write_text(json.dumps(outline_payload), encoding="utf-8")
    (outline / "outline_validation_report.json").write_text(json.dumps({"validation_ok": True}), encoding="utf-8")
    (outline / "outline_generation_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "safety_policy": {"uses_ground_truth": False}}), encoding="utf-8")
    pd.DataFrame([{"section_id": "S2", "source_filename": "a.pdf"}]).to_csv(outline / "outline_paper_mapping.csv", index=False)
    (thematic / "thematic_analysis_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "safety_policy": {"uses_ground_truth": False}}), encoding="utf-8")
    (thematic / "thematic_validation_report.json").write_text(json.dumps({"validation_ok": True}), encoding="utf-8")
    pd.DataFrame([{"source_filename": "a.pdf", "title": "A"}]).to_csv(thematic / "kb_final_for_thematic_analysis.csv", index=False)
    pd.DataFrame([{"source_filename": "a.pdf", "chunk_id": "c1", "text": "accuracy 95%"}]).to_csv(chunks / "chunks_clean_for_rag.csv", index=False)
    (rag / "chroma_index_manifest.json").write_text(json.dumps({"experiment_id": experiment_id}), encoding="utf-8")

    if quantitative:
        qdir = outputs / "02_scientific_knowledge_base"
        qdir.mkdir(parents=True)
        pd.DataFrame([{"source_filename": "a.pdf", "chunk_id": "c1", "value": "95%", "verification_status": "confirmed_in_source_chunk"}]).to_csv(qdir / "quantitative_comparative_table.csv", index=False)
        pd.DataFrame([{"source_filename": "a.pdf", "dataset": "D"}]).to_csv(qdir / "dataset_technique_summary.csv", index=False)
        (qdir / "quantitative_extraction_manifest.json").write_text(json.dumps({"experiment_id": experiment_id}), encoding="utf-8")

    state = PipelineState(identity=PipelineIdentity(experiment_id=experiment_id, run_id="run", created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z", schema_version="1.0"))
    store = StateStore(state_dir / "pipeline_state.json")
    store.initialize(state)
    return experiment, store


def _client_factory(path=None, **kwargs):
    return type("Client", (), {"list_collections": lambda self: [type("Named", (), {"name": "col"})()]})()


def _approved_result(agent_input, output_dir: Path) -> AgentResult:
    artifacts = {}
    versions = LEGACY_RUNTIME_VERSIONS if agent_input.policy["retrieval_strategy"].startswith("legacy") else HYBRID_RUNTIME_VERSIONS
    for name in REQUIRED_DRAFT_ARTIFACTS:
        path = output_dir / name
        if name.endswith(".json"):
            payload = {"ok": True}
            if name == "draft_generation_manifest.json":
                payload = {
                    "fingerprint": agent_input.policy["current_fingerprint"],
                    "versions": {
                        "stage": versions["stage_version"],
                        "rag": versions["rag_version"],
                        "validation": versions["validation_version"],
                    },
                }
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text("x", encoding="utf-8")
        artifacts[name] = ArtifactReference(str(path), "hash")
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo(code="DRAFT_APPROVED", rationale="ok"),
        quality_metrics={"technical": {"validation_ok": True}},
        warnings=(),
        requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, target_stage=None, reason_code="APPROVED"),
        output_artifacts=artifacts,
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
    )


def _revision_result(agent_input) -> AgentResult:
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code="DRAFT_NEEDS_REVISION", rationale="invalid"),
        quality_metrics={"technical": {"validation_ok": False}},
        warnings=(),
        requested_transition=RequestedTransition(action=TransitionAction.RETRY, target_stage="06_agente_redactor", reason_code="REVISION"),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=agent_input.attempt_number,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
    )


class TestAgent06RuntimeIntegrationV17(unittest.TestCase):
    def _cfg(self, root: Path, overrides=None, quantitative=True):
        _write_project(root, quantitative=quantitative)
        return load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides=overrides)

    def test_default_policy_is_legacy(self):
        self.assertEqual(build_runtime_draft_policy()["retrieval_strategy"], "legacy_chroma_then_csv_restricted")

    def test_explicit_hybrid_policy_has_v17_versions(self):
        policy = build_runtime_draft_policy({"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})
        for key, value in HYBRID_RUNTIME_VERSIONS.items():
            self.assertEqual(policy[key], value)

    def test_contradictory_declared_versions_are_overridden(self):
        policy = build_runtime_draft_policy({"retrieval_strategy": "legacy_chroma_then_csv_restricted", "stage_version": "fake", "rag_version": "fake"})
        self.assertEqual(policy["stage_version"], LEGACY_RUNTIME_VERSIONS["stage_version"])
        self.assertEqual(policy["rag_version"], LEGACY_RUNTIME_VERSIONS["rag_version"])

    def test_same_configuration_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg1 = self._cfg(root)
            fp1 = build_draft_agent_input(cfg1).policy["current_fingerprint"]
            cfg2 = load_draft_configuration(root, chroma_client_factory=_client_factory)
            fp2 = build_draft_agent_input(cfg2).policy["current_fingerprint"]
            self.assertEqual(fp1, fp2)

    def test_key_order_does_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._cfg(root)
            a = {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced", "rrf_k": 60}
            b = {"rrf_k": 60, "retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"}
            fp1 = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides=a)).policy["current_fingerprint"]
            fp2 = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides=b)).policy["current_fingerprint"]
            self.assertEqual(fp1, fp2)

    def test_strategy_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._cfg(root)
            legacy = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory)).policy["current_fingerprint"]
            hybrid = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})).policy["current_fingerprint"]
            self.assertNotEqual(legacy, hybrid)

    def test_each_relevant_parameter_changes_fingerprint(self):
        keys = {
            "candidate_multiplier": 4,
            "rrf_k": 61,
            "max_evidence_chars": 17000,
            "max_candidates_per_source": 23,
            "quantitative_evidence_quota": 1,
            "max_quantitative_rows_per_section": 11,
            "organizational_target_words": 39,
            "organizational_minimum_words": 2,
            "organizational_maximum_words": 81,
            "substantive_minimum_ratio": 0.60,
            "substantive_maximum_ratio": 1.50,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._cfg(root)
            base = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})).policy["current_fingerprint"]
            for key, value in keys.items():
                with self.subTest(key=key):
                    fp = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced", key: value})).policy["current_fingerprint"]
                    self.assertNotEqual(base, fp)

    def test_quota_and_top_k_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._cfg(root)
            base = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})).policy["current_fingerprint"]
            changed = {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced", "top_k_evidence_per_section": 9, "chroma_quota": 4, "csv_quota": 3, "rrf_quota": 2}
            fp = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides=changed)).policy["current_fingerprint"]
            self.assertNotEqual(base, fp)

    def test_agent_input_transports_hybrid_fields_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root, {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})
            agent_input = build_draft_agent_input(cfg)
            for key in ("candidate_multiplier", "chroma_quota", "csv_quota", "rrf_quota", "rrf_k", "max_evidence_chars", "quantitative_evidence_quota", "organizational_target_words"):
                self.assertIn(key, agent_input.policy)
            self.assertIn("quantitative_table", agent_input.dependencies)
            self.assertEqual(agent_input.agent_context.runtime_resources["chroma_collection_name"], "col")
            self.assertTrue(agent_input.agent_context.runtime_resources["chroma_dir"].endswith("04_chroma_index"))

    def test_quantitative_block_is_optional(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root, {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"}, quantitative=False)
            agent_input = build_draft_agent_input(cfg)
            self.assertNotIn("quantitative_table", agent_input.dependencies)

    def test_build_real_execution_uses_single_contractual_agent_and_doubles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._cfg(root)
            collection = _Collection()
            agent, agent_input, cfg = build_real_draft_execution(root, collection_factory=lambda _: collection, runtime_factory=lambda model, temperature, collection, project_dir=None: _Runtime(collection), chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})
            self.assertEqual(agent.__class__.__name__, "DraftWritingAgent")
            self.assertIs(agent.runtime.collection, collection)
            self.assertEqual(agent_input.policy["stage_version"], HYBRID_RUNTIME_VERSIONS["stage_version"])

    def test_prepare_does_not_publish_or_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            state = store.load()
            self.assertEqual(state.pending_execution.decision_id, prepared.decision_id)
            self.assertFalse(Path(agent_input.agent_context.output_directory).exists())

    def test_execute_persists_without_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            output = Path(agent_input.agent_context.output_directory); output.mkdir(parents=True)
            result = _approved_result(agent_input, output)
            agent = Mock(); agent.execute.return_value = result
            executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
            self.assertTrue(Path(executed.persisted_result_path).is_file())
            self.assertIsNotNone(store.load().pending_execution)
            self.assertNotIn("06_agente_redactor", store.load().stages)

    def test_commit_only_approved_complete_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            output = Path(agent_input.agent_context.output_directory); output.mkdir(parents=True)
            agent = Mock(); agent.execute.return_value = _approved_result(agent_input, output)
            executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
            state = commit_executed_draft(store=store, executed=executed)
            self.assertIsNone(state.pending_execution)
            self.assertEqual(state.stages["06_agente_redactor"].quality_status.value, "APPROVED")

    def test_no_commit_for_needs_revision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            agent = Mock(); agent.execute.return_value = _revision_result(agent_input)
            executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
            with self.assertRaisesRegex(RuntimeError, "DRAFT_COMMIT_REQUIRES_APPROVED_RESULT"):
                commit_executed_draft(store=store, executed=executed)

    def test_resume_valid_identical_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            output = Path(agent_input.agent_context.output_directory); output.mkdir(parents=True)
            store.persist_agent_result(prepared.decision_id, _approved_result(agent_input, output))
            resolution = resume_draft_execution(store=store, agent_input=agent_input)
            self.assertEqual(resolution.action, "COMMITTED")

    def test_resume_rejects_different_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            legacy = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=legacy)
            output = Path(legacy.agent_context.output_directory); output.mkdir(parents=True)
            store.persist_agent_result(prepared.decision_id, _approved_result(legacy, output))
            hybrid = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides={"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"}))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_FINGERPRINT_MISMATCH"):
                resume_draft_execution(store=store, agent_input=hybrid)

    def test_resume_rejects_missing_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store = _write_project(root)
            agent_input = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            output = Path(agent_input.agent_context.output_directory); output.mkdir(parents=True)
            result = _approved_result(agent_input, output)
            result.output_artifacts.pop if False else None
            payload = result.to_dict(); payload["output_artifacts"].pop("draft_sections.csv")
            store.persist_agent_result(prepared.decision_id, AgentResult.from_dict(payload))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_COMMIT_INCOMPLETE_ARTIFACTS"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_invalid_policy_override_fails_deterministically(self):
        with self.assertRaisesRegex(ValueError, "DRAFT_POLICY_INVALID_TYPE:rrf_k:expected_integer"):
            build_runtime_draft_policy({"rrf_k": "bad"})


if __name__ == "__main__":
    unittest.main()
