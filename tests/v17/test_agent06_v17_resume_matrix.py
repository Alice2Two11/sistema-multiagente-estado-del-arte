from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.adapters.draft_writing_runtime import (
    HYBRID_RUNTIME_VERSIONS,
    LEGACY_RUNTIME_VERSIONS,
    REQUIRED_DRAFT_ARTIFACTS,
    build_draft_agent_input,
    commit_executed_draft,
    load_draft_configuration,
    prepare_draft_execution,
    resume_draft_execution,
    ExecutedDraftExecution,
)
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import AgentResult, QualityStatus
from src.state.pipeline_state import ArtifactState
from src.state.state_store import StateStore
from src.state.fingerprints import sha256_file
from tests.v16.test_agent06_runtime_integration_v17 import (
    _approved_result,
    _client_factory,
    _revision_result,
    _write_project,
)


HYBRID = {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"}


class TestAgent06V17ResumeMatrix(unittest.TestCase):
    def _input_store(self, root: Path, overrides=None):
        _, store = _write_project(root, quantitative=True)
        cfg = load_draft_configuration(
            root,
            chroma_client_factory=_client_factory,
            policy_overrides=overrides,
        )
        return build_draft_agent_input(cfg), store

    def _base_result(self, agent_input, output_dir: Path) -> AgentResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = _approved_result(agent_input, output_dir)
        manifest_path = Path(
            result.output_artifacts["draft_generation_manifest.json"].path
        )
        strategy = agent_input.policy["retrieval_strategy"]
        versions = (
            HYBRID_RUNTIME_VERSIONS
            if strategy == "hybrid_chroma_csv_rrf_balanced"
            else LEGACY_RUNTIME_VERSIONS
        )
        payload = {
            "fingerprint": agent_input.policy["current_fingerprint"],
            "retrieval_strategy": strategy,
            "versions": {
                "stage": versions["stage_version"],
                "rag": versions["rag_version"],
                "validation": versions["validation_version"],
                "normalization": versions["normalization_version"],
            },
        }
        if strategy == "hybrid_chroma_csv_rrf_balanced":
            payload["versions"].update(
                quantitative_selection=versions["quantitative_selection_version"],
                budget=versions["budget_version"],
            )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        refs = dict(result.output_artifacts)
        refs["draft_generation_manifest.json"] = ArtifactReference(
            str(manifest_path), "hash"
        )
        return replace(result, output_artifacts=refs)

    def _real_hash_result(self, agent_input, output_dir: Path) -> AgentResult:
        result = self._base_result(agent_input, output_dir)
        refs = {}
        for name, ref in result.output_artifacts.items():
            path = Path(ref.path)
            refs[name] = ArtifactReference(str(path), sha256_file(path))
        return replace(result, output_artifacts=refs)

    def _prepare_result(self, store, agent_input, result):
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        store.persist_agent_result(prepared.decision_id, result)
        return prepared

    def _commit(self, store, agent_input, result):
        prepared = self._prepare_result(store, agent_input, result)
        executed = ExecutedDraftExecution(
            prepared.decision_id,
            agent_input,
            result,
            "unused",
        )
        commit_executed_draft(store=store, executed=executed)

    def _rewrite_manifest(self, result, mutate):
        path = Path(result.output_artifacts["draft_generation_manifest.json"].path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        refs = dict(result.output_artifacts)
        refs["draft_generation_manifest.json"] = ArtifactReference(
            str(path), sha256_file(path)
        )
        return replace(result, output_artifacts=refs)

    def test_pending_valid_candidate_uses_unified_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._prepare_result(store, agent_input, result)
            resolution = resume_draft_execution(store=store, agent_input=agent_input)
            self.assertEqual(resolution.action, "COMMITTED")
            self.assertIsNotNone(resolution.committed_result)

    def test_committed_valid_candidate_is_reused_without_execution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            resolution = resume_draft_execution(store=store, agent_input=agent_input)
            self.assertEqual(resolution.action, "NO_PENDING")
            self.assertEqual(resolution.committed_result.quality_status, QualityStatus.APPROVED)

    def test_policy_changes_rejected_after_commit(self):
        changes = {
            "rrf_k": 61,
            "quantitative_evidence_quota": 1,
            "organizational_target_words": 39,
            "top_k_evidence_per_section": 9,
        }
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                original, store = self._input_store(root, HYBRID)
                result = self._real_hash_result(original, root / "draft")
                self._commit(store, original, result)
                overrides = dict(HYBRID)
                overrides[key] = value
                if key == "top_k_evidence_per_section":
                    overrides.update(chroma_quota=4, csv_quota=3, rrf_quota=2)
                changed = build_draft_agent_input(load_draft_configuration(
                    root,
                    chroma_client_factory=_client_factory,
                    policy_overrides=overrides,
                ))
                with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_FINGERPRINT_MISMATCH"):
                    resume_draft_execution(store=store, agent_input=changed)

    def test_individual_quota_changes_rejected_after_commit(self):
        variants = [
            {"top_k_evidence_per_section": 9, "chroma_quota": 4, "csv_quota": 3, "rrf_quota": 2},
            {"top_k_evidence_per_section": 9, "chroma_quota": 3, "csv_quota": 4, "rrf_quota": 2},
            {"top_k_evidence_per_section": 9, "chroma_quota": 3, "csv_quota": 3, "rrf_quota": 3},
        ]
        for overrides_extra in variants:
            with self.subTest(overrides=overrides_extra), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                original, store = self._input_store(root, HYBRID)
                result = self._real_hash_result(original, root / "draft")
                self._commit(store, original, result)
                changed = build_draft_agent_input(load_draft_configuration(
                    root,
                    chroma_client_factory=_client_factory,
                    policy_overrides={**HYBRID, **overrides_extra},
                ))
                with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_FINGERPRINT_MISMATCH"):
                    resume_draft_execution(store=store, agent_input=changed)

    def test_v17_output_rejects_v16_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hybrid, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(hybrid, root / "draft")
            self._commit(store, hybrid, result)
            legacy = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_VERSION_MISMATCH"):
                resume_draft_execution(store=store, agent_input=legacy)

    def test_v16_output_rejects_v17_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy, store = self._input_store(root)
            result = self._real_hash_result(legacy, root / "draft")
            self._commit(store, legacy, result)
            hybrid = build_draft_agent_input(load_draft_configuration(root, chroma_client_factory=_client_factory, policy_overrides=HYBRID))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_VERSION_MISMATCH"):
                resume_draft_execution(store=store, agent_input=hybrid)

    def test_manifest_missing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            Path(result.output_artifacts["draft_generation_manifest.json"].path).unlink()
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_MANIFEST_NOT_FOUND"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_required_artifact_missing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            Path(result.output_artifacts["draft_sections.csv"].path).unlink()
            with self.assertRaisesRegex(RuntimeError, "DRAFT_COMMIT_INCOMPLETE_ARTIFACTS"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_corrupt_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._base_result(agent_input, root / "draft")
            path = Path(result.output_artifacts["draft_generation_manifest.json"].path)
            path.write_text("{broken", encoding="utf-8")
            refs = dict(result.output_artifacts)
            refs["draft_generation_manifest.json"] = ArtifactReference(str(path), "not-a-sha")
            result = replace(result, output_artifacts=refs)
            self._commit(store, agent_input, result)
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_MANIFEST_INVALID"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_missing_manifest_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._base_result(agent_input, root / "draft")
            result = self._rewrite_manifest(result, lambda payload: payload.pop("fingerprint", None))
            self._commit(store, agent_input, result)
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_FINGERPRINT_MISMATCH"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_reference_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._base_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            state = store.load()
            artifacts = dict(state.artifacts)
            current = artifacts["draft_sections.csv"]
            artifacts["draft_sections.csv"] = ArtifactState(
                reference=ArtifactReference(current.reference.path, "different"),
                created_at=current.created_at,
            )
            store.save(replace(state, artifacts=artifacts))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_ARTIFACT_REFERENCE_MISMATCH"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_sha256_disk_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            Path(result.output_artifacts["draft_sections.csv"].path).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_ARTIFACT_HASH_MISMATCH"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_needs_revision_pending_rejected_and_cancelled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            self._prepare_result(store, agent_input, _revision_result(agent_input))
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_REQUIRES_APPROVED_RESULT"):
                resume_draft_execution(store=store, agent_input=agent_input)
            self.assertIsNone(store.load().pending_execution)

    def test_needs_revision_committed_history_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = _revision_result(agent_input)
            prepared = self._prepare_result(store, agent_input, result)
            store.commit_execution(
                decision_id=prepared.decision_id,
                result=result,
                stage_name=agent_input.stage_name,
                fingerprints={"input": "i", "config": "c", "dependencies": "d", "composite": "x"},
            )
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RESUME_REQUIRES_APPROVED_RESULT"):
                resume_draft_execution(store=store, agent_input=agent_input)

    def test_resume_resolution_interface_is_real(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_input, store = self._input_store(root, HYBRID)
            result = self._real_hash_result(agent_input, root / "draft")
            self._commit(store, agent_input, result)
            resolution = resume_draft_execution(store=store, agent_input=agent_input)
            self.assertIn(resolution.action, {"NO_PENDING", "COMMITTED", "REEXECUTE"})
            self.assertTrue(hasattr(resolution, "state"))
            self.assertTrue(hasattr(resolution, "committed_result"))


if __name__ == "__main__":
    unittest.main()
