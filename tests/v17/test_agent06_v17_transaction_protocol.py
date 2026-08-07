from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.draft_writing_runtime import (
    ExecutedDraftExecution,
    build_real_draft_execution,
    commit_executed_draft,
    execute_prepared_draft,
    prepare_draft_execution,
)
from src.contracts.agent_result import QualityStatus
from tests.v17.agent06_v17_test_support import EvidenceAwareLLM, SyntheticCollection, chroma_client_factory, runtime_factory_for, write_synthetic_project
from tests.v17.test_agent06_v17_end_to_end_integration import HYBRID


class TestAgent06V17TransactionProtocol(unittest.TestCase):
    def _setup(self, mode="valid"):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        _, store, rows = write_synthetic_project(root)
        llm = EvidenceAwareLLM(mode)
        agent, agent_input, cfg = build_real_draft_execution(
            root,
            collection_factory=lambda _: SyntheticCollection(rows),
            runtime_factory=runtime_factory_for(llm),
            chroma_client_factory=chroma_client_factory,
            policy_overrides=HYBRID,
        )
        return root, store, llm, agent, agent_input, cfg

    def test_prepare_is_non_executing_and_non_publishing(self):
        root, store, llm, agent, agent_input, cfg = self._setup()
        stage_before = store.load().stages.get("06_agente_redactor")
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        self.assertEqual(llm.calls, 0)
        self.assertIsNotNone(store.load().pending_execution)
        self.assertEqual(store.load().stages.get("06_agente_redactor"), stage_before)
        self.assertFalse((Path(cfg["output_dir"]) / "state_of_art_draft.json").exists())
        self.assertEqual(prepared.agent_input.policy["current_fingerprint"], agent_input.policy["current_fingerprint"])

    def test_execute_approved_does_not_commit_automatically(self):
        root, store, llm, agent, agent_input, cfg = self._setup()
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        self.assertIsNotNone(store.load().pending_execution)
        self.assertNotIn("06_agente_redactor", store.load().stages)

    def test_execute_failed_preserves_diagnostic_and_commit_rejected(self):
        root, store, llm, agent, agent_input, cfg = self._setup("unsupported_numeric")
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        self.assertEqual(executed.result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertIn("draft_validation_report.json", executed.result.output_artifacts)
        with self.assertRaisesRegex(RuntimeError, "DRAFT_COMMIT_REQUIRES_APPROVED_RESULT"):
            commit_executed_draft(store=store, executed=executed)
        self.assertIsNotNone(store.load().pending_execution)

    def test_commit_is_atomic_when_artifact_missing(self):
        root, store, llm, agent, agent_input, cfg = self._setup()
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        artifacts = dict(executed.result.output_artifacts)
        artifacts.pop("draft_sections.csv")
        broken_result = type(executed.result)(
            execution_status=executed.result.execution_status,
            quality_status=executed.result.quality_status,
            decision=executed.result.decision,
            quality_metrics=executed.result.quality_metrics,
            warnings=executed.result.warnings,
            failure_reason_codes=executed.result.failure_reason_codes,
            requested_transition=executed.result.requested_transition,
            output_artifacts=artifacts,
            tool_usage=executed.result.tool_usage,
            attempt_number=executed.result.attempt_number,
            started_at=executed.result.started_at,
            completed_at=executed.result.completed_at,
        )
        broken = ExecutedDraftExecution(executed.decision_id, executed.agent_input, broken_result, executed.persisted_result_path)
        state_before = store.load().to_dict()
        with self.assertRaisesRegex(RuntimeError, "DRAFT_COMMIT_INCOMPLETE_ARTIFACTS"):
            commit_executed_draft(store=store, executed=broken)
        self.assertEqual(store.load().to_dict(), state_before)

    def test_commit_updates_only_temporary_store_and_never_next_stage(self):
        root, store, llm, agent, agent_input, cfg = self._setup()
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        committed = commit_executed_draft(store=store, executed=executed)
        self.assertIsNone(committed.pending_execution)
        self.assertEqual(committed.stages["06_agente_redactor"].quality_status, QualityStatus.APPROVED)
        self.assertNotIn("07_agente_verificador", committed.stages)
        self.assertTrue(str(store.state_path).startswith(str(root)))


if __name__ == "__main__":
    unittest.main()
