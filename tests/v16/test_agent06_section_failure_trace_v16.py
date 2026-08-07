from __future__ import annotations
import json
import unittest
from src.agents.draft_writing_agent import DraftWritingAgent
from src.adapters.draft_writing_runtime import DraftWritingRuntime
from src.contracts.agent_result import ExecutionStatus, QualityStatus, TransitionAction
from tests.v16.test_agent06_v16 import Env


class TestAgent06SectionFailureTraceV16(unittest.TestCase):
    def test_three_failed_generation_attempts_persist_partial_report_and_retry(self):
        env = Env(attempt=1)

        def always_invalid(_prompt):
            return json.dumps({
                "section_id": "S1",
                "section_title": "Methods",
                "draft_text": "This substantive scientific sentence has no valid supporting citation and cannot pass the original validation rules.",
                "claims": [],
            })

        env.agent = DraftWritingAgent(DraftWritingRuntime(always_invalid, env.collection))
        result = env.agent.execute(env.ai)

        self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED)
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.RETRY)
        self.assertEqual(result.failure_reason_codes, ("SECTION_VALIDATION_FAILED",))
        self.assertEqual(result.tool_usage.llm_calls, 3)
        self.assertEqual(result.tool_usage.validation_calls, 3)
        self.assertFalse((env.out / "state_of_art_draft.json").exists())
        self.assertFalse((env.out / "state_of_art_draft.md").exists())

        report_path = env.out / "draft_validation_report.json"
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["validation_ok"])
        self.assertEqual(report["failed_section"], "S1")
        self.assertEqual(report["section_attempts"], 3)
        self.assertEqual(len(report["generation_attempts"]["S1"]), 3)
        self.assertTrue(report["last_attempt_errors"])

        for attempt in (1, 2, 3):
            raw = env.out / "raw_section_outputs" / f"S1_attempt_{attempt}.txt"
            validation = env.out / "raw_section_outputs" / f"S1_attempt_{attempt}_validation.json"
            self.assertTrue(raw.exists())
            self.assertTrue(validation.exists())
            payload = json.loads(validation.read_text(encoding="utf-8"))
            self.assertEqual(payload["section_id"], "S1")
            self.assertEqual(payload["generation_attempt"], attempt)
            self.assertFalse(payload["validation_ok"])
            for key in (
                "validation_errors",
                "invalid_citations",
                "unsupported_claims",
                "substantive_sentences_without_claim",
                "substantive_sentences_without_citation",
                "claim_sentence_mismatches",
                "numeric_support_errors",
                "word_count",
                "citation_count",
            ):
                self.assertIn(key, payload)
        env.close()

    def test_attempt2_halts_after_internal_attempts_exhausted(self):
        env = Env(attempt=2)
        env.agent = DraftWritingAgent(DraftWritingRuntime(
            lambda _prompt: json.dumps({
                "section_id": "S1",
                "section_title": "Methods",
                "draft_text": "This substantive scientific sentence has no valid supporting citation and cannot pass the original validation rules.",
                "claims": [],
            }),
            env.collection,
        ))
        result = env.agent.execute(env.ai)
        self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED)
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.HALT_STAGE)
        env.close()


if __name__ == "__main__":
    unittest.main()
