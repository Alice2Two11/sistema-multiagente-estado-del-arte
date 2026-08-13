from __future__ import annotations
import json
import unittest
from src.agents.draft_writing_agent import DraftWritingAgent
from src.adapters.draft_writing_runtime import DraftWritingRuntime
from src.contracts.agent_result import ExecutionStatus, QualityStatus, TransitionAction
from tests.v16.test_agent06_v16 import Env


class TestAgent06RagAttemptTraceV16(unittest.TestCase):
    def test_failed_attempts_persist_retrieval_allowed_and_llm_citations(self):
        env = Env(attempt=2)

        def invalid_numeric(_prompt):
            return json.dumps({
                "section_id": "S1",
                "section_title": "Methods",
                "draft_text": (
                    "This substantive scientific sentence reports 99% with "
                    "the citation selected by the model [a.pdf | c1]."
                ),
                "claims": [{
                    "claim": "This substantive scientific sentence reports 99% with the citation selected by the model",
                    "supporting_citations": ["[a.pdf | c1]"],
                }],
            })

        env.agent = DraftWritingAgent(DraftWritingRuntime(invalid_numeric, env.collection))
        result = env.agent.execute(env.ai)
        self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED)
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.HALT_STAGE)

        report = json.loads((env.out / "draft_validation_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["validation_ok"])
        for attempt in (1, 2, 3):
            trace_path = env.out / "raw_section_outputs" / "agent_attempt_02" / f"S1_attempt_{attempt}_rag_trace.json"
            self.assertTrue(trace_path.exists())
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["section_id"], "S1")
            self.assertEqual(trace["generation_attempt"], attempt)
            self.assertTrue(trace["query"])
            self.assertTrue(trace["retrieved_chunks"])
            self.assertIn("score", trace["retrieved_chunks"][0])
            self.assertIn("retrieval_method", trace["retrieved_chunks"][0])
            self.assertEqual(trace["allowed_citations"], ["[a.pdf | c1]"])
            self.assertIn(["a.pdf", "c1"], trace["llm_citations"])
            self.assertIn("rag_trace_path", report["generation_attempts"]["S1"][attempt - 1])
        env.close()


if __name__ == "__main__":
    unittest.main()
