from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.adapters.draft_writing_runtime import build_runtime_draft_policy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestAgent06PipelineStateIsolationV17(unittest.TestCase):
    def test_real_or_frozen_pipeline_state_is_unchanged(self):
        candidates = [
            Path("/content/proyecto_estado_arte/experimento_paper_02/05_outputs/00_orchestrator_planner/pipeline_state.json"),
            Path(__file__).parents[2] / "frozen" / "pipeline_state.json",
        ]
        existing = next((path for path in candidates if path.is_file()), None)
        before = _sha(existing) if existing else None
        with tempfile.TemporaryDirectory() as td:
            temp_state = Path(td) / "pipeline_state.json"
            temp_state.write_text("{}", encoding="utf-8")
            build_runtime_draft_policy({"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"})
            self.assertEqual(temp_state.read_text(encoding="utf-8"), "{}")
        after = _sha(existing) if existing else None
        self.assertEqual(before, after)

    def test_runtime_source_does_not_import_experimental_agent_or_runtime(self):
        source = (Path(__file__).parents[2] / "src" / "adapters" / "draft_writing_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("draft_writing_agent_hybrid_experimental", source)
        self.assertNotIn("draft_writing_hybrid_runtime", source)

    def test_tests_require_no_openai_or_chroma_real(self):
        source = (Path(__file__).parents[0] / "test_agent06_runtime_integration_v17.py").read_text(encoding="utf-8")
        self.assertIn("_Collection", source)
        self.assertIn("_Runtime", source)
        self.assertNotIn("OPENAI_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
