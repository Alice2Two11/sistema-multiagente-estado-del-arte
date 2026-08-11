from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.adapters.outline_generation_runtime import (
    build_openai_outline_runtime,
    build_real_outline_execution,
)


class _FakeMessage:
    def __init__(self, *, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeChatOpenAI:
    created = []

    def __init__(self, *, model, temperature):
        self.model = model
        self.temperature = temperature
        self.calls = 0
        self.__class__.created.append(self)

    def invoke(self, messages):
        self.calls += 1
        return _FakeResponse(
            '```json\n'
            '{"title": "ok", "objective": "obj", "narrative_strategy": "ns", "sections": []}\n'
            '```'
        )


def _fake_langchain_modules():
    openai_module = types.ModuleType("langchain_openai")
    openai_module.ChatOpenAI = _FakeChatOpenAI
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.HumanMessage = _FakeMessage
    core_module = types.ModuleType("langchain_core")
    core_module.messages = messages_module
    return {
        "langchain_openai": openai_module,
        "langchain_core": core_module,
        "langchain_core.messages": messages_module,
    }


def _write_realistic_project(root: Path) -> None:
    experiment_id = "exp_real_structure"
    active = {
        "active_experiment_id": experiment_id,
        "run_id": "run_real_structure",
        "openai_model": "gpt-test",
        "generation_profile": {
            "min_sections": 4,
            "max_sections": 5,
            "output_language": "español académico",
        },
        "experiment_profile": {},
        "topic_profile": {},
        "rag_policy": {},
        "outline_generation_policy": {},
    }
    (root / "active_experiment.json").write_text(
        json.dumps(active), encoding="utf-8"
    )
    outputs = root / experiment_id / "05_outputs"
    thematic = outputs / "03_thematic_analysis"
    state_dir = outputs / "00_orchestrator_planner"
    thematic.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (state_dir / "pipeline_state.json").write_text("{}", encoding="utf-8")

    (thematic / "thematic_analysis.json").write_text(
        json.dumps({"themes": []}), encoding="utf-8"
    )
    (thematic / "thematic_analysis_manifest.json").write_text(
        json.dumps({"fingerprint": "fp04"}), encoding="utf-8"
    )
    (thematic / "thematic_validation_report.json").write_text(
        json.dumps({"validation_ok": True}), encoding="utf-8"
    )
    (thematic / "suggested_state_of_art_structure.json").write_text(
        json.dumps([{"section_id": "S1"}]), encoding="utf-8"
    )
    pd.DataFrame(
        [{"theme_id": "T1", "theme_name": "Tema", "description": "d"}]
    ).to_csv(thematic / "themes_summary.csv", index=False)
    pd.DataFrame(
        [{"gap_id": "G1", "description": "gap", "basis": "b"}]
    ).to_csv(thematic / "research_gaps.csv", index=False)
    pd.DataFrame(
        [{"source_filename": "paper.pdf", "title": "Paper"}]
    ).to_csv(thematic / "comparative_table_papers.csv", index=False)
    pd.DataFrame(
        [{"source_filename": "paper.pdf", "title": "Paper"}]
    ).to_csv(thematic / "kb_final_for_thematic_analysis.csv", index=False)


class TestAgent05RuntimeResolution(unittest.TestCase):
    def setUp(self):
        _FakeChatOpenAI.created.clear()

    def test_openai_runtime_builds_without_project_llm_utils(self):
        fake_modules = _fake_langchain_modules()
        with patch.dict(sys.modules, fake_modules, clear=False), patch(
            "src.io.credentials.load_runtime_credential", return_value="secret"
        ) as credential:
            runtime = build_openai_outline_runtime(
                "gpt-test", project_dir="/content/proyecto_estado_arte"
            )
            parsed = runtime.parse(runtime.invoke("prompt"))

        credential.assert_called_once_with(
            "OPENAI_API_KEY", project_dir="/content/proyecto_estado_arte"
        )
        # Contrato real del parser estricto de esquema
        # (src/tools/outline_generation/response_parsing.py::
        # _is_outline_root): exige los 4 campos raíz completos
        # (title/objective/narrative_strategy/sections) -- un JSON
        # incompleto como {"title": "ok"} se rechaza deliberadamente
        # (INVALID_LLM_OUTPUT), ya no se acepta como antes.
        self.assertEqual(
            parsed,
            {"title": "ok", "objective": "obj", "narrative_strategy": "ns", "sections": []},
        )
        self.assertEqual(_FakeChatOpenAI.created[0].temperature, 0)
        self.assertEqual(_FakeChatOpenAI.created[0].calls, 1)

    def test_build_real_execution_matches_repository_without_src_llm_utils(self):
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / "proyecto_estado_arte"
            project_dir.mkdir()
            _write_realistic_project(project_dir)
            self.assertFalse((project_dir / "src" / "llm_utils.py").exists())

            fake_modules = _fake_langchain_modules()
            with patch.dict(sys.modules, fake_modules, clear=False), patch(
                "src.io.credentials.load_runtime_credential", return_value="secret"
            ):
                agent, agent_input, configuration = build_real_outline_execution(
                    project_dir, attempt_number=1
                )

            self.assertEqual(agent_input.stage_name, "05_generador_esquema")
            self.assertEqual(agent_input.attempt_number, 1)
            self.assertEqual(configuration["project_dir"], project_dir.resolve())
            self.assertEqual(_FakeChatOpenAI.created[0].temperature, 0)
            self.assertIsNotNone(agent)


    def test_build_real_execution_at_exact_colab_project_path(self):
        project_dir = Path("/content/proyecto_estado_arte")
        backup = None
        if project_dir.exists():
            backup = Path("/content/proyecto_estado_arte_agent05_runtime_test_backup")
            if backup.exists():
                import shutil
                shutil.rmtree(backup)
            project_dir.rename(backup)
        try:
            project_dir.mkdir(parents=True)
            _write_realistic_project(project_dir)
            self.assertFalse((project_dir / "src" / "llm_utils.py").exists())
            fake_modules = _fake_langchain_modules()
            with patch.dict(sys.modules, fake_modules, clear=False), patch(
                "src.io.credentials.load_runtime_credential", return_value="secret"
            ):
                agent, agent_input, configuration = build_real_outline_execution(
                    "/content/proyecto_estado_arte", attempt_number=1
                )
            self.assertEqual(agent_input.attempt_number, 1)
            self.assertEqual(configuration["project_dir"], project_dir.resolve())
            self.assertIsNotNone(agent)
        finally:
            import shutil
            if project_dir.exists():
                shutil.rmtree(project_dir)
            if backup is not None and backup.exists():
                backup.rename(project_dir)

    def test_runtime_source_has_no_llm_utils_dependency(self):
        runtime_path = (
            Path(__file__).parents[2]
            / "src"
            / "adapters"
            / "outline_generation_runtime.py"
        )
        source = runtime_path.read_text(encoding="utf-8")
        self.assertNotIn("from llm_utils import", source)
        self.assertNotIn("from src.llm_utils import", source)
        self.assertNotIn("src/llm_utils.py", source)
        self.assertIn("from langchain_openai import ChatOpenAI", source)
        self.assertIn("temperature=0", source)


if __name__ == "__main__":
    unittest.main()
