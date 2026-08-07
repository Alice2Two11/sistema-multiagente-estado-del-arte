import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "04_agente_analisis_tematico_migrado_v16.ipynb"


def notebook_code_cells():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code"]


def load_resolver():
    source = notebook_code_cells()[2]
    module = ast.parse(source)
    function_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_real_pipeline_state_path")
    fragment = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(fragment, "<resolver>", "exec"), namespace)
    return namespace["resolve_real_pipeline_state_path"]


class Agent04ColabOperationalFixTests(unittest.TestCase):
    def test_bootstrap_precedes_src_import_and_clones_missing_code_root(self):
        cells = notebook_code_cells()
        combined = cells[0] + "\n" + cells[1]
        self.assertLess(combined.index("git\", \"clone"), combined.index("from src.adapters"))
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            repository = temp / "repository"
            repository.mkdir()
            shutil.copytree(ROOT / "src", repository / "src")
            subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
            subprocess.run(["git", "add", "src"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "test"], cwd=repository, check=True, capture_output=True)
            code_root = temp / "missing_code_root"
            project = temp / "project"
            project.mkdir()
            script = cells[0] + "\n" + cells[1] + "\nprint('SRC_IMPORTED_AFTER_BOOTSTRAP')\n"
            env = os.environ.copy()
            env.update({
                "THESIS_CODE_ROOT": str(code_root),
                "THESIS_PROJECT_DIR": str(project),
                "PROJECT_SOURCE_URL": str(repository),
                "GIT_BRANCH": "main",
                "NOTEBOOK_BOOTSTRAP_TEST_MODE": "0",
            })
            completed = subprocess.run([sys.executable, "-c", script], env=env, cwd=temp, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((code_root / "src").is_dir())
            self.assertIn("SRC_IMPORTED_AFTER_BOOTSTRAP", completed.stdout)
            self.assertNotIn("ModuleNotFoundError", completed.stderr + completed.stdout)

    def test_resolves_canonical_pipeline_state(self):
        resolver = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = project / "exp" / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
            state.parent.mkdir(parents=True)
            state.write_text("{}", encoding="utf-8")
            self.assertEqual(resolver(project, "exp"), state.resolve())

    def test_rglob_accepts_only_one_existing_state_and_does_not_create_another(self):
        resolver = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            experiment = project / "exp"
            state = experiment / "legacy" / "pipeline_state.json"
            state.parent.mkdir(parents=True)
            state.write_text("{}", encoding="utf-8")
            resolved = resolver(project, "exp")
            self.assertEqual(resolved, state.resolve())
            canonical = experiment / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
            self.assertFalse(canonical.exists())

    def test_attempt2_reads_attempt1_from_resolved_state(self):
        from src.adapters.thematic_analysis_runtime import _previous_attempt_from_state
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "pipeline_state.json"
            state.write_text(json.dumps({
                "stages": {
                    "04_agente_analisis_tematico": {
                        "quality_status": "NEEDS_REVISION",
                        "requested_transition": {"action": "RETRY"},
                        "failure_reason_codes": ["INVALID_THEMATIC_SCHEMA"],
                        "warnings": [],
                    }
                }
            }), encoding="utf-8")
            summary = _previous_attempt_from_state({"attempt_number": 2, "state_path": state})
            self.assertEqual(summary.quality_status, "NEEDS_REVISION")
            self.assertIn("INVALID_THEMATIC_SCHEMA", summary.failure_reason_codes)

    def test_same_resolved_state_is_used_before_all_execution_modes(self):
        source = notebook_code_cells()[2]
        assignment = 'configuration["state_path"] = resolved_state_path'
        store = "store = StateStore(resolved_state_path)"
        self.assertIn(assignment, source)
        self.assertIn(store, source)
        self.assertLess(source.index(assignment), source.index("if not RUN_REAL_AGENT04"))
        self.assertLess(source.index(store), source.index("elif RUN_DETERMINISTIC_THEMATIC_REPAIR"))
        self.assertNotIn('StateStore(configuration["state_path"])', source)
        self.assertNotIn("El intento 2 requiere pipeline_state del intento 1", source)


if __name__ == "__main__":
    unittest.main()
