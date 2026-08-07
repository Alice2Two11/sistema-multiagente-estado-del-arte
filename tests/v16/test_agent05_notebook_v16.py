import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]


class TestNotebook05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((ROOT / '05_generador_esquema_migrado_v16.ipynb').read_text())
        cls.code = '\n'.join(
            ''.join(cell.get('source', []))
            for cell in cls.nb['cells']
            if cell.get('cell_type') == 'code'
        )

    def test_shell_only(self):
        for forbidden in [
            'def repair_outline_sources',
            'def build_outline_generation_prompt',
            'get_close_matches(',
            'atomic_write_json(',
        ]:
            self.assertNotIn(forbidden, self.code)

    def test_bootstrap_before_src(self):
        self.assertLess(self.code.find('git clone'), self.code.find('from src.'))

    def test_real_repository_url_default(self):
        self.assertIn(
            'https://github.com/Alice2Two11/tesis-sistema-multiagente.git',
            self.code,
        )
        self.assertNotIn('REPLACE_WITH_PROJECT_REPOSITORY', self.code)

    def test_bootstrap_configuration_precedes_src_import(self):
        self.assertLess(self.code.find('REPOSITORY_URL ='), self.code.find('git clone'))
        self.assertLess(self.code.find('git clone'), self.code.find('importlib.invalidate_caches()'))
        self.assertLess(
            self.code.find('importlib.invalidate_caches()'),
            self.code.find('sys.path.insert(0,str(CODE_ROOT))'),
        )
        self.assertLess(
            self.code.find('sys.path.insert(0,str(CODE_ROOT))'),
            self.code.find('from src.'),
        )

    def test_modes_visible(self):
        self.assertIn('RUN_REAL_AGENT05', self.code)
        self.assertIn('RUN_PRECHECK_AGENT05', self.code)

    def test_no_agent06(self):
        self.assertNotIn('06_agente_redactor', self.code)


class TestNotebook05CleanClone(unittest.TestCase):
    def test_clean_runtime_clone_uses_default_repository_without_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            child_script = textwrap.dedent(
                f'''
                import json
                import os
                import subprocess
                from pathlib import Path
                from unittest.mock import patch

                root = Path({str(ROOT)!r})
                nb = json.loads((root / "05_generador_esquema_migrado_v16.ipynb").read_text())
                config = "".join(nb["cells"][1].get("source", []))
                bootstrap = "".join(nb["cells"][2].get("source", []))
                calls = []

                def fake_run(command, check=True, capture_output=True, text=True):
                    calls.append(list(command))
                    if command[:2] == ["git", "clone"]:
                        dst = Path(command[-1])
                        files = {{
                            "src/__init__.py": "",
                            "src/adapters/__init__.py": "",
                            "src/adapters/outline_generation_runtime.py": (
                                "def build_real_outline_execution(*a, **k): pass\\n"
                                "def load_outline_configuration(*a, **k): pass\\n"
                            ),
                            "src/runtime/__init__.py": "",
                            "src/runtime/outline_generation_protocol.py": (
                                "def execute_outline_runtime_transaction(*a, **k): pass\\n"
                            ),
                            "src/state/__init__.py": "",
                            "src/state/state_store.py": "class StateStore: pass\\n",
                        }}
                        for relative, content in files.items():
                            path = dst / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(content)
                    return subprocess.CompletedProcess(command, 0, "", "")

                os.environ.pop("PROJECT_SOURCE_URL", None)
                os.environ.update({{
                    "THESIS_CODE_ROOT": {str(Path(td) / 'tesis_codigo')!r},
                    "THESIS_PROJECT_DIR": {str(Path(td) / 'proyecto_estado_arte')!r},
                    "RUN_PRECHECK_AGENT05": "1",
                    "RUN_REAL_AGENT05": "0",
                    "NOTEBOOK_BOOTSTRAP_TEST_MODE": "0",
                }})

                namespace = {{}}
                with patch("subprocess.run", side_effect=fake_run):
                    exec(compile(config, "config_cell", "exec"), namespace, namespace)
                    exec(compile(bootstrap, "bootstrap_cell", "exec"), namespace, namespace)

                clone_calls = [call for call in calls if call[:2] == ["git", "clone"]]
                assert len(clone_calls) == 1, clone_calls
                assert (
                    "https://github.com/Alice2Two11/tesis-sistema-multiagente.git"
                    in clone_calls[0]
                ), clone_calls
                '''
            )
            result = subprocess.run(
                [sys.executable, '-c', child_script],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + '\n' + result.stderr,
            )
