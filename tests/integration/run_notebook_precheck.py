from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "03B_extraccion_cuantitativa_kb_migrado_v16.ipynb"


def create_project(root: Path) -> Path:
    project = root / "proyecto_estado_arte"
    experiment = project / "precheck_exp"
    (experiment / "05_outputs").mkdir(parents=True, exist_ok=True)
    (project / "active_experiment.json").write_text(
        json.dumps({
            "active_experiment_id": "precheck_exp",
            "run_id": "precheck_run",
            "openai_model": "not-used-in-precheck",
            "quantitative_extraction_policy": {
                "temperature": 0.1,
                "auto_rebuild": True,
                "force_rebuild": False,
                "only_include_state_of_art_papers": True,
                "verify_values_against_source_chunks": True,
                "allow_all_clean_chunks_fallback": True,
                "max_attempts": 1,
                "deterministic_flattening_repair": False,
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return project


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        project = create_project(Path(tmp))
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        old = {key: os.environ.get(key) for key in (
            "THESIS_CODE_ROOT", "THESIS_PROJECT_DIR", "RUN_REAL_03B",
            "RUN_DETERMINISTIC_FLATTENING_REPAIR", "NOTEBOOK_BOOTSTRAP_TEST_MODE",
        )}
        os.environ.update({
            "THESIS_CODE_ROOT": str(ROOT),
            "THESIS_PROJECT_DIR": str(project),
            "RUN_REAL_03B": "0",
            "RUN_DETERMINISTIC_FLATTENING_REPAIR": "0",
            "NOTEBOOK_BOOTSTRAP_TEST_MODE": "1",
        })
        try:
            executed = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        outputs = "\n".join(
            "".join(output.get("text", []))
            for cell in executed.cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        if "MODE: PRECHECK" not in outputs or "transaction_executed: False" not in outputs:
            raise RuntimeError("El notebook 03B no reportó PRECHECK correctamente.")
        report = {
            "status": "OK",
            "notebook": NOTEBOOK.name,
            "mode": "PRECHECK",
            "clean_kernel": True,
            "executed_code_cells": sum(
                1 for cell in executed.cells
                if cell.cell_type == "code" and cell.execution_count is not None
            ),
            "transaction_executed": False,
            "real_services_called": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
