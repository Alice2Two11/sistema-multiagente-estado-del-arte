from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "03_agente_extraccion_kb_migrado_v16.ipynb"
PROJECT = Path(
    os.environ.get(
        "AGENT03_REAL_PROJECT_DIR",
        "/mnt/data/agent03_v16_real_project/proyecto_estado_arte",
    )
).resolve()


def main() -> int:
    active = json.loads((PROJECT / "active_experiment.json").read_text())
    experiment = Path(active["experiment_dir"])
    outputs = experiment / "05_outputs"
    stage_state = outputs / "00_orchestrator_planner"
    extraction_dir = outputs / "01_scientific_extraction"
    kb_dir = outputs / "02_scientific_knowledge_base"
    for path in (stage_state, extraction_dir, kb_dir):
        shutil.rmtree(path, ignore_errors=True)
    outputs.mkdir(parents=True, exist_ok=True)

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    old = {key: os.environ.get(key) for key in (
        "THESIS_CODE_ROOT", "THESIS_PROJECT_DIR", "RUN_REAL_EXTRACTION",
        "NOTEBOOK_BOOTSTRAP_TEST_MODE", "AGENT03_ATTEMPT_NUMBER",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    )}
    os.environ.update({
        "THESIS_CODE_ROOT": str(ROOT),
        "THESIS_PROJECT_DIR": str(PROJECT),
        "RUN_REAL_EXTRACTION": "1",
        "NOTEBOOK_BOOTSTRAP_TEST_MODE": "1",
        "AGENT03_ATTEMPT_NUMBER": "1",
        # The execution environment has no DNS. Offline mode makes the
        # external dependency failure deterministic instead of hanging.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    try:
        executed = NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    text = "\n".join(
        "".join(output.get("text", []))
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    state_path = stage_state / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    stage = state["stages"]["03_agente_extraccion_kb"]
    result_files = list((stage_state / "pipeline_state_agent_results").glob("*.json"))
    persisted = json.loads(result_files[-1].read_text()) if result_files else {}
    artifact_paths = [
        Path(item["path"])
        for item in persisted.get("output_artifacts", {}).values()
    ]
    artifacts_exist = bool(artifact_paths) and all(path.is_file() for path in artifact_paths)
    report = {
        "mode": "REAL MODE",
        "run_real_extraction": True,
        "services": "REAL_OPENAI_REAL_CHROMA_REQUESTED_NO_DOUBLES",
        "transaction_executed": '"transaction_executed": true' in text,
        "prepare_executed": bool(state.get("decision_log")),
        "agent_result_persisted": bool(result_files),
        "commit_executed": state.get("pending_execution") is None and bool(state.get("decision_log")),
        "execution_status": stage.get("execution_status"),
        "quality_status": stage.get("quality_status"),
        "failure_reason_codes": stage.get("failure_reason_codes", []),
        "artifact_count": len(artifact_paths),
        "artifacts_exist": artifacts_exist,
        "network_dns_available": False,
        "external_blocker": (
            "The validation container cannot resolve api.openai.com or "
            "huggingface.co; successful real extraction cannot be asserted here."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    success = (
        report["transaction_executed"]
        and report["prepare_executed"]
        and report["agent_result_persisted"]
        and report["commit_executed"]
        and report["execution_status"] == "COMPLETED"
        and report["artifacts_exist"]
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
