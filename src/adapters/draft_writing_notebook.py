from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import shutil
import sys
import time

import pandas as pd

from src.adapters.draft_writing_runtime import (
    REQUIRED_DRAFT_ARTIFACTS,
    ExecutedDraftExecution,
    build_chroma_collection,
    build_openai_draft_runtime,
    commit_executed_draft,
    execute_prepared_draft,
    prepare_draft_execution,
)
from src.agents.draft_writing_agent import DraftWritingAgent
from src.state.fingerprints import sha256_file
from src.state.state_store import StateStore


APPROVED_QUALITY_STATUSES = {
    "APPROVED",
    "APPROVED_WITH_WARNINGS",
    "APPROVED_AFTER_MANUAL_REVIEW",
}

COMMIT_CONFIRMATION = "CONFIRMAR_COMMIT_AGENTE_06"


@dataclass(frozen=True)
class DraftNotebookPaths:
    code_root: Path
    project_root: Path
    experiment_dir: Path
    candidate_output_dir: Path
    official_output_dir: Path
    operational_state_path: Path
    isolated_state_path: Path


@dataclass(frozen=True)
class DraftExecutionOutcome:
    executed: Any
    duration_seconds: float


def bootstrap_code_root(code_root: str | Path) -> Path:
    """Añade el repositorio a sys.path sin limpiar cachés ni módulos."""
    root = Path(code_root).resolve()

    if not (root / "src").is_dir():
        raise FileNotFoundError(
            f"No existe el paquete src dentro de CODE_ROOT: {root}"
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    return root


def resolve_notebook_paths(
    cfg: dict[str, Any],
    *,
    code_root: str | Path,
    candidate_directory_name: str = "05_draft_v17_candidate",
) -> DraftNotebookPaths:
    """Deriva todas las rutas desde la configuración activa del notebook 00."""
    resolved_code_root = Path(code_root).resolve()
    project_root = Path(cfg["project_dir"]).resolve()
    experiment_dir = Path(cfg["experiment_dir"]).resolve()
    official_output_dir = Path(cfg["output_dir"]).resolve()
    candidate_output_dir = (
        official_output_dir.parent / candidate_directory_name
    ).resolve()
    operational_state_path = Path(cfg["state_path"]).resolve()
    isolated_state_path = (
        candidate_output_dir
        / "transaction_state"
        / "pipeline_state.json"
    ).resolve()

    if candidate_output_dir == official_output_dir:
        raise RuntimeError(
            "La salida candidata no puede coincidir con la salida oficial."
        )

    return DraftNotebookPaths(
        code_root=resolved_code_root,
        project_root=project_root,
        experiment_dir=experiment_dir,
        candidate_output_dir=candidate_output_dir,
        official_output_dir=official_output_dir,
        operational_state_path=operational_state_path,
        isolated_state_path=isolated_state_path,
    )


def configure_candidate_output(
    cfg: dict[str, Any],
    paths: DraftNotebookPaths,
) -> dict[str, Any]:
    """Devuelve una copia superficial de cfg apuntando al staging aislado."""
    candidate_cfg = dict(cfg)
    candidate_cfg["output_dir"] = paths.candidate_output_dir
    return candidate_cfg


def prepare_isolated_draft_execution(
    *,
    paths: DraftNotebookPaths,
    agent_input,
    force_reexecution: bool,
):
    """Prepara una copia transaccional sin modificar el PipelineState oficial."""
    if not paths.operational_state_path.is_file():
        raise FileNotFoundError(
            "No existe el PipelineState operacional: "
            f"{paths.operational_state_path}"
        )

    transaction_dir = paths.isolated_state_path.parent
    isolated_results_dir = transaction_dir / "agent_results"
    transaction_dir.mkdir(parents=True, exist_ok=True)

    if force_reexecution or not paths.isolated_state_path.exists():
        if paths.isolated_state_path.exists():
            paths.isolated_state_path.unlink()
        if isolated_results_dir.exists():
            shutil.rmtree(isolated_results_dir)

        shutil.copy2(
            paths.operational_state_path,
            paths.isolated_state_path,
        )

    store = StateStore(
        paths.isolated_state_path,
        agent_results_directory=isolated_results_dir,
    )

    isolated_state = store.load()
    if isolated_state.pending_execution is not None:
        store.cancel_pending_execution()

    prepared = prepare_draft_execution(
        store=store,
        agent_input=agent_input,
    )
    return store, prepared


def execute_draft_candidate(
    *,
    cfg: dict[str, Any],
    agent_input,
    store,
    prepared,
) -> DraftExecutionOutcome:
    """Construye herramientas y ejecuta el Agente 06 sin COMMIT automático."""
    started = time.perf_counter()

    collection = build_chroma_collection(cfg)
    runtime = build_openai_draft_runtime(
        cfg["model"],
        agent_input.policy["temperature"],
        collection,
        project_dir=cfg["project_dir"],
    )
    agent = DraftWritingAgent(runtime)

    executed = execute_prepared_draft(
        store=store,
        agent=agent,
        prepared=prepared,
    )

    return DraftExecutionOutcome(
        executed=executed,
        duration_seconds=time.perf_counter() - started,
    )


def enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def artifact_map(result) -> dict[str, Any]:
    return dict(getattr(result, "output_artifacts", {}) or {})


def find_artifact_reference(result, filename: str):
    artifacts = artifact_map(result)

    if filename in artifacts:
        return artifacts[filename]

    for reference in artifacts.values():
        reference_path = getattr(reference, "path", None)
        if reference_path and Path(reference_path).name == filename:
            return reference

    return None


def find_artifact_path(
    result,
    filename: str,
    candidate_output_dir: str | Path,
) -> Path | None:
    reference = find_artifact_reference(result, filename)

    if reference is not None:
        referenced_path = Path(reference.path)
        if referenced_path.is_file():
            return referenced_path

    fallback = Path(candidate_output_dir) / filename
    return fallback if fallback.is_file() else None


def read_csv_artifact(
    result,
    filename: str,
    candidate_output_dir: str | Path,
) -> tuple[pd.DataFrame, Path | None]:
    path = find_artifact_path(result, filename, candidate_output_dir)

    if path is None:
        return pd.DataFrame(), None

    try:
        return pd.read_csv(path), path
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), path


def read_json_artifact(
    result,
    filename: str,
    candidate_output_dir: str | Path,
) -> tuple[Any, Path | None]:
    path = find_artifact_path(result, filename, candidate_output_dir)

    if path is None:
        return None, None

    return json.loads(path.read_text(encoding="utf-8")), path


def summarize_draft_result(
    result,
    *,
    duration_seconds: float | None,
) -> dict[str, Any]:
    quality_metrics = (
        result.quality_metrics
        if isinstance(result.quality_metrics, dict)
        else {}
    )
    scientific = quality_metrics.get("scientific", {})
    technical = quality_metrics.get("technical", {})

    def metric(name: str, default=None):
        for source in (quality_metrics, scientific, technical):
            if isinstance(source, dict) and name in source:
                return source[name]
        return default

    transition = getattr(result, "requested_transition", None)
    transition_action = getattr(transition, "action", transition)

    return {
        "execution_status": enum_value(result.execution_status),
        "quality_status": enum_value(result.quality_status),
        "decision_code": result.decision.code,
        "requested_transition": enum_value(transition_action),
        "validation_ok": metric("validation_ok"),
        "invalid_citation_count": metric("invalid_citation_count", 0),
        "numeric_failure_count": metric("numeric_failure_count", 0),
        "sections_outside_word_range": metric(
            "sections_outside_word_range",
            [],
        ),
        "retrieval_rounds": result.tool_usage.retrieval_rounds,
        "llm_calls": result.tool_usage.llm_calls,
        "validation_calls": result.tool_usage.validation_calls,
        "duration_seconds": (
            round(duration_seconds, 3)
            if duration_seconds is not None
            else None
        ),
    }


def audit_required_artifacts(
    result,
    *,
    candidate_output_dir: str | Path,
) -> pd.DataFrame:
    rows = []

    for filename in REQUIRED_DRAFT_ARTIFACTS:
        path = find_artifact_path(
            result,
            filename,
            candidate_output_dir,
        )
        rows.append({
            "artifact": filename,
            "exists": path is not None,
            "size_bytes": path.stat().st_size if path else 0,
            "sha256": sha256_file(path) if path else None,
            "path": str(path) if path else None,
        })

    return pd.DataFrame(rows)


def commit_approved_draft(
    *,
    paths: DraftNotebookPaths,
    agent_input,
    executed,
    confirmation_text: str,
):
    """Realiza COMMIT solo con calidad aprobada y confirmación textual."""
    quality = enum_value(executed.result.quality_status)

    if confirmation_text != COMMIT_CONFIRMATION:
        raise PermissionError("Confirmación textual de COMMIT inválida.")

    if quality not in APPROVED_QUALITY_STATUSES:
        raise RuntimeError(
            f"No se puede hacer COMMIT con quality_status={quality}."
        )

    operational_store = StateStore(paths.operational_state_path)

    prepared_real = prepare_draft_execution(
        store=operational_store,
        agent_input=agent_input,
    )
    executed_real = ExecutedDraftExecution(
        decision_id=prepared_real.decision_id,
        agent_input=agent_input,
        result=executed.result,
        persisted_result_path=executed.persisted_result_path,
    )

    try:
        committed = commit_executed_draft(
            store=operational_store,
            executed=executed_real,
            observations={
                "source_notebook": "06_agente_redactor_v17_LIMPIO.ipynb",
            },
        )
    except Exception:
        operational_store.cancel_pending_execution()
        raise

    if operational_store.load().pending_execution is not None:
        raise RuntimeError(
            "El COMMIT terminó dejando pending_execution activo."
        )

    return committed


def build_execution_audit(
    *,
    execution_mode: str,
    paths: DraftNotebookPaths,
    agent_input,
    executed,
    duration_seconds: float | None,
    commit_performed: bool,
    artifact_audit: pd.DataFrame | None,
) -> dict[str, Any]:
    rows = (
        artifact_audit.to_dict("records")
        if artifact_audit is not None
        else []
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "fingerprint": agent_input.policy["current_fingerprint"],
        "isolated_state_path": str(paths.isolated_state_path),
        "operational_state_path": str(paths.operational_state_path),
        "execution_duration_seconds": duration_seconds,
        "commit_performed": commit_performed,
        "decision": (
            executed.result.decision.code if executed is not None else None
        ),
        "quality_status": (
            enum_value(executed.result.quality_status)
            if executed is not None
            else None
        ),
        "artifact_count_registered": (
            len(artifact_map(executed.result))
            if executed is not None
            else 0
        ),
        "artifact_count_found": sum(
            1 for row in rows if row.get("exists")
        ),
        "artifact_hashes": {
            row["artifact"]: row["sha256"]
            for row in rows
            if row.get("sha256")
        },
        "warnings": [
            "El Agente 07 no se ejecuta desde este notebook.",
            (
                "La salida oficial fue actualizada mediante COMMIT."
                if commit_performed
                else "La ejecución permanece en staging aislado."
            ),
        ],
    }
