from __future__ import annotations
import json
from dataclasses import dataclass
from src.io.credentials import load_runtime_credential
from src.tools.thematic_analysis.prompting import build_thematic_prompt
@dataclass(frozen=True)
class ThematicRuntimeDependencies:
    invoke: object; parse: object; build_prompt: object=build_thematic_prompt

def parse_json(value):
    if isinstance(value,dict):return value
    if hasattr(value,'content'):value=value.content
    if not isinstance(value,str):raise ValueError('INVALID_LLM_OUTPUT')
    text=value.strip();
    if text.startswith('```'): text=text.strip('`').replace('json\n','',1)
    return json.loads(text)

def build_real_thematic_dependencies(model,temperature):
    load_runtime_credential('OPENAI_API_KEY')
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm=ChatOpenAI(model=model,temperature=temperature)
    return ThematicRuntimeDependencies(invoke=lambda prompt:llm.invoke([HumanMessage(content=prompt)]).content,parse=parse_json)

from pathlib import Path
from src.contracts.agent_input import (
    AgentInput,
    AgentContext,
    ArtifactReference,
    ExecutionMode,
    PreviousAttemptSummary,
)
from src.state.fingerprints import sha256_file
from src.config.thematic_analysis_policy_config import get_thematic_analysis_policy
from src.agents.thematic_analysis_agent import ThematicAnalysisAgent


def load_thematic_configuration(project_dir: str | Path, attempt_number: int = 1):
    root = Path(project_dir).resolve()
    active_path = root / "active_experiment.json"
    if not active_path.is_file():
        raise FileNotFoundError("active_experiment.json no existe")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    experiment_id = active["active_experiment_id"]
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    kb_dir = outputs / "02_scientific_knowledge_base"
    extraction_dir = outputs / "01_scientific_extraction"
    quant_dir = kb_dir
    thematic_dir = outputs / "03_thematic_analysis"
    policy = get_thematic_analysis_policy(active.get("thematic_analysis_policy", {}))
    generation_profile = active.get("generation_profile", {})
    if isinstance(generation_profile, dict):
        policy.setdefault("min_sections", generation_profile.get("min_sections"))
        policy.setdefault("max_sections", generation_profile.get("max_sections"))
    return {
        "project_dir": root,
        "experiment_id": experiment_id,
        "run_id": active.get("run_id", experiment_id),
        "attempt_number": int(attempt_number),
        "model": active.get("openai_model", "gpt-4o-mini"),
        "policy": policy,
        "output_dir": thematic_dir,
        "state_path": experiment_dir / "pipeline_state.json",
        "paths": {
            "scientific_knowledge_base_csv": kb_dir / "scientific_knowledge_base.csv",
            "scientific_knowledge_base_jsonl": kb_dir / "scientific_knowledge_base.jsonl",
            "scientific_extraction_manifest": extraction_dir / "scientific_extraction_manifest.json",
            "quantitative_comparative_table": quant_dir / "quantitative_comparative_table.csv",
            "quantitative_datasets_table": quant_dir / "quantitative_datasets_table.csv",
            "quantitative_techniques_table": quant_dir / "quantitative_techniques_table.csv",
            "dataset_technique_summary": quant_dir / "dataset_technique_summary.csv",
            "quantitative_extraction_manifest": quant_dir / "quantitative_extraction_manifest.json",
        },
    }


def _previous_attempt_from_state(configuration):
    if configuration["attempt_number"] != 2:
        return None
    state_path = Path(configuration["state_path"])
    if not state_path.is_file():
        raise RuntimeError("El intento 2 requiere pipeline_state del intento 1.")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    stage = payload.get("stages", {}).get("04_agente_analisis_tematico", {})
    if stage.get("requested_transition", {}).get("action") != "RETRY":
        raise RuntimeError("El intento 2 requiere una transición RETRY persistida.")
    return PreviousAttemptSummary(
        quality_status=stage.get("quality_status", "NEEDS_REVISION"),
        quality_metrics={},
        blocking_warnings=tuple(
            str(item.get("code", ""))
            for item in stage.get("warnings", [])
            if item.get("blocking") and item.get("code")
        ),
        failure_reason_codes=tuple(stage.get("failure_reason_codes", [])),
        previous_artifacts={},
    )


def build_thematic_agent_input(configuration):
    paths = configuration["paths"]
    required = {
        name: paths[name]
        for name in (
            "scientific_knowledge_base_csv",
            "scientific_knowledge_base_jsonl",
            "scientific_extraction_manifest",
        )
    }
    for name, path in required.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} no existe: {path}")
    dependencies = {
        name: ArtifactReference(path=str(path), hash=sha256_file(path))
        for name, path in required.items()
    }
    optional_names = (
        "quantitative_comparative_table",
        "quantitative_datasets_table",
        "quantitative_techniques_table",
        "dataset_technique_summary",
        "quantitative_extraction_manifest",
    )
    existing = [name for name in optional_names if Path(paths[name]).is_file()]
    if existing and len(existing) != len(optional_names):
        raise FileNotFoundError("03B está parcialmente presente.")
    for name in existing:
        dependencies[name] = ArtifactReference(
            path=str(paths[name]),
            hash=sha256_file(paths[name]),
        )
    return AgentInput(
        experiment_id=configuration["experiment_id"],
        run_id=configuration["run_id"],
        stage_name="04_agente_analisis_tematico",
        attempt_number=configuration["attempt_number"],
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(
            allowed_tools=("llm", "atomic_write", "thematic_validation"),
            output_directory=str(configuration["output_dir"]),
            runtime_resources={"model": configuration["model"]},
        ),
        dependencies=dependencies,
        policy=configuration["policy"],
        previous_attempt=_previous_attempt_from_state(configuration),
    )


def build_real_thematic_execution(project_dir: str | Path, attempt_number: int = 1):
    configuration = load_thematic_configuration(project_dir, attempt_number)
    dependencies = build_real_thematic_dependencies(
        configuration["model"],
        float(configuration["policy"].get("temperature", 0.1)),
    )
    return (
        ThematicAnalysisAgent(dependencies),
        build_thematic_agent_input(configuration),
        configuration,
    )
