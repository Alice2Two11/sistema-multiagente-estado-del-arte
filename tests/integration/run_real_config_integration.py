from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from cryptography.fernet import Fernet
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]

REAL_RAG_POLICY = {
    "exclude_review_sections_from_reference_papers": True,
    "excluded_reference_section_types": [
        "related_work", "literature_review", "state_of_the_art",
        "background", "theoretical_background", "previous_work", "prior_work",
    ],
    "ground_truth_usage": "evaluation_only",
    "use_ground_truth_for_generation": False,
    "use_ground_truth_for_rag": False,
    "use_ground_truth_for_verification": False,
    "use_ground_truth_for_evaluation": True,
    "retrieval_profiles": {
        "default": {"top_k": 8, "fetch_k": 35, "max_per_source": 2},
        "compact": {"top_k": 6, "fetch_k": 35, "max_per_source": 2},
        "strict": {"top_k": 10, "fetch_k": 40, "max_per_source": 2},
        "testing": {"top_k": 5, "fetch_k": 30, "max_per_source": 2},
    },
    "indexing": {"batch_size": 200},
    "generation": {"temperature": 0.1, "answer_max_words": 120},
}

REAL_EXTRACTION_POLICY = {
    "max_chunks_per_paper": 10,
    "max_context_chars": 18000,
    "repair_max_chunks_per_paper": 18,
    "repair_max_context_chars": 26000,
    "temperature": 0.1,
    "repair_temperature": 0.0,
    "retrieval_profile": "strict",
    "retrieval_queries": [
        "research problem objective scientific contribution",
        "methodology methods models algorithms experimental design",
        "dataset data sources input variables study population case study",
        "evaluation metrics experimental results comparative performance",
        "main findings conclusions limitations research gaps future work",
    ],
    "title_repair_first_chunks": 3,
    "auto_rebuild": True,
    "force_rebuild": False,
}

LLM_UTILS_SOURCE = r'''import json
import os
import re
from pathlib import Path
from cryptography.fernet import Fernet

PROJECT_DIR = Path("/content/proyecto_estado_arte")
SECRETS_DIR = PROJECT_DIR / ".secrets"
KEY_FILE = SECRETS_DIR / "openai_api_key.key"
ENC_FILE = SECRETS_DIR / "openai_api_key.enc"

def load_openai_key_encrypted():
    if not ENC_FILE.exists() or not KEY_FILE.exists():
        return ""
    return Fernet(KEY_FILE.read_bytes()).decrypt(
        ENC_FILE.read_bytes()
    ).decode("utf-8").strip()

def ensure_openai_key(allow_prompt=True, persist_if_prompted=True):
    value = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not value:
        value = load_openai_key_encrypted()
    if not value:
        raise ValueError("No se configuró OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = value
    return value

def parse_json_safely(text):
    text = str(text).strip()
    text = re.sub(r"^```json", "", text, flags=re.I)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No se pudo parsear JSON.")
'''

PROMPTS_SOURCE = r'''import json

def build_scientific_extraction_prompt(source_filename, context, experiment_profile):
    return (
        "EXTRACTION\n"
        f"source_filename={source_filename}\n"
        f"context={context}\n"
        f"profile={json.dumps(experiment_profile, ensure_ascii=False)}"
    )

def build_relevance_classification_prompt(card, experiment_profile):
    return (
        "RELEVANCE\n"
        f"card={json.dumps(card, ensure_ascii=False)}\n"
        f"profile={json.dumps(experiment_profile, ensure_ascii=False)}"
    )
'''


GENERATION_CONFIG_SOURCE = r'''
def get_generation_profile():
    return {
        "output_language": "es",
        "output_language_label": "español académico",
        "length_profile": "large",
        "writing_mode": "critical",
        "focus_mode": "methods",
        "citation_style": "IEEE",
        "embedding_model": "all-MiniLM-L6-v2",
    }
'''


def complete_card(source_filename: str) -> dict[str, Any]:
    return {
        "source_filename": source_filename,
        "title": "Integrated paper",
        "paper_type": "research",
        "research_problem": "problem",
        "objective": "objective",
        "task_type": "prediction",
        "target_domain": "scientific domain",
        "target_variable_or_object": "target",
        "temporal_horizon_or_scope": "scope",
        "methods_or_models": ["model"],
        "method_families": ["family"],
        "datasets_or_case_study": "dataset",
        "input_variables_or_data_sources": ["input"],
        "evaluation_metrics": ["RMSE"],
        "main_results": "result",
        "reported_best_method_or_model": "model",
        "limitations_or_gaps": "limitation",
        "contribution": "contribution",
        "relevance_for_state_of_art": "relevant",
        "domain_specific_notes": "notes",
        "evidence": [{
            "claim": "claim", "supporting_quote": "quote",
            "chunk_id": f"{source_filename}-0",
        }],
    }


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, **settings):
        self.settings = settings
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        prompt = messages[0].content
        if prompt.startswith("EXTRACTION"):
            source = prompt.split("source_filename=", 1)[1].split("\n", 1)[0]
            return FakeResponse(json.dumps(complete_card(source), ensure_ascii=False))
        if prompt.startswith("RELEVANCE"):
            return FakeResponse(json.dumps({
                "task_type": "prediction",
                "target_domain": "scientific domain",
                "method_families": ["family"],
                "relevance_level": "high",
                "include_in_state_of_art": True,
                "relevance_reason": "directly relevant",
            }))
        raise AssertionError("Unexpected fake LLM prompt")


class FakeCollection:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.query_calls = []

    def count(self):
        return len(self.dataframe)

    def query(self, *, query_texts, n_results, where):
        self.query_calls.append({
            "query_texts": list(query_texts), "n_results": n_results,
            "where": dict(where),
        })
        source = where["source_filename"]
        rows = self.dataframe[
            self.dataframe["source_filename"] == source
        ].sort_values("chunk_index").head(n_results)
        return {
            "documents": [[str(row["text"]) for _, row in rows.iterrows()]],
            "metadatas": [[{
                "chunk_id": str(row["chunk_id"]),
                "chunk_index": int(row["chunk_index"]),
                "source_pdf_path": str(row["source_pdf_path"]),
                "source_filename": str(row["source_filename"]),
            } for _, row in rows.iterrows()]],
            "distances": [[0.1 + i * 0.01 for i in range(len(rows))]],
        }


class FakeChromaClient:
    def __init__(self, *, path, collection):
        self.path = path
        self.collection = collection

    def get_collection(self, *, name, embedding_function):
        if name != "reference_papers_chunks":
            raise KeyError(name)
        return self.collection


def create_real_project(root: Path) -> tuple[Path, str]:
    project_dir = root / "proyecto_estado_arte"
    experiment_id = "experimento_paper_02"
    experiment_dir = project_dir / experiment_id
    chunks_dir = experiment_dir / "03_chunks"
    chroma_dir = experiment_dir / "04_chroma_index"
    outputs_dir = experiment_dir / "05_outputs"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(parents=True, exist_ok=True)

    active = {
        "active_experiment_id": experiment_id,
        "run_id": "run-integrated-03",
        "experiment_dir": str(experiment_dir),
        "project_dir": str(project_dir),
        "last_action": "resume",
        "updated_at": "2026-07-16T00:00:00",
        "generation_profile": {
            "output_language": "es",
            "output_language_label": "español académico",
            "length_profile": "large",
            "writing_mode": "critical",
            "focus_mode": "methods",
            "citation_style": "IEEE",
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "topic_profile": {
            "topic_name": "Integration topic",
            "research_scope": "Integration scope",
            "domain_terms": ["term"],
            "method_dimensions": ["family"],
            "analysis_dimensions": ["objective"],
            "relevance_rules": "Include relevant work.",
            "excluded_domains": [],
            "relevance_levels_included": ["high", "medium"],
        },
        "openai_model": "fake-model",
        "embedding_model": "all-MiniLM-L6-v2",
        "chroma_collection_name": "reference_papers_chunks",
        "rag_policy": REAL_RAG_POLICY,
        "extraction_policy": REAL_EXTRACTION_POLICY,
    }
    (project_dir / "active_experiment.json").write_text(
        json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (project_dir / "src" / "llm_utils.py").write_text(
        LLM_UTILS_SOURCE, encoding="utf-8"
    )
    (project_dir / "src" / "prompts.py").write_text(
        PROMPTS_SOURCE, encoding="utf-8"
    )
    (project_dir / "src" / "generation_config.py").write_text(
        GENERATION_CONFIG_SOURCE, encoding="utf-8"
    )

    secret = "integration-secret-value"
    secrets_dir = project_dir / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    (secrets_dir / "openai_api_key.key").write_bytes(key)
    (secrets_dir / "openai_api_key.enc").write_bytes(
        Fernet(key).encrypt(secret.encode("utf-8"))
    )

    dataframe = pd.DataFrame([
        {
            "chunk_id": "paper.pdf-0", "source_filename": "paper.pdf",
            "source_pdf_path": "/tmp/paper.pdf", "chunk_index": 0,
            "text": "Research problem and objective.", "chars": 31,
            "is_review_section_chunk": False,
            "is_bibliography_chunk": False, "excluded_from_rag": False,
        },
        {
            "chunk_id": "paper.pdf-1", "source_filename": "paper.pdf",
            "source_pdf_path": "/tmp/paper.pdf", "chunk_index": 1,
            "text": "Method metric result limitation.", "chars": 32,
            "is_review_section_chunk": False,
            "is_bibliography_chunk": False, "excluded_from_rag": False,
        },
    ])
    chunks_path = chunks_dir / "chunks_clean_for_rag.csv"
    snapshot_path = chroma_dir / "df_chunks_clean_used_for_chroma.csv"
    dataframe.to_csv(chunks_path, index=False)
    dataframe.to_csv(snapshot_path, index=False)
    (chroma_dir / "chroma.sqlite3").write_bytes(b"fake-index-sentinel")
    manifest = {
        "created_at": "2026-07-16T00:00:00",
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "chroma_dir": str(chroma_dir),
        "collection_name": "reference_papers_chunks",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunks_source_file": str(chunks_path),
        "snapshot_file": str(snapshot_path),
        "num_chunks_indexed": len(dataframe),
        "num_rows_in_chunks_clean": len(dataframe),
        "num_unique_papers": 1,
        "ground_truth_indexed": False,
        "review_sections_indexed": False,
        "bibliography_indexed": False,
        "excluded_chunks_indexed": False,
        "rag_policy": REAL_RAG_POLICY,
        "generation_profile": active["generation_profile"],
    }
    (chroma_dir / "chroma_index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return project_dir, secret


def run_integration() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, expected_secret = create_real_project(Path(tmp))
        os.environ["THESIS_CODE_ROOT"] = str(CODE_ROOT)
        os.environ["THESIS_PROJECT_DIR"] = str(project_dir)
        os.environ.pop("OPENAI_API_KEY", None)

        from src.adapters.extraction_runtime import (
            build_agent_input, build_extraction_runtime,
            load_runtime_configuration, resolve_openai_api_key,
            resolve_project_runtime_components,
        )
        from src.agents.extraction_agent import ExtractionAgent
        from src.runtime.extraction_protocol import (
            execute_extraction_transaction, resolve_extraction_resume,
        )
        from src.state.pipeline_state import PipelineIdentity, PipelineState
        from src.state.state_store import StateStore

        api_key = resolve_openai_api_key(
            project_dir=project_dir, required=True
        )
        if api_key != expected_secret:
            raise AssertionError("Encrypted credential was not recovered")

        configuration = load_runtime_configuration(
            project_dir, code_root=CODE_ROOT
        )
        if configuration.retrieval_config["profile_name"] != "strict":
            raise AssertionError("Real retrieval profile was not loaded")
        if configuration.rag_policy["use_ground_truth_for_evaluation"] is not True:
            raise AssertionError("Evaluation flag was not preserved")

        components = resolve_project_runtime_components(
            project_dir,
            human_message_factory=FakeMessage,
        )
        dataframe = pd.read_csv(configuration.chunks_clean_path)
        collection = FakeCollection(dataframe)
        chat_clients = []

        def chat_factory(**settings):
            client = FakeLLM(**settings)
            chat_clients.append(client)
            return client

        runtime = build_extraction_runtime(
            configuration,
            api_key=api_key,
            components=components,
            chat_model_factory=chat_factory,
            embedding_factory=lambda **settings: {"settings": settings},
            chroma_client_factory=lambda **settings: FakeChromaClient(
                path=settings["path"], collection=collection
            ),
        )
        agent_input = build_agent_input(
            configuration,
            runtime_resources={
                "df_chunks_clean": runtime.dataframe,
                "collection": runtime.collection,
            },
        )
        state_path = (
            configuration.outputs_dir / "00_orchestrator_planner" /
            "pipeline_state.json"
        )
        store = StateStore(state_path)
        timestamp = datetime.now(timezone.utc).isoformat()
        store.initialize(PipelineState(
            identity=PipelineIdentity(
                experiment_id=configuration.experiment_id,
                run_id=configuration.run_id,
                created_at=timestamp,
                updated_at=timestamp,
                schema_version="1.0",
            ),
            generation_config_snapshot=configuration.to_dict(),
        ))
        transaction = execute_extraction_transaction(
            store=store,
            agent=ExtractionAgent(runtime.dependencies),
            agent_input=agent_input,
            observations={"integration": "00_to_03"},
        )
        if transaction.agent_result.execution_status.value != "COMPLETED":
            raise AssertionError("ExtractionAgent did not complete")
        resume = resolve_extraction_resume(store=store, agent_input=agent_input)
        if resume.action != "NO_PENDING":
            raise AssertionError("Resume did not observe committed state")

        result = {
            "status": "OK",
            "experiment_id": configuration.experiment_id,
            "retrieval_profile": configuration.retrieval_config["profile_name"],
            "rag_policy_technical_keys": sorted([
                key for key in (
                    "retrieval_profiles", "indexing", "generation",
                    "use_ground_truth_for_evaluation",
                ) if key in configuration.rag_policy
            ]),
            "runtime_built": True,
            "agent_status": transaction.agent_result.execution_status.value,
            "resume_action": resume.action,
            "artifacts": len(transaction.agent_result.output_artifacts),
            "main_llm_calls": len(chat_clients[0].calls),
            "repair_llm_calls": len(chat_clients[1].calls),
            "collection_queries": len(collection.query_calls),
        }
        serialized = json.dumps(result, ensure_ascii=False)
        if expected_secret in serialized:
            raise AssertionError("Secret leaked into integration report")
        return result


def main() -> int:
    try:
        result = run_integration()
    except Exception as error:
        print(json.dumps({
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
