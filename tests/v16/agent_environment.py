from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile

import pandas as pd

from src.agents.extraction_agent import (
    ExtractionAgent,
    ExtractionAgentDependencies,
)
from src.contracts.agent_input import (
    AgentContext,
    AgentInput,
    ArtifactReference,
    ExecutionMode,
)
from src.contracts.agent_result import (
    ExecutionStatus,
    QualityStatus,
    TransitionAction,
    WarningSeverity,
)
from src.state.fingerprints import sha256_file
from src.tools.extraction.card_validation import (
    CARD_LIST_FIELDS,
    CARD_REQUIRED_FIELDS,
)
from tests.v16.extraction_agent_doubles import (
    DeterministicClock,
    FakeCollection,
    FakeMessage,
    FakeRawWriter,
    RecordingPrinter,
    RoutedLLM,
    complete_card,
    extraction_prompt_builder,
    relevance_prompt_builder,
)


ERROR_COLUMNS = [
    "source_filename",
    "stage",
    "error_type",
    "error_message",
    "created_at",
]

TRACE_COLUMNS = [
    "source_filename",
    "retrieval_query",
    "chunk_id",
    "chunk_index",
    "score",
    "retrieval_profile",
    "retrieval_mode",
]


class ExtractionAgentEnvironment:
    def __init__(
        self,
        *,
        extraction_cards=None,
        main_extraction_errors=None,
        repair_cards=None,
        repair_errors=None,
        repaired_titles=None,
        title_errors=None,
        classification=None,
        classification_errors=None,
        invalid_chunks=False,
        use_runtime_collection=True,
    ):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.outputs_dir = (
            self.root / "05_outputs"
        )
        self.chunks_dir = (
            self.root / "03_chunks"
        )
        self.chroma_dir = (
            self.root / "04_chroma"
        )
        self.chunks_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.chroma_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunks_path = (
            self.chunks_dir
            / "chunks_clean_for_rag.csv"
        )
        self.chroma_manifest_path = (
            self.chroma_dir
            / "chroma_index_manifest.json"
        )

        rows = []
        for source in [
            "a.pdf",
            "b.pdf",
        ]:
            for index in [0, 1]:
                rows.append({
                    "chunk_id": (
                        f"{source}-{index}"
                    ),
                    "source_filename": source,
                    "source_pdf_path": (
                        str(
                            self.root
                            / source
                        )
                    ),
                    "chunk_index": index,
                    "text": (
                        f"Text {source} "
                        f"{index}"
                    ),
                    "chars": 20,
                    "is_review_section_chunk": (
                        invalid_chunks
                        and source == "a.pdf"
                        and index == 0
                    ),
                    "is_bibliography_chunk": (
                        False
                    ),
                    "excluded_from_rag": (
                        False
                    ),
                })

        self.dataframe = pd.DataFrame(
            rows
        )
        self.dataframe.to_csv(
            self.chunks_path,
            index=False,
        )

        self.collection = FakeCollection(
            self.dataframe
        )
        self.collection_loader_calls = []

        self.chroma_manifest_path.write_text(
            json.dumps({
                "experiment_id": "exp-1",
                "collection_name": (
                    "collection-1"
                ),
                "chunks_source_file": str(
                    self.chunks_path
                ),
            }),
            encoding="utf-8",
        )

        self.dir_extraction = (
            self.outputs_dir
            / "01_scientific_extraction"
        )
        self.dir_kb = (
            self.outputs_dir
            / "02_scientific_knowledge_base"
        )
        self.paths = {
            "OUTPUTS_DIR": (
                self.outputs_dir
            ),
            "DIR_EXTRACTION": (
                self.dir_extraction
            ),
            "DIR_KB": self.dir_kb,
            "CARDS_JSONL_PATH": (
                self.dir_extraction
                / "scientific_cards.jsonl"
            ),
            "CARDS_SUMMARY_CSV_PATH": (
                self.dir_extraction
                / "scientific_cards_summary.csv"
            ),
            "CARDS_ERRORS_CSV_PATH": (
                self.dir_extraction
                / "scientific_cards_errors.csv"
            ),
            "CARDS_QUALITY_CSV_PATH": (
                self.dir_extraction
                / "scientific_cards_quality_check.csv"
            ),
            "CARDS_REVISION_PLAN_CSV_PATH": (
                self.dir_extraction
                / "scientific_cards_revision_plan.csv"
            ),
            "RETRIEVAL_TRACE_CSV_PATH": (
                self.dir_extraction
                / "extraction_retrieval_trace.csv"
            ),
            "EXTRACTION_MANIFEST_PATH": (
                self.dir_extraction
                / "scientific_extraction_manifest.json"
            ),
            "CHUNKS_VALIDATION_REPORT_PATH": (
                self.dir_extraction
                / "chunks_clean_validation_report.json"
            ),
            "KB_CSV_PATH": (
                self.dir_kb
                / "scientific_knowledge_base.csv"
            ),
            "KB_JSONL_PATH": (
                self.dir_kb
                / "scientific_knowledge_base.jsonl"
            ),
        }

        base_cards = {
            "a.pdf": complete_card(
                "a.pdf"
            ),
            "b.pdf": complete_card(
                "b.pdf"
            ),
        }
        if extraction_cards:
            base_cards.update(
                extraction_cards
            )

        base_classification = {
            source: {
                "task_type": (
                    f"classified-{source}"
                ),
                "target_domain": "domain",
                "method_families": [
                    "family"
                ],
                "relevance_level": "high",
                "include_in_state_of_art": (
                    True
                ),
                "relevance_reason": (
                    f"reason-{source}"
                ),
            }
            for source in [
                "a.pdf",
                "b.pdf",
            ]
        }
        if classification:
            base_classification.update(
                classification
            )

        self.main_llm = RoutedLLM(
            extraction_cards=base_cards,
            classification=(
                base_classification
            ),
            extraction_errors=(
                main_extraction_errors
            ),
            classification_errors=(
                classification_errors
            ),
        )
        self.repair_llm = RoutedLLM(
            repair_cards=repair_cards,
            repaired_titles=(
                repaired_titles
            ),
            repair_errors=repair_errors,
            title_errors=title_errors,
        )
        self.raw_writer = FakeRawWriter()
        self.printer = RecordingPrinter()
        self.clock = DeterministicClock()

        def load_collection(agent_input):
            self.collection_loader_calls.append(
                agent_input.run_id
            )
            return self.collection

        self.dependencies = (
            ExtractionAgentDependencies(
                main_llm=self.main_llm,
                repair_llm=self.repair_llm,
                extraction_prompt_builder=(
                    extraction_prompt_builder
                ),
                relevance_prompt_builder=(
                    relevance_prompt_builder
                ),
                json_parser=json.loads,
                message_factory=FakeMessage,
                load_collection=load_collection,
                raw_writer=self.raw_writer,
                now_factory=self.clock,
                print_fn=self.printer,
            )
        )

        runtime_resources = {}
        if use_runtime_collection:
            runtime_resources[
                "collection"
            ] = self.collection

        self.policy = {
            "paths": {
                key: str(value)
                for key, value
                in self.paths.items()
            },
            "signature": {
                "experiment_dir": str(
                    self.root
                ),
                "chroma_collection_name": (
                    "collection-1"
                ),
                "openai_model": (
                    "fake-main"
                ),
                "embedding_model_name": (
                    "fake-embedding"
                ),
                "experiment_profile": {
                    "topic": "test"
                },
                "topic_profile": {
                    "topic": "test"
                },
                "generation_profile": {
                    "temperature": 0.1
                },
                "rag_policy": {
                    "profile": "strict"
                },
                "extraction_policy": {
                    "auto_rebuild": True
                },
                "card_required_fields": (
                    list(
                        CARD_REQUIRED_FIELDS
                    )
                ),
                "card_list_fields": list(
                    CARD_LIST_FIELDS
                ),
                "classification_fields": [
                    "relevance_level",
                    "include_in_state_of_art",
                    "relevance_reason",
                ],
                "extraction_prompt_version": (
                    "v4"
                ),
                "relevance_prompt_version": (
                    "v4"
                ),
                "kb_schema_version": "v3",
                "rag_clean_validation_version": (
                    "v3"
                ),
            },
            "retrieval": {
                "queries": ["query"],
                "profile": "strict",
                "profile_config": {
                    "fetch_k": 5
                },
                "max_chunks_per_paper": (
                    2
                ),
                "max_context_chars": (
                    18000
                ),
                "repair_max_chunks_per_paper": (
                    2
                ),
                "repair_max_context_chars": (
                    26000
                ),
            },
            "auto_rebuild_extraction": True,
            "force_rebuild_extraction": (
                False
            ),
            "title_repair_first_chunks": 2,
            "error_columns": list(
                ERROR_COLUMNS
            ),
            "trace_columns": list(
                TRACE_COLUMNS
            ),
            "next_stage": (
                "04_analisis_tematico"
            ),
        }

        self.agent_input = AgentInput(
            experiment_id="exp-1",
            run_id="run-1",
            stage_name=(
                "03_agente_extraccion_kb"
            ),
            attempt_number=1,
            mode=ExecutionMode.FULL_RUN,
            agent_context=AgentContext(
                allowed_tools=(),
                output_directory=str(
                    self.outputs_dir
                ),
                runtime_resources=(
                    runtime_resources
                ),
            ),
            dependencies={
                "chunks_clean": (
                    ArtifactReference(
                        path=str(
                            self.chunks_path
                        ),
                        hash=sha256_file(
                            self.chunks_path
                        ),
                    )
                ),
                "chroma_manifest": (
                    ArtifactReference(
                        path=str(
                            self.chroma_manifest_path
                        ),
                        hash=sha256_file(
                            self.chroma_manifest_path
                        ),
                    )
                ),
            },
            policy=self.policy,
        )

    def new_input(
        self,
        *,
        policy=None,
        dependencies=None,
        runtime_resources=None,
        run_id="run-2",
    ):
        return AgentInput(
            experiment_id="exp-1",
            run_id=run_id,
            stage_name=(
                "03_agente_extraccion_kb"
            ),
            attempt_number=1,
            mode=ExecutionMode.FULL_RUN,
            agent_context=AgentContext(
                allowed_tools=(),
                output_directory=str(
                    self.outputs_dir
                ),
                runtime_resources=(
                    runtime_resources
                    if runtime_resources
                    is not None
                    else {
                        "collection": (
                            self.collection
                        )
                    }
                ),
            ),
            dependencies=(
                dependencies
                if dependencies is not None
                else self.agent_input.dependencies
            ),
            policy=(
                policy
                if policy is not None
                else self.policy
            ),
        )

    def read_cards(self):
        records = []
        with self.paths[
            "CARDS_JSONL_PATH"
        ].open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.strip():
                    records.append(
                        json.loads(line)
                    )
        return records

    def close(self):
        self.temp.cleanup()


