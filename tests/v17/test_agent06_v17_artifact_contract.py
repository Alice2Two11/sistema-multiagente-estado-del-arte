from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.adapters.draft_writing_runtime import (
    build_real_draft_execution,
    execute_prepared_draft,
    prepare_draft_execution,
)
from src.state.fingerprints import sha256_file
from src.tools.draft_writing.artifacts import (
    DATASET_TECHNIQUE_USED_COLUMNS,
    QUANTITATIVE_USED_COLUMNS,
    write_draft_artifacts,
)
from tests.v17.agent06_v17_test_support import (
    CONTRACT_ARTIFACTS,
    EvidenceAwareLLM,
    SyntheticCollection,
    chroma_client_factory,
    runtime_factory_for,
    write_synthetic_project,
)
from tests.v17.test_agent06_v17_end_to_end_integration import HYBRID


class TestAgent06V17ArtifactContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _, self.store, rows = write_synthetic_project(self.root)
        self.llm = EvidenceAwareLLM("valid")
        agent, self.agent_input, self.cfg = build_real_draft_execution(
            self.root,
            collection_factory=lambda _: SyntheticCollection(rows),
            runtime_factory=runtime_factory_for(self.llm),
            chroma_client_factory=chroma_client_factory,
            policy_overrides=HYBRID,
        )
        prepared = prepare_draft_execution(
            store=self.store,
            agent_input=self.agent_input,
        )
        self.executed = execute_prepared_draft(
            store=self.store,
            agent=agent,
            prepared=prepared,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_twelve_contractual_artifacts(self):
        result = self.executed.result
        contractual = {
            name for name in result.output_artifacts if name != "raw_section_outputs"
        }
        self.assertEqual(contractual, set(CONTRACT_ARTIFACTS))
        self.assertTrue(Path(result.output_artifacts["raw_section_outputs"].path).is_dir())
        for name in CONTRACT_ARTIFACTS:
            reference = result.output_artifacts[name]
            self.assertTrue(Path(reference.path).is_file(), name)
            self.assertEqual(reference.hash, sha256_file(reference.path))

    def test_json_artifact_schemas_and_cross_file_coherence(self):
        result = self.executed.result
        draft = json.loads(
            Path(result.output_artifacts["state_of_art_draft.json"].path).read_text()
        )
        validation = json.loads(
            Path(result.output_artifacts["draft_validation_report.json"].path).read_text()
        )
        manifest = json.loads(
            Path(result.output_artifacts["draft_generation_manifest.json"].path).read_text()
        )
        self.assertEqual(manifest["experiment_id"], self.agent_input.experiment_id)
        self.assertEqual(
            manifest["fingerprint"],
            self.agent_input.policy["current_fingerprint"],
        )
        self.assertTrue(validation["validation_ok"])
        self.assertEqual(len(draft["sections"]), validation["section_count"])
        self.assertEqual(
            manifest["versions"]["stage"],
            draft["generation_summary"]["stage_version"],
        )

    def test_csv_artifact_minimum_schemas(self):
        result = self.executed.result
        required_columns = {
            "draft_sections.csv": {
                "section_id",
                "draft_text",
                "target_words",
                "within_section_range",
            },
            "draft_rag_evidence.csv": {
                "section_id",
                "source_filename",
                "chunk_id",
                "text",
                "selection_bucket",
            },
            "draft_quality_check.csv": {
                "section_id",
                "section_validation_ok",
                "invalid_citation_count",
            },
            "draft_length_check.csv": {
                "section_id",
                "word_count",
                "minimum_words",
                "maximum_words",
            },
            "draft_claim_evidence.csv": {
                "section_id",
                "claim_text",
                "source_filename",
                "chunk_id",
                "allowed_for_section",
            },
            "numeric_hallucination_check.csv": {
                "numeric_value",
                "found_in_cited_chunks",
                "risk",
            },
        }
        for name, columns in required_columns.items():
            frame = pd.read_csv(result.output_artifacts[name].path)
            self.assertTrue(columns.issubset(frame.columns), (name, set(frame.columns)))

    def test_empty_quantitative_artifacts_are_readable_with_exact_schema(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            artifacts = write_draft_artifacts(
                output,
                {"title": "Draft", "sections": []},
                [],
                {"validation_ok": True},
                pd.DataFrame(),
                pd.DataFrame(),
                {"fingerprint": "fp"},
                [],
                [],
                [],
                [],
            )
            quantitative = pd.read_csv(
                artifacts["quantitative_comparative_table_used.csv"].path
            )
            datasets = pd.read_csv(
                artifacts["dataset_technique_summary_used.csv"].path
            )
            self.assertEqual(len(quantitative), 0)
            self.assertEqual(list(quantitative.columns), list(QUANTITATIVE_USED_COLUMNS))
            self.assertEqual(len(datasets), 0)
            self.assertEqual(
                list(datasets.columns),
                list(DATASET_TECHNIQUE_USED_COLUMNS),
            )

    def test_zero_row_frames_keep_contract_columns_and_legitimate_extras(self):
        quant = pd.DataFrame(columns=["source_filename", "custom_quant_field"])
        datasets = pd.DataFrame(columns=["source_filename", "custom_dataset_field"])
        with tempfile.TemporaryDirectory() as td:
            artifacts = write_draft_artifacts(
                Path(td),
                {"title": "Draft", "sections": []},
                [],
                {"validation_ok": True},
                quant,
                datasets,
                {"fingerprint": "fp"},
                [],
                [],
                [],
                [],
            )
            quant_read = pd.read_csv(
                artifacts["quantitative_comparative_table_used.csv"].path
            )
            datasets_read = pd.read_csv(
                artifacts["dataset_technique_summary_used.csv"].path
            )
            self.assertEqual(
                list(quant_read.columns),
                list(QUANTITATIVE_USED_COLUMNS) + ["custom_quant_field"],
            )
            self.assertEqual(
                list(datasets_read.columns),
                list(DATASET_TECHNIQUE_USED_COLUMNS) + ["custom_dataset_field"],
            )

    def test_nonempty_quantitative_data_is_not_lost_or_reordered(self):
        quant_row = {
            column: f"q_{index}" for index, column in enumerate(QUANTITATIVE_USED_COLUMNS)
        }
        quant_row["extra_quant"] = "preserved"
        dataset_row = {
            "source_filename": "paper.pdf",
            "paper_title": "Paper",
            "techniques": "Technique",
            "datasets": "Dataset",
            "extra_dataset": "preserved",
        }
        with tempfile.TemporaryDirectory() as td:
            artifacts = write_draft_artifacts(
                Path(td),
                {"title": "Draft", "sections": []},
                [],
                {"validation_ok": True},
                pd.DataFrame([quant_row]),
                pd.DataFrame([dataset_row]),
                {"fingerprint": "fp"},
                [],
                [],
                [],
                [],
            )
            quant_read = pd.read_csv(
                artifacts["quantitative_comparative_table_used.csv"].path,
                dtype=str,
                keep_default_na=False,
            )
            datasets_read = pd.read_csv(
                artifacts["dataset_technique_summary_used.csv"].path,
                dtype=str,
                keep_default_na=False,
            )
            self.assertEqual(
                list(quant_read.columns),
                list(QUANTITATIVE_USED_COLUMNS) + ["extra_quant"],
            )
            self.assertEqual(quant_read.iloc[0].to_dict(), quant_row)
            self.assertEqual(datasets_read.iloc[0].to_dict(), dataset_row)

    def test_v17_without_quantitative_context_publishes_readable_empty_csvs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, store, rows = write_synthetic_project(root, quantitative="none")
            llm = EvidenceAwareLLM("valid")
            agent, agent_input, _ = build_real_draft_execution(
                root,
                collection_factory=lambda _: SyntheticCollection(rows),
                runtime_factory=runtime_factory_for(llm),
                chroma_client_factory=chroma_client_factory,
                policy_overrides=HYBRID,
            )
            prepared = prepare_draft_execution(store=store, agent_input=agent_input)
            executed = execute_prepared_draft(
                store=store,
                agent=agent,
                prepared=prepared,
            )
            self.assertEqual(executed.result.quality_status.value, "APPROVED")
            quantitative = pd.read_csv(
                executed.result.output_artifacts[
                    "quantitative_comparative_table_used.csv"
                ].path
            )
            datasets = pd.read_csv(
                executed.result.output_artifacts[
                    "dataset_technique_summary_used.csv"
                ].path
            )
            self.assertEqual(len(quantitative), 0)
            self.assertEqual(list(quantitative.columns), list(QUANTITATIVE_USED_COLUMNS))
            self.assertEqual(len(datasets), 0)
            self.assertEqual(
                list(datasets.columns),
                list(DATASET_TECHNIQUE_USED_COLUMNS),
            )

    def test_markdown_contains_all_sections_and_citations(self):
        result = self.executed.result
        markdown = Path(result.output_artifacts["state_of_art_draft.md"].path).read_text()
        draft = json.loads(
            Path(result.output_artifacts["state_of_art_draft.json"].path).read_text()
        )
        for section in draft["sections"]:
            self.assertIn(section["section_title"], markdown)
            self.assertIn(section["draft_text"], markdown)


if __name__ == "__main__":
    unittest.main()
