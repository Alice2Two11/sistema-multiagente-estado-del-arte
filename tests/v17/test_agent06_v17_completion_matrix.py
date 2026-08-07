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
from src.contracts.agent_result import QualityStatus
from src.state.fingerprints import sha256_file
from tests.v17.agent06_v17_test_support import (
    EvidenceAwareLLM,
    SyntheticCollection,
    chroma_client_factory,
    runtime_factory_for,
    write_synthetic_project,
)
from tests.v17.test_agent06_v17_end_to_end_integration import (
    HYBRID,
    _LegacyLengthLLM,
)


class TestAgent06V17CompletionMatrix(unittest.TestCase):
    def _execute(
        self,
        *,
        policy: dict | None = None,
        quantitative: str = "complete",
        sections: list[dict] | None = None,
        target_total_words: int = 120,
        llm=None,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _, store, rows = write_synthetic_project(
            root,
            quantitative=quantitative,
            sections=sections,
            target_total_words=target_total_words,
        )
        selected_llm = llm or EvidenceAwareLLM("valid")
        agent, agent_input, config = build_real_draft_execution(
            root,
            collection_factory=lambda _: SyntheticCollection(rows),
            runtime_factory=runtime_factory_for(selected_llm),
            chroma_client_factory=chroma_client_factory,
            policy_overrides=policy,
        )
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(
            store=store,
            agent=agent,
            prepared=prepared,
        )
        return root, store, selected_llm, agent_input, config, executed

    def test_validation_call_cost_is_observable_for_v16_and_v17(self):
        _, _, hybrid_llm, _, _, hybrid = self._execute(
            policy=HYBRID,
            quantitative="none",
        )
        _, _, legacy_llm, _, _, legacy = self._execute(
            policy={},
            quantitative="none",
            llm=_LegacyLengthLLM("valid"),
        )
        self.assertEqual(hybrid.result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(legacy.result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(hybrid_llm.calls, 3)
        self.assertEqual(legacy_llm.calls, 3)
        self.assertEqual(hybrid.result.tool_usage.validation_calls, 7)
        self.assertEqual(legacy.result.tool_usage.validation_calls, 4)
        self.assertEqual(
            hybrid.result.tool_usage.validation_calls
            - legacy.result.tool_usage.validation_calls,
            3,
        )

    def test_integrated_budgets_support_multiple_organizational_sections_and_residue(self):
        sections = [
            {
                "section_id": "O1",
                "section_title": "Introducción",
                "section_type": "introduction",
                "requires_sources": False,
                "purpose": "Presentar la revisión",
                "papers_to_use": [],
            },
            {
                "section_id": "O2",
                "section_title": "Cierre",
                "source_requirement": "source_free",
                "purpose": "Cerrar la revisión",
                "papers_to_use": [],
            },
            {
                "section_id": "S1",
                "section_title": "Resultados",
                "section_type": "substantive",
                "requires_sources": True,
                "purpose": "comparar accuracy dataset model",
                "key_arguments": ["accuracy"],
                "evidence_needs": ["resultados"],
                "papers_to_use": [
                    {"source_filename": "paper_a.pdf", "title": "A"},
                    {"source_filename": "paper_b.pdf", "title": "B"},
                ],
            },
        ]
        policy = {
            **HYBRID,
            "organizational_target_words": 10,
            "organizational_minimum_words": 1,
            "organizational_maximum_words": 40,
            "substantive_minimum_ratio": 0.40,
            "substantive_maximum_ratio": 2.00,
        }
        _, _, _, _, _, executed = self._execute(
            policy=policy,
            quantitative="none",
            sections=sections,
            target_total_words=61,
        )
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        frame = pd.read_csv(
            executed.result.output_artifacts["draft_sections.csv"].path
        )
        targets = frame.set_index("section_id")["target_words"].astype(int).to_dict()
        self.assertEqual(sum(targets.values()), 61)
        self.assertEqual(targets, {"O1": 10, "O2": 10, "S1": 41})
        source_free = (
            frame.set_index("section_id")[
                "source_free_organizational_section"
            ].astype(bool).to_dict()
        )
        self.assertEqual(source_free, {"O1": True, "O2": True, "S1": False})

    def test_global_length_below_minimum_remains_blocking(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _, store, rows = write_synthetic_project(root, quantitative="none")
        active_path = root / "active_experiment.json"
        active = json.loads(active_path.read_text(encoding="utf-8"))
        active["generation_profile"]["min_total_words"] = 999
        active_path.write_text(json.dumps(active), encoding="utf-8")
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
        self.assertEqual(executed.result.quality_status, QualityStatus.NEEDS_REVISION)
        report = json.loads(
            Path(
                executed.result.output_artifacts[
                    "draft_validation_report.json"
                ].path
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(report["global_length_valid"])
        self.assertNotIn("state_of_art_draft.json", executed.result.output_artifacts)
        self.assertNotIn("state_of_art_draft.md", executed.result.output_artifacts)

    def test_validation_error_union_is_stable_and_without_duplicates(self):
        results = []
        for _ in range(2):
            _, _, _, _, config, executed = self._execute(
                policy=HYBRID,
                llm=EvidenceAwareLLM("invalid_citation"),
            )
            self.assertEqual(
                executed.result.quality_status,
                QualityStatus.NEEDS_REVISION,
            )
            paths = sorted(
                (Path(config["output_dir"]) / "raw_section_outputs").glob(
                    "S1_attempt_*_validation.json"
                )
            )
            errors = [
                json.loads(path.read_text(encoding="utf-8"))["validation_errors"]
                for path in paths
            ]
            self.assertEqual(len(errors), 3)
            for item in errors:
                self.assertEqual(len(item), len(dict.fromkeys(item)))
                self.assertIn("invalid_citation", item)
                self.assertIn("EMPTY_DRAFT_TEXT", item)
            results.append(errors)
        self.assertEqual(results[0], results[1])

    def test_artifact_references_match_real_files_and_hashes(self):
        _, _, _, agent_input, _, executed = self._execute(
            policy=HYBRID,
            quantitative="none",
        )
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        for name, reference in executed.result.output_artifacts.items():
            if name == "raw_section_outputs":
                continue
            path = Path(reference.path)
            self.assertTrue(path.is_file(), name)
            self.assertEqual(reference.hash, sha256_file(path), name)
        manifest = json.loads(
            Path(
                executed.result.output_artifacts[
                    "draft_generation_manifest.json"
                ].path
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["fingerprint"],
            agent_input.policy["current_fingerprint"],
        )

    def test_quantitative_planning_rows_do_not_authorize_invalid_citations(self):
        _, _, _, _, config, executed = self._execute(policy=HYBRID)
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        traces = sorted(
            (Path(config["output_dir"]) / "raw_section_outputs").glob(
                "*_rag_trace.json"
            )
        )
        self.assertTrue(traces)
        for path in traces:
            trace = json.loads(path.read_text(encoding="utf-8"))
            allowed = set(trace["allowed_citations"])
            self.assertNotIn("[unauthorized.pdf | u1]", allowed)
            self.assertNotIn("[paper_a.pdf | missing]", allowed)
            expected = {
                f"[{row['source_filename']} | {row['chunk_id']}]"
                for row in trace["retrieved_chunks"]
            }
            self.assertEqual(allowed, expected)

    def test_productive_modules_do_not_import_experimental_agent_or_runtime(self):
        root = Path(__file__).resolve().parents[2]
        productive = [
            root / "src" / "agents" / "draft_writing_agent.py",
            root / "src" / "adapters" / "draft_writing_runtime.py",
        ]
        forbidden = (
            "draft_writing_agent_hybrid_experimental",
            "draft_writing_hybrid_runtime",
        )
        for path in productive:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
