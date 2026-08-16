from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents.draft_writing_agent import (
    DraftWritingAgent,
    HYBRID_RETRIEVAL_STRATEGY,
    LEGACY_RETRIEVAL_STRATEGY,
    HYBRID_VERSIONS,
    LEGACY_VERSIONS,
)
from src.config.draft_writing_policy_config import get_draft_writing_policy
from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode
from src.contracts.agent_result import QualityStatus, TransitionAction
from src.tools.draft_writing.artifacts import NAMES


class FakeRuntime:
    def __init__(self, outputs=None):
        self.collection = object()
        self.outputs = list(outputs or [])
        self.prompts: list[str] = []
        self.invoke_calls = 0

    def invoke(self, prompt):
        self.prompts.append(prompt)
        self.invoke_calls += 1
        if self.outputs:
            return self.outputs.pop(0)
        return {
            "section_id": "S1",
            "section_title": "Methods",
            "draft_text": "Supported statement [a.pdf | c1].",
            "claims": [
                {
                    "claim": "Supported statement",
                    "supporting_citations": ["[a.pdf | c1]"],
                }
            ],
        }

    def parse(self, raw):
        return raw if isinstance(raw, dict) else json.loads(raw)


class TestAgent06ContractualAgentIntegrationV17(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "draft"
        self.output.mkdir()
        self.pipeline_state = self.root / "pipeline_state.json"
        self.pipeline_state.write_text('{"sentinel":"unchanged"}', encoding="utf-8")
        self.pipeline_before = self.pipeline_state.read_bytes()

    def tearDown(self):
        self.assertEqual(self.pipeline_state.read_bytes(), self.pipeline_before)
        self.temp.cleanup()

    def make_sections(self):
        return [
            {
                "section_id": "S0",
                "section_title": "Introducción",
                "section_type": "introduction",
                "requires_sources": False,
                "purpose": "Organize",
                "papers_to_use": [],
            },
            {
                "section_id": "S1",
                "section_title": "Methods",
                "section_type": "substantive",
                "requires_sources": True,
                "purpose": "Compare methods",
                "key_arguments": ["accuracy"],
                "evidence_needs": ["results"],
                "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}],
            },
            {
                "section_id": "S2",
                "section_title": "Results",
                "section_type": "substantive",
                "requires_sources": True,
                "purpose": "Compare results",
                "key_arguments": ["error"],
                "evidence_needs": ["metrics"],
                "papers_to_use": [{"source_filename": "b.pdf", "title": "B"}],
            },
        ]

    def make_bundle(self, sections=None, quantitative_rows=None):
        sections = sections or self.make_sections()
        chunks = pd.DataFrame(
            [
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Accuracy reached 95% in dataset A.",
                },
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c2",
                    "text": "Error was 1.3 units.",
                },
                {
                    "source_filename": "b.pdf",
                    "chunk_id": "b1",
                    "text": "Accuracy reached 90% in dataset B.",
                },
            ]
        )
        quantitative = pd.DataFrame(quantitative_rows or [])
        dataset_summary = pd.DataFrame(
            [{"source_filename": "a.pdf", "dataset": "A"}]
        )
        return {
            "outline": {"title": "Draft", "topic": "T", "sections": sections},
            "chunks": chunks,
            "quantitative": quantitative,
            "dataset_summary": dataset_summary,
        }

    def make_input(self, strategy, *, attempt=1, overrides=None):
        policy_overrides = {
            "retrieval_strategy": strategy,
            "top_k_evidence_per_section": 8,
            "chroma_quota": 3,
            "csv_quota": 3,
            "rrf_quota": 2,
        }
        policy_overrides.update(overrides or {})
        policy = get_draft_writing_policy(policy_overrides)
        policy.update(
            {
                "target_total_words": 1000,
                "min_total_words": 1,
                "max_total_words": 5000,
                "output_language": "español",
                "writing_mode": "síntesis crítica",
                "focus_mode": "comparativo",
                "citation_style": "trazable",
                "current_fingerprint": "fingerprint-v17-test",
            }
        )
        return AgentInput(
            experiment_id="exp",
            run_id="run",
            stage_name="06_agente_redactor",
            attempt_number=attempt,
            mode=ExecutionMode.FULL_RUN,
            agent_context=AgentContext(
                allowed_tools=("double",), output_directory=str(self.output)
            ),
            dependencies={},
            policy=policy,
        )

    @staticmethod
    def valid_section_validation():
        return {
            "validation_ok": True,
            "errors": [],
            "citation_errors": [],
            "claim_errors": [],
            "numeric_errors": [],
        }

    def successful_patches(self, bundle):
        return (
            patch(
                "src.agents.draft_writing_agent.validate_draft_dependencies",
                return_value=bundle,
            ),
            patch(
                "src.agents.draft_writing_agent.normalize_generated_section",
                side_effect=lambda parsed, allowed: dict(parsed),
            ),
            patch(
                "src.agents.draft_writing_agent.validate_generated_section",
                return_value=self.valid_section_validation(),
            ),
            patch(
                "src.agents.draft_writing_agent.build_draft_reports",
                return_value=({}, [], [], [], []),
            ),
            patch(
                "src.agents.draft_writing_agent.validate_draft_global",
                return_value={
                    "validation_ok": True,
                    "errors": [],
                    "global_length_valid": True,
                },
            ),
        )

    def run_success(
        self,
        strategy,
        bundle,
        legacy_evidence=None,
        hybrid_evidence=None,
        augmented=None,
        policy_overrides=None,
    ):
        runtime = FakeRuntime(
            outputs=[
                {
                    "section_id": "S1",
                    "section_title": "Methods",
                    "draft_text": "Supported statement [a.pdf | c1].",
                    "claims": [
                        {
                            "claim": "Supported statement",
                            "supporting_citations": ["[a.pdf | c1]"],
                        }
                    ],
                },
                {
                    "section_id": "S2",
                    "section_title": "Results",
                    "draft_text": "Supported result [b.pdf | b1].",
                    "claims": [
                        {
                            "claim": "Supported result",
                            "supporting_citations": ["[b.pdf | b1]"],
                        }
                    ],
                },
            ]
        )
        agent = DraftWritingAgent(runtime)
        input_value = self.make_input(
            strategy, overrides=policy_overrides
        )
        legacy_evidence = legacy_evidence or [
            {
                "source_filename": "a.pdf",
                "chunk_id": "c1",
                "text": "Accuracy reached 95% in dataset A.",
                "score": 0.9,
                "retrieval_method": "CHROMA",
            }
        ]
        hybrid_evidence = hybrid_evidence or legacy_evidence
        augmented = augmented or hybrid_evidence
        patches = self.successful_patches(bundle)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence",
            return_value=legacy_evidence,
        ) as legacy_mock, patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid",
            return_value=hybrid_evidence,
        ) as hybrid_mock, patch(
            "src.agents.draft_writing_agent.augment_evidence_with_quantitative_chunks_greedy",
            return_value=augmented,
        ) as quantitative_mock:
            result = agent.execute(input_value)
        return result, runtime, legacy_mock, hybrid_mock, quantitative_mock

    def test_legacy_branch_uses_only_legacy_retrieval(self):
        bundle = self.make_bundle()
        result, _, legacy, hybrid, quantitative = self.run_success(
            LEGACY_RETRIEVAL_STRATEGY, bundle
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(legacy.call_count, 3)
        hybrid.assert_not_called()
        quantitative.assert_not_called()

    def test_legacy_manifest_contains_only_legacy_versions(self):
        result, _, _, _, _ = self.run_success(
            LEGACY_RETRIEVAL_STRATEGY, self.make_bundle()
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        manifest = json.loads(
            (self.output / "draft_generation_manifest.json").read_text()
        )
        self.assertEqual(manifest["retrieval_strategy"], LEGACY_RETRIEVAL_STRATEGY)
        self.assertEqual(
            manifest["versions"]["stage"], "06_AGENTIC_V16_BEHAVIOR_PRESERVING"
        )
        self.assertEqual(
            manifest["versions"]["rag"], "legacy_chroma_then_csv_restricted_v1"
        )
        self.assertNotIn("quantitative_selection", manifest["versions"])
        self.assertNotIn("budget", manifest["versions"])

    def test_hybrid_branch_calls_retrieval_and_augmentation_per_substantive_section(self):
        result, _, legacy, hybrid, quantitative = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY, self.make_bundle()
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        legacy.assert_not_called()
        self.assertEqual(hybrid.call_count, 2)
        self.assertEqual(quantitative.call_count, 2)
        self.assertEqual(result.tool_usage.retrieval_rounds, 2)

    def test_hybrid_passes_policy_configuration_to_modules(self):
        bundle = self.make_bundle()
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        policy = self.make_input(HYBRID_RETRIEVAL_STRATEGY).policy
        section = bundle["outline"]["sections"][1]
        quant_context = agent._quant_context(section, bundle, 12)
        with patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid",
            return_value=[],
        ) as hybrid, patch(
            "src.agents.draft_writing_agent.augment_evidence_with_quantitative_chunks_greedy",
            return_value=[],
        ) as quantitative:
            agent._retrieve_section_evidence(
                section, bundle, policy, HYBRID_RETRIEVAL_STRATEGY, quant_context
            )
        kwargs = hybrid.call_args.kwargs
        self.assertEqual(kwargs["candidate_multiplier"], 3)
        self.assertEqual(kwargs["chroma_quota"], 3)
        self.assertEqual(kwargs["csv_quota"], 3)
        self.assertEqual(kwargs["rrf_quota"], 2)
        self.assertEqual(kwargs["rrf_k"], 60)
        self.assertEqual(kwargs["top_k_evidence_per_section"], 8)
        self.assertEqual(kwargs["max_evidence_chars"], 18000)
        self.assertEqual(kwargs["max_candidates_per_source"], 24)
        qkwargs = quantitative.call_args.kwargs
        self.assertEqual(qkwargs["quantitative_evidence_quota"], 2)
        self.assertEqual(qkwargs["top_k_evidence_per_section"], 8)
        self.assertEqual(qkwargs["max_quantitative_rows_per_section"], 12)

    def test_source_aware_budgets_are_used_only_in_hybrid_mode(self):
        sections = self.make_sections()
        legacy_policy = self.make_input(LEGACY_RETRIEVAL_STRATEGY).policy
        hybrid_policy = self.make_input(HYBRID_RETRIEVAL_STRATEGY).policy
        legacy = DraftWritingAgent._section_budgets(
            sections, legacy_policy, LEGACY_RETRIEVAL_STRATEGY
        )
        hybrid = DraftWritingAgent._section_budgets(
            sections, hybrid_policy, HYBRID_RETRIEVAL_STRATEGY
        )
        self.assertNotIn("budget_type", legacy["S0"])
        self.assertEqual(hybrid["S0"]["budget_type"], "source_free_organizational")
        self.assertEqual(hybrid["S0"]["target_words"], 40)
        self.assertEqual(sum(row["target_words"] for row in hybrid.values()), 1000)
        self.assertEqual(hybrid["S1"]["target_words"], 480)
        self.assertEqual(hybrid["S2"]["target_words"], 480)

    def test_hybrid_prompt_allows_only_final_evidence_citations(self):
        hybrid = [
            {
                "source_filename": "a.pdf",
                "chunk_id": "c1",
                "text": "base",
                "retrieval_sources": ["chroma", "csv"],
                "rrf_score": 0.2,
            }
        ]
        final = [
            {
                "source_filename": "a.pdf",
                "chunk_id": "c2",
                "text": "Error was 1.3 units.",
                "selection_bucket": "quantitative_greedy",
                "quantitative_values": ["1.3"],
            }
        ]
        result, runtime, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY,
            self.make_bundle(
                quantitative_rows=[
                    {
                        "source_filename": "a.pdf",
                        "chunk_id": "c2",
                        "verification_status": "confirmed_in_source_chunk",
                        "value": "1.3",
                    }
                ]
            ),
            hybrid_evidence=hybrid,
            augmented=final,
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        substantive_prompt = next(p for p in runtime.prompts if '"section_id": "S1"' in p)
        allowed_block = substantive_prompt.split("ALLOWED_CITATIONS:", 1)[1].split(
            "EVIDENCIA:", 1
        )[0]
        self.assertIn("[a.pdf | c2]", allowed_block)
        self.assertNotIn("[a.pdf | c1]", allowed_block)

    def test_quantitative_context_does_not_create_allowed_citations(self):
        result, runtime, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY,
            self.make_bundle(
                quantitative_rows=[
                    {
                        "source_filename": "a.pdf",
                        "chunk_id": "missing",
                        "verification_status": "confirmed_in_source_chunk",
                        "value": "99%",
                    }
                ]
            ),
            augmented=[
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Accuracy reached 95%.",
                }
            ],
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        prompt = next(p for p in runtime.prompts if '"section_id": "S1"' in p)
        allowed = prompt.split("ALLOWED_CITATIONS:", 1)[1].split("EVIDENCIA:", 1)[0]
        self.assertNotIn("missing", allowed)
        self.assertIn("99%", prompt.split("CONTEXTO CUANTITATIVO CONFIRMADO:", 1)[1])

    def test_hybrid_manifest_publishes_v17_versions(self):
        result, _, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY, self.make_bundle()
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        manifest = json.loads(
            (self.output / "draft_generation_manifest.json").read_text()
        )
        self.assertEqual(
            manifest["versions"]["stage"],
            "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE",
        )
        self.assertEqual(
            manifest["versions"]["rag"], "hybrid_chroma_csv_rrf_balanced_v1"
        )
        self.assertEqual(
            manifest["versions"]["quantitative_selection"],
            "confirmed_literal_greedy_coverage_v1",
        )
        self.assertEqual(
            manifest["versions"]["budget"], "source_aware_exact_total_v1"
        )
        self.assertEqual(
            manifest["versions"]["validation"],
            "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
        )

    def test_final_evidence_metadata_is_preserved_in_rag_artifact(self):
        final = [
            {
                "source_filename": "a.pdf",
                "chunk_id": "c2",
                "text": "Error was 1.3 units.",
                "retrieval_source": "hybrid",
                "retrieval_sources": ["chroma", "csv"],
                "chroma_rank": 4,
                "csv_rank": 1,
                "rrf_score": 0.04,
                "selection_bucket": "quantitative_greedy",
                "selection_order": 1,
                "quantitative_values": ["1.3"],
                "quantitative_coverage_keys": ["rmse|1.3||||"],
                "quantitative_marginal_gain": 1,
                "quantitative_row_ids": ["r1"],
                "verification_statuses": ["confirmed_in_source_chunk"],
            }
        ]
        result, _, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY,
            self.make_bundle(),
            augmented=final,
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        frame = pd.read_csv(self.output / "draft_rag_evidence.csv")
        for column in (
            "retrieval_sources",
            "chroma_rank",
            "csv_rank",
            "rrf_score",
            "selection_bucket",
            "selection_order",
            "quantitative_values",
            "quantitative_coverage_keys",
            "quantitative_marginal_gain",
            "quantitative_row_ids",
            "verification_statuses",
        ):
            self.assertIn(column, frame.columns)

    def test_success_publishes_exactly_twelve_contractual_artifacts(self):
        result, _, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY, self.make_bundle()
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(set(result.output_artifacts) - {"raw_section_outputs"}, set(NAMES))
        self.assertEqual(len(NAMES), 12)

    def test_three_failed_attempts_publish_only_validation_and_raw_outputs(self):
        sections = [self.make_sections()[1]]
        bundle = self.make_bundle(sections=sections)
        runtime = FakeRuntime(outputs=[{}, {}, {}])
        agent = DraftWritingAgent(runtime)
        input_value = self.make_input(HYBRID_RETRIEVAL_STRATEGY, attempt=2)
        invalid = {
            "validation_ok": False,
            "errors": ["invalid"],
            "citation_errors": [],
            "claim_errors": [],
            "numeric_errors": [{"reason": "unsupported_number"}],
        }
        with patch(
            "src.agents.draft_writing_agent.validate_draft_dependencies",
            return_value=bundle,
        ), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid",
            return_value=[
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Evidence.",
                }
            ],
        ), patch(
            "src.agents.draft_writing_agent.augment_evidence_with_quantitative_chunks_greedy",
            return_value=[
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Evidence.",
                }
            ],
        ), patch(
            "src.agents.draft_writing_agent.normalize_generated_section",
            return_value={"draft_text": "Unsupported 99% [a.pdf | c1].", "claims": []},
        ), patch(
            "src.agents.draft_writing_agent.validate_generated_section",
            return_value=invalid,
        ):
            result = agent.execute(input_value)
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.HALT_STAGE)
        self.assertEqual(runtime.invoke_calls, 3)
        self.assertEqual(
            set(result.output_artifacts),
            {"draft_validation_report.json", "raw_section_outputs"},
        )
        self.assertFalse((self.output / "state_of_art_draft.json").exists())

    def test_invalid_global_validation_does_not_publish_draft(self):
        sections = [self.make_sections()[1]]
        bundle = self.make_bundle(sections=sections)
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        input_value = self.make_input(HYBRID_RETRIEVAL_STRATEGY)
        with patch(
            "src.agents.draft_writing_agent.validate_draft_dependencies",
            return_value=bundle,
        ), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid",
            return_value=[
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Evidence.",
                }
            ],
        ), patch(
            "src.agents.draft_writing_agent.augment_evidence_with_quantitative_chunks_greedy",
            return_value=[
                {
                    "source_filename": "a.pdf",
                    "chunk_id": "c1",
                    "text": "Evidence.",
                }
            ],
        ), patch(
            "src.agents.draft_writing_agent.normalize_generated_section",
            side_effect=lambda parsed, allowed: dict(parsed),
        ), patch(
            "src.agents.draft_writing_agent.validate_generated_section",
            return_value=self.valid_section_validation(),
        ), patch(
            "src.agents.draft_writing_agent.build_draft_reports",
            return_value=({}, [], [], [], []),
        ), patch(
            "src.agents.draft_writing_agent.validate_draft_global",
            return_value={"validation_ok": False, "errors": ["length"]},
        ):
            result = agent.execute(input_value)
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.RETRY)
        self.assertFalse((self.output / "state_of_art_draft.json").exists())

    def test_no_quantitative_tables_passes_empty_context_to_augmentation(self):
        sections = [self.make_sections()[1]]
        bundle = self.make_bundle(sections=sections)
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        input_value = self.make_input(HYBRID_RETRIEVAL_STRATEGY)
        patches = self.successful_patches(bundle)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid",
            return_value=[
                {"source_filename": "a.pdf", "chunk_id": "c1", "text": "Evidence."}
            ],
        ), patch(
            "src.agents.draft_writing_agent.augment_evidence_with_quantitative_chunks_greedy",
            return_value=[
                {"source_filename": "a.pdf", "chunk_id": "c1", "text": "Evidence."}
            ],
        ) as quantitative:
            result = agent.execute(input_value)
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        context = quantitative.call_args.args[2]
        self.assertEqual(context["quantitative_results"], [])

    def test_unsupported_strategy_fails_without_retrieval(self):
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        input_value = self.make_input(LEGACY_RETRIEVAL_STRATEGY)
        object.__setattr__(
            input_value,
            "policy",
            {**dict(input_value.policy), "retrieval_strategy": "unknown"},
        )
        with patch(
            "src.agents.draft_writing_agent.validate_draft_dependencies",
            return_value=self.make_bundle(),
        ), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence"
        ) as legacy, patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence_hybrid"
        ) as hybrid:
            result = agent.execute(input_value)
        self.assertEqual(result.quality_status, QualityStatus.REJECTED)
        legacy.assert_not_called()
        hybrid.assert_not_called()

    def test_repeated_hybrid_execution_is_deterministic_with_doubles(self):
        bundle = self.make_bundle()
        first, _, _, _, _ = self.run_success(HYBRID_RETRIEVAL_STRATEGY, bundle)
        first_manifest = (self.output / "draft_generation_manifest.json").read_text()
        first_evidence = (self.output / "draft_rag_evidence.csv").read_text()
        for path in self.output.iterdir():
            if path.is_file():
                path.unlink()
        second, _, _, _, _ = self.run_success(HYBRID_RETRIEVAL_STRATEGY, bundle)
        second_manifest = (self.output / "draft_generation_manifest.json").read_text()
        second_evidence = (self.output / "draft_rag_evidence.csv").read_text()
        self.assertEqual(first.quality_status, second.quality_status)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_evidence, second_evidence)

    def test_test_module_uses_only_isolated_contractual_imports(self):
        import ast

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("src.state.state_store", imported_modules)
        self.assertNotIn("src.adapters.draft_writing_runtime", imported_modules)
        self.assertFalse(any(module.startswith("openai") for module in imported_modules))


    def test_legacy_ignores_injected_hybrid_versions_everywhere(self):
        injected = {
            "stage_version": HYBRID_VERSIONS["stage_version"],
            "rag_version": HYBRID_VERSIONS["rag_version"],
            "validation_version": "arbitrary_validation_v999",
        }
        result, _, _, _, _ = self.run_success(
            LEGACY_RETRIEVAL_STRATEGY,
            self.make_bundle(),
            policy_overrides=injected,
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        manifest = json.loads(
            (self.output / "draft_generation_manifest.json").read_text()
        )
        report = json.loads(
            (self.output / "draft_validation_report.json").read_text()
        )
        draft = json.loads(
            (self.output / "state_of_art_draft.json").read_text()
        )
        self.assertEqual(
            manifest["versions"]["stage"], LEGACY_VERSIONS["stage_version"]
        )
        self.assertEqual(
            manifest["versions"]["rag"], LEGACY_VERSIONS["rag_version"]
        )
        self.assertEqual(
            manifest["versions"]["validation"],
            LEGACY_VERSIONS["validation_version"],
        )
        self.assertNotIn("quantitative_selection", manifest["versions"])
        self.assertNotIn("budget", manifest["versions"])
        self.assertEqual(
            report["validation_version"],
            LEGACY_VERSIONS["validation_version"],
        )
        summary = draft["generation_summary"]
        self.assertEqual(summary["stage_version"], LEGACY_VERSIONS["stage_version"])
        self.assertEqual(summary["rag_version"], LEGACY_VERSIONS["rag_version"])
        self.assertEqual(
            summary["validation_version"],
            LEGACY_VERSIONS["validation_version"],
        )
        self.assertNotIn("quantitative_selection_version", summary)
        self.assertNotIn("budget_version", summary)

    def test_hybrid_ignores_injected_legacy_versions_everywhere(self):
        injected = {
            "stage_version": LEGACY_VERSIONS["stage_version"],
            "rag_version": LEGACY_VERSIONS["rag_version"],
            "validation_version": "legacy_override_should_not_win",
        }
        result, _, _, _, _ = self.run_success(
            HYBRID_RETRIEVAL_STRATEGY,
            self.make_bundle(),
            policy_overrides=injected,
        )
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        manifest = json.loads(
            (self.output / "draft_generation_manifest.json").read_text()
        )
        report = json.loads(
            (self.output / "draft_validation_report.json").read_text()
        )
        draft = json.loads(
            (self.output / "state_of_art_draft.json").read_text()
        )
        self.assertEqual(
            manifest["versions"]["stage"], HYBRID_VERSIONS["stage_version"]
        )
        self.assertEqual(
            manifest["versions"]["rag"], HYBRID_VERSIONS["rag_version"]
        )
        self.assertEqual(
            manifest["versions"]["quantitative_selection"],
            HYBRID_VERSIONS["quantitative_selection_version"],
        )
        self.assertEqual(
            manifest["versions"]["budget"], HYBRID_VERSIONS["budget_version"]
        )
        self.assertEqual(
            manifest["versions"]["validation"],
            HYBRID_VERSIONS["validation_version"],
        )
        self.assertEqual(
            report["validation_version"],
            HYBRID_VERSIONS["validation_version"],
        )
        summary = draft["generation_summary"]
        for key, expected in HYBRID_VERSIONS.items():
            self.assertEqual(summary[key], expected)

    def test_legacy_ignores_arbitrary_policy_versions(self):
        arbitrary = {
            "stage_version": "stage_false",
            "rag_version": "rag_false",
            "validation_version": "validation_false",
        }
        versions = DraftWritingAgent._effective_versions(
            arbitrary, LEGACY_RETRIEVAL_STRATEGY
        )
        self.assertEqual(versions, LEGACY_VERSIONS)

    def test_hybrid_ignores_arbitrary_policy_versions(self):
        arbitrary = {
            "stage_version": "stage_false",
            "rag_version": "rag_false",
            "validation_version": "validation_false",
        }
        versions = DraftWritingAgent._effective_versions(
            arbitrary, HYBRID_RETRIEVAL_STRATEGY
        )
        self.assertEqual(versions, HYBRID_VERSIONS)

if __name__ == "__main__":
    unittest.main()
