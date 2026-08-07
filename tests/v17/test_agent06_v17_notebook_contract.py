from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import nbformat

from src.adapters.draft_writing_runtime import build_runtime_draft_policy
from src.config.draft_writing_policy_config import LEGACY_RETRIEVAL_STRATEGY


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "06_agente_redactor_v17.ipynb"
EXPECTED_ARTIFACTS = (
    "state_of_art_draft.json",
    "state_of_art_draft.md",
    "draft_sections.csv",
    "draft_rag_evidence.csv",
    "draft_quality_check.csv",
    "draft_length_check.csv",
    "draft_claim_evidence.csv",
    "numeric_hallucination_check.csv",
    "draft_validation_report.json",
    "quantitative_comparative_table_used.csv",
    "dataset_technique_summary_used.csv",
    "draft_generation_manifest.json",
)


class Agent06V17NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.cells = cls.notebook.cells
        cls.code = "\n\n".join(
            cell.source for cell in cls.cells if cell.cell_type == "code"
        )
        cls.markdown = "\n\n".join(
            cell.source for cell in cls.cells if cell.cell_type == "markdown"
        )

    def test_notebook_is_valid_and_has_exact_cell_contract(self):
        self.assertEqual(self.notebook.nbformat, 4)
        self.assertEqual(len(self.cells), 20)
        self.assertEqual(self.cells[0].cell_type, "markdown")
        self.assertTrue(all(cell.cell_type == "code" for cell in self.cells[1:]))
        for index, cell in enumerate(self.cells[1:], start=2):
            with self.subTest(cell=index):
                compile(cell.source, f"cell_{index}", "exec")

    def test_title_scope_and_v16_v17_explanation(self):
        first = self.cells[0].source
        self.assertIn("Agente 06 — Redacción del Estado del Arte", first)
        self.assertIn("Versión V17 híbrida, cuantitativa y source-aware", first)
        self.assertIn("V16 legacy", first)
        self.assertIn("V17 híbrida", first)
        self.assertIn("no contiene la lógica interna", first)
        self.assertIn("estado operacional no se modifica", first)

    def test_definitive_imports_and_no_experimental_imports(self):
        required = (
            "src.adapters.draft_writing_runtime",
            "src.agents.draft_writing_agent",
            "src.config.draft_writing_policy_config",
        )
        for module in required:
            self.assertIn(module, self.code)
        forbidden = (
            "draft_writing_agent_hybrid_experimental",
            "draft_writing_hybrid_runtime",
        )
        for token in forbidden:
            self.assertNotIn(token, self.code)

    def test_notebook_does_not_reimplement_productive_algorithms(self):
        tree = ast.parse(self.code)
        function_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forbidden_functions = {
            "query_chroma_candidates",
            "query_csv_ranked_candidates",
            "deduplicate_candidates",
            "reciprocal_rank_fusion",
            "balanced_hybrid_selection",
            "retrieve_section_evidence_hybrid",
            "augment_evidence_with_quantitative_chunks_greedy",
            "assign_source_aware_section_budgets",
            "validate_generated_section",
            "write_draft_artifacts",
            "_validate_resume_candidate",
        }
        self.assertTrue(function_names.isdisjoint(forbidden_functions))
        self.assertNotIn("1 / (rrf_k + rank)", self.code)
        self.assertNotIn("marginal_coverage", self.code)

    def test_visible_configuration_and_safe_defaults(self):
        config = self.cells[3].source
        self.assertIn('EXECUTION_MODE = "v17_hybrid"', config)
        self.assertIn('"v16_legacy"', config)
        self.assertIn("COMMIT_ENABLED = False", config)
        self.assertIn("RUN_EXECUTE = False", config)
        self.assertIn("FORCE_REEXECUTION = False", config)
        for key, value in {
            "candidate_multiplier": "3",
            "top_k_evidence_per_section": "8",
            "chroma_quota": "3",
            "csv_quota": "3",
            "rrf_quota": "2",
            "rrf_k": "60",
            "max_evidence_chars": "18000",
            "max_candidates_per_source": "24",
            "quantitative_evidence_quota": "2",
            "organizational_target_words": "40",
            "organizational_minimum_words": "25",
            "organizational_maximum_words": "70",
            "substantive_minimum_ratio": "0.65",
            "substantive_maximum_ratio": "1.40",
        }.items():
            self.assertIn(f'"{key}": {value}', config)

    def test_policy_cell_uses_runtime_builder_and_keeps_product_default_legacy(self):
        self.assertIn("build_runtime_draft_policy", self.cells[4].source)
        v17 = build_runtime_draft_policy(
            {"retrieval_strategy": "hybrid_chroma_csv_rrf_balanced"}
        )
        legacy = build_runtime_draft_policy()
        self.assertEqual(v17["retrieval_strategy"], "hybrid_chroma_csv_rrf_balanced")
        self.assertEqual(legacy["retrieval_strategy"], LEGACY_RETRIEVAL_STRATEGY)

    def test_paths_are_centralized_and_output_is_isolated(self):
        paths = self.cells[1].source
        self.assertIn('/content/proyecto_estado_arte', paths)
        self.assertIn('experimento_paper_02', paths)
        self.assertIn('05_draft_v17_candidate', paths)
        self.assertNotIn('ISOLATED_OUTPUT_DIR = EXPERIMENT_DIR / "05_outputs" / "05_draft"', paths)
        self.assertIn("baseline 05_draft", paths)
        self.assertEqual(self.code.count("05_draft_v17_candidate"), 1)

    def test_dependency_validation_distinguishes_optional_quantitative_context(self):
        source = self.cells[5].source
        self.assertIn("missing_required", source)
        self.assertIn("INVALID_QUANTITATIVE_CONTEXT", source)
        self.assertIn("all(quant_present.values())", source)
        self.assertIn("any(quant_present.values())", source)
        self.assertIn('"complete"', source)
        self.assertIn('"absent"', source)

    def test_agent_input_uses_runtime_and_does_not_print_full_papers(self):
        source = self.cells[6].source
        self.assertIn("build_draft_agent_input(BASE_CFG)", source)
        self.assertIn("section_count", source)
        self.assertIn("allowed_paper_count", source)
        self.assertIn("fingerprint", source)
        self.assertNotIn("chunks_clean_for_rag.csv).read_text", source)

    def test_prepare_uses_isolated_state_copy(self):
        source = self.cells[7].source
        self.assertIn("shutil.copy2(REAL_STATE_PATH, ISOLATED_STATE_PATH)", source)
        self.assertIn("StateStore", source)
        self.assertIn("prepare_draft_execution", source)
        self.assertIn("Estado operacional modificado: False", source)

    def test_execute_guard_prevents_accidental_llm_call(self):
        self.assertIn("RUN_EXECUTE = False", self.cells[8].source)
        self.assertIn("Ejecución detenida antes de llamar al LLM", self.cells[8].source)
        execute_source = self.cells[9].source
        self.assertIn("if RUN_EXECUTE:", execute_source)
        self.assertIn("build_openai_draft_runtime", execute_source)
        self.assertIn("DraftWritingAgent", execute_source)
        self.assertIn("execute_prepared_draft", execute_source)
        self.assertIn("no se construyó OpenAI", execute_source)

    def test_result_sections_evidence_and_validation_views_exist(self):
        result_source = self.cells[10].source
        for field in (
            "execution_status", "quality_status", "decision_code",
            "requested_transition", "published_draft", "validation_ok",
            "total_words", "target_words", "invalid_citation_count",
            "numeric_failure_count", "sections_outside_word_range",
            "llm_calls", "validation_calls",
        ):
            self.assertIn(field, result_source)
        for field in (
            "budget_type", "actual_words", "attempts", "evidence_count"
        ):
            self.assertIn(field, self.cells[11].source)
        for field in (
            "retrieval_sources", "chroma_rank", "csv_rank", "rrf_score",
            "selection_bucket", "quantitative_values", "selection_order",
        ):
            self.assertIn(field, self.cells[12].source)
        validation_source = self.cells[13].source
        self.assertIn("original_validation", validation_source)
        self.assertIn("normalized_validation", validation_source)
        self.assertIn("Citas inválidas", validation_source)
        self.assertIn("Claims sin soporte", validation_source)
        self.assertIn("Números no respaldados", validation_source)

    def test_exact_twelve_artifacts_are_listed(self):
        source = self.cells[14].source
        for artifact in EXPECTED_ARTIFACTS:
            self.assertEqual(source.count(f'"{artifact}"'), 1)
        self.assertIn("len(EXECUTED.result.output_artifacts)", self.cells[19].source)
        self.assertNotIn('"raw_section_outputs"', source)

    def test_resume_is_manual_and_uses_definitive_runtime(self):
        source = self.cells[16].source
        self.assertIn("CHECK_RESUME = False", source)
        self.assertIn("resume_draft_execution", source)
        self.assertIn("RESUME no ejecutado automáticamente", source)

    def test_commit_has_two_guards_and_no_agent07_execution(self):
        barrier = self.cells[17].source
        commit = self.cells[18].source
        self.assertIn("COMMIT_ENABLED = False", barrier)
        self.assertIn('CONFIRM_COMMIT_TEXT = ""', barrier)
        self.assertIn('CONFIRM_COMMIT_TEXT == "CONFIRMAR_COMMIT_AGENTE_06"', barrier)
        self.assertIn("quality_status.value in APPROVED_QUALITY_STATUSES", barrier)
        self.assertIn("commit_executed_draft", commit)
        self.assertIn("cancel_pending_execution", commit)
        self.assertIn("Agente 07 ejecutado: False", commit)
        self.assertNotIn("Agent07", self.code)
        self.assertNotIn("07_agente_verificador", self.code)

    def test_notebook_has_no_stored_outputs_or_execution_count(self):
        for cell in self.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_notebook_json_is_stable_and_readable(self):
        payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        self.assertEqual(len(payload["cells"]), 20)


if __name__ == "__main__":
    unittest.main()
