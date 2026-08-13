from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.adapters.draft_writing_runtime import (
    build_real_draft_execution,
    commit_executed_draft,
    execute_prepared_draft,
    prepare_draft_execution,
)
from src.contracts.agent_result import QualityStatus, TransitionAction
from tests.v17.agent06_v17_test_support import (
    EvidenceAwareLLM,
    SyntheticCollection,
    chroma_client_factory,
    runtime_factory_for,
    write_synthetic_project,
)
from tests.v17.test_agent06_v17_end_to_end_integration import HYBRID


class TestAgent06V17ScientificFailures(unittest.TestCase):
    def _execute(self, mode: str, *, policy_overrides=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        _, store, chunk_rows = write_synthetic_project(root)
        llm = EvidenceAwareLLM(mode)
        agent, agent_input, cfg = build_real_draft_execution(
            root,
            collection_factory=lambda _: SyntheticCollection(chunk_rows),
            runtime_factory=runtime_factory_for(llm),
            chroma_client_factory=chroma_client_factory,
            policy_overrides=HYBRID if policy_overrides is None else policy_overrides,
        )
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(
            store=store,
            agent=agent,
            prepared=prepared,
        )
        return root, store, llm, cfg, executed

    @staticmethod
    def _attempt_validations(cfg: dict, section_id: str = "S1") -> list[dict]:
        raw = Path(cfg["output_dir"]) / "raw_section_outputs"
        # raw_section_outputs ahora versiona por agent_attempt_NN/ (ver
        # draft_writing_agent.py) -- rglob localiza los archivos de
        # validación sin importar bajo qué intento externo quedaron,
        # sin necesitar conocer el número exacto aquí.
        paths = sorted(raw.rglob(f"{section_id}_attempt_*_validation.json"))
        return [json.loads(path.read_text()) for path in paths]

    def _assert_section_failure(self, mode: str, expected_fragment: str):
        _, store, llm, cfg, executed = self._execute(mode)
        result = executed.result
        self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
        self.assertEqual(result.requested_transition.action, TransitionAction.RETRY)
        self.assertEqual(llm.calls, 3)
        self.assertNotIn("state_of_art_draft.json", result.output_artifacts)
        self.assertNotIn("state_of_art_draft.md", result.output_artifacts)

        report = json.loads(
            Path(result.output_artifacts["draft_validation_report.json"].path).read_text()
        )
        self.assertFalse(report["validation_ok"])
        self.assertIn(expected_fragment, json.dumps(report, ensure_ascii=False))

        validations = self._attempt_validations(cfg)
        self.assertEqual(len(validations), 3)
        for attempt in validations:
            self.assertIn("original_validation", attempt)
            self.assertIn("normalized_validation", attempt)
            self.assertFalse(attempt["validation_ok"])
            self.assertIn(expected_fragment, json.dumps(attempt, ensure_ascii=False))

        raw = Path(cfg["output_dir"]) / "raw_section_outputs"
        self.assertEqual(len(list(raw.rglob("S1_attempt_*.txt"))), 3)
        self.assertEqual(len(list(raw.rglob("S1_attempt_*_rag_trace.json"))), 3)
        with self.assertRaisesRegex(
            RuntimeError,
            "DRAFT_COMMIT_REQUIRES_APPROVED_RESULT",
        ):
            commit_executed_draft(store=store, executed=executed)

    def test_claim_missing_is_observable_before_normalization(self):
        self._assert_section_failure("claim_missing", "missing_claim_for_sentence")

    def test_claim_mismatch_is_observable_before_normalization(self):
        self._assert_section_failure("claim_mismatch", "missing_claim_for_sentence")

    def test_invalid_citation_is_not_replaced_only_by_empty_text(self):
        _, _, _, cfg, executed = self._execute("invalid_citation")
        self.assertEqual(executed.result.quality_status, QualityStatus.NEEDS_REVISION)
        attempts = self._attempt_validations(cfg)
        self.assertEqual(len(attempts), 3)
        for attempt in attempts:
            self.assertIn("invalid_citation", attempt["validation_errors"])
            # Fail-closed sin heurísticas (ver normalization.py): una
            # oración con cita inválida ya NUNCA se convierte en
            # EMPTY_DRAFT_TEXT -- se preserva en draft_text sin cita
            # inventada, y el motivo real (invalid_citation) es la única
            # causa reportada.
            self.assertNotIn("EMPTY_DRAFT_TEXT", attempt["validation_errors"])
            self.assertIn(
                "invalid_citation",
                attempt["original_validation"]["citation_errors"],
            )

    def test_uncited_sentence_is_not_replaced_only_by_empty_text(self):
        _, _, _, cfg, executed = self._execute("uncited")
        self.assertEqual(executed.result.quality_status, QualityStatus.NEEDS_REVISION)
        attempts = self._attempt_validations(cfg)
        self.assertEqual(len(attempts), 3)
        for attempt in attempts:
            self.assertIn("uncited_substantive_sentence", attempt["validation_errors"])
            # Mismo contrato: una oración sustantiva sin cita se preserva
            # en draft_text (nunca se borra, nunca se vacía la sección
            # completa) -- EMPTY_DRAFT_TEXT nunca debe aparecer como
            # consecuencia de esto.
            self.assertNotIn("EMPTY_DRAFT_TEXT", attempt["validation_errors"])

    def test_unsupported_numeric_value_remains_unchanged(self):
        self._assert_section_failure(
            "unsupported_numeric",
            "UNSUPPORTED_NUMERIC_VALUE:777.7",
        )

    def test_valid_original_and_normalized_output_approves(self):
        _, _, llm, cfg, executed = self._execute("valid")
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(executed.result.decision.code, "DRAFT_APPROVED")
        self.assertGreater(llm.calls, 0)
        first_attempt = self._attempt_validations(cfg)[0]
        self.assertTrue(first_attempt["original_validation"]["validation_ok"])
        self.assertTrue(first_attempt["normalized_validation"]["validation_ok"])
        self.assertTrue(first_attempt["validation_ok"])

    def test_legacy_keeps_normalize_then_validate_behavior(self):
        _, _, _, cfg, _ = self._execute("claim_missing", policy_overrides={})
        attempt = self._attempt_validations(cfg)[0]
        self.assertNotIn("original_validation", attempt)
        self.assertNotIn("normalized_validation", attempt)
        self.assertTrue(attempt["validation_ok"])
        self.assertNotIn("missing_claim_for_sentence", attempt["validation_errors"])

    def test_length_failures_remain_blocking(self):
        for mode in ("too_short", "too_long"):
            with self.subTest(mode=mode):
                _, store, _, _, executed = self._execute(mode)
                result = executed.result
                self.assertEqual(result.quality_status, QualityStatus.NEEDS_REVISION)
                self.assertFalse(
                    json.loads(
                        Path(
                            result.output_artifacts[
                                "draft_validation_report.json"
                            ].path
                        ).read_text()
                    )["validation_ok"]
                )
                self.assertNotIn("state_of_art_draft.json", result.output_artifacts)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "DRAFT_COMMIT_REQUIRES_APPROVED_RESULT",
                ):
                    commit_executed_draft(store=store, executed=executed)

    def test_partial_quantitative_context_fails_before_llm(self):
        modes = (
            "table",
            "summary",
            "manifest",
            "table,summary",
            "table,manifest",
            "summary,manifest",
        )
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                write_synthetic_project(root, quantitative=mode)
                llm = EvidenceAwareLLM("valid")
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "INVALID_QUANTITATIVE_CONTEXT",
                ):
                    build_real_draft_execution(
                        root,
                        collection_factory=lambda _: SyntheticCollection([]),
                        runtime_factory=runtime_factory_for(llm),
                        chroma_client_factory=chroma_client_factory,
                        policy_overrides=HYBRID,
                    )
                self.assertEqual(llm.calls, 0)
                draft_dir = root / "exp_synthetic" / "05_outputs" / "05_draft"
                self.assertFalse((draft_dir / "state_of_art_draft.json").exists())


if __name__ == "__main__":
    unittest.main()
