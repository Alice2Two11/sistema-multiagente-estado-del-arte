from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from src.agents.extraction_agent import ExtractionAgent
from src.contracts.agent_input import AgentInput, PreviousAttemptSummary
from src.contracts.agent_result import ExecutionStatus, QualityStatus, TransitionAction
from src.tools.extraction.revision_strategy import (
    InvalidCardSchemaError,
    build_revision_plan,
    normalize_card_payload,
)
from tests.v16.agent_environment import ExtractionAgentEnvironment
from tests.v16.extraction_agent_doubles import complete_card


def attempt_two_input(agent_input, first_result):
    payload = agent_input.to_dict()
    payload["attempt_number"] = 2
    payload["previous_attempt"] = PreviousAttemptSummary(
        quality_status=first_result.quality_status.value,
        quality_metrics=first_result.quality_metrics,
        failure_reason_codes=first_result.failure_reason_codes,
        previous_artifacts=first_result.output_artifacts,
    ).to_dict()
    return AgentInput.from_dict(payload)


class DirectedAttempt2Tests(unittest.TestCase):
    def test_singleton_object_list_is_normalized(self):
        card = complete_card("a.pdf")
        self.assertEqual(normalize_card_payload([card]), card)

    def test_ambiguous_list_is_invalid_llm_output(self):
        with self.assertRaisesRegex(InvalidCardSchemaError, "INVALID_LLM_OUTPUT"):
            normalize_card_payload([{"title": "a"}, {"title": "b"}])

    def test_title_only_repair_preserves_scientific_fields(self):
        original = complete_card("b.pdf", title="no especificado")
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": original},
            repaired_titles={"b.pdf": "Exact title"},
        )
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertIn(result.quality_status, {
                QualityStatus.APPROVED,
                QualityStatus.APPROVED_WITH_WARNINGS,
            })
            card = {item["source_filename"]: item for item in env.read_cards()}["b.pdf"]
            self.assertEqual(card["title"], "Exact title")
            for field in (
                "research_problem", "objective", "methods_or_models",
                "evaluation_metrics", "main_results", "evidence",
            ):
                self.assertEqual(card[field], original[field])
        finally:
            env.close()

    def test_attempt_two_never_retries_repair_for_a_card_already_quarantined(self):
        # Comportamiento actualizado (pre-eligibilidad documental, ver
        # corpus_eligibility.py): una card con output LLM ambiguo
        # (title="error", evidence=[]) queda QUARANTINE
        # DEFINITIVAMENTE en el intento 1 mismo -- el intento 2 NUNCA
        # vuelve a intentar reparar su título (a diferencia del
        # comportamiento histórico, que la reparaba vía el ciclo de
        # dos intentos). No se llama a repair_llm para esa fuente en
        # ningún momento.
        ambiguous = [complete_card("b.pdf"), complete_card("b.pdf")]
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": ambiguous},
            repair_cards={"b.pdf": complete_card("b.pdf", title="Recovered")},
        )
        try:
            first = ExtractionAgent(env.dependencies).execute(env.agent_input)
            second = ExtractionAgent(env.dependencies).execute(
                attempt_two_input(env.agent_input, first)
            )
            self.assertEqual(second.execution_status, ExecutionStatus.COMPLETED)
            self.assertIsNone(second.error)
            repaired_sources = [
                call[0].content.split("::", 2)[1]
                for call in env.repair_llm.calls
                if call[0].content.startswith("EXTRACT::")
            ]
            self.assertNotIn("b.pdf", repaired_sources)
            cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
            b_card = next(c for c in cards if c["source_filename"] == "b.pdf")
            self.assertEqual(b_card["corpus_eligibility"], "QUARANTINE")
        finally:
            env.close()

    def test_retrieval_sufficient_strategy_does_not_expand(self):
        card = complete_card("paper.pdf", title="error")
        errors = [{
            "source_filename": "paper.pdf",
            "stage": "initial_extraction",
            "error_type": "TypeError",
            "error_message": "list indices must be integers or slices, not str",
        }]
        trace = [
            {"source_filename": "paper.pdf", "chunk_id": f"c{i}"}
            for i in range(18)
        ]
        row = build_revision_plan([card], errors, trace)[0]
        self.assertEqual(row["recommended_strategy"], "REPAIR_SCHEMA_REUSE_RETRIEVAL")

    def test_low_chunk_strategy_expands_evidence(self):
        card = complete_card("paper.pdf", title="error")
        errors = [{
            "source_filename": "paper.pdf",
            "stage": "initial_extraction",
            "error_type": "TypeError",
            "error_message": "list indices must be integers or slices, not str",
        }]
        trace = [
            {"source_filename": "paper.pdf", "chunk_id": f"c{i}"}
            for i in range(3)
        ]
        row = build_revision_plan([card], errors, trace)[0]
        self.assertEqual(row["recommended_strategy"], "REPAIR_SCHEMA_EXPANDED_EVIDENCE")

    def test_no_attempt_three(self):
        env = ExtractionAgentEnvironment()
        try:
            payload = env.agent_input.to_dict()
            payload["attempt_number"] = 3
            result = ExtractionAgent(env.dependencies).execute(AgentInput.from_dict(payload))
            self.assertEqual(result.execution_status, ExecutionStatus.FAILED)
            self.assertEqual(result.requested_transition.action, TransitionAction.HALT_STAGE)
        finally:
            env.close()

    def test_quality_and_revision_plan_exist_before_retry(self):
        # Comportamiento actualizado (pre-eligibilidad documental, ver
        # corpus_eligibility.py): una card con output LLM ambiguo/
        # inválido (title="error", evidence=[]) ahora se captura como
        # QUARANTINE en FASE 1, ANTES de llegar al quality gate
        # científico -- ya no aparece en el revision_plan como
        # INVALID_LLM_OUTPUT bloqueante, y queda auditada.
        ambiguous = [complete_card("b.pdf"), complete_card("b.pdf")]
        env = ExtractionAgentEnvironment(extraction_cards={"b.pdf": ambiguous})
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertTrue(env.paths["CARDS_SUMMARY_CSV_PATH"].is_file())
            self.assertTrue(env.paths["CARDS_QUALITY_CSV_PATH"].is_file())
            self.assertTrue(env.paths["CARDS_REVISION_PLAN_CSV_PATH"].is_file())
            plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
            self.assertNotIn("b.pdf", plan["source_filename"].tolist() if len(plan) else [])
            self.assertTrue(env.paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"].is_file())
            quarantine = pd.read_csv(env.paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"])
            self.assertIn("b.pdf", quarantine["source_filename"].tolist())
            cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
            b_card = next(c for c in cards if c["source_filename"] == "b.pdf")
            self.assertEqual(b_card["corpus_eligibility"], "QUARANTINE")
        finally:
            env.close()

    def test_kb_is_built_after_sufficient_attempt_two(self):
        # Comportamiento actualizado: una card QUARANTINE nunca
        # impide que la KB se construya -- solo depende de que exista
        # al menos una card INCLUDE válida. Este escenario ya no
        # necesita un segundo intento para "reparar" la card ambigua
        # (queda QUARANTINE definitivamente desde el intento 1), así
        # que verifica directamente que el KB se construye con esa
        # mezcla, y que un segundo intento (llamado igualmente, sin
        # cambios pendientes) sigue siendo consistente.
        ambiguous = [complete_card("b.pdf"), complete_card("b.pdf")]
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": ambiguous},
            repair_cards={"b.pdf": complete_card("b.pdf", title="Recovered")},
        )
        try:
            first = ExtractionAgent(env.dependencies).execute(env.agent_input)
            second = ExtractionAgent(env.dependencies).execute(
                attempt_two_input(env.agent_input, first)
            )
            self.assertEqual(second.execution_status, ExecutionStatus.COMPLETED)
            self.assertIsNone(second.error)
            self.assertTrue(env.paths["KB_CSV_PATH"].is_file())
            self.assertTrue(env.paths["KB_JSONL_PATH"].is_file())
            self.assertTrue(env.paths["EXTRACTION_MANIFEST_PATH"].is_file())
            cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
            eligibility_by_source = {c["source_filename"]: c.get("corpus_eligibility") for c in cards}
            self.assertEqual(eligibility_by_source.get("a.pdf"), "INCLUDE")
            self.assertEqual(eligibility_by_source.get("b.pdf"), "QUARANTINE")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
