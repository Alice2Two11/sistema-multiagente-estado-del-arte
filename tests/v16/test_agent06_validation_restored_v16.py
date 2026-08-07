from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from src.tools.draft_writing.prompting import build_section_prompt, assign_section_budgets
from src.tools.draft_writing.validation import build_draft_reports
from src.tools.draft_writing.input_validation import validate_draft_dependencies
from src.tools.draft_writing.artifacts import write_raw_section_output
from tests.v16.test_agent06_v16 import Env


class TestAgent06ValidationRestored(unittest.TestCase):
    def _policy(self, **overrides):
        policy = {
            "target_total_words": 100,
            "min_total_words": 60,
            "max_total_words": 140,
            "max_evidence_chars": 18000,
            "output_language": "español",
            "writing_mode": "síntesis crítica",
            "focus_mode": "comparativo",
            "citation_style": "trazable",
        }
        policy.update(overrides)
        return policy

    def _outline(self):
        return [{
            "section_id": "S1",
            "section_title": "Métodos",
            "section_type": "linea_tematica",
            "purpose": "Comparar métodos",
            "key_arguments": ["desempeño"],
            "evidence_needs": ["resultados"],
            "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}],
        }]

    def _evidence(self):
        return {"S1": [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "El método reporta 95 por ciento de precisión con evidencia experimental suficiente."}]}

    def _section(self, text, validation=None, claims=None):
        return [{
            "section_id": "S1",
            "section_title": "Métodos",
            "draft_text": text,
            "claims": claims or [],
            "section_validation": validation or {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
        }]

    def test_total_words_outside_range_invalidates(self):
        sections = self._section("Texto breve [a.pdf | c1].")
        report, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(min_total_words=100))
        self.assertFalse(report["global_length_valid"])
        self.assertFalse(report["validation_ok"])

    def test_section_outside_budget_invalidates(self):
        sections = self._section("Esta oración sustantiva tiene evidencia válida y suficiente [a.pdf | c1].")
        report, _, rows, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(target_total_words=300, min_total_words=1, max_total_words=500))
        self.assertFalse(rows[0]["within_section_range"])
        self.assertIn("S1", report["sections_outside_word_range"])
        self.assertFalse(report["validation_ok"])

    def test_low_citation_density_invalidates(self):
        text = "Esta oración sustantiva presenta una comparación metodológica detallada sin evidencia documental. Otra oración sí está respaldada [a.pdf | c1]."
        sections = self._section(text)
        report, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(min_total_words=1, max_total_words=500))
        self.assertTrue(report["sections_with_low_citation_density"])
        self.assertFalse(report["validation_ok"])

    def test_section_without_valid_citations_invalidates(self):
        sections = self._section("Esta sección presenta evidencia científica detallada pero no contiene una cita válida.")
        report, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(min_total_words=1, max_total_words=500))
        self.assertEqual(report["sections_without_valid_citations"], ["S1"])
        self.assertFalse(report["validation_ok"])

    def test_claim_support_errors_invalidate_global(self):
        sections = self._section(
            "Esta oración sustantiva está respaldada por evidencia trazable [a.pdf | c1].",
            {"validation_ok": False, "claim_errors": ["missing_claim_for_sentence"], "numeric_errors": []},
        )
        report, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(min_total_words=1, max_total_words=500))
        self.assertTrue(report["sections_with_claim_support_errors"])
        self.assertFalse(report["validation_ok"])

    def test_numeric_errors_invalidate_global(self):
        sections = self._section(
            "Esta oración sustantiva reporta 99 por ciento de precisión [a.pdf | c1].",
            {"validation_ok": False, "claim_errors": [], "numeric_errors": ["UNSUPPORTED_NUMERIC_VALUE:99"]},
        )
        report, *_ = build_draft_reports(sections, self._outline(), self._evidence(), self._policy(min_total_words=1, max_total_words=500))
        self.assertTrue(report["sections_with_quantitative_support_errors"])
        self.assertFalse(report["validation_ok"])

    def test_prompt_restores_modes_style_and_word_budget(self):
        outline = self._outline()
        policy = self._policy(target_total_words=120)
        policy["outline_sections"] = outline
        policy["section_budgets"] = assign_section_budgets(outline, 120)
        prompt = build_section_prompt(outline[0], self._evidence()["S1"], {}, [], policy)
        self.assertIn("Modo de escritura: síntesis crítica", prompt)
        self.assertIn("Enfoque: comparativo", prompt)
        self.assertIn("estilo bibliográfico trazable", prompt)
        self.assertIn("Extensión objetivo: 120 palabras", prompt)
        self.assertIn("rango orientativo: 78-168", prompt)

    def test_chroma_collection_mismatch_is_rejected(self):
        env = Env()
        payload = json.loads((env.inp / "chroma_manifest.json").read_text())
        payload["collection_name"] = "otra"
        (env.inp / "chroma_manifest.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "CHROMA_COLLECTION_MISMATCH"):
            validate_draft_dependencies(env.ai)
        env.close()

    def test_chroma_embedding_mismatch_is_rejected(self):
        env = Env()
        payload = json.loads((env.inp / "chroma_manifest.json").read_text())
        payload["embedding_model"] = "otro"
        (env.inp / "chroma_manifest.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "CHROMA_EMBEDDING_MODEL_MISMATCH"):
            validate_draft_dependencies(env.ai)
        env.close()

    def test_ground_truth_review_bibliography_and_excluded_chunks_rejected(self):
        for column in ("is_review_section_chunk", "is_bibliography_chunk", "excluded_from_rag"):
            env = Env()
            frame = pd.read_csv(env.inp / "chunks.csv")
            frame[column] = [True, False]
            frame.to_csv(env.inp / "chunks.csv", index=False)
            with self.assertRaisesRegex(ValueError, "UNSAFE_CHUNKS"):
                validate_draft_dependencies(env.ai)
            env.close()
        env = Env()
        frame = pd.read_csv(env.inp / "chunks.csv")
        frame.loc[0, "source_filename"] = "ground_truth.pdf"
        frame.to_csv(env.inp / "chunks.csv", index=False)
        with self.assertRaisesRegex(ValueError, "GROUND_TRUTH_POLICY_VIOLATION"):
            validate_draft_dependencies(env.ai)
        env.close()

    def test_outline_mapping_inconsistency_rejected(self):
        env = Env()
        frame = pd.read_csv(env.inp / "mapping.csv")
        frame.loc[0, "source_filename"] = "b.pdf"
        frame.to_csv(env.inp / "mapping.csv", index=False)
        with self.assertRaisesRegex(ValueError, "OUTLINE_MAPPING_INCONSISTENT"):
            validate_draft_dependencies(env.ai)
        env.close()

    def test_raw_outputs_use_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def fake_atomic(path, text):
                calls.append((Path(path), text))
                Path(path).write_text(text, encoding="utf-8")
            with patch("src.tools.draft_writing.artifacts.atomic_write_text", side_effect=fake_atomic):
                path = write_raw_section_output(directory, "S1", 2, "raw")
            self.assertEqual(path.name, "S1_attempt_2.txt")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], path)
            self.assertEqual(path.read_text(), "raw")


if __name__ == "__main__":
    unittest.main()
