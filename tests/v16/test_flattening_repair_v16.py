from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.tools.quantitative_extraction.normalization import flatten_results
from src.tools.quantitative_extraction.quality import calculate_diagnostic_metrics, diagnostic_quality_status


class FlatteningRepairV16Tests(unittest.TestCase):
    def sample(self):
        return [{
            "source_filename": "paper.pdf",
            "paper_title": "Paper",
            "source_text_evidence": "paper evidence",
            "quantitative_results": {
                "modelo_ANN": {
                    "MBE": "0.21 MJ·m−2",
                    "R": 0.96,
                    "RMSE": "1.34 MJ·m−2",
                    "neurons_hidden_layer": 7,
                }
            },
            "techniques": ["ANN", "Regresión múltiple"],
            "datasets": [{"dataset_name": "Existing dataset", "case_study": "Site A"}],
        }]

    def test_nested_model_metrics_produce_four_rows(self):
        result = flatten_results(self.sample())
        self.assertEqual(len(result.quantitative), 4)
        by_metric = {row["metric"]: row for row in result.quantitative}
        self.assertEqual(by_metric["RMSE"]["model_or_method"], "modelo_ANN")
        self.assertEqual(by_metric["RMSE"]["numeric_value"], 1.34)
        self.assertEqual(by_metric["RMSE"]["unit"], "MJ·m−2")
        self.assertEqual(by_metric["neurons_hidden_layer"]["condition"], "model_parameter")

    def test_string_techniques_produce_rows_without_inventing_fields(self):
        rows = flatten_results(self.sample()).techniques
        self.assertEqual([row["technique_name"] for row in rows], ["ANN", "Regresión múltiple"])
        self.assertTrue(all(row["technique_family"] == "" and row["role"] == "" for row in rows))

    def test_existing_dataset_dict_is_not_degraded(self):
        rows = flatten_results(self.sample()).datasets
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_name"], "Existing dataset")
        self.assertEqual(rows[0]["case_study"], "Site A")

    def test_dataset_string_is_supported(self):
        data = self.sample(); data[0]["datasets"] = ["Wind farm A"]
        row = flatten_results(data).datasets[0]
        self.assertEqual(row["dataset_name"], "Wind farm A")

    def test_numeric_and_unit_values_are_preserved(self):
        rows = flatten_results(self.sample()).quantitative
        r = next(row for row in rows if row["metric"] == "R")
        rmse = next(row for row in rows if row["metric"] == "RMSE")
        self.assertEqual(r["numeric_value"], 0.96)
        self.assertEqual(r["value"], "0.96")
        self.assertEqual(rmse["unit"], "MJ·m−2")

    def test_raw_path_and_raw_value_are_preserved(self):
        row = next(row for row in flatten_results(self.sample()).quantitative if row["metric"] == "RMSE")
        self.assertEqual(row["raw_path"], "quantitative_results.modelo_ANN.RMSE")
        self.assertIn("1.34", row["raw_value"])

    def test_unknown_structure_generates_warning_and_discard_count(self):
        data = self.sample(); data[0]["quantitative_results"] = {"model": {"bad": {1, 2}}}
        result = flatten_results(data)
        self.assertGreaterEqual(len(result.issues), 1)
        self.assertGreaterEqual(result.flattened_summary["discarded_record_count"], 1)
        self.assertEqual(result.issues[0]["error_type"], "UNSUPPORTED_SCHEMA")

    def test_raw_candidates_with_empty_table_is_needs_revision(self):
        metrics = calculate_diagnostic_metrics(
            papers_processed=1, quantitative_rows=[], dataset_rows=[], technique_rows=[], error_rows=[],
            raw_summary={"raw_quantitative_candidate_count": 2, "raw_technique_candidate_count": 0},
            flattened_summary={"flattened_quantitative_rows": 0, "flattened_technique_rows": 0, "flattened_dataset_rows": 0, "discarded_record_count": 2, "normalization_warning_count": 2},
        )
        status, reasons = diagnostic_quality_status(metrics, fallback_used=False, error_count=0)
        self.assertEqual(status, "NEEDS_REVISION")
        self.assertIn("FLATTENING_FAILED", reasons)

    def test_no_quantitative_data_only_when_raw_and_flat_are_zero(self):
        metrics = calculate_diagnostic_metrics(
            papers_processed=1, quantitative_rows=[], dataset_rows=[], technique_rows=[], error_rows=[],
            raw_summary={"raw_quantitative_candidate_count": 0, "raw_technique_candidate_count": 0},
            flattened_summary={"flattened_quantitative_rows": 0, "flattened_technique_rows": 0, "flattened_dataset_rows": 0, "discarded_record_count": 0, "normalization_warning_count": 0},
        )
        status, reasons = diagnostic_quality_status(metrics, fallback_used=False, error_count=0)
        self.assertEqual(status, "APPROVED_WITH_WARNINGS")
        self.assertEqual(reasons, ("NO_QUANTITATIVE_DATA_OBSERVED",))

    def test_evidence_is_inherited_without_invention(self):
        rows = flatten_results(self.sample()).quantitative
        self.assertTrue(all(row["source_text_evidence"] == "paper evidence" for row in rows))

    def test_deterministic_repair_preserves_json_and_skips_llm(self):
        from tests.v16.test_quantitative_capability_v16 import make_fixture, make_input, capability, FailingLLM
        from src.state.fingerprints import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, deps, _ = make_fixture(root)
            out.mkdir(parents=True, exist_ok=True)
            structured = out / "structured_quantitative_extraction.json"
            raw = out / "structured_quantitative_extraction_raw.jsonl"
            payload = self.sample()
            structured.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            raw.write_text(json.dumps({"status": "ok", "parsed": payload[0]}, ensure_ascii=False) + "\n", encoding="utf-8")
            before = (sha256_file(structured), sha256_file(raw))
            policy = dict(make_input(root, deps=deps).policy)
            policy["deterministic_flattening_repair"] = True
            policy["force_rebuild"] = True
            inp = make_input(root, deps=deps, policy=policy)
            result = capability(FailingLLM()).execute(inp)
            self.assertEqual(result.execution_status.value, "COMPLETED")
            self.assertEqual(result.tool_usage.llm_calls, 0)
            self.assertEqual(before, (sha256_file(structured), sha256_file(raw)))
            table = Path(result.output_artifacts["quantitative_comparative_table.csv"].path).read_text(encoding="utf-8")
            self.assertIn("modelo_ANN", table)
            manifest = json.loads(Path(result.output_artifacts["quantitative_extraction_manifest.json"].path).read_text(encoding="utf-8"))
            self.assertTrue(manifest["repair"]["deterministic_flattening_repair"])
            self.assertFalse(manifest["repair"]["openai_called"])


if __name__ == "__main__":
    unittest.main()
