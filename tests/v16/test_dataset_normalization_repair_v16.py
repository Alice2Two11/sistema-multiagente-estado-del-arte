from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.quantitative_extraction.normalization import flatten_results
from src.state.fingerprints import sha256_file


class DatasetNormalizationRepairV16Tests(unittest.TestCase):
    def _paper(self, datasets):
        return [{
            "source_filename": "paper.pdf",
            "paper_title": "Paper",
            "source_text_evidence": "paper evidence",
            "quantitative_results": {"ANN": {"RMSE": 1.2}},
            "techniques": ["ANN"],
            "datasets": datasets,
        }]

    def test_description_is_used_as_provisional_dataset_name(self):
        payload = {"description": "Measurements from a mountain station"}
        result = flatten_results(self._paper([payload]))
        self.assertEqual(len(result.datasets), 1)
        self.assertEqual(result.datasets[0]["dataset_name"], payload["description"])
        self.assertEqual(result.datasets[0]["description"], payload["description"])
        self.assertFalse(result.issues[0]["discarded"])
        self.assertEqual(result.issues[0]["error_code"], "INVALID_DATASET_SCHEMA")

    def test_coordinates_are_preserved(self):
        payload = {"description": "Station dataset", "coordinates": {"lat": -0.18, "lon": -78.47}}
        row = flatten_results(self._paper([payload])).datasets[0]
        self.assertIn('"lat": -0.18', row["coordinates"])
        self.assertIn('"lon": -78.47', row["coordinates"])
        self.assertIn('"coordinates"', row["raw_value"])

    def test_altitude_masl_is_preserved(self):
        payload = {"description": "High-altitude station", "altitude_masl": 2850}
        row = flatten_results(self._paper([payload])).datasets[0]
        self.assertEqual(row["altitude_masl"], "2850")
        self.assertIn("2850", row["raw_value"])

    def test_data_split_is_preserved(self):
        payload = {"description": "Operational records", "data_split": {"train": "70%", "test": "30%"}}
        row = flatten_results(self._paper([payload])).datasets[0]
        self.assertIn('"train": "70%"', row["data_split"])
        self.assertIn('"test": "30%"', row["data_split"])

    def test_dataset_without_descriptive_text_is_kept_not_discarded(self):
        payload = {"coordinates": [-0.18, -78.47], "altitude_masl": 2850, "data_split": "80/20"}
        result = flatten_results(self._paper([payload]))
        self.assertEqual(result.datasets[0]["dataset_name"], "Dataset no nombrado")
        self.assertFalse(result.issues[0]["discarded"])
        self.assertEqual(result.flattened_summary["discarded_record_count"], 0)

    def test_named_dataset_behavior_is_not_degraded(self):
        payload = {"dataset_name": "Wind Farm A", "case_study": "Site A", "data_type": "SCADA"}
        result = flatten_results(self._paper([payload]))
        self.assertEqual(result.datasets[0]["dataset_name"], "Wind Farm A")
        self.assertEqual(result.datasets[0]["case_study"], "Site A")
        self.assertEqual(result.datasets[0]["data_type"], "SCADA")
        self.assertEqual(result.issues, [])

    def test_six_descriptive_datasets_are_all_kept(self):
        datasets = [
            {"description": f"Dataset description {i}", "coordinates": {"lat": i}, "altitude_masl": 1000+i, "data_split": "70/30"}
            for i in range(6)
        ]
        result = flatten_results(self._paper(datasets))
        self.assertEqual(len(result.datasets), 6)
        self.assertEqual(len(result.issues), 6)
        self.assertTrue(all(not issue["discarded"] for issue in result.issues))
        self.assertEqual(result.flattened_summary["discarded_record_count"], 0)

    def test_deterministic_repair_preserves_json_and_regenerates_dataset_table(self):
        from tests.v16.test_quantitative_capability_v16 import make_fixture, make_input, capability, FailingLLM
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, deps, _ = make_fixture(root)
            out.mkdir(parents=True, exist_ok=True)
            structured = out / "structured_quantitative_extraction.json"
            raw = out / "structured_quantitative_extraction_raw.jsonl"
            payload = self._paper([{
                "description": "Mountain station measurements",
                "coordinates": {"lat": -0.18, "lon": -78.47},
                "altitude_masl": 2850,
                "data_split": {"train": "70%", "test": "30%"},
            }])
            structured.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            raw.write_text(json.dumps({"status": "ok", "parsed": payload[0]}, ensure_ascii=False)+"\n", encoding="utf-8")
            before = (sha256_file(structured), sha256_file(raw))
            policy = dict(make_input(root, deps=deps).policy)
            policy["deterministic_flattening_repair"] = True
            policy["force_rebuild"] = True
            result = capability(FailingLLM()).execute(make_input(root, deps=deps, policy=policy))
            self.assertEqual(result.execution_status.value, "COMPLETED")
            self.assertEqual(result.tool_usage.llm_calls, 0)
            self.assertEqual(before, (sha256_file(structured), sha256_file(raw)))
            dataset_path = Path(result.output_artifacts["quantitative_datasets_table.csv"].path)
            with dataset_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["dataset_name"], "Mountain station measurements")
            self.assertEqual(rows[0]["altitude_masl"], "2850")
            errors_path = Path(result.output_artifacts["quantitative_extraction_errors.csv"].path)
            with errors_path.open(encoding="utf-8", newline="") as handle:
                errors = list(csv.DictReader(handle))
            self.assertEqual(errors[0]["error_code"], "INVALID_DATASET_SCHEMA")
            self.assertEqual(errors[0]["discarded"], "False")
            manifest = json.loads(Path(result.output_artifacts["quantitative_extraction_manifest.json"].path).read_text(encoding="utf-8"))
            self.assertTrue(manifest["repair"]["deterministic_flattening_repair"])
            self.assertFalse(manifest["repair"]["openai_called"])


if __name__ == "__main__":
    unittest.main()
