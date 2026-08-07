from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from src.adapters.draft_writing_runtime import resolve_chroma_dir, load_draft_configuration


class NamedCollection:
    def __init__(self, name):
        self.name = name


class ClientFactory:
    def __init__(self, mapping):
        self.mapping = {str(Path(k).resolve()): list(v) for k, v in mapping.items()}

    def __call__(self, path=None, **kwargs):
        resolved = str(Path(path).resolve())
        names = self.mapping.get(resolved, [])
        return type("Client", (), {"list_collections": lambda self: [NamedCollection(n) for n in names]})()


def make_db(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "chroma.sqlite3").write_bytes(b"persistent-index")


class TestAgent06ChromaDirResolution(unittest.TestCase):
    def test_empty_04_chroma_and_valid_04_chroma_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            empty = exp / "04_chroma"
            valid = exp / "04_chroma_index"
            empty.mkdir(parents=True)
            make_db(valid)
            factory = ClientFactory({valid: ["reference_papers_chunks"]})
            resolved = resolve_chroma_dir(exp, "reference_papers_chunks", empty, client_factory=factory)
            self.assertEqual(resolved, valid.resolve())

    def test_persisted_real_path_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            valid = exp / "04_chroma_index"
            make_db(valid)
            factory = ClientFactory({valid: ["reference_papers_chunks"]})
            self.assertEqual(resolve_chroma_dir(exp, "reference_papers_chunks", None, client_factory=factory), valid.resolve())

    def test_no_expected_collection_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            wrong = exp / "04_chroma_index"
            make_db(wrong)
            factory = ClientFactory({wrong: ["other_collection"]})
            with self.assertRaisesRegex(FileNotFoundError, "CHROMA_COLLECTION_NOT_FOUND"):
                resolve_chroma_dir(exp, "reference_papers_chunks", None, client_factory=factory)

    def test_multiple_valid_paths_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            first = exp / "04_chroma_index"
            second = exp / "backup_index"
            make_db(first); make_db(second)
            factory = ClientFactory({first: ["reference_papers_chunks"], second: ["reference_papers_chunks"]})
            with self.assertRaisesRegex(RuntimeError, "CHROMA_DIR_AMBIGUOUS"):
                resolve_chroma_dir(exp, "reference_papers_chunks", first, client_factory=factory)

    def test_wrong_collection_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            valid_db = exp / "04_chroma_index"
            make_db(valid_db)
            factory = ClientFactory({valid_db: ["scientific_chunks"]})
            with self.assertRaisesRegex(FileNotFoundError, "reference_papers_chunks"):
                resolve_chroma_dir(exp, "reference_papers_chunks", valid_db, client_factory=factory)

    def test_configuration_uses_real_path_and_manifest_is_not_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eid = "experiment"
            exp = root / eid
            outputs = exp / "05_outputs"
            (outputs / "00_orchestrator_planner").mkdir(parents=True)
            state = outputs / "00_orchestrator_planner" / "pipeline_state.json"
            state.write_text(json.dumps({"stages": {}}), encoding="utf-8")
            (exp / "04_chroma").mkdir(parents=True)
            real = exp / "04_chroma_index"
            make_db(real)
            (root / "active_experiment.json").write_text(json.dumps({
                "active_experiment_id": eid,
                "run_id": "run",
                "chroma_collection_name": "reference_papers_chunks",
                "chroma_dir": str(exp / "04_chroma"),
                "generation_profile": {},
                "draft_generation_policy": {},
            }), encoding="utf-8")
            factory = ClientFactory({real: ["reference_papers_chunks"]})
            cfg = load_draft_configuration(root, 1, chroma_client_factory=factory)
            self.assertEqual(cfg["chroma_dir"], real.resolve())
            self.assertFalse(cfg["paths"]["chroma_manifest"].exists())


if __name__ == "__main__":
    unittest.main()
