from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.adapters.draft_writing_runtime import DraftWritingRuntime
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore


CONTRACT_ARTIFACTS = (
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


class SyntheticCollection:
    """Deterministic Chroma double restricted by source_filename."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = [dict(row) for row in rows]
        self.query_calls: list[dict[str, Any]] = []

    def query(self, *, query_texts, n_results, where, include):
        self.query_calls.append(
            {
                "query_texts": list(query_texts),
                "n_results": n_results,
                "where": dict(where),
                "include": list(include),
            }
        )
        source = str(where.get("source_filename", ""))
        selected = [row for row in self.rows if row["source_filename"] == source]
        selected.sort(key=lambda row: (float(row.get("distance", 1.0)), row["chunk_id"]))
        selected = selected[:n_results]
        return {
            "documents": [[row["text"] for row in selected]],
            "metadatas": [[
                {"source_filename": row["source_filename"], "chunk_id": row["chunk_id"]}
                for row in selected
            ]],
            "distances": [[float(row.get("distance", 0.2)) for row in selected]],
        }


class EvidenceAwareLLM:
    """LLM double that derives output only from the prompt evidence."""

    def __init__(self, mode: str = "valid"):
        self.mode = mode
        self.calls = 0
        self.prompts: list[str] = []

    @staticmethod
    def _json_block(prompt: str, start: str, end: str) -> Any:
        content = prompt.split(start, 1)[1].split(end, 1)[0].strip()
        return json.loads(content)

    @staticmethod
    def _number(text: str) -> str | None:
        match = re.search(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?", text)
        return None if match is None else match.group(0)

    def __call__(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        section = self._json_block(prompt, "SECCIÓN DEL ESQUEMA:\n", "\n\nALLOWED_CITATIONS:")
        evidence = self._json_block(prompt, "EVIDENCIA:\n", "\n\nCONTEXTO CUANTITATIVO CONFIRMADO:")
        sid = str(section["section_id"])
        title = str(section["section_title"])
        if not evidence:
            return {"section_id": sid, "section_title": title, "draft_text": "", "claims": []}

        chosen = next((row for row in evidence if self._number(str(row.get("text", "")))), evidence[0])
        source = str(chosen["source_filename"])
        chunk_id = str(chosen["chunk_id"])
        citation = f"[{source} | {chunk_id}]"
        number = self._number(str(chosen.get("text", ""))) or ""
        sentence = (
            f"La evidencia comparativa documenta que el enfoque evaluado obtuvo {number} "
            f"en el escenario descrito, lo que permite contrastar de manera verificable "
            f"su comportamiento con las demás propuestas analizadas {citation}."
        )
        claim = re.sub(r"\s*\[[^\]]+\]", "", sentence).rstrip(".")
        citations = [citation]

        if self.mode == "unsupported_numeric":
            sentence = sentence.replace(number, "777.7", 1)
            claim = claim.replace(number, "777.7", 1)
        elif self.mode == "invalid_citation":
            bad = "[unauthorized.pdf | missing_chunk]"
            sentence = sentence.replace(citation, bad)
            citations = [bad]
        elif self.mode == "claim_missing":
            return {"section_id": sid, "section_title": title, "draft_text": sentence, "claims": []}
        elif self.mode == "claim_mismatch":
            claim = "Este claim no corresponde literalmente con la oración sustantiva"
        elif self.mode == "uncited":
            sentence = sentence.replace(f" {citation}", "")
            citations = []
        elif self.mode == "too_short":
            sentence = f"Resultado {number} {citation}."
            claim = f"Resultado {number}"
        elif self.mode == "too_long":
            words = " ".join(["evidencia"] * 180)
            sentence = f"{words} {number} {citation}."
            claim = f"{words} {number}"

        return {
            "section_id": sid,
            "section_title": title,
            "draft_text": sentence,
            "claims": [{"claim": claim, "supporting_citations": citations}],
        }


class ForbiddenRuntime:
    def __call__(self, *args, **kwargs):
        raise AssertionError("A real runtime dependency was invoked")


def _sections() -> list[dict[str, Any]]:
    return [
        {
            "section_id": "S0",
            "section_title": "Introducción",
            "section_type": "introduction",
            "requires_sources": False,
            "purpose": "Organizar la revisión",
            "papers_to_use": [],
        },
        {
            "section_id": "S1",
            "section_title": "Modelos predictivos",
            "section_type": "substantive",
            "requires_sources": True,
            "purpose": "comparar accuracy dataset model",
            "key_arguments": ["accuracy", "dataset"],
            "evidence_needs": ["resultados cuantitativos"],
            "papers_to_use": [
                {"source_filename": "paper_a.pdf", "title": "A"},
                {"source_filename": "paper_b.pdf", "title": "B"},
            ],
        },
        {
            "section_id": "S2",
            "section_title": "Errores de estimación",
            "section_type": "substantive",
            "requires_sources": True,
            "purpose": "comparar error rmse method",
            "key_arguments": ["error", "rmse"],
            "evidence_needs": ["métricas"],
            "papers_to_use": [
                {"source_filename": "paper_a.pdf", "title": "A"},
                {"source_filename": "paper_b.pdf", "title": "B"},
            ],
        },
        {
            "section_id": "S3",
            "section_title": "Comparación de rendimiento",
            "section_type": "substantive",
            "requires_sources": True,
            "purpose": "comparar rendimiento precision evaluation",
            "key_arguments": ["rendimiento", "precision"],
            "evidence_needs": ["comparación"],
            "papers_to_use": [
                {"source_filename": "paper_b.pdf", "title": "B"},
                {"source_filename": "paper_c.pdf", "title": "C"},
            ],
        },
    ]


def synthetic_chunks() -> list[dict[str, Any]]:
    return [
        {"source_filename": "paper_a.pdf", "chunk_id": "a_chroma", "text": "Model accuracy evidence reached 91% on dataset A with method alpha.", "distance": 0.05},
        {"source_filename": "paper_a.pdf", "chunk_id": "a_shared", "text": "Comparative accuracy dataset evidence reported 92% using method alpha.", "distance": 0.10},
        {"source_filename": "paper_a.pdf", "chunk_id": "a_quant", "text": "The confirmed RMSE error was 1.3 units for method alpha on dataset A.", "distance": 0.90},
        {"source_filename": "paper_b.pdf", "chunk_id": "b_chroma", "text": "Evaluation precision and performance reached 88% on dataset B.", "distance": 0.08},
        {"source_filename": "paper_b.pdf", "chunk_id": "b_shared", "text": "Comparative error evaluation measured 2.4 units using method beta.", "distance": 0.12},
        {"source_filename": "paper_b.pdf", "chunk_id": "b_quant", "text": "Confirmed accuracy was 95% for method beta on dataset B under condition test.", "distance": 0.95},
        {"source_filename": "paper_c.pdf", "chunk_id": "c_csv", "text": "Rendimiento precision comparison evaluation reported 86% for method gamma.", "distance": 0.70},
        {"source_filename": "unauthorized.pdf", "chunk_id": "u1", "text": "Accuracy reached 99% in an unauthorized source.", "distance": 0.01},
    ]


def quantitative_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "q1",
            "source_filename": "paper_a.pdf",
            "chunk_id": "a_quant",
            "verification_status": "confirmed_in_source_chunk",
            "value": "1.3",
            "metric": "RMSE",
            "unit": "units",
            "method": "alpha",
            "dataset": "A",
        },
        {
            "row_id": "q2",
            "source_filename": "paper_b.pdf",
            "chunk_id": "b_quant",
            "verification_status": "confirmed_in_source_chunk",
            "value": "95%",
            "metric": "accuracy",
            "method": "beta",
            "dataset": "B",
            "condition": "test",
        },
        {
            "row_id": "q3",
            "source_filename": "paper_b.pdf",
            "chunk_id": "b_shared",
            "verification_status": "not_confirmed",
            "value": "2.4",
            "metric": "error",
        },
        {
            "row_id": "q4",
            "source_filename": "unauthorized.pdf",
            "chunk_id": "u1",
            "verification_status": "confirmed_in_source_chunk",
            "value": "99%",
            "metric": "accuracy",
        },
        {
            "row_id": "q5",
            "source_filename": "paper_a.pdf",
            "chunk_id": "missing",
            "verification_status": "confirmed_in_source_chunk",
            "value": "77%",
            "metric": "accuracy",
        },
    ]


def write_synthetic_project(
    root: Path,
    *,
    quantitative: str = "complete",
    sections: list[dict[str, Any]] | None = None,
    target_total_words: int = 120,
) -> tuple[Path, StateStore, list[dict[str, Any]]]:
    experiment_id = "exp_synthetic"
    experiment = root / experiment_id
    outputs = experiment / "05_outputs"
    outline_dir = outputs / "04_outline"
    thematic = outputs / "03_thematic_analysis"
    draft = outputs / "05_draft"
    chunks_dir = experiment / "03_chunks"
    rag = outputs / "01_rag"
    state_dir = outputs / "00_orchestrator_planner"
    chroma = experiment / "04_chroma_index"
    for path in (outline_dir, thematic, draft, chunks_dir, rag, state_dir, chroma):
        path.mkdir(parents=True, exist_ok=True)
    (chroma / "chroma.sqlite3").write_bytes(b"synthetic")

    active = {
        "active_experiment_id": experiment_id,
        "run_id": "run_synthetic",
        "openai_model": "forbidden-real-model",
        "embedding_model_name": "forbidden-real-embedding",
        "chroma_collection_name": "reference_papers_chunks",
        "generation_profile": {
            "output_language": "español",
            "writing_mode": "síntesis crítica",
            "focus_mode": "comparativo",
            "citation_style": "trazable",
            "target_total_words": target_total_words,
            "min_total_words": 1,
            "max_total_words": 1000,
        },
        "draft_generation_policy": {},
        "rag_policy": {},
    }
    (root / "active_experiment.json").write_text(json.dumps(active), encoding="utf-8")

    outline = {"title": "Estado del arte sintético", "topic": "Tema sintético", "sections": sections or _sections()}
    (outline_dir / "state_of_art_outline.json").write_text(json.dumps(outline), encoding="utf-8")
    pd.DataFrame([{"section_id": item["section_id"], "source_filename": paper["source_filename"], "title": paper["title"]} for item in outline["sections"] for paper in item.get("papers_to_use", [])]).to_csv(outline_dir / "outline_paper_mapping.csv", index=False)
    (outline_dir / "outline_validation_report.json").write_text(json.dumps({"validation_ok": True, "unresolved_references": [], "papers_missing_from_coverage": [], "coverage_entries_not_used": [], "duplicate_coverage_sources": [], "coverage_used_sections_mismatches": []}), encoding="utf-8")
    (outline_dir / "outline_generation_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "validation_ok": True, "safety_policy": {"uses_ground_truth": False, "uses_external_knowledge": False, "source_filenames_validated": True, "titles_validated": True}}), encoding="utf-8")

    chunk_rows = synthetic_chunks()
    pd.DataFrame(chunk_rows).drop(columns=["distance"]).to_csv(chunks_dir / "chunks_clean_for_rag.csv", index=False)
    pd.DataFrame([{"source_filename": "paper_a.pdf", "title": "A"}, {"source_filename": "paper_b.pdf", "title": "B"}, {"source_filename": "paper_c.pdf", "title": "C"}]).to_csv(thematic / "kb_final_for_thematic_analysis.csv", index=False)
    (thematic / "thematic_analysis_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "validation_ok": True, "safety_policy": {"uses_ground_truth": False, "uses_external_knowledge": False}}), encoding="utf-8")
    (thematic / "thematic_validation_report.json").write_text(json.dumps({"validation_ok": True}), encoding="utf-8")
    (rag / "chroma_index_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "collection_name": "reference_papers_chunks", "embedding_model": "forbidden-real-embedding", "num_chunks_indexed": len(chunk_rows), "safety_policy": {"uses_ground_truth": False, "uses_external_knowledge": False}, "ground_truth_indexed": False, "review_sections_indexed": False, "bibliography_indexed": False, "excluded_chunks_indexed": False}), encoding="utf-8")

    qdir = outputs / "02_scientific_knowledge_base"
    if quantitative != "none":
        qdir.mkdir(parents=True, exist_ok=True)
        files = {
            "table": qdir / "quantitative_comparative_table.csv",
            "summary": qdir / "dataset_technique_summary.csv",
            "manifest": qdir / "quantitative_extraction_manifest.json",
        }
        requested = {"table", "summary", "manifest"} if quantitative == "complete" else set(quantitative.split(","))
        if "table" in requested:
            qrows = []
            for row in quantitative_rows():
                qrows.append({
                    **row,
                    "model_or_method": row.get("method", ""),
                    "dataset_or_case": row.get("dataset", ""),
                    "evaluation_scope": row.get("condition", ""),
                    "data_resolution": "synthetic",
                    "value_found_in_source_chunk": row.get("verification_status") in {"confirmed_in_source_chunk", "confirmed_literal_in_source_chunk"},
                })
            pd.DataFrame(qrows).to_csv(files["table"], index=False)
        if "summary" in requested:
            pd.DataFrame([{"source_filename": "paper_a.pdf", "dataset": "A"}, {"source_filename": "paper_b.pdf", "dataset": "B"}]).to_csv(files["summary"], index=False)
        if "manifest" in requested:
            files["manifest"].write_text(json.dumps({"experiment_id": experiment_id, "safety_policy": {"uses_ground_truth": False, "uses_review_sections": False, "uses_bibliography": False, "uses_external_knowledge": False}}), encoding="utf-8")

    state = PipelineState(
        identity=PipelineIdentity(
            experiment_id=experiment_id,
            run_id="run_synthetic",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            schema_version="1.0",
        )
    )
    store = StateStore(state_dir / "pipeline_state.json")
    store.initialize(state)
    return experiment, store, chunk_rows


def chroma_client_factory(path=None, **kwargs):
    return type("Client", (), {"list_collections": lambda self: [type("Named", (), {"name": "reference_papers_chunks"})()]})()


def runtime_factory_for(llm: EvidenceAwareLLM):
    def factory(model, temperature, collection, *, project_dir=None):
        return DraftWritingRuntime(llm, collection)
    return factory
