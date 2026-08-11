from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.adapters.draft_writing_runtime import (
    build_real_draft_execution,
    commit_executed_draft,
    execute_prepared_draft,
    prepare_draft_execution,
)
from src.contracts.agent_result import QualityStatus
from tests.v17.agent06_v17_test_support import (
    CONTRACT_ARTIFACTS,
    EvidenceAwareLLM,
    SyntheticCollection,
    chroma_client_factory,
    runtime_factory_for,
    write_synthetic_project,
)




class _LegacyLengthLLM(EvidenceAwareLLM):
    """Produce a legacy-valid section length without using external services."""

    def __call__(self, prompt: str):
        payload = super().__call__(prompt)
        filler = " ".join(["evidencia"] * 25)
        draft_text = str(payload.get("draft_text", ""))
        base, separator, citation_tail = draft_text.partition(" [")
        if separator:
            payload["draft_text"] = f"{base} {filler}{separator}{citation_tail}"
        claims = payload.get("claims") or []
        if claims:
            claims[0]["claim"] = f"{claims[0]['claim']} {filler}"
        return payload


def _normalize_validation_paths(report: dict) -> dict:
    normalized = json.loads(json.dumps(report))
    normalized.pop("raw_section_outputs_directory", None)
    for attempts in (normalized.get("generation_attempts") or {}).values():
        for attempt in attempts:
            if "attempt_validation_path" in attempt:
                attempt["attempt_validation_path"] = Path(
                    attempt["attempt_validation_path"]
                ).name
            if "rag_trace_path" in attempt:
                attempt["rag_trace_path"] = Path(attempt["rag_trace_path"]).name
    return normalized


def _normalize_draft_claim_identity(sections: list) -> list:
    """claim_uid es un UUID4 aleatorio y opaco, minteado de nuevo en
    CADA ejecución (ver src/tools/draft_writing/claim_identity.py --
    decisión explícita: identidad persistente NO determinista, nunca un
    hash de contenido). Es correcto -- y esperado -- que dos corridas
    independientes del mismo INITIAL_DRAFT produzcan claim_uid distintos
    para "el mismo" claim en términos de contenido. Este test compara
    determinismo del contenido CIENTÍFICO (texto, citas, veredictos),
    no de la identidad -- se excluye claim_uid antes de comparar, igual
    que ya se excluyen rutas específicas del entorno arriba."""
    normalized = json.loads(json.dumps(sections))
    for section in normalized:
        for claim in section.get("claims") or []:
            claim.pop("claim_uid", None)
    return normalized

HYBRID = {
    "retrieval_strategy": "hybrid_chroma_csv_rrf_balanced",
    "top_k_evidence_per_section": 5,
    "chroma_quota": 2,
    "csv_quota": 2,
    "rrf_quota": 1,
    "candidate_multiplier": 3,
    "quantitative_evidence_quota": 1,
    "max_candidates_per_source": 3,
    "max_evidence_chars": 5000,
    "organizational_target_words": 20,
    "organizational_minimum_words": 1,
    "organizational_maximum_words": 40,
    "substantive_minimum_ratio": 0.60,
    "substantive_maximum_ratio": 1.60,
}


class TestAgent06V17EndToEndIntegration(unittest.TestCase):
    def _run(self, *, quantitative="complete", mode="valid"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        _, store, chunk_rows = write_synthetic_project(root, quantitative=quantitative)
        collection = SyntheticCollection(chunk_rows)
        llm = EvidenceAwareLLM(mode)
        agent, agent_input, cfg = build_real_draft_execution(
            root,
            collection_factory=lambda _: collection,
            runtime_factory=runtime_factory_for(llm),
            chroma_client_factory=chroma_client_factory,
            policy_overrides=HYBRID,
        )
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        return temp, root, store, collection, llm, agent_input, cfg, prepared, executed

    def test_v17_approved_end_to_end_and_commit(self):
        temp, root, store, collection, llm, agent_input, cfg, prepared, executed = self._run()
        self.addCleanup(temp.cleanup)
        result = executed.result
        self.assertEqual(result.execution_status.value, "COMPLETED")
        self.assertEqual(result.quality_status, QualityStatus.APPROVED)
        self.assertEqual(result.decision.code, "DRAFT_APPROVED")
        report = json.loads(Path(result.output_artifacts["draft_validation_report.json"].path).read_text())
        self.assertTrue(report["validation_ok"])
        self.assertEqual(report["invalid_citation_count"], 0)
        self.assertEqual(report["numeric_failure_count"], 0)
        self.assertTrue(report["global_length_valid"])
        self.assertEqual(report["sections_outside_word_range"], [])
        self.assertEqual(set(name for name in result.output_artifacts if name != "raw_section_outputs"), set(CONTRACT_ARTIFACTS))

        manifest = json.loads(Path(result.output_artifacts["draft_generation_manifest.json"].path).read_text())
        self.assertEqual(manifest["retrieval_strategy"], "hybrid_chroma_csv_rrf_balanced")
        self.assertEqual(manifest["versions"]["stage"], "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE")
        self.assertEqual(manifest["fingerprint"], agent_input.policy["current_fingerprint"])

        evidence = pd.read_csv(result.output_artifacts["draft_rag_evidence.csv"].path)
        self.assertFalse(evidence.duplicated(["section_id", "source_filename", "chunk_id"]).any())
        self.assertNotIn("unauthorized.pdf", set(evidence["source_filename"]))
        self.assertIn("quantitative_greedy", set(evidence["selection_bucket"].dropna()))
        self.assertTrue(any("chroma" in str(value) for value in evidence["retrieval_sources"].dropna()))
        self.assertTrue(any("csv" in str(value) for value in evidence["retrieval_sources"].dropna()))
        self.assertTrue(evidence["rrf_score"].notna().any())
        self.assertLessEqual(len(evidence), 15)
        for _, group in evidence.groupby("section_id"):
            self.assertLessEqual(len(group), HYBRID["top_k_evidence_per_section"])
            self.assertTrue((group.groupby("source_filename").size() <= HYBRID["max_candidates_per_source"]).all())
            self.assertLessEqual(sum(len(str(value)) for value in group["text"]), HYBRID["max_evidence_chars"])

        traces = list((Path(cfg["output_dir"]) / "raw_section_outputs").glob("*_rag_trace.json"))
        self.assertTrue(traces)
        for trace_path in traces:
            trace = json.loads(trace_path.read_text())
            allowed = set(trace["allowed_citations"])
            final = {f"[{row['source_filename']} | {row['chunk_id']}]" for row in trace["retrieved_chunks"]}
            self.assertEqual(allowed, final)

        state_before = store.load()
        self.assertIsNotNone(state_before.pending_execution)
        committed = commit_executed_draft(store=store, executed=executed)
        self.assertIsNone(committed.pending_execution)
        self.assertEqual(committed.stages["06_agente_redactor"].quality_status, QualityStatus.APPROVED)
        self.assertEqual(llm.calls, 3)
        self.assertGreater(len(collection.query_calls), 0)

    def test_without_quantitative_context_still_approves_and_returns_slots(self):
        temp, root, store, collection, llm, agent_input, cfg, prepared, executed = self._run(quantitative="none")
        self.addCleanup(temp.cleanup)
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        evidence = pd.read_csv(executed.result.output_artifacts["draft_rag_evidence.csv"].path)
        self.assertNotIn("quantitative_greedy", set(evidence.get("selection_bucket", pd.Series(dtype=str)).dropna()))
        quant_used = pd.read_csv(executed.result.output_artifacts["quantitative_comparative_table_used.csv"].path)
        self.assertTrue(quant_used.empty)
        dataset_used = pd.read_csv(executed.result.output_artifacts["dataset_technique_summary_used.csv"].path)
        self.assertTrue(dataset_used.empty)
        self.assertEqual(llm.calls, 3)

    def test_v16_legacy_complete_uses_same_agent_and_v16_versions(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        _, store, chunk_rows = write_synthetic_project(root, quantitative="none")
        collection = SyntheticCollection(chunk_rows)
        llm = _LegacyLengthLLM("valid")
        agent, agent_input, cfg = build_real_draft_execution(
            root,
            collection_factory=lambda _: collection,
            runtime_factory=runtime_factory_for(llm),
            chroma_client_factory=chroma_client_factory,
        )
        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        self.assertEqual(executed.result.quality_status, QualityStatus.APPROVED)
        manifest = json.loads(Path(executed.result.output_artifacts["draft_generation_manifest.json"].path).read_text())
        self.assertEqual(manifest["versions"]["stage"], "06_AGENTIC_V16_BEHAVIOR_PRESERVING")
        self.assertNotIn("quantitative_selection", manifest["versions"])
        self.assertNotIn("budget", manifest["versions"])
        self.assertEqual(agent_input.policy["retrieval_strategy"], "legacy_chroma_then_csv_restricted")

    def test_v17_deterministic_scientific_outputs(self):
        outputs = []
        for _ in range(2):
            temp, root, store, collection, llm, agent_input, cfg, prepared, executed = self._run()
            self.addCleanup(temp.cleanup)
            result = executed.result
            outputs.append({
                "evidence": pd.read_csv(
                    result.output_artifacts["draft_rag_evidence.csv"].path,
                    keep_default_na=False,
                ).to_dict("records"),
                "validation": json.loads(Path(result.output_artifacts["draft_validation_report.json"].path).read_text()),
                "draft": json.loads(Path(result.output_artifacts["state_of_art_draft.json"].path).read_text()),
                "fingerprint": agent_input.policy["current_fingerprint"],
            })
        # Fingerprints include temporary absolute paths, so scientific output must match while fingerprints differ by isolated project path.
        self.assertEqual(outputs[0]["evidence"], outputs[1]["evidence"])
        self.assertEqual(
            _normalize_validation_paths(outputs[0]["validation"]),
            _normalize_validation_paths(outputs[1]["validation"]),
        )
        self.assertEqual(
            _normalize_draft_claim_identity(outputs[0]["draft"]["sections"]),
            _normalize_draft_claim_identity(outputs[1]["draft"]["sections"]),
        )


if __name__ == "__main__":
    unittest.main()
