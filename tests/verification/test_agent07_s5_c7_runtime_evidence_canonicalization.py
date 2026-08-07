from __future__ import annotations

from copy import deepcopy

import pytest

from src.adapters.verification_runtime import (
    VerificationRuntimeDependencies,
    _base_metrics,
    _blocked_runtime_result,
    _independent_retrieve_claim,
    validate_agent07_runtime_result_contract,
)


def _binding() -> dict[str, str]:
    return {
        "experiment_id": "experimento_paper_02",
        "collection_name": "reference_papers_chunks",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chroma_manifest_fingerprint": "a" * 64,
        "chunks_manifest_fingerprint": "b" * 64,
    }


class Retriever:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)

    def retrieve_more(self, request):
        return {"rounds_executed": 1, "selected_candidates": self.candidates}


def _dependencies(candidates) -> VerificationRuntimeDependencies:
    return VerificationRuntimeDependencies(
        retrieval_tool=Retriever(candidates),
        retriever_binding=_binding(),
    )


def _s5_c7_context() -> dict:
    # Sanitized shape of the productive S5_C7 condition: three inherited rows,
    # unique both by evidence_id and by physical (source_filename, chunk_id).
    return {
        "claim_id": "S5_C7",
        "section_id": "S5",
        "authorized_source_filenames": ("paper-a.pdf", "paper-b.pdf"),
        "eligible_evidence": (
            {
                "evidence_id": "chunk-a-01",
                "source_filename": "paper-a.pdf",
                "chunk_id": "chunk-a-01",
                "canonical_text": "Evidence text for the first inherited chunk.",
                "authorized_for_section": True,
                "usage_role": "ELIGIBLE",
            },
            {
                "evidence_id": "chunk-a-02",
                "source_filename": "paper-a.pdf",
                "chunk_id": "chunk-a-02",
                "canonical_text": "Evidence text for the second inherited chunk.",
                "authorized_for_section": True,
                "usage_role": "ELIGIBLE",
            },
            {
                "evidence_id": "chunk-b-01",
                "source_filename": "paper-b.pdf",
                "chunk_id": "chunk-b-01",
                "canonical_text": "Evidence text for the third inherited chunk.",
                "authorized_for_section": True,
                "usage_role": "ELIGIBLE",
            },
        ),
    }


def _alias_candidate(*, text="Evidence text for the second inherited chunk.") -> dict:
    return {
        "source_filename": "paper-a.pdf",
        "chunk_id": "chunk-a-02",
        "text": text,
        "query_ids": ("S5_C7",),
    }


def test_s5_c7_retrieval_alias_is_canonicalized_before_snapshot_and_runtime_validation():
    original = _s5_c7_context()
    updated, record = _independent_retrieve_claim(
        deepcopy(original),
        _dependencies((_alias_candidate(),)),
    )

    assert original == _s5_c7_context()  # producer does not mutate the handoff
    assert len(updated["eligible_evidence"]) == 3
    assert len({(e["source_filename"], e["chunk_id"]) for e in updated["eligible_evidence"]}) == 3

    merged = next(e for e in updated["eligible_evidence"] if e["chunk_id"] == "chunk-a-02")
    assert merged["evidence_id"] == "chunk-a-02"
    assert merged["evidence_id_aliases"] == ("chunk-a-02", "paper-a.pdf::chunk-a-02")
    assert set(merged["retrieval_origins"]) == {"AGENT06_INHERITED", "AGENT07_INDEPENDENT_RAG"}
    assert set(merged["usage_roles"]) == {"ELIGIBLE", "SUPPORT"}

    snapshot = record["verification_context_snapshot"]["eligible_evidence"]
    assert len(snapshot) == 3
    assert len({(e["source_filename"], e["chunk_id"]) for e in snapshot}) == 3
    assert sum(e["chunk_id"] == "chunk-a-02" for e in snapshot) == 1

    assert record["retrieved_candidate_ids"] == ("chunk-a-02",)
    assert record["retrieved_candidate_records"][0]["evidence_id"] == "chunk-a-02"
    assert record["retrieved_candidate_records"][0]["query_ids"] == ("S5_C7",)

    metrics = _base_metrics(
        claims_processed=1,
        independent_rag_claims=1,
        independent_rag_claims_with_results=1,
        independent_rag_claim_records=(record,),
    )
    runtime = _blocked_runtime_result(
        stage="VERIFICATION",
        claim_id="S5_C7",
        section_id="S5",
        error_code="AGENT07_TEST_AUDIT_ONLY",
        classification="CONTRACTUAL",
        schema_versions={},
        metrics=metrics,
    )
    validated = validate_agent07_runtime_result_contract(runtime)
    assert validated["execution_metrics"]["independent_rag_claim_records"][0]["retrieved_candidate_ids"] == ("chunk-a-02",)


def test_new_physical_chunk_remains_an_additional_row():
    candidate = {
        "source_filename": "paper-b.pdf",
        "chunk_id": "chunk-b-02",
        "text": "A genuinely new authorized chunk.",
        "query_ids": ("S5_C7",),
    }
    updated, record = _independent_retrieve_claim(_s5_c7_context(), _dependencies((candidate,)))
    assert len(updated["eligible_evidence"]) == 4
    assert record["retrieved_candidate_ids"] == ("paper-b.pdf::chunk-b-02",)


def test_same_physical_pair_with_semantically_different_text_remains_blocked():
    with pytest.raises(ValueError, match="AGENT07_CONTEXT_ADAPTER_EVIDENCE_TEXT_CONFLICT"):
        _independent_retrieve_claim(
            _s5_c7_context(),
            _dependencies((_alias_candidate(text="Scientifically different evidence text."),)),
        )


def test_same_physical_pair_with_contradictory_authorization_remains_blocked():
    context = _s5_c7_context()
    rows = list(context["eligible_evidence"])
    rows[1] = {**rows[1], "authorized_for_section": False}
    context["eligible_evidence"] = tuple(rows)
    with pytest.raises(ValueError, match="AGENT07_CONTEXT_ADAPTER_EVIDENCE_AUTHORIZATION_CONFLICT"):
        _independent_retrieve_claim(context, _dependencies((_alias_candidate(),)))
