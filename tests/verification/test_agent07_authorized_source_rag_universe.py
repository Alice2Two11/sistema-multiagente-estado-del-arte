from copy import deepcopy
import hashlib
import pytest

from src.adapters.verification_runtime import (
    VerificationRuntimeDependencies,
    _independent_retrieve_claim,
    create_agent07_runtime_result,
)
from src.adapters.agent06_verification_handoff import Agent07RetrieverBinding
from tests.verification.agent07c_test_support import terminal_handoff_args
from tests.verification.test_agent07c_terminal_safety_closure import _prepare_args, _fp
from src.adapters.agent07c_handoff import prepare_agent07c_input_from_agent07, REQUIRED_SAFETY_POLICY


def _binding():
    return Agent07RetrieverBinding("exp", "collection", "embed", "a" * 64, "b" * 64)


def _deps(retriever):
    b = _binding()
    return VerificationRuntimeDependencies(
        retrieval_tool=retriever,
        retriever_binding={
            "experiment_id": b.experiment_id,
            "collection_name": b.collection_name,
            "embedding_model": b.embedding_model,
            "chroma_manifest_fingerprint": b.chroma_manifest_fingerprint,
            "chunks_manifest_fingerprint": b.chunks_manifest_fingerprint,
        },
    )


def _ctx(*, evidence=(), sources=("paper.pdf",)):
    return {
        "claim_id": "c1",
        "section_id": "s1",
        "eligible_evidence": tuple(evidence),
        "authorized_source_filenames": tuple(sources),
    }


class Retriever:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.last_request = None

    def retrieve_more(self, request):
        self.last_request = deepcopy(request)
        return {"rounds_executed": 1, "selected_candidates": self.candidates}


def candidate(*, source="paper.pdf", chunk="new-chunk", claim="c1", text="new evidence"):
    return {"source_filename": source, "chunk_id": chunk, "text": text, "query_ids": (claim,)}


def test_claim_without_inherited_evidence_can_retrieve_authorized_chunk():
    retriever = Retriever((candidate(),))
    updated, record = _independent_retrieve_claim(_ctx(evidence=()), _deps(retriever))
    assert retriever.last_request["allowed_source_filenames"] == ("paper.pdf",)
    assert "allowed_source_pairs" not in retriever.last_request
    assert record["retrieval_status"] == "COMPLETED_WITH_RESULTS"
    assert record["retrieved_candidate_ids"] == ("paper.pdf::new-chunk",)
    assert updated["eligible_evidence"][0]["chunk_id"] == "new-chunk"


def test_new_chunk_from_authorized_source_not_inherited_is_accepted():
    inherited = ({
        "evidence_id": "paper.pdf::old", "source_filename": "paper.pdf",
        "chunk_id": "old", "text": "old evidence", "authorized_for_section": True,
    },)
    updated, record = _independent_retrieve_claim(_ctx(evidence=inherited), _deps(Retriever((candidate(chunk="new"),))))
    assert {e["chunk_id"] for e in updated["eligible_evidence"]} == {"old", "new"}
    assert record["retrieved_candidate_ids"] == ("paper.pdf::new",)


def test_chunk_from_unauthorized_source_is_rejected():
    with pytest.raises(ValueError, match="OUTLINE_VIOLATION"):
        _independent_retrieve_claim(_ctx(), _deps(Retriever((candidate(source="other.pdf"),))))


def test_empty_retrieval_is_completed_without_results_and_counts_as_execution():
    updated, record = _independent_retrieve_claim(_ctx(evidence=()), _deps(Retriever(())))
    assert updated["eligible_evidence"] == ()
    assert record["retrieval_status"] == "COMPLETED_NO_RESULTS"
    assert record["retrieved_candidate_ids"] == ()
    assert record["retrieved_candidate_records"] == ()


def test_retrieval_status_contradicting_candidate_ids_is_rejected():
    draft, contexts, bundle, resolution, args = _prepare_args()
    runtime = deepcopy(args["runtime_result"])
    row = runtime["execution_metrics"]["independent_rag_claim_records"][0]
    row["retrieval_status"] = "COMPLETED_NO_RESULTS"
    row["retrieved_candidate_ids"] = ("paper.pdf::x",)
    row["retrieved_candidate_records"] = ({"evidence_id":"paper.pdf::x","source_filename":"paper.pdf","chunk_id":"x","query_ids":("c1",),"text_fingerprint":hashlib.sha256(b"x").hexdigest()},)
    row["verification_context_snapshot"] = {"claim_id":"c1","section_id":"s1","eligible_evidence":({"evidence_id":"paper.pdf::x","source_filename":"paper.pdf","chunk_id":"x","authorized_for_section":True,"text_fingerprint":hashlib.sha256(b"x").hexdigest()},)}
    with pytest.raises(ValueError, match="STATUS_CONTRADICTION"):
        create_agent07_runtime_result(
            provisional_bundle=runtime["provisional_bundle"],
            multi_proposal_resolution_result=runtime["multi_proposal_resolution_result"],
            candidate_artifact_inventory=runtime["candidate_artifact_inventory"],
            execution_metrics=runtime["execution_metrics"],
            runtime_warnings=(), runtime_issue_codes=(), runtime_error_records=(),
            blocked_runtime_audit_record=None, runtime_status=runtime["runtime_status"],
            correction_applied=False, official_artifacts_created=False,
            evaluation_ready_emitted=False,
        )


def test_with_and_without_results_counts_partition_independent_rag_claims():
    _, contexts, bundle, resolution, args = _prepare_args()
    metrics = deepcopy(args["runtime_result"]["execution_metrics"])
    row = metrics["independent_rag_claim_records"][0]
    row["retrieval_status"] = "COMPLETED_NO_RESULTS"
    row["retrieved_candidate_ids"] = ()
    row["retrieved_candidate_records"] = ()
    metrics["independent_rag_claims_with_results"] = 0
    metrics["independent_rag_claims_without_results"] = 1
    runtime = create_agent07_runtime_result(
        provisional_bundle=bundle.to_dict(), multi_proposal_resolution_result=resolution.to_dict(),
        candidate_artifact_inventory=args["runtime_result"]["candidate_artifact_inventory"],
        execution_metrics=metrics, runtime_warnings=(), runtime_issue_codes=(),
        runtime_error_records=(), blocked_runtime_audit_record=None,
        runtime_status=args["runtime_result"]["runtime_status"], correction_applied=False,
        official_artifacts_created=False, evaluation_ready_emitted=False,
    )
    assert runtime.execution_metrics["independent_rag_claims_with_results"] == 0
    assert runtime.execution_metrics["independent_rag_claims_without_results"] == 1


def test_all_claims_with_executed_search_including_no_results_allow_safety():
    draft, contexts, bundle, resolution, args = _prepare_args()
    args = deepcopy(args)
    metrics = args["runtime_result"]["execution_metrics"]
    row = metrics["independent_rag_claim_records"][0]
    row["retrieval_status"] = "COMPLETED_NO_RESULTS"
    row["retrieved_candidate_ids"] = ()
    row["retrieved_candidate_records"] = ()
    metrics["independent_rag_claims_with_results"] = 0
    metrics["independent_rag_claims_without_results"] = 1
    prepared = prepare_agent07c_input_from_agent07(
        provisional_bundle=bundle.to_dict(), resolution_result=resolution.to_dict(),
        source_draft=draft, source_draft_markdown=draft["sections"][0]["text"],
        experiment_id="exp", committed_source_draft_fingerprint=_fp(draft),
        claim_source_contexts=contexts, safety_policy=REQUIRED_SAFETY_POLICY, **args,
    )
    assert prepared.result_contract_valid is True
