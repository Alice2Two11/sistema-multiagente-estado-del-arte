from copy import deepcopy
import hashlib
import pytest

from src.adapters.verification_runtime import (
    VerificationRuntimeDependencies,
    _independent_retrieve_claim,
    create_agent07_runtime_result,
    _candidate_inventory,
    _base_metrics,
    _resolution_to_runtime_status,
)
from src.adapters.agent06_verification_handoff import Agent07RetrieverBinding
from src.adapters.agent07c_handoff import prepare_agent07c_input_from_agent07, REQUIRED_SAFETY_POLICY
from src.tools.verification.traceability import ClaimEvidenceTraceabilityRow
from src.tools.verification.validation import create_provisional_verification_traceability_bundle
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from src.tools.verification.corrections import fingerprint_text
from tests.verification.agent07c_test_support import terminal_handoff_args
from tests.verification.test_agent07c_terminal_safety_closure import _source_context, _fp
from tests.verification.test_multi_proposal_resolution_phase66 import claim, metrics


def binding():
    return Agent07RetrieverBinding("exp", "c", "m", "a" * 64, "b" * 64)


def deps(retriever):
    b = binding()
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


class Retriever:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
    def retrieve_more(self, request):
        return {"rounds_executed": 1, "selected_candidates": self.candidates}


def candidate(*, source="p.pdf", chunk="new", text="new evidence", claim_id="c1"):
    return {"source_filename": source, "chunk_id": chunk, "text": text, "query_ids": (claim_id,)}


def context(evidence=()):
    return {"claim_id": "c1", "section_id": "s1", "eligible_evidence": tuple(evidence), "authorized_source_filenames": ("p.pdf",)}


def bundle_with_evidence(*, evidence_id="p.pdf::new", text="new evidence", text_fp=None):
    row = ClaimEvidenceTraceabilityRow(
        "c1", "s1", evidence_id, "p.pdf", "new",
        text_fp or fingerprint_text(text), "SUPPORT", True, True, "NOT_EVALUATED",
    ).to_dict()
    return create_provisional_verification_traceability_bundle(
        claim_traceability_rows=(claim(corrections=()),), correction_traceability_rows=(),
        claim_evidence_traceability_rows=(row,), correction_evidence_traceability_rows=(),
        reverification_traceability_rows=(), metrics=metrics(True), aggregation_status="VALID",
        metrics_status="COMPUTED", partial_reason_codes=(), aggregation_issue_codes=(),
        aggregation_warnings=(), normalized_bundle_status="COMPUTED",
        normalized_bundle_fingerprint="1" * 64, aggregation_audit_fingerprint="2" * 64,
        input_collection_fingerprints={}, policy_versions={"verification": "v1"},
        schema_versions={"bundle": "v4"}, correction_applied=False,
        official_artifacts_created=False, additional_llm_calls=0, additional_retrieval_rounds=0,
    )


def runtime_for_new_evidence(bundle, resolution, source_contexts, *, include_retrieved=True, snapshot_fp=None):
    seed = {"evidence_id":"p.pdf::new","source_filename":"p.pdf","chunk_id":"new","text":"new evidence","authorized_for_section":True}
    seeded_contexts=tuple(dict(c, eligible_evidence=(seed,)) for c in source_contexts)
    base = terminal_handoff_args(bundle, resolution, seeded_contexts)
    # The scientific source handoff remains the caller-provided Agent 06 context;
    # the runtime snapshot is independently retained below.
    base["agent06_handoff"]["claim_verification_contexts"] = tuple(source_contexts)
    runtime = deepcopy(base["runtime_result"])
    rec = runtime["execution_metrics"]["independent_rag_claim_records"][0]
    fp = snapshot_fp or fingerprint_text("new evidence")
    if include_retrieved:
        rec.update({
            "retrieval_status": "COMPLETED_WITH_RESULTS",
            "retrieved_candidate_ids": ("p.pdf::new",),
            "retrieved_candidate_records": ({
                "evidence_id": "p.pdf::new", "source_filename": "p.pdf", "chunk_id": "new",
                "query_ids": ("c1",), "text_fingerprint": fp,
            },),
            "verification_context_snapshot": {
                "claim_id": "c1", "section_id": "s1", "eligible_evidence": ({
                    "evidence_id": "p.pdf::new", "source_filename": "p.pdf", "chunk_id": "new",
                    "authorized_for_section": True, "text_fingerprint": fp,
                },),
            },
        })
        runtime["execution_metrics"]["independent_rag_claims_with_results"] = 1
        runtime["execution_metrics"]["independent_rag_claims_without_results"] = 0
    else:
        rec["retrieval_status"]="COMPLETED_NO_RESULTS"
        rec["retrieved_candidate_ids"]=()
        rec["retrieved_candidate_records"]=()
        rec["verification_context_snapshot"]={"claim_id":"c1","section_id":"s1","eligible_evidence":()}
        runtime["execution_metrics"]["independent_rag_claims_with_results"] = 0
        runtime["execution_metrics"]["independent_rag_claims_without_results"] = 1
    return create_agent07_runtime_result(
        provisional_bundle=bundle.to_dict(), multi_proposal_resolution_result=resolution.to_dict(),
        candidate_artifact_inventory=_candidate_inventory(bundle.to_dict(), resolution.to_dict(), {"provisional_bundle":"v4","multi_proposal_resolution":"v1"}),
        execution_metrics=runtime["execution_metrics"], runtime_warnings=(), runtime_issue_codes=(),
        runtime_error_records=(), blocked_runtime_audit_record=None,
        runtime_status=_resolution_to_runtime_status(resolution.resolution_status), correction_applied=False,
        official_artifacts_created=False, evaluation_ready_emitted=False,
    ).to_dict(), base


def test_original_context_without_evidence_snapshot_contains_retrieved_candidate():
    updated, record = _independent_retrieve_claim(context(()), deps(Retriever((candidate(),))))
    assert updated["eligible_evidence"][0]["evidence_id"] == "p.pdf::new"
    snap = record["verification_context_snapshot"]["eligible_evidence"][0]
    assert snap["evidence_id"] == "p.pdf::new"
    assert snap["text_fingerprint"] == fingerprint_text("new evidence")


def test_new_retrieved_and_used_evidence_is_authorized_for_agent07c():
    draft, contexts = _source_context()
    contexts = (dict(contexts[0], authorized_source_filenames=("p.pdf",)),)
    b = bundle_with_evidence(); r = resolve_multiple_correction_proposals(b)
    runtime, args = runtime_for_new_evidence(b, r, contexts)
    prepared = prepare_agent07c_input_from_agent07(
        provisional_bundle=b.to_dict(), resolution_result=r.to_dict(), source_draft=draft,
        source_draft_markdown=draft["sections"][0]["text"], experiment_id="exp",
        committed_source_draft_fingerprint=_fp(draft), claim_source_contexts=contexts,
        safety_policy=REQUIRED_SAFETY_POLICY, runtime_result=runtime,
        retriever_binding=args["retriever_binding"], agent06_handoff=args["agent06_handoff"],
    )
    assert prepared.result_contract_valid is True


def test_inherited_evidence_remains_authorized():
    draft, contexts = _source_context()
    inherited = {"evidence_id":"p.pdf::new","source_filename":"p.pdf","chunk_id":"new","text":"new evidence","authorized_for_section":True}
    contexts = (dict(contexts[0], eligible_evidence=(inherited,), authorized_source_filenames=("p.pdf",)),)
    b = bundle_with_evidence(); r = resolve_multiple_correction_proposals(b)
    args = terminal_handoff_args(b, r, contexts)
    prepared = prepare_agent07c_input_from_agent07(
        provisional_bundle=b.to_dict(), resolution_result=r.to_dict(), source_draft=draft,
        source_draft_markdown=draft["sections"][0]["text"], experiment_id="exp",
        committed_source_draft_fingerprint=_fp(draft), claim_source_contexts=contexts,
        safety_policy=REQUIRED_SAFETY_POLICY, **args,
    )
    assert prepared.result_contract_valid is True



def test_terminal_evidence_alias_change_preserves_physical_provenance():
    draft, contexts = _source_context()
    contexts = (dict(contexts[0], authorized_source_filenames=("p.pdf",)),)
    b = bundle_with_evidence(evidence_id="E01")
    r = resolve_multiple_correction_proposals(b)
    runtime, _ = runtime_for_new_evidence(b, r, contexts)
    assert runtime["result_contract_valid"] is True


def test_evidence_outside_agent06_and_agent07_retrieval_is_rejected():
    draft, contexts = _source_context(); contexts=(dict(contexts[0],authorized_source_filenames=("p.pdf",)),)
    b=bundle_with_evidence(); r=resolve_multiple_correction_proposals(b)
    with pytest.raises(ValueError, match="TERMINAL_EVIDENCE_CONTEXT_MISMATCH"):
        runtime_for_new_evidence(b,r,contexts,include_retrieved=False)


def test_verifier_terminal_evidence_not_in_snapshot_is_rejected():
    draft, contexts = _source_context(); contexts=(dict(contexts[0],authorized_source_filenames=("p.pdf",)),)
    b=bundle_with_evidence(); r=resolve_multiple_correction_proposals(b)
    with pytest.raises(ValueError, match="TERMINAL_EVIDENCE_CONTEXT_MISMATCH"):
        runtime_for_new_evidence(b, r, contexts, include_retrieved=False)


def test_same_evidence_id_with_altered_text_is_rejected():
    inherited=({"evidence_id":"p.pdf::new","source_filename":"p.pdf","chunk_id":"new","text":"old","authorized_for_section":True},)
    with pytest.raises(ValueError, match="CANDIDATE_CONFLICT"):
        _independent_retrieve_claim(context(inherited), deps(Retriever((candidate(text="changed"),))))


def test_incorrect_text_fingerprint_is_rejected():
    draft, contexts = _source_context(); contexts=(dict(contexts[0],authorized_source_filenames=("p.pdf",)),)
    b=bundle_with_evidence(); r=resolve_multiple_correction_proposals(b)
    with pytest.raises(ValueError, match="TERMINAL_EVIDENCE_CONTEXT_MISMATCH|SNAPSHOT_MISMATCH"):
        runtime_for_new_evidence(b, r, contexts, snapshot_fp="0"*64)


def test_identical_candidate_duplicate_is_deduplicated():
    c=candidate()
    updated, record = _independent_retrieve_claim(context(()), deps(Retriever((c, deepcopy(c)))))
    assert len(record["retrieved_candidate_records"]) == 1
    assert len(updated["eligible_evidence"]) == 1


def test_conflicting_candidate_duplicate_is_blocked():
    with pytest.raises(ValueError, match="CANDIDATE_CONFLICT"):
        _independent_retrieve_claim(context(()), deps(Retriever((candidate(text="a"), candidate(text="b")))))
