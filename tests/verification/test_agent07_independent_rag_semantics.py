from copy import deepcopy
import hashlib, json
import pytest

from src.adapters.verification_runtime import (
    VerificationRuntimeDependencies,
    _independent_retrieve_claim,
)
from src.adapters.agent06_verification_handoff import Agent07RetrieverBinding
from src.adapters.agent07c_handoff import prepare_agent07c_input_from_agent07, REQUIRED_SAFETY_POLICY
from tests.verification.agent07c_test_support import terminal_handoff_args
from tests.verification.test_agent07c_terminal_safety_closure import _prepare_args, _fp


def _binding():
    return Agent07RetrieverBinding("exp","collection","embed","a"*64,"b"*64)


def _ctx():
    return {
        "claim_id":"c1","section_id":"s1","authorized_source_filenames":("p.pdf",),
        "eligible_evidence":({"evidence_id":"p.pdf::ch1","source_filename":"p.pdf","chunk_id":"ch1","text":"e","authorized_for_section":True},),
    }


class Retriever:
    def __init__(self, *, rounds=1, claim_id="c1"):
        self.rounds=rounds; self.claim_id=claim_id
    def retrieve_more(self, request):
        return {
            "rounds_executed":self.rounds,
            "selected_candidates":({"source_filename":"p.pdf","chunk_id":"ch1","text":"e","query_ids":(self.claim_id,)},),
        }


def _deps(retriever, binding=None):
    return VerificationRuntimeDependencies(retrieval_tool=retriever,retriever_binding=(binding or _binding()).__dict__ if hasattr((binding or _binding()),'__dict__') else {
        "experiment_id":(binding or _binding()).experiment_id,
        "collection_name":(binding or _binding()).collection_name,
        "embedding_model":(binding or _binding()).embedding_model,
        "chroma_manifest_fingerprint":(binding or _binding()).chroma_manifest_fingerprint,
        "chunks_manifest_fingerprint":(binding or _binding()).chunks_manifest_fingerprint,
    })


def test_selector_without_retrieval_does_not_count_as_rag():
    with pytest.raises(ValueError, match="ROUND_MISSING"):
        _independent_retrieve_claim(_ctx(), _deps(Retriever(rounds=0)))


def test_real_retrieval_counts_and_is_claim_linked():
    updated, record = _independent_retrieve_claim(_ctx(), _deps(Retriever()))
    assert record["retrieval_requested"] == 1
    assert record["retrieval_rounds"] == 1
    assert record["claim_id"] == "c1"
    assert record["retrieved_candidate_ids"] == ("p.pdf::ch1",)
    assert any(e["retrieval_origin"] == "AGENT07_INDEPENDENT_RAG" for e in updated["eligible_evidence"])


def test_retrieval_for_other_claim_is_rejected():
    with pytest.raises(ValueError, match="CLAIM_MISMATCH"):
        _independent_retrieve_claim(_ctx(), _deps(Retriever(claim_id="other")))


def test_wrong_retriever_binding_does_not_support_safety():
    draft,contexts,b,r,args=_prepare_args()
    args=deepcopy(args)
    args["runtime_result"]["execution_metrics"]["independent_rag_claim_records"][0]["retriever_binding_fingerprint"]="0"*64
    with pytest.raises(ValueError, match="INDEPENDENT_RAG_UNPROVEN"):
        prepare_agent07c_input_from_agent07(
            provisional_bundle=b.to_dict(), resolution_result=r.to_dict(), source_draft=draft,
            source_draft_markdown=draft["sections"][0]["text"], experiment_id="exp",
            committed_source_draft_fingerprint=_fp(draft), claim_source_contexts=contexts,
            safety_policy=REQUIRED_SAFETY_POLICY, **args,
        )


def test_one_claim_without_retrieval_fails_independent_rag_safety():
    draft,contexts,b,r,args=_prepare_args()
    args=deepcopy(args)
    args["runtime_result"]["execution_metrics"]["independent_rag_claim_records"]=()
    args["runtime_result"]["execution_metrics"]["independent_rag_claims"]=0
    args["runtime_result"]["execution_metrics"]["independent_rag_claims_with_results"]=0
    args["runtime_result"]["execution_metrics"]["independent_rag_claims_without_results"]=0
    with pytest.raises(ValueError, match="INDEPENDENT_RAG_UNPROVEN"):
        prepare_agent07c_input_from_agent07(
            provisional_bundle=b.to_dict(), resolution_result=r.to_dict(), source_draft=draft,
            source_draft_markdown=draft["sections"][0]["text"], experiment_id="exp",
            committed_source_draft_fingerprint=_fp(draft), claim_source_contexts=contexts,
            safety_policy=REQUIRED_SAFETY_POLICY, **args,
        )


def test_all_claims_with_real_retrieval_allow_safety_true():
    draft,contexts,b,r,args=_prepare_args()
    prepared=prepare_agent07c_input_from_agent07(
        provisional_bundle=b.to_dict(), resolution_result=r.to_dict(), source_draft=draft,
        source_draft_markdown=draft["sections"][0]["text"], experiment_id="exp",
        committed_source_draft_fingerprint=_fp(draft), claim_source_contexts=contexts,
        safety_policy=REQUIRED_SAFETY_POLICY, **args,
    )
    assert prepared.result_contract_valid is True
