from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.adapters.claim_verification_context import (
    build_claim_verification_context_from_agent06_handoff,
)
from src.tools.verification.validation import validate_claim_verification_context


_FIXTURE = Path(__file__).parent / "fixtures" / "s2_c4_evidence_merge.json"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _base_context() -> dict:
    return deepcopy(_fixture()["handoff_context"])


def _retrieved(index: int = 0) -> dict:
    return deepcopy(_fixture()["retrieved_candidates"][index])


def test_real_s2_c4_inherited_and_retrieved_equivalent_pair_is_merged_and_validated():
    source = _base_context()
    inherited = deepcopy(source["eligible_evidence"][0])
    retrieved = _retrieved(0)
    assert (inherited["source_filename"], inherited["chunk_id"]) == (
        retrieved["source_filename"], retrieved["chunk_id"]
    )
    assert inherited["text"] != retrieved["text"]  # spaces versus line breaks in the real execution
    before = deepcopy(source)
    source["eligible_evidence"] = (inherited, retrieved)
    source["agent07_independent_retrieval_executed"] = True
    source["agent07_independent_retrieval_rounds"] = 1
    source["agent07_independent_retrieval_status"] = "COMPLETED_WITH_RESULTS"

    adapted = build_claim_verification_context_from_agent06_handoff(
        source, verification_policy={}
    )
    validated = validate_claim_verification_context(adapted)

    assert validated["claim_id"] == "S2_C4"
    assert validated["allowed_source_pairs"] == ((inherited["source_filename"], inherited["chunk_id"]),)
    assert len(validated["inherited_evidence_assessment"]["evidence_rows"]) == 1
    assert len(validated["retrieval_result"]["selected_candidates"]) == 1
    merged = validated["retrieval_result"]["selected_candidates"][0]
    assert merged["evidence_id"] == inherited["chunk_id"]
    assert set(merged["evidence_id_aliases"]) == {inherited["evidence_id"], retrieved["evidence_id"]}
    assert set(merged["retrieval_origins"]) == {"AGENT06_INHERITED", "AGENT07_INDEPENDENT_RAG"}
    assert set(merged["usage_roles"]) == {"ELIGIBLE", "SUPPORT"}
    assert merged["usage_role"] == "SUPPORT"
    assert source["eligible_evidence"] == (inherited, retrieved)
    assert before["eligible_evidence"] == [inherited]


def test_identical_duplicate_is_deduplicated_canonically():
    source = _base_context()
    row = deepcopy(source["eligible_evidence"][0])
    source["eligible_evidence"] = (row, deepcopy(row))
    adapted = build_claim_verification_context_from_agent06_handoff(source, verification_policy={})
    assert len(adapted["inherited_evidence_assessment"]["evidence_rows"]) == 1


def test_representation_equivalent_duplicate_normalizes_whitespace_and_sequences():
    source = _base_context()
    first = deepcopy(source["eligible_evidence"][0])
    first["query_ids"] = ["S2_C4"]
    second = deepcopy(first)
    second["text"] = "\n  " + first["text"].replace(" ", "\n", 8) + "  \n"
    second["canonical_text"] = second.pop("text")
    second["query_ids"] = ("S2_C4",)
    source["eligible_evidence"] = (first, second)
    adapted = build_claim_verification_context_from_agent06_handoff(source, verification_policy={})
    row = adapted["inherited_evidence_assessment"]["evidence_rows"][0]
    assert "\n" not in row["canonical_text"]
    assert row["query_ids"] == ("S2_C4",)


def test_same_pair_with_different_text_is_a_specific_semantic_conflict():
    source = _base_context()
    first = deepcopy(source["eligible_evidence"][0])
    second = deepcopy(first)
    second["evidence_id"] = "retrieved-alias"
    second["text"] = "Texto científicamente distinto para el mismo chunk."
    source["eligible_evidence"] = (first, second)
    with pytest.raises(ValueError, match="AGENT07_CONTEXT_ADAPTER_EVIDENCE_TEXT_CONFLICT"):
        build_claim_verification_context_from_agent06_handoff(source, verification_policy={})


def test_same_pair_with_contradictory_authorization_is_a_specific_conflict():
    source = _base_context()
    first = deepcopy(source["eligible_evidence"][0])
    second = deepcopy(first)
    second["evidence_id"] = "retrieved-alias"
    second["authorized_for_section"] = False
    source["eligible_evidence"] = (first, second)
    with pytest.raises(ValueError, match="AGENT07_CONTEXT_ADAPTER_EVIDENCE_AUTHORIZATION_CONFLICT"):
        build_claim_verification_context_from_agent06_handoff(source, verification_policy={})


def test_inherited_plus_new_independent_retrieval_keeps_both_authorized_pairs():
    source = _base_context()
    inherited = deepcopy(source["eligible_evidence"][0])
    retrieved = _retrieved(1)
    source["eligible_evidence"] = (inherited, retrieved)
    source["agent07_independent_retrieval_executed"] = True
    source["agent07_independent_retrieval_rounds"] = 1
    source["agent07_independent_retrieval_status"] = "COMPLETED_WITH_RESULTS"
    adapted = build_claim_verification_context_from_agent06_handoff(source, verification_policy={})
    assert len(adapted["inherited_evidence_assessment"]["evidence_rows"]) == 1
    assert len(adapted["retrieval_result"]["selected_candidates"]) == 1
    assert set(adapted["allowed_source_pairs"]) == {
        (inherited["source_filename"], inherited["chunk_id"]),
        (retrieved["source_filename"], retrieved["chunk_id"]),
    }
