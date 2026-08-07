from __future__ import annotations

from hashlib import sha256
import copy

import pytest

from src.config.verification_policy_config import (
    REVERIFICATION_EXECUTION_STATUSES,
    get_verification_input_policy,
)
from src.tools.verification.corrections import build_virtual_corrected_claim, compute_correction_proposal_fingerprint, fingerprint_text
from src.tools.verification.validation import run_virtual_reverification_prechecks


def fp(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def refresh_proposal_fingerprint(c):
    c["correction_validation_result"].setdefault("prompt_version", c["policy"]["correction_user_prompt_version"])
    c["proposal_fingerprint"] = compute_correction_proposal_fingerprint(
        original_claim_fingerprint=c["original_claim_fingerprint"],
        original_section_fingerprint=c["original_section_fingerprint"],
        target_text_fingerprint=fingerprint_text(c["target_span_in_claim"]["text"]),
        claim_id=c["claim_id"],
        action_type=c["correction_action_type"],
        target_span=c["target_span_in_claim"],
        replacement_text=c["replacement_text"],
        evidence_ids=c["evidence_ids"],
        prompt_version=c["correction_validation_result"]["prompt_version"],
    )
    return c


def base_context(action="REPLACE_NUMERIC_VALUE"):
    original="El modelo obtuvo 94 %."
    proposed="El modelo obtuvo 95 %."
    section=original
    cfp=fp(original); sfp=fp(section)
    cv={
        "proposal_status":"ACCEPTED_FOR_REVERIFICATION",
        "correction_applied":False,
        "new_numeric_pairs":[["95","%"]],
        "metric_context":"modelo",
        "new_attribution_elements":[],
        "new_attributions":[],
        "attribution_relation":"",
        "new_citation_refs":[],
        "citation_text_span":None,
        "new_conditions":[],
    }
    c = {
        "correction_id":"a"*64,
        "claim_id":"C1",
        "section_id":"S1",
        "original_claim_text":original,
        "proposed_claim_text":proposed,
        "section_text":section,
        "source_verdict":"PARTIALLY_SUPPORTED",
        "source_issue_codes":["UNSUPPORTED_NUMERIC_VALUE"],
        "target_issue_codes":["UNSUPPORTED_NUMERIC_VALUE"],
        "correction_action_type":action,
        "claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":sfp,"start":0,"end":len(original),"text":original},
        "target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":cfp,"start":17,"end":21,"text":"94 %"},
        "replacement_text":"95 %",
        "evidence_ids":["E1"],
        "authorized_evidence":[{"evidence_id":"E1","source_filename":"paper.pdf","chunk_id":"c1","authorized_for_section":True,"canonical_text":"El modelo obtuvo 95 %.","usage_role":"SUPPORT"}],
        "correction_validation_result":cv,
        "proposal_fingerprint":"b"*64,
        "proposed_claim_text_fingerprint":fp(proposed),
        "original_claim_fingerprint":cfp,
        "original_section_fingerprint":sfp,
        "base_claim_fingerprint":cfp,
        "base_section_fingerprint":sfp,
        "application_order_key":["S1",0,17,"a"*64],
        "attempt_context":{"attempt_number":1},
        "policy":get_verification_input_policy(),
    }
    return refresh_proposal_fingerprint(c)


def test_failed_execution_status_available():
    assert "FAILED" in REVERIFICATION_EXECUTION_STATUSES

@pytest.mark.parametrize("key",[
    "require_frozen_reverification_evidence",
    "require_same_risk_policy_version",
    "require_virtual_proposed_claim_reconstruction",
])
def test_policy_invariants_cannot_be_false(key):
    with pytest.raises(ValueError, match="must_be_true"):
        get_verification_input_policy({key:False})


def test_virtual_reconstruction_correct():
    c=base_context()
    assert build_virtual_corrected_claim(c["original_claim_text"],c["target_span_in_claim"],c["replacement_text"])==c["proposed_claim_text"]


def test_precheck_passed_and_zero_llm_no_application():
    r=run_virtual_reverification_prechecks(base_context())
    assert r["precheck_status"]=="PRECHECK_PASSED"
    assert r["llm_calls"]==0
    assert r["correction_applied"] is False


def mutate(path, value):
    c=base_context(); obj=c
    for key in path[:-1]: obj=obj[key]
    obj[path[-1]]=value
    return c


def test_proposed_text_mismatch_rejected():
    r=run_virtual_reverification_prechecks(mutate(["proposed_claim_text"],"otro"))
    assert r["precheck_status"]=="PRECHECK_REJECTED"
    assert "PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH" in r["reason_codes"]


def test_proposed_fingerprint_mismatch_rejected():
    r=run_virtual_reverification_prechecks(mutate(["proposed_claim_text_fingerprint"],"0"*64))
    assert "PROPOSED_CLAIM_TEXT_FINGERPRINT_MISMATCH" in r["reason_codes"]


def test_real_claim_fingerprint_mismatch_blocked():
    c=base_context(); c["original_claim_fingerprint"]="0"*64; c["base_claim_fingerprint"]="0"*64; c["target_span_in_claim"]["base_text_fingerprint"]="0"*64
    r=run_virtual_reverification_prechecks(c)
    assert r["precheck_status"]=="PRECHECK_BLOCKED"


def test_real_section_fingerprint_mismatch_blocked():
    c=base_context(); c["original_section_fingerprint"]="0"*64; c["base_section_fingerprint"]="0"*64; c["claim_span_in_section"]["base_text_fingerprint"]="0"*64
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_claim_span_out_of_range():
    c=base_context(); c["claim_span_in_section"]["end"]=999
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_target_span_out_of_range():
    c=base_context(); c["target_span_in_claim"]["end"]=999
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_target_span_text_mismatch():
    c=base_context(); c["target_span_in_claim"]["text"]="93 %"
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_claim_span_must_equal_original_claim():
    c=base_context(); c["section_text"]="X"+c["section_text"]; c["original_section_fingerprint"]=fp(c["section_text"]); c["base_section_fingerprint"]=fp(c["section_text"]); c["claim_span_in_section"]["base_text_fingerprint"]=fp(c["section_text"])
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_new_evidence_blocked():
    c=base_context(); c["evidence_ids"].append("E2")
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_corpus_wide_evidence_blocked():
    c=base_context(); c["authorized_evidence"][0]["outside_section_sources"]=True
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_incomplete_document_identity_blocked():
    c=base_context(); c["authorized_evidence"][0]["chunk_id"]=""
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_numeric_unit_incompatible_rejected():
    c=base_context(); c["authorized_evidence"][0]["canonical_text"]="El modelo obtuvo 95 ms."
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_unsupported_attribution_rejected():
    c=base_context("CORRECT_ATTRIBUTION"); c["replacement_text"]="Modelo de Pérez"; c["proposed_claim_text"]="El modelo obtuvo Modelo de Pérez."; c["proposed_claim_text_fingerprint"]=fp(c["proposed_claim_text"]); c["correction_validation_result"].update({"new_attribution_elements":["Pérez","modelo"],"attribution_relation":"PROPOSED_BY","attribution_subject":"Pérez","attribution_object":"modelo"})
    refresh_proposal_fingerprint(c)
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_stale_citation_rejected():
    c=base_context("REPLACE_CITATION"); c["correction_validation_result"].update({"citation_text_span":{"start":0,"end":3,"text":"bad"},"old_citation_refs":[{"source_filename":"old.pdf","chunk_id":"old1"}],"new_citation_refs":[{"source_filename":"paper.pdf","chunk_id":"c1"}]})
    refresh_proposal_fingerprint(c)
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_scope_expansion_rejected():
    c=base_context("NARROW_SCOPE"); c["correction_validation_result"]["new_conditions"]=["todo el mundo"]
    refresh_proposal_fingerprint(c)
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_invented_qualification_rejected():
    c=base_context("ADD_QUALIFICATION"); c["correction_validation_result"]["new_conditions"]=["en Marte"]
    refresh_proposal_fingerprint(c)
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_removal_that_changes_negation_rejected():
    c=base_context("REMOVE_UNSUPPORTED_FRAGMENT")
    original="El modelo no supera al baseline."; proposed="El modelo supera al baseline."
    c.update(original_claim_text=original,proposed_claim_text=proposed,section_text=original,replacement_text="")
    cfp=fp(original); c.update(original_claim_fingerprint=cfp,base_claim_fingerprint=cfp,original_section_fingerprint=cfp,base_section_fingerprint=cfp,proposed_claim_text_fingerprint=fp(proposed))
    c["claim_span_in_section"].update(base_text_fingerprint=cfp,start=0,end=len(original),text=original)
    start=original.index("no "); c["target_span_in_claim"].update(base_text_fingerprint=cfp,start=start,end=start+3,text="no ")
    c["application_order_key"]=["S1",0,start,"a"*64]
    refresh_proposal_fingerprint(c)
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_REJECTED"


def test_contractual_missing_section_text_blocked():
    c=base_context(); c.pop("section_text")
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"
