from __future__ import annotations

from hashlib import sha256

from src.tools.verification.corrections import compute_correction_proposal_fingerprint, fingerprint_text
from src.tools.verification.validation import run_virtual_reverification_prechecks
from test_reverification_prechecks_phase62 import base_context, refresh_proposal_fingerprint


def fp(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def rebuild(c, original, proposed, target, replacement):
    c["original_claim_text"] = original
    c["proposed_claim_text"] = proposed
    c["section_text"] = original
    c["replacement_text"] = replacement
    cfp = fp(original)
    c.update(original_claim_fingerprint=cfp, base_claim_fingerprint=cfp,
             original_section_fingerprint=cfp, base_section_fingerprint=cfp,
             proposed_claim_text_fingerprint=fp(proposed))
    c["claim_span_in_section"] = {"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":cfp,"start":0,"end":len(original),"text":original}
    start = original.index(target)
    c["target_span_in_claim"] = {"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":cfp,"start":start,"end":start+len(target),"text":target}
    c["application_order_key"] = [c["section_id"],0,start,c["correction_id"]]
    refresh_proposal_fingerprint(c)
    return c


def set_action(c, action, target_issue):
    c["correction_action_type"] = action
    c["source_issue_codes"] = [target_issue]
    c["target_issue_codes"] = [target_issue]
    refresh_proposal_fingerprint(c)
    return c


def test_modified_proposal_with_old_fingerprint_is_blocked():
    c=base_context()
    c["correction_validation_result"]["prompt_version"]="tampered-version"
    r=run_virtual_reverification_prechecks(c)
    assert r["precheck_status"]=="PRECHECK_BLOCKED"
    assert "PROPOSAL_FINGERPRINT_MISMATCH" in r["reason_codes"]


def test_fingerprints_valid_false_when_proposed_text_fp_wrong():
    c=base_context(); c["proposed_claim_text_fingerprint"]="0"*64
    r=run_virtual_reverification_prechecks(c)
    assert r["fingerprints_valid"] is False


def attribution_context(evidence_text, relation="PROPOSED_BY", obj="ANN"):
    c=base_context("CORRECT_ATTRIBUTION")
    set_action(c,"CORRECT_ATTRIBUTION","ATTRIBUTION_ERROR")
    original="El método fue atribuido incorrectamente."
    proposed="El método ANN fue propuesto por Pérez."
    rebuild(c,original,proposed,"fue atribuido incorrectamente","ANN fue propuesto por Pérez")
    c["authorized_evidence"][0]["canonical_text"]=evidence_text
    c["correction_validation_result"].update({"new_attribution_elements":["Pérez",obj],"new_attributions":["Pérez"],"attribution_relation":relation,"attribution_subject":"Pérez","attribution_object":obj})
    refresh_proposal_fingerprint(c)
    return c


def test_author_present_but_relation_wrong():
    r=run_virtual_reverification_prechecks(attribution_context("Pérez evaluó el método ANN."))
    assert "ATTRIBUTION_RELATION_NOT_SUPPORTED" in r["reason_codes"]


def test_attribution_object_wrong():
    r=run_virtual_reverification_prechecks(attribution_context("El método ANN fue propuesto por Pérez.",obj="CNN"))
    assert "ATTRIBUTION_OBJECT_NOT_SUPPORTED" in r["reason_codes"]


def test_narrow_scope_that_expands_scope():
    c=set_action(base_context("NARROW_SCOPE"),"NARROW_SCOPE","UNSUPPORTED_EXTRAPOLATION")
    rebuild(c,"El modelo funciona solo en Dataset A.","El modelo funciona en todos los datasets.","solo en Dataset A","en todos los datasets")
    c["correction_validation_result"].update({"new_conditions":["en todos los datasets"],"old_conditions":["solo en Dataset A"]})
    c["authorized_evidence"][0]["canonical_text"]="El modelo funciona en todos los datasets."
    refresh_proposal_fingerprint(c)
    r=run_virtual_reverification_prechecks(c)
    assert "SCOPE_EXPANSION_DETECTED" in r["reason_codes"]


def test_narrow_scope_removes_original_restriction():
    c=set_action(base_context("NARROW_SCOPE"),"NARROW_SCOPE","UNSUPPORTED_EXTRAPOLATION")
    rebuild(c,"El modelo funciona solo en Dataset A.","El modelo funciona bajo evaluación.","solo en Dataset A","bajo evaluación")
    c["correction_validation_result"].update({"new_conditions":["bajo evaluación"],"old_conditions":["solo en Dataset A"]})
    c["authorized_evidence"][0]["canonical_text"]="El modelo funciona bajo evaluación."
    refresh_proposal_fingerprint(c)
    r=run_virtual_reverification_prechecks(c)
    assert "ORIGINAL_SCOPE_CONDITION_REMOVED" in r["reason_codes"]


def test_generic_qualifier_substring_rejected():
    c=set_action(base_context("ADD_QUALIFICATION"),"ADD_QUALIFICATION","UNSUPPORTED_EXTRAPOLATION")
    c["correction_validation_result"]["new_conditions"]=["en"]
    refresh_proposal_fingerprint(c)
    assert "QUALIFICATION_DIFFERENTIAL_INVALID" in run_virtual_reverification_prechecks(c)["reason_codes"]


def test_qualification_increases_certainty():
    c=set_action(base_context("ADD_QUALIFICATION"),"ADD_QUALIFICATION","UNSUPPORTED_EXTRAPOLATION")
    rebuild(c,"El modelo puede mejorar.","El modelo definitivamente mejora.","puede mejorar","definitivamente mejora")
    c["correction_validation_result"]["new_conditions"]=["definitivamente"]
    c["authorized_evidence"][0]["canonical_text"]="El modelo definitivamente mejora."
    refresh_proposal_fingerprint(c)
    assert "QUALIFICATION_INCREASES_CERTAINTY" in run_virtual_reverification_prechecks(c)["reason_codes"]


def citation_context(evidence_text, include_new_marker=True):
    c=set_action(base_context("REPLACE_CITATION"),"REPLACE_CITATION","INVALID_CITATION")
    old="[old.pdf | old1]"; new="[paper.pdf | c1]" if include_new_marker else "[referencia]"
    original=f"El modelo mejora {old}."; proposed=f"El modelo mejora {new}."
    rebuild(c,original,proposed,old,new)
    st=original.index(old)
    c["correction_validation_result"].update({"citation_text_span":{"start":st,"end":st+len(old),"text":old},"old_citation_refs":[{"source_filename":"old.pdf","chunk_id":"old1"}],"new_citation_refs":[{"source_filename":"paper.pdf","chunk_id":"c1"}]})
    c["authorized_evidence"][0]["canonical_text"]=evidence_text
    refresh_proposal_fingerprint(c)
    return c


def test_new_citation_marker_missing():
    r=run_virtual_reverification_prechecks(citation_context("El modelo mejora.",False))
    assert "NEW_CITATION_MARKER_MISSING" in r["reason_codes"]


def test_authorized_citation_not_supporting_claim():
    r=run_virtual_reverification_prechecks(citation_context("La muestra contiene datos meteorológicos."))
    assert "NEW_CITATION_DOES_NOT_SUPPORT_PROPOSED_CLAIM" in r["reason_codes"]


def test_removal_of_temporal_operator_rejected():
    c=set_action(base_context("REMOVE_UNSUPPORTED_FRAGMENT"),"REMOVE_UNSUPPORTED_FRAGMENT","CLAIM_EVIDENCE_CONFLICT")
    rebuild(c,"El modelo actualmente supera al baseline.","El modelo supera al baseline.","actualmente ","")
    c["correction_validation_result"]["unsupported_fragment"]="actualmente "
    refresh_proposal_fingerprint(c)
    assert "REMOVAL_ALTERS_SUPPORTED_MEANING" in run_virtual_reverification_prechecks(c)["reason_codes"]


def test_numeric_action_with_citation_target_issue_rejected():
    c=base_context(); c["source_issue_codes"]=["INVALID_CITATION"]; c["target_issue_codes"]=["INVALID_CITATION"]
    refresh_proposal_fingerprint(c)
    assert "ACTION_TARGET_ISSUE_MISMATCH" in run_virtual_reverification_prechecks(c)["reason_codes"]


def test_different_evidence_order_blocked_after_valid_proposal_hash():
    c=base_context()
    c["evidence_ids"]=["E2","E1"]
    c["authorized_evidence"]=[
        {"evidence_id":"E1","source_filename":"p1.pdf","chunk_id":"c1","authorized_for_section":True,"canonical_text":"95 %","usage_role":"SUPPORT"},
        {"evidence_id":"E2","source_filename":"p2.pdf","chunk_id":"c2","authorized_for_section":True,"canonical_text":"95 %","usage_role":"SUPPORT"},
    ]
    refresh_proposal_fingerprint(c)
    r=run_virtual_reverification_prechecks(c)
    assert r["precheck_status"]=="PRECHECK_BLOCKED"
    assert "REVERIFICATION_EVIDENCE_ORDER_MISMATCH" in r["reason_codes"]


def test_fingerprint_mismatch_is_blocked():
    c=base_context(); c["proposal_fingerprint"]="0"*64
    assert run_virtual_reverification_prechecks(c)["precheck_status"]=="PRECHECK_BLOCKED"


def test_precheck_result_preserves_fingerprints():
    c=base_context(); r=run_virtual_reverification_prechecks(c)
    assert r["proposal_fingerprint"]==c["proposal_fingerprint"]
    assert r["base_claim_fingerprint"]==c["base_claim_fingerprint"]
    assert r["base_section_fingerprint"]==c["base_section_fingerprint"]
    assert r["virtual_proposed_claim_text_fingerprint"]==c["proposed_claim_text_fingerprint"]
