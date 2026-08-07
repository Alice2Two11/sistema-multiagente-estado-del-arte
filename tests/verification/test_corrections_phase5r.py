from __future__ import annotations
import json, unittest
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.corrections import (
    fingerprint_text, locate_target, postproposal_batch_conflict_analysis,
    propose_correction, validate_correction_response,
)

class Double:
    def __init__(self, values=None, exc=None): self.values=list(values or []); self.exc=exc; self.calls=0
    def invoke(self, messages):
        self.calls+=1
        if self.exc: raise self.exc
        value=self.values[min(self.calls-1,len(self.values)-1)]
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)

def response(**kw):
    base=dict(claim_id="C1",correction_decision="PROPOSE_CHANGE",action_type="REPLACE_NUMERIC_VALUE",
        target_text="95 %",replacement_text="94 %",evidence_ids=["E01"],reason_codes=["LOCALIZED_NUMERIC_ERROR"],
        change_scope="TOKEN",semantic_change_level="MINIMAL",old_citation_refs=[],new_citation_refs=[],
        old_numeric_pairs=[["95","%"]],new_numeric_pairs=[["94","%"]],metric_context="accuracy",unit_context="%",
        old_attribution_elements=[],new_attribution_elements=[],attribution_relation=None,new_entities=[],new_attributions=[],
        new_conditions=[],new_technical_terms=[],citation_text_span=None,llm_correction_recommendation=True)
    base.update(kw); return base

def context(text="Accuracy was 95 %.", section=None):
    section=section or text
    claim_start=section.index(text)
    return dict(claim_id="C1",section_id="S1",original_claim_text=text,section_text=section,
        claim_fingerprint=fingerprint_text(text),section_fingerprint=fingerprint_text(section),
        claim_span_in_section={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(section),"start":claim_start,"end":claim_start+len(text),"text":text},
        scientific_verdict="PARTIALLY_SUPPORTED",deterministic_issue_codes=("UNSUPPORTED_NUMERIC_VALUE",),semantic_issue_codes=(),
        final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE",
        eligible_evidence=({"evidence_id":"E01","source_filename":"a.pdf","chunk_id":"c1","text":"Accuracy was 94 %.","authorized_for_section":True,"usage_role":"NUMERIC"},),
        existing_correction_proposals=(),policy=get_verification_input_policy())

class TestPhase5R(unittest.TestCase):
    def test_claim_fingerprint_missing(self):
        c=context(); c.pop("claim_fingerprint")
        with self.assertRaisesRegex(ValueError,"CLAIM_FINGERPRINT_REQUIRED"): propose_correction(c,llm=Double([response()]))
    def test_section_fingerprint_missing(self):
        c=context(); c.pop("section_fingerprint")
        with self.assertRaisesRegex(ValueError,"SECTION_FINGERPRINT_REQUIRED"): propose_correction(c,llm=Double([response()]))
    def test_section_fingerprint_wrong(self):
        c=context(); c["section_fingerprint"]="0"*64
        with self.assertRaisesRegex(ValueError,"SECTION_FINGERPRINT_MISMATCH"): propose_correction(c,llm=Double([response()]))
    def test_claim_span_separate_from_target(self):
        c=context("Accuracy was 95 %.","Intro. Accuracy was 95 %. End.")
        out=propose_correction(c,llm=Double([response()]))
        self.assertNotEqual(out.claim_span_in_section["start"],out.target_span_in_claim["start"])
    def test_same_local_offsets_different_claims_no_false_conflict(self):
        section="Alpha Beta"
        fp=fingerprint_text(section)
        def p(cid,ss,se): return {"correction_id":cid,"section_id":"S","claim_id":cid,
          "target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(section[ss:se]),"start":0,"end":2,"text":section[ss:ss+2]},
          "claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":ss,"end":se,"text":section[ss:se]}}
        self.assertEqual(postproposal_batch_conflict_analysis([p("A",0,5),p("B",6,10)]),())
    def test_real_cross_claim_conflict_after_section_conversion(self):
        fp=fingerprint_text("abcdef")
        a={"correction_id":"A","section_id":"S","claim_id":"A","target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("abcd"),"start":2,"end":4,"text":"cd"},"claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":0,"end":4,"text":"abcd"}}
        b={"correction_id":"B","section_id":"S","claim_id":"B","target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("cdef"),"start":0,"end":2,"text":"cd"},"claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":2,"end":6,"text":"cdef"}}
        self.assertEqual(postproposal_batch_conflict_analysis([a,b])[0]["conflict_type"],"SPAN_OVERLAP")
    def test_llm_exception(self):
        out=propose_correction(context(),llm=Double(exc=RuntimeError("x")))
        self.assertEqual(out.proposal_status,"DEFERRED"); self.assertIn("CORRECTION_LLM_INVOCATION_FAILED",out.validation_issue_codes)
    def test_format_retry_limit(self):
        c=context(); c["policy"]=get_verification_input_policy({"max_correction_llm_attempts":3,"max_correction_format_repair_attempts":1})
        d=Double(["not json","still bad",response()]); out=propose_correction(c,llm=d)
        self.assertEqual(d.calls,2); self.assertEqual(out.retry_metrics["format_retries"],2)
    def test_proposal_without_evidence(self):
        with self.assertRaisesRegex(ValueError,"AUTOMATIC_PROPOSAL_REQUIRES_EVIDENCE"):
            validate_correction_response(response(evidence_ids=[]),allowed_evidence_ids=("E01",))
    def test_empty_replacement_non_removal(self):
        with self.assertRaisesRegex(ValueError,"REPLACEMENT_TEXT_REQUIRED"):
            validate_correction_response(response(replacement_text=""),allowed_evidence_ids=("E01",))
    def test_numeric_action_requires_pairs(self):
        with self.assertRaisesRegex(ValueError,"NUMERIC_PAIRS_REQUIRED"):
            validate_correction_response(response(old_numeric_pairs=[],new_numeric_pairs=[]),allowed_evidence_ids=("E01",))
    def test_new_attribution_unsupported(self):
        c=context("Alpha was proposed by Doe."); c["eligible_evidence"]=(dict(c["eligible_evidence"][0],text="Alpha was reported by Smith."),)
        r=response(action_type="CORRECT_ATTRIBUTION",target_text="Doe",replacement_text="Jones",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",old_attribution_elements=["Doe"],new_attribution_elements=["Jones"],new_attributions=["Jones"],attribution_relation="PROPOSED_BY",reason_codes=["LOCALIZED_ATTRIBUTION_ERROR"])
        out=propose_correction(c,llm=Double([r])); self.assertIn("UNSUPPORTED_NEW_ATTRIBUTION",out.validation_issue_codes)
    def test_percent_unit_supported(self):
        out=propose_correction(context(),llm=Double([response()])); self.assertNotIn("UNSUPPORTED_NEW_NUMERIC_VALUE",out.validation_issue_codes)
    def test_non_alphanumeric_unit(self):
        c=context("Speed was 20 m/s."); c["eligible_evidence"]=(dict(c["eligible_evidence"][0],text="Speed was 18 m/s."),)
        r=response(target_text="20 m/s",replacement_text="18 m/s",old_numeric_pairs=[["20","m/s"]],new_numeric_pairs=[["18","m/s"]],metric_context="Speed",unit_context="m/s")
        out=propose_correction(c,llm=Double([r])); self.assertNotIn("UNSUPPORTED_NEW_NUMERIC_VALUE",out.validation_issue_codes)
    def test_metric_context_token_delimited(self):
        c=context(); c["eligible_evidence"]=(dict(c["eligible_evidence"][0],text="Inaccuracy was 94 %."),)
        out=propose_correction(c,llm=Double([response()])); self.assertIn("NUMERIC_CONTEXT_MISMATCH",out.validation_issue_codes)
    def test_normalized_match_in_long_claim(self):
        t="Inicio "*20+'El método  “A”  mejora.'+" Fin"*20
        span,method=locate_target(t,'El método "A" mejora.',claim_fingerprint=fingerprint_text(t))
        self.assertEqual(method,"NORMALIZED_UNIQUE_MATCH"); self.assertEqual(t[span.start:span.end],span.text)
    def test_prior_fingerprint_conflict(self):
        c=context(); c["existing_correction_proposals"]=({"claim_id":"C1","section_id":"S1","claim_span_in_section":c["claim_span_in_section"],"target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":"x"*64,"start":13,"end":17,"text":"95 %"}},)
        out=propose_correction(c,llm=Double([response()])); self.assertIn("STALE_PRIOR_CORRECTION_PROPOSAL",out.validation_issue_codes)

if __name__=='__main__': unittest.main()
