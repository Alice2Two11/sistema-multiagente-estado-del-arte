from __future__ import annotations
import json, unittest
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.corrections import fingerprint_text, propose_correction, validate_correction_response

class Double:
    def __init__(self, values): self.values=list(values); self.calls=0
    def invoke(self, messages):
        self.calls += 1
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

def context():
    text="Accuracy was 95 %."; section="Intro. "+text+" End."; start=section.index(text)
    return dict(claim_id="C1",section_id="S1",original_claim_text=text,section_text=section,
        claim_fingerprint=fingerprint_text(text),section_fingerprint=fingerprint_text(section),
        claim_span_in_section={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(section),"start":start,"end":start+len(text),"text":text},
        final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE", scientific_verdict="PARTIALLY_SUPPORTED",
        deterministic_issue_codes=("UNSUPPORTED_NUMERIC_VALUE",),semantic_issue_codes=(),
        eligible_evidence=({"evidence_id":"E01","source_filename":"a.pdf","chunk_id":"c1","canonical_text":"Accuracy was 94 %.","contractual_text":"Accuracy was 93 %.","text":"Accuracy was 92 %.","authorized_for_section":True,"usage_role":"NUMERIC"},),
        existing_correction_proposals=(),policy=get_verification_input_policy())

class TestPhase5S(unittest.TestCase):
    def test_claim_id_mismatch(self):
        with self.assertRaisesRegex(ValueError,"CORRECTION_RESPONSE_CLAIM_ID_MISMATCH"):
            validate_correction_response(response(claim_id="OTHER"),allowed_evidence_ids=("E01",),expected_claim_id="C1")
    def test_section_text_absent(self):
        c=context(); c.pop("section_text")
        out=propose_correction(c,llm=Double([response()])); self.assertEqual(out.proposal_status,"DEFERRED"); self.assertIn("CLAIM_SPAN_IN_SECTION_REQUIRED",out.validation_issue_codes)
    def test_claim_span_absent(self):
        c=context(); c.pop("claim_span_in_section")
        out=propose_correction(c,llm=Double([response()])); self.assertEqual(out.proposal_status,"DEFERRED"); self.assertIn("CLAIM_SPAN_IN_SECTION_REQUIRED",out.validation_issue_codes)
    def test_canonical_priority(self):
        out=propose_correction(context(),llm=Double([response()])); self.assertTrue(out.accepted_for_reverification)
    def test_numeric_action_cannot_change_citation(self):
        with self.assertRaisesRegex(ValueError,"CORRECTION_ACTION_FIELD_MATRIX_VIOLATION"):
            validate_correction_response(response(new_citation_refs=[{"source_filename":"a.pdf","chunk_id":"c1"}]),allowed_evidence_ids=("E01",),expected_claim_id="C1")
    def test_attribution_action_cannot_change_number(self):
        r=response(action_type="CORRECT_ATTRIBUTION",target_text="A",replacement_text="B",old_attribution_elements=["A"],new_attribution_elements=["B"],new_attributions=["B"],attribution_relation="REPORTED_BY")
        with self.assertRaisesRegex(ValueError,"CORRECTION_ACTION_FIELD_MATRIX_VIOLATION"):
            validate_correction_response(r,allowed_evidence_ids=("E01",),expected_claim_id="C1")
    def test_false_recommendation_with_propose(self):
        with self.assertRaisesRegex(ValueError,"CORRECTION_RECOMMENDATION_CONTRADICTION"):
            validate_correction_response(response(llm_correction_recommendation=False),allowed_evidence_ids=("E01",),expected_claim_id="C1")
    def test_citation_span_not_old_citation(self):
        text="Method improved results [Smith, 2020]."; c=context(); c["original_claim_text"]=text; c["claim_fingerprint"]=fingerprint_text(text); c["section_text"]=text; c["section_fingerprint"]=fingerprint_text(text); c["claim_span_in_section"]={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(text),"start":0,"end":len(text),"text":text}
        bad={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(text),"start":0,"end":6,"text":"Method"}
        r=response(action_type="REPLACE_CITATION",target_text="[Smith, 2020]",replacement_text="[Jones, 2021]",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",old_citation_refs=[{"source_filename":"smith.pdf","chunk_id":"c1"}],new_citation_refs=[{"source_filename":"a.pdf","chunk_id":"c1"}],citation_text_span=bad,reason_codes=["INVALID_CITATION_WITH_VALID_REPLACEMENT"])
        out=propose_correction(c,llm=Double([r])); self.assertIn("CITATION_TEXT_REFERENCE_MISMATCH",out.validation_issue_codes)
    def test_attribution_substring_not_supported(self):
        c=context(); c["original_claim_text"]="Attributed to Ann."; c["claim_fingerprint"]=fingerprint_text(c["original_claim_text"]); c["section_text"]=c["original_claim_text"]; c["section_fingerprint"]=fingerprint_text(c["section_text"]); c["claim_span_in_section"]={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":c["section_fingerprint"],"start":0,"end":len(c["section_text"]),"text":c["section_text"]}; c["eligible_evidence"]=(dict(c["eligible_evidence"][0],canonical_text="Reported by Joann."),)
        r=response(action_type="CORRECT_ATTRIBUTION",target_text="Ann",replacement_text="Ann",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",old_attribution_elements=["Ann"],new_attribution_elements=["Ann"],new_attributions=["Ann"],attribution_relation="REPORTED_BY",reason_codes=["LOCALIZED_ATTRIBUTION_ERROR"])
        out=propose_correction(c,llm=Double([r])); self.assertIn("UNSUPPORTED_NEW_ATTRIBUTION",out.validation_issue_codes)
    def test_exact_format_retry_calls(self):
        c=context(); c["policy"]=get_verification_input_policy({"max_correction_llm_attempts":5,"max_correction_format_repair_attempts":2})
        d=Double(["bad","bad again","still bad",response()]); out=propose_correction(c,llm=d)
        self.assertEqual(d.calls,3); self.assertEqual(out.retry_metrics["format_retries"],3)
    def test_authorized_evidence_unavailable(self):
        c=context(); c["eligible_evidence"]=()
        out=propose_correction(c,llm=Double([response()])); self.assertIn("AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE",out.validation_issue_codes)
    def test_llm_unavailable(self):
        out=propose_correction(context(),llm=None); self.assertIn("CORRECTION_LLM_UNAVAILABLE",out.validation_issue_codes)
    def test_malformed_prior_ignored_but_audited(self):
        c=context(); c["existing_correction_proposals"]=({"claim_id":"C1"},)
        out=propose_correction(c,llm=Double([response()])); self.assertTrue(out.accepted_for_reverification); self.assertIn("MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED",out.validation_issue_codes)

if __name__=='__main__': unittest.main()
