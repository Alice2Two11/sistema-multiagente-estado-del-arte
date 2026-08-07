from __future__ import annotations
import json, unittest
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.corrections import (
    apply_localized_change, fingerprint_text, locate_target, postproposal_batch_conflict_analysis,
    propose_correction, validate_correction_response, validate_text_integrity,
)

class Double:
    def __init__(self, value): self.value=value; self.calls=0
    def invoke(self, messages): self.calls+=1; return json.dumps(self.value, ensure_ascii=False)

def response(**kw):
    base=dict(claim_id="C1",correction_decision="PROPOSE_CHANGE",action_type="REPLACE_NUMERIC_VALUE",
        target_text="95 %",replacement_text="94 %",evidence_ids=["E01"],reason_codes=["LOCALIZED_NUMERIC_ERROR"],
        change_scope="TOKEN",semantic_change_level="MINIMAL",old_citation_refs=[],new_citation_refs=[],
        old_numeric_pairs=[["95","%"]],new_numeric_pairs=[["94","%"]],metric_context="accuracy",unit_context="%",
        old_attribution_elements=[],new_attribution_elements=[],attribution_relation=None,new_entities=[],new_attributions=[],
        new_conditions=[],new_technical_terms=[],citation_text_span=None,llm_correction_recommendation=True)
    base.update(kw); return base

def context(text="Accuracy was 95 %."):
    return dict(claim_id="C1",section_id="S1",original_claim_text=text,section_text=text,
        claim_fingerprint=fingerprint_text(text),section_fingerprint=fingerprint_text(text),
        claim_span_in_section={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text(text),"start":0,"end":len(text),"text":text},scientific_verdict="PARTIALLY_SUPPORTED",
        deterministic_issue_codes=("UNSUPPORTED_NUMERIC_VALUE",),semantic_issue_codes=(),final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE",
        eligible_evidence=({"evidence_id":"E01","source_filename":"a.pdf","chunk_id":"c1","text":"Accuracy was 94 %.","authorized_for_section":True,"usage_role":"NUMERIC"},),
        existing_correction_proposals=(),policy=get_verification_input_policy())

class TestPhase5(unittest.TestCase):
    def test_decision_manual_review_separate_from_action(self):
        value=response(correction_decision="DEFER_TO_MANUAL_REVIEW",action_type=None,llm_correction_recommendation=False)
        self.assertEqual(validate_correction_response(value,allowed_evidence_ids=("E01",))["correction_decision"],"DEFER_TO_MANUAL_REVIEW")
    def test_span_wrong_base(self):
        t="abc"; fp=fingerprint_text(t)
        with self.assertRaisesRegex(ValueError,"SPAN_COORDINATE_BASE_INVALID"):
            locate_target(t,"a",claim_fingerprint=fp,explicit_span={"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":0,"end":1,"text":"a"})
    def test_normalized_offsets_map_to_original(self):
        t='El método  “A”  mejora.'; fp=fingerprint_text(t)
        span, method=locate_target(t,'El método "A" mejora.',claim_fingerprint=fp)
        self.assertEqual(t[span.start:span.end],span.text); self.assertEqual(method,"NORMALIZED_UNIQUE_MATCH")
    def test_removal_changes_negation(self):
        c=context("El método no mejora."); c["eligible_evidence"]=(dict(c["eligible_evidence"][0],text="El método no mejora."),)
        out=propose_correction(c,llm=Double(response(action_type="REMOVE_UNSUPPORTED_FRAGMENT",target_text="no ",replacement_text="",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",citation_text_span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("Accuracy was 95 %."),"start":13,"end":17,"text":"95 %"})))
        self.assertIn("REMOVAL_ALTERS_SUPPORTED_MEANING",out.validation_issue_codes)
    def test_supported_qualification(self):
        c=context("El método mejora."); c["eligible_evidence"]=(dict(c["eligible_evidence"][0],text="El método mejora en Dataset X."),)
        out=propose_correction(c,llm=Double(response(action_type="ADD_QUALIFICATION",target_text="mejora",replacement_text="mejora en Dataset X",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",new_conditions=["Dataset X"],reason_codes=["LOCALIZED_EXTRAPOLATION"])))
        self.assertNotIn("UNSUPPORTED_NEW_INFORMATION",out.validation_issue_codes)
    def test_invented_qualification(self):
        c=context("El método mejora.")
        out=propose_correction(c,llm=Double(response(action_type="ADD_QUALIFICATION",target_text="mejora",replacement_text="mejora en Dataset Z",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",new_conditions=["Dataset Z"],reason_codes=["LOCALIZED_EXTRAPOLATION"])))
        self.assertIn("UNSUPPORTED_NEW_INFORMATION",out.validation_issue_codes)
    def test_new_entity_unsupported(self):
        with self.assertRaisesRegex(ValueError,"CORRECTION_ACTION_FIELD_MATRIX_VIOLATION"):
            validate_correction_response(response(new_entities=["ModelZ"]),allowed_evidence_ids=("E01",),expected_claim_id="C1")
    def test_structured_authorized_citation(self):
        v=response(action_type="REPLACE_CITATION",target_text="95 %",replacement_text="94 % [a]",old_citation_refs=[{"source_filename":"old.pdf","chunk_id":"old"}],new_citation_refs=[{"source_filename":"a.pdf","chunk_id":"c1"}],old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",citation_text_span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("Accuracy was 95 %."),"start":13,"end":17,"text":"95 %"})
        out=propose_correction(context(),llm=Double(v)); self.assertNotIn("UNAUTHORIZED_NEW_CITATION",out.validation_issue_codes)
    def test_structured_unauthorized_citation(self):
        v=response(action_type="REPLACE_CITATION",target_text="95 %",replacement_text="94 % [b]",old_citation_refs=[{"source_filename":"old.pdf","chunk_id":"old"}],new_citation_refs=[{"source_filename":"b.pdf","chunk_id":"x"}],old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",citation_text_span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fingerprint_text("Accuracy was 95 %."),"start":13,"end":17,"text":"95 %"})
        out=propose_correction(context(),llm=Double(v)); self.assertIn("UNAUTHORIZED_NEW_CITATION",out.validation_issue_codes)
    def test_numeric_metric_context_mismatch(self):
        v=response(metric_context="F1")
        out=propose_correction(context(),llm=Double(v)); self.assertIn("NUMERIC_CONTEXT_MISMATCH",out.validation_issue_codes)
    def test_wrong_attribution_relation(self):
        with self.assertRaisesRegex(ValueError,"ATTRIBUTION_RELATION_INVALID"):
            validate_correction_response(response(action_type="CORRECT_ATTRIBUTION",attribution_relation="OWNS"),allowed_evidence_ids=("E01",))
    def test_split_claim_deferred(self):
        v=response(action_type="SPLIT_CLAIM",target_text="95 %",replacement_text="",old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="")
        out=propose_correction(context(),llm=Double(v)); self.assertEqual(out.proposal_status,"DEFERRED")
    def test_pre_conflict(self):
        c=context(); c["existing_correction_proposals"]=({"claim_id":"C1","section_id":"S1","claim_span_in_section":c["claim_span_in_section"],"target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":c["claim_fingerprint"],"start":10,"end":20,"text":c["original_claim_text"][10:20]}},)
        out=propose_correction(c,llm=Double(response())); self.assertIn("MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED",out.validation_issue_codes)
    def test_post_conflict(self):
        fp=fingerprint_text("abcdefghij"); p=[{"correction_id":"A","section_id":"S","claim_id":"C1","target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":0,"end":5,"text":"abcde"},"claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":0,"end":5,"text":"abcde"}}, {"correction_id":"B","section_id":"S","claim_id":"C2","target_span_in_claim":{"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":0,"end":4,"text":"efgh"},"claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":fp,"start":4,"end":8,"text":"efgh"}}]
        self.assertEqual(postproposal_batch_conflict_analysis(p)[0]["conflict_type"],"SPAN_OVERLAP")
    def test_proposed_claim_deterministic(self):
        out=propose_correction(context(),llm=Double(response()))
        self.assertEqual(out.proposed_claim_text,"Accuracy was 94 %."); self.assertFalse(out.correction_applied)
    def test_fingerprint_differs_by_prompt_version(self):
        c1=context(); c2=context(); c2["policy"]=get_verification_input_policy({"correction_user_prompt_version":"AGENT07_CORRECTION_USER_V2"})
        self.assertNotEqual(propose_correction(c1,llm=Double(response())).proposal_fingerprint,propose_correction(c2,llm=Double(response())).proposal_fingerprint)
    def test_punctuation_broken(self): self.assertIn("PUNCTUATION_INTEGRITY_INVALID",validate_text_integrity("Texto,."))
    def test_unbalanced_brackets(self): self.assertIn("BRACKET_BALANCE_INVALID",validate_text_integrity("Texto (x"))
    def test_invalid_spaces(self): self.assertIn("WHITESPACE_INTEGRITY_INVALID",validate_text_integrity("Texto  doble"))
    def test_not_proposed(self):
        c=context(); c["final_correction_eligibility"]="NO_CORRECTION_NEEDED"
        out=propose_correction(c,llm=Double(response())); self.assertEqual(out.proposal_status,"NOT_PROPOSED")

if __name__=='__main__': unittest.main()
