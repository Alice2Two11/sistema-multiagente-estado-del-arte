from __future__ import annotations
import json
import unittest

from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.corrections import fingerprint_text, propose_correction


class Double:
    def __init__(self, value):
        self.value = value
        self.calls = 0
    def invoke(self, messages):
        self.calls += 1
        return self.value if isinstance(self.value, str) else json.dumps(self.value, ensure_ascii=False)


def response(**kw):
    base = dict(
        claim_id="C1", correction_decision="PROPOSE_CHANGE", action_type="REPLACE_NUMERIC_VALUE",
        target_text="95 %", replacement_text="94 %", evidence_ids=["E01"],
        reason_codes=["LOCALIZED_NUMERIC_ERROR"], change_scope="TOKEN",
        semantic_change_level="MINIMAL", old_citation_refs=[], new_citation_refs=[],
        old_numeric_pairs=[["95", "%"]], new_numeric_pairs=[["94", "%"]],
        metric_context="accuracy", unit_context="%", old_attribution_elements=[],
        new_attribution_elements=[], attribution_relation=None, new_entities=[],
        new_attributions=[], new_conditions=[], new_technical_terms=[],
        citation_text_span=None, llm_correction_recommendation=True,
    )
    base.update(kw)
    return base


def context(text="Accuracy was 95 %.", section=None):
    section = section or text
    start = section.index(text)
    return dict(
        claim_id="C1", section_id="S1", original_claim_text=text, section_text=section,
        claim_fingerprint=fingerprint_text(text), section_fingerprint=fingerprint_text(section),
        claim_span_in_section={
            "coordinate_base":"SECTION_TEXT", "coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint":fingerprint_text(section), "start":start,
            "end":start+len(text), "text":text,
        },
        final_correction_eligibility="POTENTIALLY_AUTO_CORRECTABLE",
        scientific_verdict="PARTIALLY_SUPPORTED",
        deterministic_issue_codes=("UNSUPPORTED_NUMERIC_VALUE",), semantic_issue_codes=(),
        eligible_evidence=({
            "evidence_id":"E01", "source_filename":"a.pdf", "chunk_id":"c1",
            "canonical_text":"Accuracy was 94 %. Clear-sky conditions were evaluated.",
            "authorized_for_section":True, "usage_role":"NUMERIC",
        },),
        existing_correction_proposals=(), policy=get_verification_input_policy(),
    )


def prior(c, *, correction_id, status="ACCEPTED_FOR_REVERIFICATION", section_id="S1", claim_fp=None,
          claim_start=None, claim_end=None, target_start=13, target_end=17):
    claim_text = c["original_claim_text"]
    section_text = c["section_text"]
    cs = c["claim_span_in_section"]
    claim_start = cs["start"] if claim_start is None else claim_start
    claim_end = cs["end"] if claim_end is None else claim_end
    claim_slice = section_text[claim_start:claim_end] if 0 <= claim_start < claim_end <= len(section_text) else claim_text
    fp = claim_fp or fingerprint_text(claim_slice)
    target_text = claim_slice[target_start:target_end] if target_end <= len(claim_slice) else "x"
    return {
        "correction_id": correction_id, "claim_id": "C1", "section_id": section_id,
        "proposal_status": status,
        "claim_span_in_section": {
            "coordinate_base":"SECTION_TEXT", "coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint":c["section_fingerprint"], "start":claim_start,
            "end":claim_end, "text":claim_slice,
        },
        "target_span_in_claim": {
            "coordinate_base":"CLAIM_TEXT", "coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint":fp, "start":target_start, "end":target_end,
            "text":target_text,
        },
    }


class TestPhase5T(unittest.TestCase):
    def test_limit_reached_without_llm_call(self):
        c=context(); c["policy"]=get_verification_input_policy({"max_correction_proposals_per_claim":2})
        c["existing_correction_proposals"]=(prior(c,correction_id="P1"),prior(c,correction_id="P2"))
        d=Double(response()); out=propose_correction(c,llm=d)
        self.assertEqual(d.calls,0); self.assertEqual(out.proposal_status,"DEFERRED")
        self.assertIn("CORRECTION_PROPOSAL_LIMIT_REACHED",out.validation_issue_codes)

    def test_rejected_proposals_do_not_consume_limit(self):
        c=context(); c["policy"]=get_verification_input_policy({"max_correction_proposals_per_claim":2})
        c["existing_correction_proposals"]=(prior(c,correction_id="P1",status="REJECTED"),prior(c,correction_id="P2",status="REJECTED"))
        d=Double(response()); out=propose_correction(c,llm=d)
        self.assertEqual(d.calls,1); self.assertTrue(out.accepted_for_reverification)

    def test_narrow_scope_authorized_condition(self):
        text="The model performs well."; c=context(text)
        r=response(action_type="NARROW_SCOPE",target_text="performs well",replacement_text="performs well under clear-sky conditions",
                   old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",
                   new_conditions=["clear-sky conditions"],reason_codes=["LOCALIZED_EXTRAPOLATION"])
        out=propose_correction(c,llm=Double(r))
        self.assertTrue(out.accepted_for_reverification)

    def test_narrow_scope_requires_condition(self):
        text="The model performs well."; c=context(text)
        r=response(action_type="NARROW_SCOPE",target_text="performs well",replacement_text="performs well in context",
                   old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",
                   new_conditions=[],reason_codes=["LOCALIZED_EXTRAPOLATION"])
        out=propose_correction(c,llm=Double(r))
        self.assertIn("NARROW_SCOPE_CONDITIONS_REQUIRED",str(out.raw_attempts))

    def test_narrow_scope_invented_condition(self):
        text="The model performs well."; c=context(text)
        r=response(action_type="NARROW_SCOPE",target_text="performs well",replacement_text="performs well at night",
                   old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",
                   new_conditions=["at night"],reason_codes=["LOCALIZED_EXTRAPOLATION"])
        out=propose_correction(c,llm=Double(r))
        self.assertIn("UNSUPPORTED_NEW_INFORMATION",out.validation_issue_codes)

    def test_bracket_span_not_linked_to_reference(self):
        text="Result [experimental finding]."; c=context(text)
        st=text.index("["); en=text.index("]")+1
        span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
              "base_text_fingerprint":fingerprint_text(text),"start":st,"end":en,"text":text[st:en]}
        r=response(action_type="REPLACE_CITATION",target_text=text[st:en],replacement_text="[a | c1]",
                   old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",
                   old_citation_refs=[{"source_filename":"smith.pdf","chunk_id":"old1"}],
                   new_citation_refs=[{"source_filename":"a.pdf","chunk_id":"c1"}],
                   citation_text_span=span,reason_codes=["INVALID_CITATION_WITH_VALID_REPLACEMENT"])
        out=propose_correction(c,llm=Double(r))
        self.assertIn("CITATION_TEXT_REFERENCE_MISMATCH",out.validation_issue_codes)

    def test_contractual_citation_marker_valid(self):
        text="Result [Smith, 2020]."; c=context(text)
        st=text.index("["); en=text.index("]")+1
        span={"coordinate_base":"CLAIM_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
              "base_text_fingerprint":fingerprint_text(text),"start":st,"end":en,"text":text[st:en]}
        r=response(action_type="REPLACE_CITATION",target_text=text[st:en],replacement_text="[a | c1]",
                   old_numeric_pairs=[],new_numeric_pairs=[],metric_context="",unit_context="",
                   old_citation_refs=[{"source_filename":"smith.pdf","chunk_id":"old1"}],
                   new_citation_refs=[{"source_filename":"a.pdf","chunk_id":"c1"}],
                   citation_text_span=span,reason_codes=["INVALID_CITATION_WITH_VALID_REPLACEMENT"])
        out=propose_correction(c,llm=Double(r))
        self.assertNotIn("CITATION_TEXT_REFERENCE_MISMATCH",out.validation_issue_codes)

    def test_prior_out_of_range_ignored(self):
        c=context(); bad=prior(c,correction_id="P",claim_start=0,claim_end=999)
        c["existing_correction_proposals"]=(bad,)
        out=propose_correction(c,llm=Double(response()))
        self.assertIn("MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED",out.validation_issue_codes)

    def test_prior_other_section_ignored(self):
        c=context(); c["existing_correction_proposals"]=(prior(c,correction_id="P",section_id="OTHER"),)
        out=propose_correction(c,llm=Double(response()))
        self.assertIn("PRIOR_CORRECTION_SECTION_MISMATCH",out.validation_issue_codes)

    def test_prior_stale_fingerprint_ignored(self):
        c=context(); c["existing_correction_proposals"]=(prior(c,correction_id="P",claim_fp="x"*64),)
        out=propose_correction(c,llm=Double(response()))
        self.assertIn("STALE_PRIOR_CORRECTION_PROPOSAL",out.validation_issue_codes)

    def test_proposal_count_deterministic(self):
        c1=context(); rows=(prior(c1,correction_id="B"),prior(c1,correction_id="A"))
        c1["existing_correction_proposals"]=rows
        c2=context(); c2["existing_correction_proposals"]=tuple(reversed(rows))
        a=propose_correction(c1,llm=Double(response())); b=propose_correction(c2,llm=Double(response()))
        self.assertEqual(a.to_dict(),b.to_dict())


if __name__ == "__main__":
    unittest.main()
