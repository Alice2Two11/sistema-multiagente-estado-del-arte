from __future__ import annotations
import json
from copy import deepcopy

from test_reverification_prechecks_phase62 import base_context
from src.tools.verification.validation import (
    run_virtual_reverification_prechecks,
    build_reverification_claim_context,
    run_independent_virtual_reverification,
    compute_reverification_context_fingerprint,
)

class Double:
    def __init__(self, responses): self.responses=list(responses); self.calls=0
    def invoke(self, messages): self.calls += 1; return self.responses.pop(0)

def setup():
    c=base_context(); p=run_virtual_reverification_prechecks(c); assert p["precheck_status"]=="PRECHECK_PASSED"; return c,p

def payload(c):
    return {
        "correction_id":c["correction_id"], "claim_id":c["claim_id"],
        "proposed_verdict":"SUPPORTED", "support_level":"STRONG",
        "evidence_ids_used":["E1"], "observed_issue_codes":[],
        "target_issues_resolved":["UNSUPPORTED_NUMERIC_VALUE"],
        "supported_meaning_preserved":True, "intended_semantic_change_valid":True,
        "unintended_semantic_change_absent":True,
        "scope_assessment":"NOT_APPLICABLE", "numeric_assessment":"VALID",
        "attribution_assessment":"NOT_APPLICABLE", "citation_assessment":"NOT_APPLICABLE",
        "manual_review_recommended":False, "reason_codes":["TARGET_ISSUE_APPEARS_RESOLVED"],
        "rationale":"La evidencia autorizada respalda el claim virtual.", "confidence":0.9,
    }

def blocked_after(mut):
    c,p=setup(); mut(c,p); d=Double([json.dumps(payload(c))])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="BLOCKED" and d.calls==0

def test_identity_mismatches_zero_calls():
    blocked_after(lambda c,p:p.__setitem__("correction_id","OTHER"))
    blocked_after(lambda c,p:p.__setitem__("claim_id","OTHER"))
    blocked_after(lambda c,p:p.__setitem__("section_id","OTHER"))

def test_evidence_snapshot_mutations_block():
    blocked_after(lambda c,p:c["authorized_evidence"][0].__setitem__("usage_role","CONTEXT"))
    blocked_after(lambda c,p:c["authorized_evidence"][0].__setitem__("canonical_text","changed"))

def test_policy_and_prompt_mutations_block():
    blocked_after(lambda c,p:c["policy"].__setitem__("max_reverification_llm_attempts",9))
    blocked_after(lambda c,p:c["policy"].__setitem__("reverification_user_prompt_version","OTHER"))

def test_unknown_reason_and_technical_observed_fail():
    for field,value in [("reason_codes",["UNKNOWN"]),("observed_issue_codes",["REVERIFICATION_LLM_INVOCATION_FAILED"])]:
        c,p=setup(); q=payload(c); q[field]=value; c["policy"]["max_reverification_llm_attempts"]=1
        # Re-freeze after changing policy limit for a valid context.
        p=run_virtual_reverification_prechecks(c)
        r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(q)]))
        assert r["reverification_execution_status"]=="FAILED"

def test_scientific_coherence_rules():
    cases=[]
    q=payload(base_context()); q["evidence_ids_used"]=[]; cases.append(q)
    q=payload(base_context()); q["support_level"]="NONE"; cases.append(q)
    q=payload(base_context()); q["observed_issue_codes"]=["UNSUPPORTED_NUMERIC_VALUE"]; cases.append(q)
    q=payload(base_context()); q["proposed_verdict"]="NOT_EVALUATED"; cases.append(q)
    for q in cases:
        c,p=setup(); c["policy"]["max_reverification_llm_attempts"]=1; p=run_virtual_reverification_prechecks(c)
        r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(q)]))
        assert r["reverification_execution_status"]=="FAILED"

def test_assessment_applicability():
    c,p=setup(); q=payload(c); q["numeric_assessment"]="NOT_APPLICABLE"
    c["policy"]["max_reverification_llm_attempts"]=1; p=run_virtual_reverification_prechecks(c)
    r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(q)]))
    assert r["reverification_execution_status"]=="FAILED"
    c,p=setup(); q=payload(c); q["scope_assessment"]="VALID"
    c["policy"]["max_reverification_llm_attempts"]=1; p=run_virtual_reverification_prechecks(c)
    r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(q)]))
    assert r["reverification_execution_status"]=="FAILED"

def test_evidence_order_normalized_and_no_acceptance_field():
    c=base_context(); c["authorized_evidence"].append({
        "evidence_id":"E2","source_filename":"paper.pdf","chunk_id":"c2",
        "authorized_for_section":True,"usage_role":"SUPPORT","canonical_text":"RMSE 1.34 m/s"
    }); c["evidence_ids"]=["E1","E2"]
    # proposal fingerprint depends on evidence order
    from src.tools.verification.validation import _recompute_phase5t_proposal_fingerprint
    c["proposal_fingerprint"]=_recompute_phase5t_proposal_fingerprint(c)
    p=run_virtual_reverification_prechecks(c); q=payload(c); q["evidence_ids_used"]=["E2","E1"]
    r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(q)]))
    assert r["evidence_ids_used"]==("E1","E2")
    assert "acceptance_decision" not in r

def test_context_fingerprint_deterministic():
    c,p=setup(); x1=build_reverification_claim_context(c,p); x2=build_reverification_claim_context(deepcopy(c),deepcopy(p))
    assert x1["reverification_context_fingerprint"]==x2["reverification_context_fingerprint"]
    assert len(x1["reverification_context_fingerprint"])==64

def test_result_preserves_snapshots():
    c,p=setup(); r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(payload(c))]))
    assert r["reverification_execution_status"]=="COMPLETED"
    assert r["frozen_evidence_snapshot_fingerprint"]==p["frozen_evidence_snapshot_fingerprint"]
    assert r["reverification_context_fingerprint"]==p["reverification_context_fingerprint"]
