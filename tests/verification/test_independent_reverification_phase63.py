from __future__ import annotations
import json
from copy import deepcopy

from test_reverification_prechecks_phase62 import base_context
from src.tools.verification.validation import (
    run_virtual_reverification_prechecks,
    build_reverification_claim_context,
    run_independent_virtual_reverification,
)

class Double:
    def __init__(self, responses=None, error=None):
        self.responses=list(responses or []); self.error=error; self.calls=0; self.messages=[]
    def invoke(self, messages):
        self.calls += 1; self.messages.append(messages)
        if self.error: raise self.error
        return self.responses.pop(0)

def valid_payload(c):
    return {
        "correction_id":c["correction_id"], "claim_id":c["claim_id"],
        "proposed_verdict":"SUPPORTED", "support_level":"STRONG",
        "evidence_ids_used":["E1"], "observed_issue_codes":[],
        "target_issues_resolved":["UNSUPPORTED_NUMERIC_VALUE"],
        "supported_meaning_preserved":True, "intended_semantic_change_valid":True,
        "unintended_semantic_change_absent":True, "scope_assessment":"NOT_APPLICABLE",
        "numeric_assessment":"VALID", "attribution_assessment":"NOT_APPLICABLE",
        "citation_assessment":"NOT_APPLICABLE", "manual_review_recommended":False,
        "reason_codes":[], "rationale":"La evidencia autorizada respalda el claim propuesto.",
        "confidence":0.9,
    }

def setup():
    c=base_context(); p=run_virtual_reverification_prechecks(c); return c,p

def test_blocked_precheck_zero_calls():
    c,p=setup(); p["precheck_status"]="PRECHECK_BLOCKED"; d=Double(["{}"])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="BLOCKED" and d.calls==0

def test_context_exact_and_retrieval_zero():
    c,p=setup(); x=build_reverification_claim_context(c,p)
    assert x["claim_text"]==p["virtual_proposed_claim_text"]
    assert x["allowed_evidence_ids"]==tuple(c["evidence_ids"])
    assert x["retrieval_allowed"] is False and x["retrieval_rounds"]==0

def test_valid_json_completed_and_no_acceptance():
    c,p=setup(); d=Double([json.dumps(valid_payload(c))])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="COMPLETED"
    assert "acceptance_decision" not in r and r["correction_applied"] is False
    assert r["reverification_llm_calls"]==1
    assert len(r["raw_attempts"][0]["raw_output_hash"])==64

def test_invalid_json_then_valid_counts_format_retry():
    c,p=setup(); d=Double(["not-json",json.dumps(valid_payload(c))])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="COMPLETED"
    assert r["format_attempts"]==2 and r["format_retries"]==1
    assert r["schema_retries"]==0

def test_schema_invalid_then_valid_counts_schema_retry():
    c,p=setup(); bad=valid_payload(c); bad.pop("rationale")
    d=Double([json.dumps(bad),json.dumps(valid_payload(c))])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="COMPLETED"
    assert r["schema_attempts"]==2 and r["schema_retries"]==1

def test_ids_evidence_issue_confidence_fail_terminal():
    mutators=[
        lambda x:x.update(correction_id="bad"), lambda x:x.update(claim_id="bad"),
        lambda x:x.update(evidence_ids_used=["NEW"]), lambda x:x.update(observed_issue_codes=["UNKNOWN_X"]),
        lambda x:x.update(confidence=2.0),
    ]
    for mutate in mutators:
        c,p=setup(); payload=valid_payload(c); mutate(payload)
        c["policy"]["max_reverification_llm_attempts"]=1
        p=run_virtual_reverification_prechecks(c)
        r=run_independent_virtual_reverification(c,p,reverification_llm=Double([json.dumps(payload)]))
        assert r["reverification_execution_status"]=="FAILED"

def test_exception_failed_without_message_leak():
    c,p=setup(); r=run_independent_virtual_reverification(c,p,reverification_llm=Double(error=RuntimeError("secret")))
    assert r["reverification_execution_status"]=="FAILED"
    assert r["decision_trace"][0]["exception_type"]=="RuntimeError"
    assert "secret" not in json.dumps(r)

def test_exact_format_repair_limit():
    c,p=setup(); c["policy"]["max_reverification_llm_attempts"]=5; c["policy"]["max_reverification_format_repair_attempts"]=2
    p=run_virtual_reverification_prechecks(c)
    d=Double(["bad","bad","bad","bad"])
    r=run_independent_virtual_reverification(c,p,reverification_llm=d)
    assert r["reverification_execution_status"]=="FAILED"
    assert d.calls==3 and r["format_retries"]==2

def test_deterministic_double():
    c,p=setup(); raw=json.dumps(valid_payload(c),sort_keys=True)
    r1=run_independent_virtual_reverification(c,p,reverification_llm=Double([raw]))
    r2=run_independent_virtual_reverification(c,p,reverification_llm=Double([raw]))
    assert r1==r2
