from copy import deepcopy
from types import SimpleNamespace
import pytest

from test_multi_proposal_resolution_phase66 import bundle as real_bundle
from src.tools.verification.resolution import resolve_multiple_correction_proposals

from src.adapters.verification_runtime import (
    Agent07RuntimeInput,
    VerificationRuntimeDependencies,
    run_agent07_in_memory,
)
from src.tools.verification import resolution as rm
from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.resolution import validate_provisional_multi_proposal_resolution_result
from tests.verification.test_multi_proposal_resolution_phase66 import bundle, corr


def resolved(*rows):
    return rm.resolve_multiple_correction_proposals(bundle(tuple(rows))).to_dict()


def replan(result, mutate):
    data = deepcopy(result)
    plan = dict(data["claim_resolution_plans"][0])
    mutate(plan)
    plan["claim_resolution_plan_fingerprint"] = rm._sha256(rm._plan_payload(plan))
    data["claim_resolution_plans"] = (plan,)
    data["multi_proposal_resolution_fingerprint"] = rm._sha256(rm._normalized_result_payload(
        data["claim_resolution_plans"], data["pair_relations"], data["resolution_status"], data["aggregation_status"], data["eligible_for_07c"]
    ))
    blocked = tuple(p["claim_id"] for p in data["claim_resolution_plans"] if p["blocks_07c"])
    data["multi_proposal_audit_fingerprint"] = rm._sha256(rm._audit_result_payload(
        data["multi_proposal_resolution_fingerprint"], data["pair_relations"], data["resolution_issue_codes"], data["resolution_warnings"], blocked, data["source_bundle_audit_fingerprint"]
    ))
    return data


def test_application_order_inverted_rejected():
    data = resolved(corr("x1",6,10,"beta","B"), corr("x2",11,16,"gamma","G"))
    bad = replan(data, lambda p: p.update(application_order=tuple(reversed(p["application_order"]))))
    with pytest.raises(ValueError, match="APPLICATION_ORDER_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(bad)


def test_virtual_result_tamper_rejected():
    data = resolved(corr("x1",6,10,"beta","B"))
    bad = replan(data, lambda p: p.update(virtual_result_text="tampered"))
    with pytest.raises(ValueError, match="VIRTUAL_RESULT_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(bad)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["selected_patch_records"][0].update(replacement_text="X"), "VIRTUAL_RESULT_MISMATCH"),
    (lambda p: p["selected_patch_records"][0]["target_span_in_claim"].update(start=5), "PATCH_SPAN_INVALID|PATCH_TARGET_MISMATCH"),
    (lambda p: p.update(selected_patch_records=()), "PATCH_MEMBERSHIP_MISMATCH"),
    (lambda p: p.update(selected_patch_records=p["selected_patch_records"] + (dict(p["selected_patch_records"][0], correction_id="extra"),)), "PATCH_MEMBERSHIP_MISMATCH"),
    (lambda p: p["selected_patch_records"][0].update(claim_id="other"), "PATCH_CONTEXT_MISMATCH"),
    (lambda p: p["selected_patch_records"][0].update(original_claim_fingerprint="0"*64), "ORIGINAL_FINGERPRINT_MISMATCH"),
    (lambda p: p["selected_patch_records"][0]["target_span_in_claim"].update(text="xxxx"), "PATCH_TARGET_MISMATCH"),
])
def test_patch_tampering_rejected(mutation,error):
    data = resolved(corr("x1",6,10,"beta","B"))
    bad = replan(data, mutation)
    with pytest.raises(ValueError, match=error):
        validate_provisional_multi_proposal_resolution_result(bad)


def test_reason_matrix_independent_rejects_redundancy_reason():
    data = resolved(corr("x1",6,10,"beta","B"), corr("x2",11,16,"gamma","G"))
    data["pair_relations"][0]["reason_codes"] = ("PAIR_SAME_SPAN_AND_REPLACEMENT",)
    with pytest.raises(ValueError, match="REASON_TYPE_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_reason_matrix_redundant_rejects_positional_reason():
    data = resolved(corr("x1",6,10,"beta","B"), corr("x2",6,10,"beta","B"))
    data["pair_relations"][0]["reason_codes"] = ("PAIR_POSITIONALLY_DISJOINT","PAIR_SEMANTIC_COMPATIBILITY_NOT_ASSERTED")
    with pytest.raises(ValueError, match="REASON_TYPE_MISMATCH"):
        validate_provisional_multi_proposal_resolution_result(data)


def test_valid_plan_reconstructs_and_patch_matches():
    data = resolved(corr("x1",6,10,"beta","B"), corr("x2",11,16,"gamma","GAMMA"))
    plan = data["claim_resolution_plans"][0]
    assert plan["application_order"] == ("x2","x1")
    assert plan["virtual_result_text"] == "Alpha B GAMMA."
    assert {p["correction_id"] for p in plan["selected_patch_records"]} == {"x1","x2"}
    validate_provisional_multi_proposal_resolution_result(data)


def test_redundancy_has_one_selected_patch():
    plan = resolved(corr("x1",6,10,"beta","B"), corr("x2",6,10,"beta","B"))["claim_resolution_plans"][0]
    assert plan["selected_correction_ids"] == ("x1",)
    assert plan["redundant_correction_ids"] == ("x2",)
    assert tuple(p["correction_id"] for p in plan["selected_patch_records"]) == ("x1",)


def test_runtime_in_memory_orchestrates_without_writes():
    events=[]
    class Agent:
        def __init__(self, **kwargs): events.append("agent")
        def verify_claim(self, context): events.append("verify"); return {"claim_id":context["claim_id"]}
    def proposal_runner(context, *, llm):
        events.append("proposal"); return {"correction_id":"x1","accepted_for_reverification":True}
    def precheck(value): events.append("precheck"); return {"precheck_status":"PRECHECK_PASSED"}
    def reverify(inp, pre, *, reverification_llm): events.append("reverify"); return {"reverification_execution_status":"COMPLETED"}
    def compare(inp, pre, rev): events.append("compare"); return {"acceptance_decision":"ACCEPT_FOR_07C"}
    def build_bundle(inp): events.append("bundle"); return real_bundle(())
    def resolve(b): events.append("resolve"); return resolve_multiple_correction_proposals(b)
    deps=VerificationRuntimeDependencies(
        verification_agent_factory=Agent, proposal_runner=proposal_runner, precheck_runner=precheck,
        reverification_runner=reverify, comparison_runner=compare, bundle_builder=build_bundle,
        resolution_runner=resolve,
        correction_context_factory=lambda c,v,cfg:{"claim_id":c["claim_id"]},
        reverification_input_factory=lambda c,v,p,cfg:{"correction_id":p["correction_id"]},
    )
    inp=Agent07RuntimeInput(
        committed_agent06_output={"commit_status":"COMMITTED","run_id":"run-06","artifact_identity":"draft-manifest","schema_version":"AGENT06_COMMITTED_V1","source_draft_fingerprint":"a"*64,"claim_verification_contexts":({"claim_id":"c1","section_id":"s1"},)},
        agent07_config={},policy_versions={"verification":"v1"},schema_versions={"runtime":"v1"},experiment_paths={"root":"/tmp/exp"},
    )
    result=run_agent07_in_memory(inp,dependencies=deps)
    assert events == ["agent","verify","proposal","precheck","reverify","compare","bundle","resolve"]
    assert not result.correction_applied and not result.official_artifacts_created and not result.evaluation_ready_emitted
    assert result.execution_metrics["official_writes"] == 0
    assert result.execution_metrics["additional_llm_calls"] == 0
    assert result.execution_metrics["additional_retrieval_rounds"] == 0


def test_runtime_requires_committed_agent06_output():
    inp=Agent07RuntimeInput({"commit_status":"DRAFT","claim_verification_contexts":()}, {}, {"p":"v"}, {"s":"v"}, {})
    deps=VerificationRuntimeDependencies(correction_context_factory=lambda *x:{},reverification_input_factory=lambda *x:{})
    with pytest.raises(ValueError, match="NOT_COMMITTED"):
        run_agent07_in_memory(inp,dependencies=deps)
