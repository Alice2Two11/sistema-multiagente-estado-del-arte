
import csv, hashlib, io, json
from copy import deepcopy
import pytest

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.agent07c_handoff import (
    REQUIRED_SAFETY_POLICY, prepare_agent07c_input_from_agent07,
    validate_original_agent07c_input_artifacts,
)
from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from test_pipeline_compatibility_handoffs import _store_fixture, _write_csv
from test_multi_proposal_resolution_phase66 import bundle, claim, corr
from agent07c_test_support import terminal_handoff_args


def _fp(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _args(draft, contexts, bundle_obj, resolution_obj):
    return dict(source_draft_markdown=draft["sections"][0]["text"],experiment_id="exp",committed_source_draft_fingerprint=_fp(draft),claim_source_contexts=contexts,safety_policy=REQUIRED_SAFETY_POLICY,**terminal_handoff_args(bundle_obj,resolution_obj,contexts))


def test_claim_without_evidence_is_included_from_authoritative_draft_inventory(tmp_path):
    store,mapping=_store_fixture(tmp_path,extra_claim_no_evidence=True)
    out=build_agent07_input_from_committed_agent06(store=store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    assert out["expected_claim_ids"] == ("c1","c2")
    assert tuple(c["claim_id"] for c in out["claim_verification_contexts"]) == ("c1","c2")
    second=out["claim_verification_contexts"][1]
    assert second["eligible_evidence"] == ()


def test_unknown_claim_and_duplicate_inventory_are_rejected(tmp_path):
    (tmp_path/"unknown").mkdir(); store,mapping=_store_fixture(tmp_path/"unknown",unknown_evidence=True)
    with pytest.raises(ValueError,match="UNKNOWN_EVIDENCE_CLAIM"):
        build_agent07_input_from_committed_agent06(store=store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    (tmp_path/"duplicate").mkdir(); store2,mapping2=_store_fixture(tmp_path/"duplicate",duplicate_inventory=True)
    with pytest.raises(ValueError,match="CLAIM_ID_DUPLICATE"):
        build_agent07_input_from_committed_agent06(store=store2,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping2)


def test_confidence_available_and_unavailable_are_exported_without_fabrication():
    text="Alpha beta gamma."; draft={"sections":[{"section_id":"s1","text":text}]}; sfp=fingerprint_text(text)
    ctx=({"claim_id":"c1","section_id":"s1","original_claim_text":text,"claim_fingerprint":sfp,"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":sfp}},)
    row=claim(corrections=()); row["source_verification_confidence"]=0.82; row["source_confidence_status"]="AVAILABLE"
    b=bundle((),row); r=resolve_multiple_correction_proposals(b)
    out=prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,**_args(draft,ctx,b,r))
    rows=list(csv.DictReader(io.StringIO(out.artifact_payloads["verification_report.csv"].decode())))
    assert rows[0]["confidence"]=="0.82" and rows[0]["confidence_status"]=="AVAILABLE"
    b2=bundle((),claim(corrections=())); r2=resolve_multiple_correction_proposals(b2)
    out2=prepare_agent07c_input_from_agent07(provisional_bundle=b2.to_dict(),resolution_result=r2.to_dict(),source_draft=draft,**_args(draft,ctx,b2,r2))
    rows2=list(csv.DictReader(io.StringIO(out2.artifact_payloads["verification_report.csv"].decode())))
    assert rows2[0]["confidence"]=="" and rows2[0]["confidence_status"]=="NOT_AVAILABLE_IN_SOURCE_CONTRACT"


def test_unbacked_safety_flag_is_rejected():
    text="Alpha beta gamma."; draft={"sections":[{"section_id":"s1","text":text}]}; sfp=fingerprint_text(text)
    ctx=({"claim_id":"c1","section_id":"s1","original_claim_text":text,"claim_fingerprint":sfp,"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":sfp}},)
    b=bundle((),claim(corrections=()));r=resolve_multiple_correction_proposals(b);args=_args(draft,ctx,b,r)
    args["runtime_result"]["execution_metrics"]["independent_rag_claims"]=0
    with pytest.raises(ValueError,match="INDEPENDENT_RAG_UNPROVEN"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,**args)


def test_artifact_gate_can_pass_while_scientific_coverage_check_blocks():
    text="Alpha beta gamma."; draft={"sections":[{"section_id":"s1","text":text}]}; sfp=fingerprint_text(text)
    ctx=({"claim_id":"c1","section_id":"s1","original_claim_text":text,"claim_fingerprint":sfp,"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":sfp}},
         {"claim_id":"c2","section_id":"s1","original_claim_text":"missing","claim_fingerprint":fingerprint_text("missing"),"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"claim_span_in_section":{"start":0,"end":7,"text":"missing","base_text_fingerprint":sfp}})
    b=bundle((),claim(corrections=()));r=resolve_multiple_correction_proposals(b)
    with pytest.raises(ValueError,match="CLAIM_COVERAGE_MISMATCH"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,**_args(draft,ctx,b,r))


def test_optional_historical_exports_are_auditable_not_required_by_07c():
    text="Alpha beta gamma."; draft={"sections":[{"section_id":"s1","text":text}]}; sfp=fingerprint_text(text)
    ctx=({"claim_id":"c1","section_id":"s1","original_claim_text":text,"claim_fingerprint":sfp,"section_fingerprint":sfp,"source_draft_fingerprint":_fp(draft),"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":sfp}},)
    b=bundle((),claim(corrections=()));r=resolve_multiple_correction_proposals(b)
    out=prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,**_args(draft,ctx,b,r))
    assert set(out.optional_artifact_payloads)=={"hallucination_report.csv","citation_check.csv","claim_atomization_log.csv"}
    assert validate_original_agent07c_input_artifacts(artifact_payloads=out.artifact_payloads,experiment_id="exp")["validation_ok"]
    report=json.loads(out.artifact_payloads["verification_validation_report.json"])
    assert report["validation_ok"] == (report["structural_validation_ok"] and report["scientific_handoff_validation_ok"] and report["original_07c_artifact_gate_ok"])
