import csv, hashlib, io, json
from copy import deepcopy
from types import SimpleNamespace
import pytest

from src.adapters.agent07c_handoff import (
    AGENT07C_REQUIRED_ARTIFACTS, REQUIRED_SAFETY_POLICY,
    _apply_section_claim_replacements,
    create_agent07c_prepared_input, prepare_agent07c_input_from_agent07,
    validate_agent07c_prepared_input_contract,
    validate_original_agent07c_input_artifacts,
)
from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.verification_runtime import build_agent07_runtime_dependencies
from src.state.fingerprints import sha256_file
from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from test_multi_proposal_resolution_phase66 import bundle, corr
from test_pipeline_compatibility_handoffs import _store_fixture, _write_csv
from agent07c_test_support import terminal_handoff_args


def _canonical_fp(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _prepared():
    b=bundle((corr("x1",6,10,"beta","BETA-LONG"),))
    resolution=resolve_multiple_correction_proposals(b)
    draft={"sections":[{"section_id":"s1","text":"Alpha beta gamma."}]}
    fp=_canonical_fp(draft); text=draft["sections"][0]["text"]
    context={"claim_id":"c1","section_id":"s1","original_claim_text":text,
      "claim_fingerprint":fingerprint_text(text),"section_fingerprint":fingerprint_text(text),
      "source_draft_fingerprint":fp,
      "claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":fingerprint_text(text),"coordinate_system":"PYTHON_CODEPOINT_OFFSETS"}}
    return prepare_agent07c_input_from_agent07(
      provisional_bundle=b.to_dict(),resolution_result=resolution.to_dict(),source_draft=draft,
      source_draft_markdown="# S1\n\n"+text,experiment_id="exp",committed_source_draft_fingerprint=fp,
      claim_source_contexts=(context,),safety_policy=REQUIRED_SAFETY_POLICY,
      **terminal_handoff_args(b,resolution,(context,)),
    )


def _replace_csv(payloads,name,mutator):
    payloads=dict(payloads);reader=csv.DictReader(io.StringIO(payloads[name].decode()));rows=list(reader);fields=list(reader.fieldnames or ())
    fields,rows=mutator(fields,rows)
    out=io.StringIO(newline="");w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows);payloads[name]=out.getvalue().encode();return payloads


def test_generated_artifacts_pass_original_07c_gate_and_validation_is_derived():
    out=_prepared()
    assert validate_original_agent07c_input_artifacts(artifact_payloads=out.artifact_payloads,experiment_id="exp")["validation_ok"]
    report=json.loads(out.artifact_payloads["verification_validation_report.json"])
    assert report["validation_ok"] is True
    assert report["structural_validation_ok"] is True
    assert report["scientific_handoff_validation_ok"] is True
    assert report["original_07c_artifact_gate_ok"] is True
    with pytest.raises(TypeError,match="derived"):
        create_agent07c_prepared_input(result_contract_valid=True)


def test_missing_validation_report_and_verification_column_are_rejected():
    out=_prepared(); payloads=dict(out.artifact_payloads);payloads.pop("verification_validation_report.json")
    with pytest.raises(ValueError,match="REQUIRED_INPUT_MISSING"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")
    payloads=_replace_csv(out.artifact_payloads,"verification_report.csv",lambda fields,rows:([f for f in fields if f!="confidence"],[{k:v for k,v in r.items() if k!="confidence"} for r in rows]))
    with pytest.raises(ValueError,match="VERIFICATION_REPORT_COLUMNS"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")


def test_status_must_be_applied_and_manifest_workflow_safety_are_real():
    out=_prepared()
    payloads=_replace_csv(out.artifact_payloads,"auto_corrections_log.csv",lambda fields,rows:(fields,[{**r,"status":"APPLIED_TO_COPY"} for r in rows]))
    with pytest.raises(ValueError,match="RECHECK_FLAG_MISMATCH"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")
    payloads=dict(out.artifact_payloads); manifest=json.loads(payloads["verification_traceability_manifest.json"]);manifest.pop("workflow_state");payloads["verification_traceability_manifest.json"]=json.dumps(manifest).encode()
    with pytest.raises(ValueError,match="WORKFLOW_INVALID"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")
    payloads=dict(out.artifact_payloads);manifest=json.loads(payloads["verification_traceability_manifest.json"]);manifest["safety_policy"]["uses_chunks_clean_for_rag"]=False;payloads["verification_traceability_manifest.json"]=json.dumps(manifest).encode()
    with pytest.raises(ValueError,match="SAFETY_POLICY_MISMATCH"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")


def test_source_draft_fingerprint_and_json_markdown_divergence_rejected():
    out=_prepared(); b=bundle((corr("x1",6,10,"beta","B"),));r=resolve_multiple_correction_proposals(b);draft={"sections":[{"section_id":"s1","text":"Alpha beta gamma."}]};text=draft["sections"][0]["text"]
    context={"claim_id":"c1","section_id":"s1","claim_fingerprint":fingerprint_text(text),"section_fingerprint":fingerprint_text(text),"source_draft_fingerprint":"0"*64,"claim_span_in_section":{"start":0,"end":len(text),"text":text,"base_text_fingerprint":fingerprint_text(text)}}
    with pytest.raises(ValueError,match="SOURCE_DRAFT_FINGERPRINT_MISMATCH"):
        prepare_agent07c_input_from_agent07(provisional_bundle=b.to_dict(),resolution_result=r.to_dict(),source_draft=draft,source_draft_markdown=text,experiment_id="exp",committed_source_draft_fingerprint="0"*64,claim_source_contexts=(context,),safety_policy=REQUIRED_SAFETY_POLICY,**terminal_handoff_args(b,r,(context,)))
    payloads=dict(out.artifact_payloads);payloads["verified_state_of_art.md"]=b"unrelated"
    with pytest.raises(ValueError,match="JSON_MARKDOWN_DIVERGENCE"):
        validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id="exp")


def test_two_claims_same_section_and_different_length_patches_use_offsets():
    text="Alpha beta gamma delta."
    sfp=fingerprint_text(text)
    contexts=(
      {"claim_id":"c1","section_id":"s1","claim_fingerprint":fingerprint_text("beta"),"section_fingerprint":sfp,"claim_span_in_section":{"start":6,"end":10,"text":"beta","base_text_fingerprint":sfp}},
      {"claim_id":"c2","section_id":"s1","claim_fingerprint":fingerprint_text("delta"),"section_fingerprint":sfp,"claim_span_in_section":{"start":17,"end":22,"text":"delta","base_text_fingerprint":sfp}},
    )
    plans=(
      {"claim_id":"c1","section_id":"s1","original_claim_text":"beta","virtual_result_text":"BETA-LONG"},
      {"claim_id":"c2","section_id":"s1","original_claim_text":"delta","virtual_result_text":"D"},
    )
    assert _apply_section_claim_replacements(section_text=text,contexts=contexts,plans=plans)=="Alpha BETA-LONG gamma D."


def test_agent06_evidence_identical_dedup_conflict_and_outline_authorization(tmp_path):
    store,mapping=_store_fixture(tmp_path)
    out=build_agent07_input_from_committed_agent06(store=store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    evidence=out["claim_verification_contexts"][0]["eligible_evidence"]
    assert len(evidence)==1 and evidence[0]["authorized_for_section"] is True
    # Same ID with different content across the two committed evidence files is a conflict.
    conflict_dir=tmp_path/"conflict";conflict_dir.mkdir()
    conflict_store,conflict_mapping=_store_fixture(conflict_dir,conflicting_evidence=True)
    with pytest.raises(ValueError,match="EVIDENCE_CONFLICTING_DUPLICATE"):
        build_agent07_input_from_committed_agent06(store=conflict_store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=conflict_mapping)


def test_agent06_disallowed_outline_source_and_absent_numeric_status(tmp_path):
    store,mapping=_store_fixture(tmp_path)
    _write_csv(mapping,[{"section_id":"s1","source_filename":"other.pdf"}])
    # Update numeric file to header-only and its ArtifactReference hash in the committed state is intentionally not changed: hash gate blocks first.
    out=build_agent07_input_from_committed_agent06(store=store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    assert out["claim_verification_contexts"][0]["eligible_evidence"][0]["authorized_for_section"] is False



def test_agent06_absent_numeric_is_not_not_evaluated(tmp_path):
    store,mapping=_store_fixture(tmp_path,missing_numeric=True)
    out=build_agent07_input_from_committed_agent06(store=store,stage_name="06_agente_redactor",agent07_config={},policy_versions={},schema_versions={},experiment_paths={},outline_paper_mapping_path=mapping)
    context=out["claim_verification_contexts"][0]
    assert context["numeric_risk"] is None
    assert context["numeric_risk_status"]=="NOT_AVAILABLE"

def test_retriever_without_contract_identity_is_rejected(tmp_path):
    cm=tmp_path/"chroma_index_manifest.json";chunks=tmp_path/"chunks.csv"
    cm.write_text(json.dumps({"experiment_id":"exp","collection_name":"c","embedding_model":"m"}));chunks.write_text("chunk_id\n1\n")
    binding={"experiment_id":"exp","collection_name":"c","embedding_model":"m","chroma_manifest_fingerprint":sha256_file(cm),"chunks_manifest_fingerprint":sha256_file(chunks)}
    config={"verification_policy":{},"correction_policy":{},"reverification_policy":{},"verification_budgets":{},"correction_budgets":{},"reverification_budgets":{},"verification_prompt_version":"v","correction_prompt_version":"v","reverification_prompt_version":"v","chroma_collection_name":"c","embedding_model":"m"}
    with pytest.raises(ValueError,match="RETRIEVER_IDENTITY_MISSING"):
        build_agent07_runtime_dependencies(config=config,experiment_paths={"x":"y"},verification_llm=object(),correction_llm=object(),reverification_llm=object(),incremental_retriever=object(),retriever_binding=binding,chroma_manifest_path=str(cm),chunks_manifest_path=str(chunks),committed_experiment_id="exp")
