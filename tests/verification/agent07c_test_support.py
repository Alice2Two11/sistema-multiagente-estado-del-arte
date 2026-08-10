import hashlib, json
from src.adapters.agent06_verification_handoff import Agent07RetrieverBinding
from src.adapters.verification_runtime import create_agent07_runtime_result, _candidate_inventory, _base_metrics, _resolution_to_runtime_status

def _fp(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def terminal_handoff_args(bundle, resolution, contexts, experiment_id="exp"):
    b=bundle.to_dict() if hasattr(bundle,"to_dict") else dict(bundle)
    r=resolution.to_dict() if hasattr(resolution,"to_dict") else dict(resolution)
    normalized_contexts=[]
    for original in contexts:
        c=dict(original)
        sources=tuple(sorted({str(e.get("source_filename") or "p.pdf") for e in c.get("eligible_evidence",()) if str(e.get("source_filename") or "")} or {"p.pdf"}))
        c["authorized_source_filenames"]=sources
        normalized_contexts.append(c)
    contexts=tuple(normalized_contexts)
    ids=tuple(sorted(str(c["claim_id"]) for c in contexts))
    binding=Agent07RetrieverBinding(experiment_id,"c","m","a"*64,"b"*64)
    binding_fp=_fp({"experiment_id":binding.experiment_id,"collection_name":binding.collection_name,"embedding_model":binding.embedding_model,"chroma_manifest_fingerprint":binding.chroma_manifest_fingerprint,"chunks_manifest_fingerprint":binding.chunks_manifest_fingerprint})
    rag_records=[]
    for c in contexts:
        candidates=[]
        for e in c.get("eligible_evidence",()):
            source=str(e.get("source_filename") or "p.pdf"); chunk=str(e.get("chunk_id") or e.get("evidence_id") or "ch1")
            eid=str(e.get("evidence_id") or f"{source}::{chunk}")
            text=str(e.get("canonical_text",e.get("text","")))
            candidates.append({"evidence_id":eid,"source_filename":source,"chunk_id":chunk,"query_ids":(str(c["claim_id"]),),"text_fingerprint":hashlib.sha256(text.encode()).hexdigest()})
        ids_for_claim=tuple(sorted(x["evidence_id"] for x in candidates))
        rag_records.append({
            "claim_id":str(c["claim_id"]),"section_id":str(c["section_id"]),
            "retrieval_requested":1,"retrieval_rounds":1,
            "retrieval_status":"COMPLETED_WITH_RESULTS" if ids_for_claim else "COMPLETED_NO_RESULTS",
            "retriever_binding_fingerprint":binding_fp,"retrieved_candidate_ids":ids_for_claim,
            "retrieved_candidate_records":tuple(candidates),
            "verification_context_snapshot":{
                "claim_id":str(c["claim_id"]),"section_id":str(c["section_id"]),
                "eligible_evidence":tuple(sorted(({
                    "evidence_id":str(e.get("evidence_id") or f"{str(e.get('source_filename') or 'p.pdf')}::{str(e.get('chunk_id') or 'ch1')}"),
                    "source_filename":str(e.get("source_filename") or "p.pdf"),
                    "chunk_id":str(e.get("chunk_id") or "ch1"),
                    "authorized_for_section":bool(e.get("authorized_for_section") is True),
                    "text_fingerprint":hashlib.sha256(str(e.get("canonical_text",e.get("text",""))).encode()).hexdigest(),
                } for e in c.get("eligible_evidence",())),key=lambda x:(x["evidence_id"],x["source_filename"],x["chunk_id"])))
            },
        })
    with_results=sum(1 for x in rag_records if x["retrieval_status"]=="COMPLETED_WITH_RESULTS")
    without_results=len(rag_records)-with_results
    runtime=create_agent07_runtime_result(
        provisional_bundle=b,multi_proposal_resolution_result=r,
        candidate_artifact_inventory=_candidate_inventory(b,r,{"provisional_bundle":"v3","multi_proposal_resolution":"v1"}),
        execution_metrics=_base_metrics(
            claims_processed=len(ids), independent_rag_claims=len(ids), independent_rag_claims_with_results=with_results,
            independent_rag_claims_without_results=without_results, evidence_candidate_validation_claims=len(ids),
            independent_rag_claim_records=tuple(rag_records),
        ),
        runtime_warnings=(),runtime_issue_codes=(),runtime_error_records=(),blocked_runtime_audit_record=None,
        runtime_status=_resolution_to_runtime_status(r["resolution_status"]),correction_applied=False,official_artifacts_created=False,evaluation_ready_emitted=False,
    ).to_dict()
    handoff={
      "commit_status":"COMMITTED","run_id":"run","experiment_id":experiment_id,"artifact_identity":"draft06","schema_version":"v1",
      "source_draft_fingerprint":str(contexts[0]["source_draft_fingerprint"]),"agent06_manifest_source_draft_fingerprint":None,
      "claim_verification_contexts":contexts,"expected_claim_ids":ids,"claim_inventory_fingerprint":_fp(ids),
      "agent06_decision_id":"d06","outline_mapping_fingerprint":"c"*64,
      "claim_identity_migration_signal":False,
      "integration_metadata":{"agent07_config_fingerprint":"d"*64,"policy_versions":{},"schema_versions":{},"experiment_paths":{}},
    }
    return {
      "runtime_result":runtime,
      "retriever_binding":binding,
      "agent06_handoff":handoff,
    }
