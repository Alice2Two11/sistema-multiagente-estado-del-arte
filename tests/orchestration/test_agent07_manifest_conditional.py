"""Pruebas del contrato condicional del manifest de 07 (parte 3 del pedido):
``writer_revision_request.json`` participa del manifest SOLO cuando la
transición es RETURN.

Usa un ``PreparedAgent07Execution`` REAL (vía ``prepare_agent07_execution``
sobre el fixture real de ``test_verification_stagespec_integration.py``)
como base de identidad/fingerprints -- pero ejercita las funciones de
CONTRATO (``_manifest_for``, ``validate_agent07_artifact_manifest_contract``,
``_expected_candidate_payload_names``, ``_validate_execution_for_commit``,
``commit_executed_agent07``, ``resume_agent07_execution``) directamente
con payloads/``ExecutedAgent07Execution`` construidos a mano, en vez de
atravesar todo el runtime de ``VerificationAgent`` (eso queda para la
prueba productiva end-to-end, documentada como bloqueo pendiente en el
informe final -- la construcción real del bundle vía
``build_provisional_verification_traceability_bundle`` exige datos de
referential/evidencia más profundos de lo que se pudo reproducir a
tiempo).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "verification"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.verification_notebook import (  # noqa: E402
    AGENT07_ARTIFACT_NAMES,
    AGENT07_CONDITIONAL_ARTIFACT_NAME,
    ExecutedAgent07Execution,
    MANIFEST_NAME,
    OPERATIONAL_AUDIT_NAME,
    SCIENTIFIC_ARTIFACT_NAMES,
    _expected_candidate_payload_names,
    _json_bytes,
    _manifest_for,
    _validate_execution_for_commit,
    commit_executed_agent07,
    prepare_agent07_execution,
    resume_agent07_execution,
    validate_agent07_artifact_manifest_contract,
    validate_executed_agent07_execution_contract,
)
from src.adapters.verification_runtime import Agent07RuntimeResult  # noqa: E402
from src.contracts.agent_input import ArtifactReference  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.state.fingerprints import sha256_bytes  # noqa: E402
from src.state.state_store import StateStore  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


def _prepared_and_store(tmp_path: Path):
    project_dir, store, mapping_path = T._seed_project(tmp_path)
    build_execution = T._deterministic_build_execution(store, mapping_path, project_dir)
    dependencies, runtime_input = build_execution(project_dir, 1)
    prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
    return project_dir, store, prepared


def _scientific_payloads():
    return {
        SCIENTIFIC_ARTIFACT_NAMES[0]: _json_bytes({"claim_traceability_rows": []}),
        SCIENTIFIC_ARTIFACT_NAMES[1]: _json_bytes({"claim_resolution_plans": []}),
        SCIENTIFIC_ARTIFACT_NAMES[2]: _json_bytes({"runtime_status": "COMPLETED"}),
    }


def _revision_request_payload():
    return {
        "experiment_id": "exp_stagespec",
        "cycle_id": "cycle_1",
        "round_number": 1,
        "correctable_claim_ids": ["c1"],
    }


def _fake_runtime_result(*, status="COMPLETED"):
    bundle = T.real_bundle(())
    from src.tools.verification.resolution import resolve_multiple_correction_proposals

    resolution = resolve_multiple_correction_proposals(bundle)
    bundle_d = bundle.to_dict()
    resolution_d = resolution.to_dict()
    inventory = (
        {
            "artifact_type": "PROVISIONAL_VERIFICATION_TRACEABILITY_BUNDLE",
            "artifact_status": "READY_CANDIDATE",
            "candidate_only": True,
            "producer": "AGENT07_RUNTIME",
            "schema_version": "v1",
            "audit_fingerprint": bundle_d["aggregation_audit_fingerprint"],
            "normalized_fingerprint": bundle_d["normalized_bundle_fingerprint"],
        },
        {
            "artifact_type": "MULTI_PROPOSAL_RESOLUTION_RESULT",
            "artifact_status": "READY_CANDIDATE",
            "candidate_only": True,
            "producer": "AGENT07_RUNTIME",
            "schema_version": "v1",
            "audit_fingerprint": resolution_d["multi_proposal_audit_fingerprint"],
            "normalized_fingerprint": resolution_d["multi_proposal_resolution_fingerprint"],
        },
    )
    return Agent07RuntimeResult(
        provisional_bundle=bundle_d,
        multi_proposal_resolution_result=resolution_d,
        candidate_artifact_inventory=inventory,
        execution_metrics={
            "claims_processed": 0, "independent_rag_claims": 0, "independent_rag_claims_with_results": 0,
            "independent_rag_claims_without_results": 0, "independent_rag_claim_records": (),
            "evidence_candidate_validation_claims": 0, "correction_proposals": 0, "reverification_inputs": 0,
            "prechecks": 0, "reverifications": 0, "comparisons": 0, "additional_llm_calls": 0,
            "additional_retrieval_rounds": 0, "official_writes": 0, "physical_corrections": 0,
        },
        runtime_warnings=(),
        runtime_issue_codes=(),
        runtime_error_records=(),
        blocked_runtime_audit_record=None,
        runtime_status=status,
        result_contract_valid=True,
    )


def _executed_with_payloads(prepared, payloads, *, output_dir: Path, runtime_status="COMPLETED", action="ADVANCE"):
    refs = {name: ArtifactReference(str(output_dir / name), sha256_bytes(data)) for name, data in payloads.items()}
    action_map = {"ADVANCE": TransitionAction.ADVANCE, "RETURN": TransitionAction.RETURN, "HALT_STAGE": TransitionAction.HALT_STAGE}
    result = AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED if action == "ADVANCE" else QualityStatus.NEEDS_REVISION,
        decision=DecisionInfo(code=f"AGENT07_{action}", rationale="prueba de manifest"),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(action=action_map[action], target_stage=None, reason_code=f"AGENT07_{action}", requires_human_confirmation=(action != "ADVANCE")),
        output_artifacts=refs,
        tool_usage=ToolUsage(),
        attempt_number=prepared.attempt_number,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    staging_manifest = output_dir / "staging_index.json"
    staging_manifest.parent.mkdir(parents=True, exist_ok=True)
    staging_manifest.write_text("{}", encoding="utf-8")
    persisted_result_path = output_dir / "persisted_result.json"
    persisted_result_path.write_text("{}", encoding="utf-8")
    return ExecutedAgent07Execution(
        decision_id=prepared.decision_id,
        runtime_input=prepared.runtime_input,
        runtime_result=_fake_runtime_result(status=runtime_status),
        candidate_payloads=payloads,
        staging_manifest_path=str(staging_manifest),
        agent_result=result,
        persisted_result_path=str(persisted_result_path),
        stage_fingerprints=prepared.stage_fingerprints,
        attempt_number=prepared.attempt_number,
        execution_fingerprint=prepared.execution_fingerprint,
    )


def _output_dir_for(prepared) -> Path:
    from src.adapters.verification_notebook import _published_dir

    class _Stub:
        runtime_input = prepared.runtime_input

    return _published_dir(_Stub())


@scenario("M01. Manifest de RETURN válido: incluye writer_revision_request.json con hash y tamaño reales")
def test_manifest_return_valid():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, prepared = _prepared_and_store(Path(tmp))
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        validated = validate_agent07_artifact_manifest_contract(manifest, artifact_bytes=payloads)
        names = {a["artifact_name"] for a in validated["artifacts"]}
        assert names == set(SCIENTIFIC_ARTIFACT_NAMES) | {AGENT07_CONDITIONAL_ARTIFACT_NAME}
        entry = next(a for a in validated["artifacts"] if a["artifact_name"] == AGENT07_CONDITIONAL_ARTIFACT_NAME)
        assert entry["sha256"] == sha256_bytes(payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME])
        assert entry["size_bytes"] == len(payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME])


@scenario("M02. Manifest de ADVANCE/HALT: sin writer_revision_request.json, no se exige ni se acepta")
def test_manifest_advance_no_conditional():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, prepared = _prepared_and_store(Path(tmp))
        payloads = _scientific_payloads()
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        validated = validate_agent07_artifact_manifest_contract(manifest, artifact_bytes=payloads)
        names = {a["artifact_name"] for a in validated["artifacts"]}
        assert names == set(SCIENTIFIC_ARTIFACT_NAMES)
        assert AGENT07_CONDITIONAL_ARTIFACT_NAME not in names


@scenario("M03. Manifest declara el condicional pero artifact_bytes no lo trae -> ValueError (hash incorrecto/faltante)")
def test_manifest_declares_conditional_missing_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, prepared = _prepared_and_store(Path(tmp))
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        payloads_without_conditional = {k: v for k, v in payloads.items() if k != AGENT07_CONDITIONAL_ARTIFACT_NAME}
        try:
            validate_agent07_artifact_manifest_contract(manifest, artifact_bytes=payloads_without_conditional)
        except ValueError as exc:
            assert "AGENT07_MANIFEST_ARTIFACT_CONTENT_MISMATCH" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError")


@scenario("M04. Hash incorrecto: el contenido real no coincide con el hash declarado en el manifest")
def test_manifest_hash_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, prepared = _prepared_and_store(Path(tmp))
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        # Alterado DESPUÉS del staging (simula un revision_request modificado
        # tras haberse fijado el manifest -- el hash ya no coincide).
        tampered = dict(payloads)
        tampered[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes({**_revision_request_payload(), "round_number": 99})
        try:
            validate_agent07_artifact_manifest_contract(manifest, artifact_bytes=tampered)
        except ValueError as exc:
            assert "AGENT07_MANIFEST_ARTIFACT_CONTENT_MISMATCH" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError")


@scenario("M05. Nombre extra que no es writer_revision_request.json -> manifest inválido")
def test_manifest_unexpected_extra_name():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, prepared = _prepared_and_store(Path(tmp))
        payloads = _scientific_payloads()
        payloads["algo_inesperado.json"] = _json_bytes({"x": 1})
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        # Forzar manualmente un manifest con un artefacto extra no permitido.
        from dataclasses import replace
        from src.adapters.verification_notebook import Agent07ManifestArtifact

        bad_manifest = replace(
            manifest,
            artifacts=manifest.artifacts + (Agent07ManifestArtifact("algo_inesperado.json", sha256_bytes(payloads["algo_inesperado.json"]), len(payloads["algo_inesperado.json"])),),
        )
        try:
            validate_agent07_artifact_manifest_contract(bad_manifest, artifact_bytes=payloads)
        except ValueError as exc:
            assert "AGENT07_MANIFEST_ARTIFACT_SET_INVALID" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError")


@scenario("M06. _expected_candidate_payload_names: RETURN exige el condicional; ADVANCE/HALT lo rechazan")
def test_expected_candidate_payload_names_conditional():
    completed = _fake_runtime_result(status="COMPLETED")
    with_request = _expected_candidate_payload_names(completed, has_revision_request=True)
    without_request = _expected_candidate_payload_names(completed, has_revision_request=False)
    assert AGENT07_CONDITIONAL_ARTIFACT_NAME in with_request
    assert AGENT07_CONDITIONAL_ARTIFACT_NAME not in without_request
    assert with_request - without_request == {AGENT07_CONDITIONAL_ARTIFACT_NAME}


@scenario("M07. validate_executed_agent07_execution_contract: deriva has_revision_request de candidate_payloads, sin parámetro extra")
def test_executed_contract_derives_conditional():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = tmp_path / "output"
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="RETURN")
        validated = validate_executed_agent07_execution_contract(executed)
        assert validated is executed


@scenario("M08. writer_revision_request ausente de candidate_payloads antes del COMMIT: HALT/ADVANCE nunca lo exigen")
def test_advance_never_requires_conditional():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = tmp_path / "output"
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="ADVANCE")
        validated = validate_executed_agent07_execution_contract(executed)
        assert AGENT07_CONDITIONAL_ARTIFACT_NAME not in validated_payload_names(validated)


def validated_payload_names(executed):
    return set(executed.candidate_payloads)


@scenario("M09. COMMIT real con RETURN: writer_revision_request.json existe en el directorio definitivo tras el COMMIT")
def test_commit_return_writes_conditional_to_definitive_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir, store, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = _output_dir_for(prepared)
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="RETURN")
        commit_executed_agent07(store=store, executed=executed)
        assert (output_dir / AGENT07_CONDITIONAL_ARTIFACT_NAME).is_file()
        for name in SCIENTIFIC_ARTIFACT_NAMES:
            assert (output_dir / name).is_file()
        assert (output_dir / MANIFEST_NAME).is_file()


@scenario("M10. COMMIT real con ADVANCE: writer_revision_request.json NO existe en el directorio definitivo")
def test_commit_advance_no_conditional_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir, store, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = _output_dir_for(prepared)
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="ADVANCE")
        commit_executed_agent07(store=store, executed=executed)
        assert not (output_dir / AGENT07_CONDITIONAL_ARTIFACT_NAME).exists()


@scenario("M11. RESUME tras COMMIT con RETURN: detecta el condicional en disco y lo valida contra el manifest")
def test_resume_after_commit_with_conditional():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir, store, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME] = _json_bytes(_revision_request_payload())
        manifest = _manifest_for(prepared, payloads, include_conditional=True)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = _output_dir_for(prepared)
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="RETURN")
        commit_executed_agent07(store=store, executed=executed)

        resume = resume_agent07_execution(store=store, runtime_input=prepared.runtime_input)
        assert resume.action == "COMMITTED", resume.action


@scenario("M12. _validate_execution_for_commit rechaza un nombre extra que no sea el condicional")
def test_validate_execution_for_commit_rejects_unexpected_extra():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _, prepared = _prepared_and_store(tmp_path)
        payloads = _scientific_payloads()
        payloads["otro_extra.json"] = _json_bytes({"x": 1})
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = tmp_path / "output"
        executed = _executed_with_payloads(prepared, payloads, output_dir=output_dir, action="ADVANCE")
        try:
            _validate_execution_for_commit(executed)
        except RuntimeError as exc:
            assert "AGENT07_COMMIT_MANIFEST_INCOMPLETE" in str(exc)
        else:
            raise AssertionError("debía lanzar RuntimeError")


if __name__ == "__main__":
    for fn in (
        test_manifest_return_valid,
        test_manifest_advance_no_conditional,
        test_manifest_declares_conditional_missing_bytes,
        test_manifest_hash_mismatch,
        test_manifest_unexpected_extra_name,
        test_expected_candidate_payload_names_conditional,
        test_executed_contract_derives_conditional,
        test_advance_never_requires_conditional,
        test_commit_return_writes_conditional_to_definitive_dir,
        test_commit_advance_no_conditional_file,
        test_resume_after_commit_with_conditional,
        test_validate_execution_for_commit_rejects_unexpected_extra,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
