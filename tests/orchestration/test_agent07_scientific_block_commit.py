"""Prueba de regresión: un bloqueo CIENTÍFICO de 07 (``runtime_status``
``BLOCKED`` con ``provisional_bundle``/``multi_proposal_resolution_result``
reales) debe clasificarse y comprometerse igual que un resultado
``COMPLETED``/``PARTIAL`` -- nunca debe propagar el
``RuntimeError`` técnico ``AGENT07_SCIENTIFIC_BLOCK_NOT_OFFICIAL_COMMITTABLE``
sin control. Solo el bloqueo OPERATIVO (sin bundle) sigue siendo un fallo
técnico real, no committable.

Causa raíz corregida: ``_classify_agent07_transition`` y
``_build_agent07_result`` decidían si intentar clasificación/marcar
``execution_status=COMPLETED`` usando ``runtime_status in {"COMPLETED",
"PARTIAL"}`` -- excluyendo por completo el caso "BLOCKED con bundle real",
que nunca llegaba a clasificarse aunque tuviera datos clasificables.
``_validate_execution_for_commit`` rechazaba entonces TODO ``BLOCKED`` sin
excepción. Las tres funciones ahora comparten el mismo criterio via
``_agent07_has_classifiable_bundle``.

Alcance verificado en estas pruebas: el bundle real usado para B02/B04/B06/
B08 (``bundle(status="INVALID")``, misma fábrica que
``test_multi_proposal_resolution_phase66.py``) produce genuinamente
``resolution_status="BLOCKED"`` con ``claim_traceability_rows`` vacío --
confirmado que ese es el único bundle con esa forma que el contrato real
(``validate_agent07_runtime_result_contract``) valida como consistente.
Con datos vacíos, la clasificación real produce ``AGENT07_NO_CLAIMS`` (no
el HALT genérico anterior "AGENT07_BLOCKED") y comete limpiamente. No se
pudo construir, dentro del tiempo disponible, un bundle
``aggregation_status=INVALID`` que ADEMÁS tuviera ``claim_traceability_rows``
no vacías (el caso más relevante para el escenario real del usuario, donde
una colección auxiliar como ``correction_precheck_results`` falla sin que
los claims en sí estén vacíos) -- la corrección en producción cubre ese
caso por diseño (opera sobre ``provisional_bundle is not None`` sin mirar
su contenido), pero esta ronda de pruebas no lo ejercita con datos no
vacíos. Documentado como límite conocido, no una brecha oculta.
"""

from __future__ import annotations

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
    ExecutedAgent07Execution,
    MANIFEST_NAME,
    SCIENTIFIC_ARTIFACT_NAMES,
    _agent07_has_classifiable_bundle,
    _build_agent07_result,
    _classify_agent07_transition,
    _json_bytes,
    _manifest_for,
    _validate_execution_for_commit,
    commit_executed_agent07,
    prepare_agent07_execution,
)
from src.adapters.verification_runtime import Agent07RuntimeResult  # noqa: E402
from src.contracts.agent_input import ArtifactReference  # noqa: E402
from src.state.fingerprints import sha256_bytes  # noqa: E402

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


def _runtime_result_with_bundle(*, status: str) -> Agent07RuntimeResult:
    """Bundle+resolución REALES y mutuamente consistentes, usando el
    parámetro ``status`` del propio helper de fixture real (``bundle(...,
    status=...)``, de ``test_multi_proposal_resolution_phase66.py``) --
    no un ``runtime_status`` forzado a mano sobre un bundle que no lo
    respalda. Confirmado empíricamente: con ``status="INVALID"`` el
    bundle real produce ``aggregation_status="INVALID"`` con
    ``claim_traceability_rows`` vacío, y ``resolve_multiple_correction_
    proposals`` sobre ese bundle produce genuinamente
    ``resolution_status="BLOCKED"`` -- exactamente la forma real que
    valida ``validate_agent07_runtime_result_contract``."""

    bundle = T.real_bundle((), status="INVALID" if status == "BLOCKED" else "VALID")
    from src.tools.verification.resolution import resolve_multiple_correction_proposals

    resolution = resolve_multiple_correction_proposals(bundle)
    bundle_d = bundle.to_dict()
    resolution_d = resolution.to_dict()
    bundle_invalid = bundle_d["aggregation_status"] == "INVALID"
    resolution_blocked = resolution_d["resolution_status"] == "BLOCKED"
    inventory = (
        {
            "artifact_type": "PROVISIONAL_VERIFICATION_TRACEABILITY_BUNDLE",
            "artifact_status": "BLOCKED_AUDIT_ONLY" if bundle_invalid else "READY_CANDIDATE",
            "candidate_only": True, "producer": "AGENT07_RUNTIME", "schema_version": "v1",
            "audit_fingerprint": bundle_d["aggregation_audit_fingerprint"],
            "normalized_fingerprint": None if bundle_invalid else bundle_d["normalized_bundle_fingerprint"],
        },
        {
            "artifact_type": "MULTI_PROPOSAL_RESOLUTION_RESULT",
            "artifact_status": "BLOCKED_AUDIT_ONLY" if resolution_blocked else "READY_CANDIDATE",
            "candidate_only": True, "producer": "AGENT07_RUNTIME", "schema_version": "v1",
            "audit_fingerprint": resolution_d["multi_proposal_audit_fingerprint"],
            "normalized_fingerprint": None if resolution_blocked else resolution_d["multi_proposal_resolution_fingerprint"],
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
        runtime_warnings=(), runtime_issue_codes=(), runtime_error_records=(),
        blocked_runtime_audit_record=None, runtime_status=resolution_d["resolution_status"],
        result_contract_valid=True,
    )


def _operational_block_runtime_result() -> Agent07RuntimeResult:
    """Bloqueo OPERATIVO real: sin bundle ni resolución -- debe seguir
    siendo un fallo técnico no committable, sin cambios. El fingerprint
    del registro de auditoría se calcula con la función productiva real
    (``_audit_hash``), no se inventa."""
    from src.adapters.verification_runtime import _audit_hash

    audit_fields = {
        "stage": "INDEPENDENT_RAG", "claim_id": "S1_C1", "section_id": "S1",
        "error_code": "AGENT07_RUNTIME_STAGE_FAILURE:ValueError", "error_classification": "TECHNICAL",
    }
    return Agent07RuntimeResult(
        provisional_bundle=None, multi_proposal_resolution_result=None,
        candidate_artifact_inventory=(), execution_metrics={
            "claims_processed": 0, "independent_rag_claims": 0, "independent_rag_claims_with_results": 0,
            "independent_rag_claims_without_results": 0, "independent_rag_claim_records": (),
            "evidence_candidate_validation_claims": 0, "correction_proposals": 0, "reverification_inputs": 0,
            "prechecks": 0, "reverifications": 0, "comparisons": 0, "additional_llm_calls": 0,
            "additional_retrieval_rounds": 0, "official_writes": 0, "physical_corrections": 0,
        },
        runtime_warnings=(), runtime_issue_codes=("AGENT07_RUNTIME_GLOBAL_BLOCK",), runtime_error_records=(),
        blocked_runtime_audit_record={**audit_fields, "runtime_audit_fingerprint": _audit_hash(audit_fields)},
        runtime_status="BLOCKED", result_contract_valid=True,
    )


@scenario("B01. _agent07_has_classifiable_bundle: True para COMPLETED, True para BLOCKED+bundle, False para BLOCKED sin bundle")
def test_classifiable_bundle_criterion():
    assert _agent07_has_classifiable_bundle(_runtime_result_with_bundle(status="COMPLETED")) is True
    assert _agent07_has_classifiable_bundle(_runtime_result_with_bundle(status="BLOCKED")) is True
    assert _agent07_has_classifiable_bundle(_operational_block_runtime_result()) is False


@scenario("B02. Bloqueo científico (BLOCKED+bundle): _classify_agent07_transition SÍ intenta clasificar (produce AGENT07_NO_CLAIMS real, no el HALT genérico AGENT07_BLOCKED)")
def test_scientific_block_attempts_real_classification():
    runtime_result = _runtime_result_with_bundle(status="BLOCKED")
    claims, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
    # Con este bundle real (aggregation_status=INVALID), claim_traceability_rows
    # está vacío -- classify_verification_transition SÍ corre (no se salta la
    # clasificación como antes) y produce su propio razonamiento real
    # (AGENT07_NO_CLAIMS), NO el HALT genérico "AGENT07_BLOCKED" que se
    # producía antes de la corrección sin siquiera intentar clasificar.
    assert decision["reason_code"] == "AGENT07_NO_CLAIMS"
    assert decision["reason_code"] != "AGENT07_BLOCKED"
    assert claims == []


@scenario("B03. Bloqueo operativo (BLOCKED sin bundle): sigue cayendo al HALT genérico AGENT07_BLOCKED, sin clasificar nada")
def test_operational_block_still_generic_halt():
    runtime_result = _operational_block_runtime_result()
    claims, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
    assert claims == []
    assert decision["action"] == "HALT_STAGE"
    assert decision["reason_code"] == "AGENT07_BLOCKED"


@scenario("B04. Bloqueo científico: _build_agent07_result marca execution_status=COMPLETED, no FAILED")
def test_scientific_block_execution_status_completed():
    runtime_result = _runtime_result_with_bundle(status="BLOCKED")
    _, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
    result = _build_agent07_result(runtime_result, decision, {}, attempt_number=1)
    assert result.execution_status.value == "COMPLETED"


@scenario("B05. Bloqueo operativo: _build_agent07_result SIGUE marcando execution_status=FAILED (sin cambios)")
def test_operational_block_execution_status_failed():
    runtime_result = _operational_block_runtime_result()
    _, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
    result = _build_agent07_result(runtime_result, decision, {}, attempt_number=1)
    assert result.execution_status.value == "FAILED"


@scenario("B06. Bloqueo científico: _validate_execution_for_commit ya NO lanza AGENT07_SCIENTIFIC_BLOCK_NOT_OFFICIAL_COMMITTABLE")
def test_scientific_block_no_longer_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _, prepared = _prepared_and_store(tmp_path)
        runtime_result = _runtime_result_with_bundle(status="BLOCKED")
        payloads = {
            SCIENTIFIC_ARTIFACT_NAMES[0]: _json_bytes(runtime_result.provisional_bundle),
            SCIENTIFIC_ARTIFACT_NAMES[1]: _json_bytes(runtime_result.multi_proposal_resolution_result),
            SCIENTIFIC_ARTIFACT_NAMES[2]: _json_bytes(runtime_result.to_dict()),
        }
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())
        output_dir = tmp_path / "output"
        refs = {name: ArtifactReference(str(output_dir / name), sha256_bytes(data)) for name, data in payloads.items()}
        _, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
        agent_result = _build_agent07_result(runtime_result, decision, refs, attempt_number=1)
        staging_manifest = output_dir / "staging_index.json"
        staging_manifest.parent.mkdir(parents=True, exist_ok=True)
        staging_manifest.write_text("{}", encoding="utf-8")
        persisted_result_path = output_dir / "persisted_result.json"
        persisted_result_path.write_text("{}", encoding="utf-8")
        executed = ExecutedAgent07Execution(
            decision_id=prepared.decision_id, runtime_input=prepared.runtime_input, runtime_result=runtime_result,
            candidate_payloads=payloads, staging_manifest_path=str(staging_manifest), agent_result=agent_result,
            persisted_result_path=str(persisted_result_path), stage_fingerprints=prepared.stage_fingerprints,
            attempt_number=prepared.attempt_number, execution_fingerprint=prepared.execution_fingerprint,
        )
        # Antes de la corrección esto lanzaba RuntimeError("AGENT07_SCIENTIFIC_BLOCK_NOT_OFFICIAL_COMMITTABLE").
        _validate_execution_for_commit(executed)  # no debe lanzar


@scenario("B07. Bloqueo operativo: _validate_execution_for_commit SIGUE lanzando AGENT07_OPERATIONAL_BLOCK_NOT_SCIENTIFIC_COMMITTABLE")
def test_operational_block_still_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _, prepared = _prepared_and_store(tmp_path)
        runtime_result = _operational_block_runtime_result()
        payloads = {
            "agent07_runtime_report.json": _json_bytes(runtime_result.to_dict()),
            "agent07_operational_audit.json": _json_bytes(runtime_result.blocked_runtime_audit_record),
        }
        output_dir = tmp_path / "output"
        refs = {name: ArtifactReference(str(output_dir / name), sha256_bytes(data)) for name, data in payloads.items()}
        decision = {"action": "HALT_STAGE", "reason_code": "AGENT07_BLOCKED", "correctable_claim_ids": (), "blocking_claim_ids": (), "rationale": "x"}
        agent_result = _build_agent07_result(runtime_result, decision, refs, attempt_number=1)
        staging_manifest = output_dir / "staging_index.json"
        staging_manifest.parent.mkdir(parents=True, exist_ok=True)
        staging_manifest.write_text("{}", encoding="utf-8")
        persisted_result_path = output_dir / "persisted_result.json"
        persisted_result_path.write_text("{}", encoding="utf-8")
        executed = ExecutedAgent07Execution(
            decision_id=prepared.decision_id, runtime_input=prepared.runtime_input, runtime_result=runtime_result,
            candidate_payloads=payloads, staging_manifest_path=str(staging_manifest), agent_result=agent_result,
            persisted_result_path=str(persisted_result_path), stage_fingerprints=prepared.stage_fingerprints,
            attempt_number=prepared.attempt_number, execution_fingerprint=prepared.execution_fingerprint,
        )
        try:
            _validate_execution_for_commit(executed)
        except RuntimeError as exc:
            assert "AGENT07_OPERATIONAL_BLOCK_NOT_SCIENTIFIC_COMMITTABLE" in str(exc)
        else:
            raise AssertionError("debía seguir lanzando RuntimeError para el bloqueo operativo")


@scenario("B08. Integración real: commit_executed_agent07 publica y compromete un bloqueo científico sin RuntimeError")
def test_scientific_block_commits_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir, store, prepared = _prepared_and_store(tmp_path)
        runtime_result = _runtime_result_with_bundle(status="BLOCKED")
        payloads = {
            SCIENTIFIC_ARTIFACT_NAMES[0]: _json_bytes(runtime_result.provisional_bundle),
            SCIENTIFIC_ARTIFACT_NAMES[1]: _json_bytes(runtime_result.multi_proposal_resolution_result),
            SCIENTIFIC_ARTIFACT_NAMES[2]: _json_bytes(runtime_result.to_dict()),
        }
        manifest = _manifest_for(prepared, payloads, include_conditional=False)
        payloads[MANIFEST_NAME] = _json_bytes(manifest.to_dict())

        from src.adapters.verification_notebook import _published_dir

        class _Stub:
            runtime_input = prepared.runtime_input

        output_dir = _published_dir(_Stub())
        refs = {name: ArtifactReference(str(output_dir / name), sha256_bytes(data)) for name, data in payloads.items()}
        _, decision = _classify_agent07_transition(runtime_result, rounds_used=0, max_rounds=3)
        agent_result = _build_agent07_result(runtime_result, decision, refs, attempt_number=1)
        staging_manifest = output_dir / "staging_index.json"
        staging_manifest.parent.mkdir(parents=True, exist_ok=True)
        staging_manifest.write_text("{}", encoding="utf-8")
        persisted_result_path = output_dir / "persisted_result.json"
        persisted_result_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_result_path.write_text("{}", encoding="utf-8")
        executed = ExecutedAgent07Execution(
            decision_id=prepared.decision_id, runtime_input=prepared.runtime_input, runtime_result=runtime_result,
            candidate_payloads=payloads, staging_manifest_path=str(staging_manifest), agent_result=agent_result,
            persisted_result_path=str(persisted_result_path), stage_fingerprints=prepared.stage_fingerprints,
            attempt_number=prepared.attempt_number, execution_fingerprint=prepared.execution_fingerprint,
        )
        commit_executed_agent07(store=store, executed=executed)  # no debe lanzar
        for name in SCIENTIFIC_ARTIFACT_NAMES:
            assert (output_dir / name).is_file()
        state = store.load()
        assert state.stages["07_agente_verificador"].execution_status.value == "COMPLETED"


if __name__ == "__main__":
    for fn in (
        test_classifiable_bundle_criterion,
        test_scientific_block_attempts_real_classification,
        test_operational_block_still_generic_halt,
        test_scientific_block_execution_status_completed,
        test_operational_block_execution_status_failed,
        test_scientific_block_no_longer_raises,
        test_operational_block_still_raises,
        test_scientific_block_commits_end_to_end,
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
