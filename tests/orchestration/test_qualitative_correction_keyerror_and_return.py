"""Cierre del flujo productivo cualitativo (claim metodológico), y las
pruebas puntuales para el ``KeyError`` que bloqueaba ``propose_correction``.

Causa exacta del ``KeyError`` original (documentada, no solo corregida):
``src/adapters/verification_runtime.py``, función
``_productive_reverification_input``, línea 180 (versión previa a esta
corrección): ``runtime_config["reverification_prompt_version"]`` —
faltaba en el ``agent07_config`` de prueba, NO en ningún fixture
productivo real (es una clave que el llamador de la etapa 07 real
siempre provee vía ``build_experimental_verification_execution``, no
investigada aquí porque requiere red/OpenAI).

Un SEGUNDO desajuste contractual real (no un ``KeyError``, un dato
faltante silencioso) se descubrió al resolver el primero:
``run_virtual_reverification_prechecks`` (el ``precheck_runner`` por
defecto, real) exige ``context.get("section_text")`` directamente sobre
el mismo dict que recibe de ``reverification_input_factory`` -- pero
``_productive_reverification_input`` (el productor real, wired como
``reverification_input_factory`` por defecto) nunca lo copiaba desde
``claim_context``. Corregido en producción: se agrega DESPUÉS de
``validate_correction_reverification_input_contract`` (no antes, para no
alterar el contrato formal de ``ReverificationInput`` ni su fingerprint).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "verification"))

import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06  # noqa: E402
from src.adapters.claim_verification_context import fingerprint_text  # noqa: E402
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever  # noqa: E402
from src.adapters.verification_notebook import commit_executed_agent07, execute_prepared_agent07, prepare_agent07_execution  # noqa: E402
from src.adapters.verification_runtime import (  # noqa: E402
    Agent07RuntimeInput,
    VerificationRuntimeDependencies,
    _productive_reverification_input,
)
from src.agents.verification_agent import VerificationAgent  # noqa: E402
from src.config.verification_policy_config import get_verification_input_policy  # noqa: E402
from src.tools.verification.corrections import propose_correction  # noqa: E402
from src.tools.verification.cycle_round_persistence import (  # noqa: E402
    read_round_status,
    round_directory,
)
from src.tools.verification.resolution import resolve_multiple_correction_proposals  # noqa: E402
from src.tools.verification.validation import build_provisional_verification_traceability_bundle  # noqa: E402

RESULTS = []
FULL_POLICY = get_verification_input_policy()

CLAIM_TEXT_WRONG = "El método Alpha utiliza una red neuronal convolucional para clasificar documentos científicos."
CLAIM_TEXT_EVIDENCE = "El método Alpha utiliza una arquitectura basada en transformadores para clasificar documentos científicos."


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


class _AuthorizedFakeChroma:
    def query(self, *, query_texts, n_results):
        return {
            "documents": [[CLAIM_TEXT_EVIDENCE]],
            "metadatas": [[{"source_filename": "paper_a.pdf", "chunk_id": "a_independent"}]],
            "distances": [[0.02]],
        }


def _make_verification_llm(claim_id="S1_C1"):
    class _LLM:
        def invoke(self, messages):
            return {
                "claim_id": claim_id, "verdict": "PARTIALLY_SUPPORTED", "support_level": "PARTIAL",
                "evidence_ids_used": ["E02"], "evidence_ids_rejected": [], "rationale": "La sección menciona una red convolucional pero la evidencia describe transformadores.",
                "contradiction_type": "NONE", "contradiction_evidence_ids": [], "numeric_assessment": "NOT_APPLICABLE",
                "attribution_assessment": "NOT_APPLICABLE", "extrapolation_assessment": "NOT_APPLICABLE", "confidence": "MEDIUM",
                "additional_retrieval_needed": False, "llm_correction_recommendation": True, "manual_review_required": False,
                "reason_codes": ["PARTIAL_EVIDENCE"],
            }
    return _LLM()


def _make_correction_llm(claim_id="S1_C1"):
    class _LLM:
        def invoke(self, messages):
            return json.dumps({
                "claim_id": claim_id, "correction_decision": "PROPOSE_CHANGE", "action_type": "REMOVE_UNSUPPORTED_FRAGMENT",
                "target_text": "una red neuronal convolucional", "replacement_text": "una arquitectura basada en transformadores",
                "evidence_ids": ["E02"], "reason_codes": ["LOCALIZED_UNSUPPORTED_FRAGMENT"], "change_scope": "PHRASE",
                "semantic_change_level": "MINIMAL", "old_citation_refs": [], "new_citation_refs": [], "old_numeric_pairs": [],
                "new_numeric_pairs": [], "metric_context": "", "unit_context": "", "old_attribution_elements": [],
                "new_attribution_elements": [], "attribution_relation": None, "new_entities": [], "new_attributions": [],
                "new_conditions": [], "new_technical_terms": [], "citation_text_span": None, "llm_correction_recommendation": True,
            })
    return _LLM()


def _make_supported_verification_llm(claim_id="S1_C1"):
    class _LLM:
        def invoke(self, messages):
            return {
                "claim_id": claim_id, "verdict": "SUPPORTED", "support_level": "STRONG",
                "evidence_ids_used": ["E01"], "evidence_ids_rejected": [], "rationale": "El texto ahora coincide con la evidencia disponible.",
                "contradiction_type": "NONE", "contradiction_evidence_ids": [], "numeric_assessment": "NOT_APPLICABLE",
                "attribution_assessment": "NOT_APPLICABLE", "extrapolation_assessment": "NOT_APPLICABLE", "confidence": "HIGH",
                "additional_retrieval_needed": False, "llm_correction_recommendation": False, "manual_review_required": False,
                "reason_codes": [],
            }
    return _LLM()


class _ReverificationLLM:
    def __init__(self, payload_holder):
        self._payload_holder = payload_holder

    def invoke(self, messages):
        return json.dumps(self._payload_holder[0])


def _clean_claim_context(*, claim_id, section_id, section_title, claim_text, evidence_text, source_draft_fingerprint):
    return {
        "claim_id": claim_id, "claim_id_origin": "inherited_agent06", "section_id": section_id,
        "section_title": section_title, "original_claim_text": claim_text, "section_text": claim_text,
        "supporting_citations": ["[paper_a.pdf | a_chroma]"], "source_free_organizational_section": False,
        "claim_span_in_section": {
            "coordinate_base": "SECTION_TEXT", "coordinate_system": "PYTHON_CODEPOINT_OFFSETS",
            "base_text_fingerprint": fingerprint_text(claim_text), "start": 0, "end": len(claim_text), "text": claim_text,
        },
        "claim_fingerprint": fingerprint_text(claim_text), "section_fingerprint": fingerprint_text(claim_text),
        "eligible_evidence": (
            {"evidence_id": "a_chroma", "source_filename": "paper_a.pdf", "chunk_id": "a_chroma",
             "text": evidence_text, "usage_role": "ELIGIBLE", "authorized_for_section": True},
        ),
        "authorized_source_filenames": ("paper_a.pdf", "paper_b.pdf"),
        "numeric_risk": None, "numeric_risk_status": "NOT_AVAILABLE",
        "source_draft_fingerprint": source_draft_fingerprint,
    }


def _run_verification(*, store, project_dir, mapping_path, claim_text, evidence_text, verification_llm=None):
    committed_agent06_output = build_agent07_input_from_committed_agent06(
        store=store, stage_name=T.DRAFT, agent07_config={}, policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
    )
    ctx = _clean_claim_context(
        claim_id="S1_C1", section_id="S1", section_title="Modelos predictivos",
        claim_text=claim_text, evidence_text=evidence_text,
        source_draft_fingerprint=committed_agent06_output["claim_verification_contexts"][0]["source_draft_fingerprint"],
    )
    committed_agent06_output = dict(committed_agent06_output)
    committed_agent06_output["claim_verification_contexts"] = (ctx,)

    runtime_input = Agent07RuntimeInput(
        committed_agent06_output=committed_agent06_output,
        agent07_config={
            "verification_policy": {}, "reverification_policy": FULL_POLICY,
            "target_issue_codes_by_correction": {}, "reverification_prompt_version": "v1",
            "reverification_budgets": {"max_llm_attempts": 1},
        },
        policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(project_dir)},
    )
    retriever = Agent07ChromaRetriever(
        collection=_AuthorizedFakeChroma(), experiment_id="exp_stagespec", collection_name="reference_papers_chunks",
        embedding_model="all-MiniLM-L6-v2", chroma_manifest_fingerprint="f" * 64, chunks_manifest_fingerprint="c" * 64,
    )

    payload_holder = [None]

    def correction_context_factory(ctx_in, verification, config):
        return {**ctx_in, "final_correction_eligibility": verification["final_correction_eligibility"],
                "policy": FULL_POLICY, "eligible_evidence": verification["eligible_evidence"]}

    def proposal_runner_wrapper(context, *, llm):
        proposal = propose_correction(context, llm=llm)
        if proposal.accepted_for_reverification:
            payload_holder[0] = {
                "correction_id": proposal.correction_id, "claim_id": "S1_C1", "proposed_verdict": "SUPPORTED",
                "support_level": "STRONG", "evidence_ids_used": ["E02"], "observed_issue_codes": [],
                "target_issues_resolved": ["PARTIAL_SUPPORT"], "supported_meaning_preserved": True,
                "intended_semantic_change_valid": True, "unintended_semantic_change_absent": True,
                "scope_assessment": "VALID", "numeric_assessment": "NOT_APPLICABLE", "attribution_assessment": "NOT_APPLICABLE",
                "citation_assessment": "NOT_APPLICABLE", "manual_review_recommended": False, "reason_codes": [],
                "rationale": "La sección corregida ahora coincide con la evidencia disponible.", "confidence": 0.9,
            }
        return proposal

    dependencies = VerificationRuntimeDependencies(
        verification_agent_factory=VerificationAgent, verification_llm=verification_llm or _make_verification_llm(), retrieval_tool=retriever,
        retriever_binding={
            "experiment_id": "exp_stagespec", "collection_name": "reference_papers_chunks",
            "embedding_model": "all-MiniLM-L6-v2", "chroma_manifest_fingerprint": "f" * 64,
            "chunks_manifest_fingerprint": "c" * 64,
        },
        correction_context_factory=correction_context_factory,
        reverification_input_factory=_productive_reverification_input,
        proposal_runner=proposal_runner_wrapper, correction_llm=_make_correction_llm(),
        reverification_llm=_ReverificationLLM(payload_holder),
        bundle_builder=build_provisional_verification_traceability_bundle,
        resolution_runner=resolve_multiple_correction_proposals,
    )
    prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
    return execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)


@scenario("R01. Campo que provocaba el KeyError: reverification_prompt_version, ahora resuelto con runtime_config completo")
def test_keyerror_field_resolved():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )
        assert executed.runtime_result.runtime_status == "COMPLETED"
        assert executed.runtime_result.blocked_runtime_audit_record is None


@scenario("R02. Contexto correctivo productivo completo: propose_correction real produce PROPOSE_CHANGE con contexto real")
def test_full_productive_correction_context():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )
        corrections = executed.runtime_result.provisional_bundle.get("correction_traceability_rows", ())
        assert len(corrections) == 1
        assert corrections[0]["proposal_status"] == "ACCEPTED_FOR_REVERIFICATION"


@scenario("R03. CorrectionProposal válida: campos reales de propose_correction (estado, requested change, evidencia, fingerprints)")
def test_correction_proposal_real_fields():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )
        c = executed.runtime_result.provisional_bundle["correction_traceability_rows"][0]
        assert c["replacement_text"] == "una arquitectura basada en transformadores"
        assert c["proposal_fingerprint"]
        assert c["original_claim_fingerprint"]


@scenario("R04. Propuesta inválida (evidencia inexistente): fail-closed, nunca corrección automática inventada")
def test_invalid_proposal_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        committed_agent06_output = build_agent07_input_from_committed_agent06(
            store=store, stage_name=T.DRAFT, agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
        )
        ctx = _clean_claim_context(
            claim_id="S1_C1", section_id="S1", section_title="Modelos predictivos",
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
            source_draft_fingerprint=committed_agent06_output["claim_verification_contexts"][0]["source_draft_fingerprint"],
        )
        committed_agent06_output = dict(committed_agent06_output)
        committed_agent06_output["claim_verification_contexts"] = (ctx,)

        class BadCorrectionLLM:
            def invoke(self, messages):
                return json.dumps({
                    "claim_id": "S1_C1", "correction_decision": "PROPOSE_CHANGE", "action_type": "REMOVE_UNSUPPORTED_FRAGMENT",
                    "target_text": "una red neuronal convolucional", "replacement_text": "una arquitectura basada en transformadores",
                    "evidence_ids": ["EVIDENCIA_INEXISTENTE"], "reason_codes": ["LOCALIZED_UNSUPPORTED_FRAGMENT"],
                    "change_scope": "PHRASE", "semantic_change_level": "MINIMAL", "old_citation_refs": [], "new_citation_refs": [],
                    "old_numeric_pairs": [], "new_numeric_pairs": [], "metric_context": "", "unit_context": "",
                    "old_attribution_elements": [], "new_attribution_elements": [], "attribution_relation": None,
                    "new_entities": [], "new_attributions": [], "new_conditions": [], "new_technical_terms": [],
                    "citation_text_span": None, "llm_correction_recommendation": True,
                })

        runtime_input = Agent07RuntimeInput(
            committed_agent06_output=committed_agent06_output,
            agent07_config={
                "verification_policy": {}, "reverification_policy": FULL_POLICY,
                "target_issue_codes_by_correction": {}, "reverification_prompt_version": "v1",
                "reverification_budgets": {"max_llm_attempts": 1},
            },
            policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(project_dir)},
        )
        retriever = Agent07ChromaRetriever(
            collection=_AuthorizedFakeChroma(), experiment_id="exp_stagespec", collection_name="reference_papers_chunks",
            embedding_model="all-MiniLM-L6-v2", chroma_manifest_fingerprint="f" * 64, chunks_manifest_fingerprint="c" * 64,
        )
        dependencies = VerificationRuntimeDependencies(
            verification_agent_factory=VerificationAgent, verification_llm=_make_verification_llm(), retrieval_tool=retriever,
            retriever_binding={
                "experiment_id": "exp_stagespec", "collection_name": "reference_papers_chunks",
                "embedding_model": "all-MiniLM-L6-v2", "chroma_manifest_fingerprint": "f" * 64,
                "chunks_manifest_fingerprint": "c" * 64,
            },
            correction_context_factory=lambda ctx_in, verification, config: {
                **ctx_in, "final_correction_eligibility": verification["final_correction_eligibility"],
                "policy": FULL_POLICY, "eligible_evidence": verification["eligible_evidence"],
            },
            reverification_input_factory=_productive_reverification_input,
            proposal_runner=propose_correction, correction_llm=BadCorrectionLLM(),
            reverification_llm=_ReverificationLLM([None]),
            bundle_builder=build_provisional_verification_traceability_bundle,
            resolution_runner=resolve_multiple_correction_proposals,
        )
        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(store=store, prepared=prepared, dependencies=dependencies)
        c = executed.runtime_result.provisional_bundle["correction_traceability_rows"][0]
        assert c["proposal_status"] in {"REJECTED", "DEFERRED"}


@scenario("R05. Claim metodológico completo: PARTIALLY_SUPPORTED, evidencia SUPPORT, RETURN real con round_01 creada")
def test_methodological_claim_to_return():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))
        executed = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )
        assert executed.agent_result.requested_transition.action.value == "RETURN"
        row = executed.runtime_result.provisional_bundle["claim_traceability_rows"][0]
        assert row["source_verdict"] == "PARTIALLY_SUPPORTED"
        evidence_rows = executed.runtime_result.provisional_bundle["claim_evidence_traceability_rows"]
        assert any(e["usage_role"] == "SUPPORT" for e in evidence_rows)

        cycle_root = Path(tmp)
        status = read_round_status(project_dir=cycle_root, experiment_id="exp_stagespec", round_number=1)
        assert status["status"] == "AWAITING_REVISION"
        directory = round_directory(cycle_root, "exp_stagespec", 1)
        assert (directory / "writer_revision_request.json").is_file()


@scenario("R06. Ciclo completo real: RETURN -> 06 REVISION real -> reverificación 07 real -> ADVANCE -> fingerprint listo para 08")
def test_full_qualitative_cycle_return_to_advance():
    import json as _json
    from unittest.mock import patch

    sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))
    import test_writer_verifier_cycle_e2e as E2E  # noqa: E402
    from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))

        # --- 1. 07 real, primera verificación: PARTIALLY_SUPPORTED -> RETURN ---
        executed_first = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )
        assert executed_first.agent_result.requested_transition.action.value == "RETURN"
        commit_executed_agent07(store=store, executed=executed_first)

        cycle_root = Path(tmp)
        revision_request = _json.loads(
            (round_directory(cycle_root, "exp_stagespec", 1) / "writer_revision_request.json").read_text()
        )

        # --- 2. 06 real, modo REVISION: corrige solo S1, S2 queda intacta ---
        unaffected_text = "Texto de control que no debe modificarse."
        runtime06 = E2E.FakeRuntime(outputs=[{
            "section_id": "S1", "section_title": "Modelos predictivos",
            "draft_text": CLAIM_TEXT_EVIDENCE + " [paper_a.pdf | a_chroma].",
            "claims": [{"claim": CLAIM_TEXT_EVIDENCE, "supporting_citations": ["[paper_a.pdf | a_chroma]"]}],
        }])
        agent06 = DraftWritingAgent(runtime06)
        output_dir = Path(tmp) / "draft_output"
        previous_draft = {"sections": [
            {"section_id": "S1", "section_title": "Modelos predictivos",
             "draft_text": CLAIM_TEXT_WRONG + " [paper_a.pdf | a_chroma].",
             "claims": [{"claim": CLAIM_TEXT_WRONG, "supporting_citations": ["[paper_a.pdf | a_chroma]"]}]},
            {"section_id": "S2", "section_title": "Seccion no afectada",
             "draft_text": unaffected_text, "claims": []},
        ]}
        agent_input = E2E._make_agent_input(output_dir, {
            "mode": "REVISION", "writer_revision_request": revision_request, "previous_draft": previous_draft,
            "round_number": 1, "cycle_project_dir": str(cycle_root), "experiment_id": "exp_stagespec",
        })
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=E2E._bundle()), \
             patch("src.agents.draft_writing_agent.retrieve_section_evidence",
                   return_value=[{"source_filename": "paper_a.pdf", "chunk_id": "a_chroma", "text": CLAIM_TEXT_EVIDENCE}]):
            result06 = agent06.execute(agent_input)
        assert result06.execution_status.value == "COMPLETED"

        status_after_06 = read_round_status(project_dir=cycle_root, experiment_id="exp_stagespec", round_number=1)
        assert status_after_06["status"] == "REVISION_COMPLETED"

        revised = _json.loads((round_directory(cycle_root, "exp_stagespec", 1) / "revised_draft.json").read_text())
        sections_by_id = {s["section_id"]: s for s in revised["sections"]}
        assert "arquitectura basada en transformadores" in sections_by_id["S1"]["draft_text"]
        assert "red neuronal convolucional" not in sections_by_id["S1"]["draft_text"]
        assert sections_by_id["S2"]["draft_text"] == unaffected_text  # byte a byte, sin tocar

        assert (round_directory(cycle_root, "exp_stagespec", 1) / "revision_changelog.json").is_file()
        assert (round_directory(cycle_root, "exp_stagespec", 1) / "revision_resolution_matrix.json").is_file()

        # --- 3. 07 real reverifica el texto corregido: SUPPORTED -> ADVANCE ---
        executed_second = _run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_EVIDENCE, evidence_text=CLAIM_TEXT_EVIDENCE,
            verification_llm=_make_supported_verification_llm(),
        )
        row = executed_second.runtime_result.provisional_bundle["claim_traceability_rows"][0]
        assert row["source_verdict"] == "SUPPORTED"
        assert executed_second.agent_result.requested_transition.action.value == "ADVANCE"
        assert executed_second.agent_result.requested_transition.target_stage == "08_evaluacion_experimental"

        # --- 4. Fingerprint de 07 disponible para propagar a 08 (real, no None) ---
        assert executed_second.execution_fingerprint
        from src.adapters.evaluation_fingerprint import build_evaluation_signature
        draft_path = Path(tmp) / "draft.json"
        draft_path.write_text("{}", encoding="utf-8")
        signature = build_evaluation_signature(
            experiment_id="exp_stagespec", evaluation_policy={"x": 1}, openai_model="gpt-4.1-mini",
            evaluation_ready_json_path=draft_path, upstream_fingerprint=executed_second.execution_fingerprint,
            ground_truth_text="GT.", chunks=[], traceability_rows=[], llm_judge_prompt_version="v1",
        )
        assert signature["upstream_fingerprint"] == executed_second.execution_fingerprint


if __name__ == "__main__":
    for fn in (
        test_keyerror_field_resolved,
        test_full_productive_correction_context,
        test_correction_proposal_real_fields,
        test_invalid_proposal_fails_closed,
        test_methodological_claim_to_return,
        test_full_qualitative_cycle_return_to_advance,
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
