"""Pruebas minimas de aceptacion del ciclo real 06 <-> 07.

Usa el patron real ya establecido en
tests/v16/test_agent06_contractual_agent_integration_v17.py
(patch de src.agents.draft_writing_agent.validate_draft_dependencies)
para atravesar DraftWritingAgent real en modo REVISION, y
_runtime_agent_result/classify_verification_transition reales de 07 con
un Agent07RuntimeResult sintetico pero con la forma exacta que produce
run_agent07_in_memory (provisional_bundle["claim_verification_records"]).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.draft_writing_agent import DraftWritingAgent
from src.config.draft_writing_policy_config import get_draft_writing_policy
from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode
from src.contracts.agent_result import ExecutionStatus, QualityStatus, TransitionAction
from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER, apply_return_with_cycle
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore

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


class FakeRuntime:
    def __init__(self, outputs=None):
        self.collection = object()
        self.outputs = list(outputs or [])
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.outputs:
            return self.outputs.pop(0)
        return {
            "section_id": "S2",
            "section_title": "Results",
            "draft_text": "Corrected statement [b.pdf | b1].",
            "claims": [{"claim": "Corrected statement", "supporting_citations": ["[b.pdf | b1]"]}],
        }

    def parse(self, raw):
        return raw if isinstance(raw, dict) else json.loads(raw)


def _sections():
    return [
        {
            "section_id": "S1",
            "section_title": "Methods",
            "section_type": "substantive",
            "requires_sources": True,
            "purpose": "Compare methods",
            "key_arguments": ["accuracy"],
            "evidence_needs": ["results"],
            "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}],
        },
        {
            "section_id": "S2",
            "section_title": "Results",
            "section_type": "substantive",
            "requires_sources": True,
            "purpose": "Compare results",
            "key_arguments": ["error"],
            "evidence_needs": ["metrics"],
            "papers_to_use": [{"source_filename": "b.pdf", "title": "B"}],
        },
    ]


def _bundle():
    chunks = pd.DataFrame(
        [
            {"source_filename": "a.pdf", "chunk_id": "a1", "text": "Accuracy reached 95% in dataset A."},
            {"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error was 1.3 units in dataset B, corrected value."},
        ]
    )
    return {
        "outline": {"title": "Draft", "topic": "T", "sections": _sections()},
        "chunks": chunks,
        "quantitative": pd.DataFrame([]),
        "dataset_summary": pd.DataFrame([{"source_filename": "a.pdf", "dataset": "A"}]),
    }


def _previous_draft():
    return {
        "sections": [
            {
                "section_id": "S1",
                "section_title": "Methods",
                "draft_text": "Original approved statement [a.pdf | a1].",
                "claims": [{"claim": "Original approved statement", "supporting_citations": ["[a.pdf | a1]"]}],
            },
            {
                "section_id": "S2",
                "section_title": "Results",
                "draft_text": "Wrong statement about error value [b.pdf | b1].",
                "claims": [{"claim": "Wrong statement about error value", "supporting_citations": ["[b.pdf | b1]"]}],
            },
        ],
        "source_draft_fingerprint": "fp_previous_draft",
    }


def _revision_request(round_number=1):
    return {
        "schema_version": "writer_revision_request_v1",
        "experiment_id": "exp1",
        "cycle_id": "cyc1",
        "round_number": round_number,
        "source_draft_path": "draft.json",
        "source_draft_fingerprint": "fp_previous_draft",
        "verification_fingerprint": "fp_verification",
        "created_at": "2026-01-01T00:00:00Z",
        "transition_reason": "AGENT07_CORRECTABLE_ISSUES",
        "summary": "1 observacion corregible.",
        "issues": [
            {
                "issue_id": "issue_c2",
                "claim_id": "c2",
                "section_id": "S2",
                "claim_text": "Wrong statement about error value",
                "problem_type": "AUTO_CORRECTABLE",
                "verdict": "UNSUPPORTED",
                "severity": "medium",
                "hallucination_risk": "MEDIUM",
                "correction_needed": True,
                "source_filename": "b.pdf",
                "chunk_id": "b1",
                "evidence_text": "Error was 1.3 units in dataset B, corrected value.",
                "citation": "[b.pdf | b1]",
                "requested_change": "Ajustar el valor de error segun el chunk citado.",
                "requested_change_is_fallback": True,
                "constraints": "No modificar claims aprobados de otras secciones.",
                "correctable": True,
            }
        ],
    }


def _make_agent_input(output_dir, policy_overrides):
    policy = get_draft_writing_policy(
        {"retrieval_strategy": "legacy_chroma_then_csv_restricted", "top_k_evidence_per_section": 8}
    )
    policy.update(
        {
            "target_total_words": 500,
            "min_total_words": 1,
            "max_total_words": 5000,
            "output_language": "espanol",
            "writing_mode": "sintesis critica",
            "focus_mode": "comparativo",
            "citation_style": "trazable",
            "current_fingerprint": "fp-test",
        }
    )
    policy.update(policy_overrides)
    return AgentInput(
        experiment_id="exp1",
        run_id="run1",
        stage_name="06_agente_redactor",
        attempt_number=1,
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("double",), output_directory=str(output_dir)),
        dependencies={},
        policy=policy,
        previous_attempt=None,
    )


@scenario("C01. REVISION: seccion sin issues se preserva byte a byte")
def test_revision_preserves_unaffected_section():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "draft"
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        agent_input = _make_agent_input(
            output_dir,
            {
                "mode": "REVISION",
                "writer_revision_request": _revision_request(),
                "previous_draft": _previous_draft(),
                "round_number": 1,
            },
        )
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=_bundle()), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence",
            return_value=[{"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error was 1.3 units in dataset B, corrected value."}],
        ):
            result = agent.execute(agent_input)

        assert result.execution_status == ExecutionStatus.COMPLETED
        revised = json.loads((output_dir / "revised_draft.json").read_text(encoding="utf-8"))
        s1 = next(s for s in revised["sections"] if s["section_id"] == "S1")
        assert s1["draft_text"] == "Original approved statement [a.pdf | a1]."


@scenario("C02. REVISION: solo la seccion afectada se regenera, con el issue en el prompt")
def test_revision_regenerates_only_affected_section():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "draft"
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        agent_input = _make_agent_input(
            output_dir,
            {
                "mode": "REVISION",
                "writer_revision_request": _revision_request(),
                "previous_draft": _previous_draft(),
                "round_number": 1,
            },
        )
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=_bundle()), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence",
            return_value=[{"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error was 1.3 units in dataset B, corrected value."}],
        ):
            result = agent.execute(agent_input)

        assert len(runtime.prompts) == 1
        assert "issue_c2" in runtime.prompts[0]
        assert "MODO REVISI" in runtime.prompts[0]

        revised = json.loads((output_dir / "revised_draft.json").read_text(encoding="utf-8"))
        s2 = next(s for s in revised["sections"] if s["section_id"] == "S2")
        assert s2["draft_text"] == "Corrected statement [b.pdf | b1]."

        changelog = json.loads((output_dir / "revision_changelog.json").read_text(encoding="utf-8"))
        actions = {c["section_id"]: c["action"] for c in changelog}
        assert actions["S1"] == "PRESERVED"
        assert actions["S2"].startswith("REVISED")

        matrix = json.loads((output_dir / "revision_resolution_matrix.json").read_text(encoding="utf-8"))
        assert matrix[0]["issue_id"] == "issue_c2"
        assert matrix[0]["result"] == "RESOLVED"

        assert result.requested_transition.action == TransitionAction.ADVANCE
        assert result.requested_transition.target_stage == "07_agente_verificador"


@scenario("C03. REVISION: no aparece Ground Truth en policy ni en el prompt")
def test_revision_never_uses_ground_truth():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "draft"
        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        agent_input = _make_agent_input(
            output_dir,
            {
                "mode": "REVISION",
                "writer_revision_request": _revision_request(),
                "previous_draft": _previous_draft(),
                "round_number": 1,
            },
        )
        assert "ground_truth" not in agent_input.policy
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=_bundle()), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence",
            return_value=[{"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error was 1.3 units in dataset B, corrected value."}],
        ):
            agent.execute(agent_input)
        for prompt in runtime.prompts:
            assert "ground_truth" not in prompt.lower() or "no uses ground truth" in prompt.lower()


def _fake_runtime_result(claims, runtime_status="COMPLETED"):
    from src.adapters.verification_runtime import Agent07RuntimeResult, _base_metrics

    bundle = {
        "claim_verification_records": tuple(
            {"section_id": c.pop("section_id"), "claim_verification_result": c} for c in claims
        )
    }
    # Construye el dataclass directamente (no via create_agent07_runtime_result,
    # que exige un ProvisionalVerificationTraceabilityBundle real y completo,
    # producido normalmente por dependencies.bundle_builder -- fuera de
    # alcance reconstruir aqui). _runtime_agent_result solo lee
    # provisional_bundle/runtime_status/execution_metrics/runtime_issue_codes,
    # que si estan presentes y son correctos.
    return Agent07RuntimeResult(
        provisional_bundle=bundle,
        multi_proposal_resolution_result={"resolution_status": "OK"},
        candidate_artifact_inventory=(),
        execution_metrics=_base_metrics(claims_processed=len(claims)),
        runtime_warnings=(),
        runtime_issue_codes=(),
        runtime_error_records=(),
        blocked_runtime_audit_record=None,
        runtime_status=runtime_status,
    )


@scenario("C04. 07 real: claim corregible con evidencia -> RETURN a 06, revision_request construido")
def test_verification_runtime_emits_return_with_evidence():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [
        {"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "NO_CORRECTION_NEEDED",
         "scientific_verdict": "SUPPORTED", "hallucination_risk": "LOW", "evidence_used": ()},
        {"claim_id": "c2", "section_id": "S2", "final_correction_eligibility": "AUTO_CORRECTION_ELIGIBLE",
         "scientific_verdict": "UNSUPPORTED", "hallucination_risk": "MEDIUM",
         "evidence_used": ({"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error corregido."},)},
    ]
    runtime_result = _fake_runtime_result(claims)
    result, revision_request = _runtime_agent_result(
        runtime_result, {}, attempt_number=1,
        experiment_id="exp1", cycle_id="cyc1",
        source_draft_path="draft.json", source_draft_fingerprint="fp_draft",
        verification_fingerprint="fp_verif", rounds_used=0, max_rounds=3,
    )
    assert result.requested_transition.action == TransitionAction.RETURN
    assert result.requested_transition.target_stage == "06_agente_redactor"
    assert revision_request is not None
    assert revision_request["issues"][0]["claim_id"] == "c2"


@scenario("C05. 07 real: todos NO_CORRECTION_NEEDED -> ADVANCE a 08")
def test_verification_runtime_emits_advance():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [{"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "NO_CORRECTION_NEEDED",
               "scientific_verdict": "SUPPORTED", "hallucination_risk": "LOW", "evidence_used": ()}]
    runtime_result = _fake_runtime_result(claims)
    result, revision_request = _runtime_agent_result(
        runtime_result, {}, attempt_number=1, rounds_used=0, max_rounds=3,
    )
    assert result.requested_transition.action == TransitionAction.ADVANCE
    assert result.requested_transition.target_stage == "08_evaluacion_experimental"
    assert revision_request is None


@scenario("C06. 07 real: elegibilidad desconocida -> HALT_STAGE, sin revision_request")
def test_verification_runtime_unknown_eligibility_halts():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [{"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "SOMETHING_NEW",
               "scientific_verdict": "SUPPORTED", "hallucination_risk": "LOW", "evidence_used": ()}]
    runtime_result = _fake_runtime_result(claims)
    result, revision_request = _runtime_agent_result(
        runtime_result, {}, attempt_number=1, rounds_used=0, max_rounds=3,
    )
    assert result.requested_transition.action == TransitionAction.HALT_STAGE
    assert revision_request is None


@scenario("C07. 07 real: issue sin evidencia -> HALT_STAGE fail-closed, sin revision_request")
def test_verification_runtime_no_evidence_halts():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [{"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "AUTO_CORRECTION_ELIGIBLE",
               "scientific_verdict": "UNSUPPORTED", "hallucination_risk": "MEDIUM", "evidence_used": ()}]
    runtime_result = _fake_runtime_result(claims)
    result, revision_request = _runtime_agent_result(
        runtime_result, {}, attempt_number=1, rounds_used=0, max_rounds=3,
    )
    assert result.requested_transition.action == TransitionAction.HALT_STAGE
    assert revision_request is None


@scenario("C08. 07 real: rondas agotadas -> HALT_STAGE aunque haya evidencia corregible")
def test_verification_runtime_max_rounds_halts():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [{"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "AUTO_CORRECTION_ELIGIBLE",
               "scientific_verdict": "UNSUPPORTED", "hallucination_risk": "MEDIUM",
               "evidence_used": ({"source_filename": "b.pdf", "chunk_id": "b1", "text": "x"},)}]
    runtime_result = _fake_runtime_result(claims)
    result, revision_request = _runtime_agent_result(
        runtime_result, {}, attempt_number=1, rounds_used=3, max_rounds=3,
    )
    assert result.requested_transition.action == TransitionAction.HALT_STAGE
    assert revision_request is None


def _new_store(tmp):
    from datetime import datetime, timezone

    store = StateStore(Path(tmp) / "pipeline_state.json")
    now = datetime.now(timezone.utc).isoformat()
    store.initialize(
        PipelineState(
            identity=PipelineIdentity(
                experiment_id="exp1", run_id="run1", created_at=now, updated_at=now, schema_version="1.0"
            )
        )
    )
    return store


@scenario("C09. CycleState real: RETURN incrementa rounds_used e invalida 06-08 sin tocar 03-05")
def test_apply_return_with_cycle_real():
    from dataclasses import replace

    from src.contracts.agent_result import ExecutionStatus as ES
    from src.state.fingerprints import build_stage_fingerprints
    from src.state.pipeline_state import StageState

    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)
        state = store.load()
        fp = build_stage_fingerprints(input_data={"x": 1}, config_data={}, dependencies_data={})
        committed_stage = StageState(execution_status=ES.COMPLETED, quality_status=QualityStatus.APPROVED, fingerprints=fp)
        stages = dict(state.stages)
        for stage_key in ("03_agente_extraccion_kb", "03B_extraccion_cuantitativa_kb",
                          "04_agente_analisis_tematico", "05_generador_esquema",
                          "06_agente_redactor", "07_agente_verificador", "08_evaluacion_experimental"):
            stages[stage_key] = committed_stage
        store.save(replace(state, stages=stages))

        outcome = apply_return_with_cycle(
            store, from_stage="07_agente_verificador", target_stage="06_agente_redactor",
            reason="AGENT07_CORRECTABLE_ISSUES", max_rounds=3, order=CANONICAL_STAGE_ORDER,
        )
        assert outcome.cycle_exhausted is False
        state = store.load()
        assert state.cycles["writer_verifier"].rounds_used == 1
        assert state.cycles["writer_verifier"].status == "ACTIVE"
        assert set(outcome.invalidated_stages) == {
            "06_agente_redactor", "07_agente_verificador", "08_evaluacion_experimental",
        }
        for early_stage in ("03_agente_extraccion_kb", "03B_extraccion_cuantitativa_kb",
                             "04_agente_analisis_tematico", "05_generador_esquema"):
            assert early_stage not in outcome.invalidated_stages


@scenario("C10. CycleState real: al agotar max_rounds no invalida y marca EXHAUSTED")
def test_apply_return_with_cycle_exhausted():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)
        for _ in range(2):
            apply_return_with_cycle(
                store, from_stage="07_agente_verificador", target_stage="06_agente_redactor",
                reason="AGENT07_CORRECTABLE_ISSUES", max_rounds=2, order=CANONICAL_STAGE_ORDER,
            )
        final_outcome = apply_return_with_cycle(
            store, from_stage="07_agente_verificador", target_stage="06_agente_redactor",
            reason="AGENT07_CORRECTABLE_ISSUES", max_rounds=2, order=CANONICAL_STAGE_ORDER,
        )
        assert final_outcome.cycle_exhausted is True
        state = store.load()
        assert state.cycles["writer_verifier"].status == "EXHAUSTED"
        assert state.cycles["writer_verifier"].rounds_used == 2


@scenario("C11. 07C no aparece en el wiring real de la transicion")
def test_no_07c_in_wiring():
    from src.adapters.verification_notebook import _runtime_agent_result

    claims = [{"claim_id": "c1", "section_id": "S1", "final_correction_eligibility": "NO_CORRECTION_NEEDED",
               "scientific_verdict": "SUPPORTED", "hallucination_risk": "LOW", "evidence_used": ()}]
    runtime_result = _fake_runtime_result(claims)
    result, _ = _runtime_agent_result(runtime_result, {}, attempt_number=1, rounds_used=0, max_rounds=3)
    assert "07C" not in str(result.requested_transition.target_stage or "")
    assert "07C" not in result.requested_transition.reason_code


@scenario("C12. Fase 07: create_round_awaiting_revision crea round_01, estado AWAITING_REVISION, no sobrescribe")
def test_round_persistence_no_overwrite():
    from src.tools.verification.cycle_round_persistence import (
        create_round_awaiting_revision,
        list_persisted_rounds,
        read_round_status,
        round_directory,
    )

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        request = _revision_request(1)
        written = create_round_awaiting_revision(
            project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
            writer_revision_request=request,
            artifacts={"writer_revision_request.json": request, "transition.json": {"action": "RETURN"}},
        )
        assert set(written) == {"writer_revision_request.json", "transition.json"}
        directory = round_directory(project_dir, "exp1", 1)
        assert (directory / "writer_revision_request.json").is_file()
        assert list_persisted_rounds(project_dir=project_dir, experiment_id="exp1") == [1]
        status = read_round_status(project_dir=project_dir, experiment_id="exp1", round_number=1)
        assert status["status"] == "AWAITING_REVISION"

        try:
            create_round_awaiting_revision(
                project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
                writer_revision_request=request, artifacts={"writer_revision_request.json": request},
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("debia lanzar FileExistsError -- no sobrescribir")


@scenario("C13. Fase 07: round_01 y round_02 coexisten sin pisarse")
def test_round_persistence_multiple_rounds():
    from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision, list_persisted_rounds

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        r1, r2 = _revision_request(1), _revision_request(2)
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id="exp1", cycle_id=r1["cycle_id"], round_number=1,
            writer_revision_request=r1, artifacts={"writer_revision_request.json": r1},
        )
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id="exp1", cycle_id=r2["cycle_id"], round_number=2,
            writer_revision_request=r2, artifacts={"writer_revision_request.json": r2},
        )
        rounds = list_persisted_rounds(project_dir=project_dir, experiment_id="exp1")
        assert rounds == [1, 2]


@scenario("C14. Modo REVISION de 06 COMPLETA (no crea) su ronda cuando 07 ya la creó")
def test_revision_mode_persists_round_on_disk():
    from src.tools.verification.cycle_round_persistence import (
        create_round_awaiting_revision,
        read_round_status,
        round_directory,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "draft"
        cycle_root = Path(tmp) / "cycle_root"
        request = _revision_request(1)

        # Fase 07 primero, como en el flujo real -- 06 nunca crea la ronda.
        create_round_awaiting_revision(
            project_dir=cycle_root, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
            writer_revision_request=request, artifacts={"writer_revision_request.json": request},
        )

        runtime = FakeRuntime()
        agent = DraftWritingAgent(runtime)
        agent_input = _make_agent_input(
            output_dir,
            {
                "mode": "REVISION",
                "writer_revision_request": request,
                "previous_draft": _previous_draft(),
                "round_number": 1,
                "cycle_project_dir": str(cycle_root),
                "experiment_id": "exp1",
            },
        )
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=_bundle()), patch(
            "src.agents.draft_writing_agent.retrieve_section_evidence",
            return_value=[{"source_filename": "b.pdf", "chunk_id": "b1", "text": "Error was 1.3 units in dataset B, corrected value."}],
        ):
            agent.execute(agent_input)

        directory = round_directory(cycle_root, "exp1", 1)
        assert (directory / "writer_revision_request.json").is_file()  # de 07, sigue ahi
        assert (directory / "revised_draft.json").is_file()  # de 06, agregado despues
        assert (directory / "revision_changelog.json").is_file()
        assert (directory / "revision_resolution_matrix.json").is_file()
        assert (directory / "fingerprint.json").is_file()
        status = read_round_status(project_dir=cycle_root, experiment_id="exp1", round_number=1)
        assert status["status"] == "REVISION_COMPLETED"


@scenario("C15. Fingerprint de 07 se propaga hacia el manifest de 08 como upstream_fingerprint (sin duplicar)")
def test_upstream_fingerprint_propagates_to_evaluation_signature():
    from src.adapters.evaluation_fingerprint import build_evaluation_signature

    with tempfile.TemporaryDirectory() as tmp:
        draft_path = Path(tmp) / "draft.json"
        draft_path.write_text("{}", encoding="utf-8")
        signature = build_evaluation_signature(
            experiment_id="exp1", evaluation_policy={"x": 1}, openai_model="gpt-4.1-mini",
            evaluation_ready_json_path=draft_path, upstream_fingerprint="fp_agent07_composite_real",
            ground_truth_text="GT.", chunks=[], traceability_rows=[],
            llm_judge_prompt_version="v1",
        )
        assert signature["upstream_fingerprint"] == "fp_agent07_composite_real"


@scenario("T01. _resolve_draft_execution_mode detecta REVISION real cuando el ciclo esta ACTIVE y la ronda AWAITING_REVISION")
def test_resolve_draft_execution_mode_detects_revision():
    from dataclasses import replace

    from src.orchestration.pipeline_orchestrator import _resolve_draft_execution_mode
    from src.state.pipeline_state import CycleState
    from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        experiment_id = "exp1"
        (project_dir / "active_experiment.json").write_text(
            json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
        )
        draft_dir = project_dir / experiment_id / "05_outputs" / "05_draft"
        draft_dir.mkdir(parents=True)
        previous_draft = _previous_draft()
        (draft_dir / "state_of_art_draft.json").write_text(json.dumps(previous_draft), encoding="utf-8")

        store = _new_store(tmp)
        state = store.load()
        cycle = CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)
        store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))

        revision_request = _revision_request(1)
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id=experiment_id, cycle_id=revision_request["cycle_id"],
            round_number=1, writer_revision_request=revision_request,
            artifacts={"writer_revision_request.json": revision_request},
        )

        overrides = _resolve_draft_execution_mode(project_dir, store)
        assert overrides is not None
        assert overrides["mode"] == "REVISION"
        assert overrides["round_number"] == 1
        assert overrides["writer_revision_request"]["issues"][0]["claim_id"] == "c2"
        assert overrides["previous_draft"]["sections"][0]["section_id"] == "S1"
        assert overrides["experiment_id"] == experiment_id


@scenario("T02. _resolve_draft_execution_mode devuelve None (INITIAL_DRAFT) cuando no hay ciclo activo")
def test_resolve_draft_execution_mode_no_cycle():
    from src.orchestration.pipeline_orchestrator import _resolve_draft_execution_mode

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        store = _new_store(tmp)
        assert _resolve_draft_execution_mode(project_dir, store) is None


@scenario("T03. Consistencia (punto 5): writer_revision_request de OTRO experimento -> RuntimeError")
def test_resolve_draft_execution_mode_experiment_mismatch():
    from dataclasses import replace

    from src.orchestration.pipeline_orchestrator import _resolve_draft_execution_mode
    from src.state.pipeline_state import CycleState
    from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        experiment_id = "exp1"
        (project_dir / "active_experiment.json").write_text(
            json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
        )
        draft_dir = project_dir / experiment_id / "05_outputs" / "05_draft"
        draft_dir.mkdir(parents=True)
        (draft_dir / "state_of_art_draft.json").write_text(json.dumps(_previous_draft()), encoding="utf-8")

        store = _new_store(tmp)
        state = store.load()
        cycle = CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)
        store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))

        wrong_experiment_request = dict(_revision_request(1))
        wrong_experiment_request["experiment_id"] = "OTRO_EXPERIMENTO"
        # create_round_awaiting_revision usa el experiment_id de disco (exp1)
        # como directorio, pero el CONTENIDO del request declara otro --
        # exactamente el caso que _resolve_draft_execution_mode debe atrapar.
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id=experiment_id, cycle_id=wrong_experiment_request["cycle_id"],
            round_number=1, writer_revision_request=wrong_experiment_request,
            artifacts={"writer_revision_request.json": wrong_experiment_request},
        )

        try:
            _resolve_draft_execution_mode(project_dir, store)
        except RuntimeError as exc:
            assert "DRAFT_REVISION_EXPERIMENT_MISMATCH" in str(exc)
        else:
            raise AssertionError("debia lanzar RuntimeError por experimento inconsistente")


@scenario("T04. Consistencia (punto 5): writer_revision_request de OTRA ronda -> RuntimeError")
def test_resolve_draft_execution_mode_round_mismatch():
    from dataclasses import replace

    from src.orchestration.pipeline_orchestrator import _resolve_draft_execution_mode
    from src.state.pipeline_state import CycleState
    from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        experiment_id = "exp1"
        (project_dir / "active_experiment.json").write_text(
            json.dumps({"active_experiment_id": experiment_id}), encoding="utf-8"
        )
        draft_dir = project_dir / experiment_id / "05_outputs" / "05_draft"
        draft_dir.mkdir(parents=True)
        (draft_dir / "state_of_art_draft.json").write_text(json.dumps(_previous_draft()), encoding="utf-8")

        store = _new_store(tmp)
        state = store.load()
        cycle = CycleState(status="ACTIVE", rounds_used=1, max_rounds=3)
        store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))

        wrong_round_request = dict(_revision_request(1))
        wrong_round_request["round_number"] = 99
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id=experiment_id, cycle_id=wrong_round_request["cycle_id"],
            round_number=1, writer_revision_request=wrong_round_request,
            artifacts={"writer_revision_request.json": wrong_round_request},
        )

        try:
            _resolve_draft_execution_mode(project_dir, store)
        except RuntimeError as exc:
            assert "DRAFT_REVISION_ROUND_MISMATCH" in str(exc)
        else:
            raise AssertionError("debia lanzar RuntimeError por ronda inconsistente")


@scenario("T05. Fase 07: fallo de serializacion no deja una ronda a medio persistir (staging se limpia)")
def test_round_persistence_failure_leaves_no_partial_state():
    from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision, round_directory

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        request = _revision_request(1)

        class Unserializable:
            def __repr__(self):
                raise TypeError("no se puede serializar")

        try:
            create_round_awaiting_revision(
                project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
                writer_revision_request=request,
                artifacts={"writer_revision_request.json": request, "b.json": Unserializable()},
            )
        except TypeError:
            pass
        else:
            raise AssertionError("debia propagar el error de serializacion, no silenciarlo")

        directory = round_directory(project_dir, "exp1", 1)
        assert not directory.exists()
        cycle_dir = project_dir / "exp1" / "05_outputs" / "writer_verifier_cycle"
        staging_dirs = list(cycle_dir.glob(".staging_*")) if cycle_dir.exists() else []
        assert staging_dirs == []


@scenario("T06. Conflicto real: 07 crea round_01 (AWAITING_REVISION), luego 06 la completa SIN FileExistsError")
def test_seven_and_six_write_same_round_sequentially():
    """Prueba integrada obligatoria: exactamente la secuencia de 7 pasos
    pedida -- 07 persiste round_01 vía RETURN, se confirma
    AWAITING_REVISION, 06 consume el MISMO writer_revision_request, 06
    completa round_01 sin FileExistsError, los artefactos de ambos
    coexisten, el estado final es REVISION_COMPLETED, y un segundo intento
    de completar la misma ronda falla."""

    from src.tools.verification.cycle_round_persistence import (
        complete_round_revision,
        create_round_awaiting_revision,
        read_round_status,
        round_directory,
    )

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        experiment_id = "exp1"
        request = _revision_request(1)

        # 1. 07 persiste round_01 mediante RETURN (create_round_awaiting_revision).
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id=experiment_id, cycle_id=request["cycle_id"], round_number=1,
            writer_revision_request=request,
            artifacts={
                "input_draft_reference.json": {"source_draft_path": "draft.json"},
                "agent07_result.json": {"decision_code": "AGENT07_CORRECTABLE_ISSUES"},
                "writer_revision_request.json": request,
                "transition.json": {"action": "RETURN", "target_stage": "06_agente_redactor"},
                "fingerprints.json": {"verification_fingerprint": "fp_verif"},
            },
        )

        # 2. round_01 esta en AWAITING_REVISION.
        status_after_07 = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=1)
        assert status_after_07["status"] == "AWAITING_REVISION"

        # 3. 06 consume ese MISMO writer_revision_request (releido de disco,
        # no reconstruido -- prueba que el hash coincide con el real).
        directory = round_directory(project_dir, experiment_id, 1)
        consumed_request = json.loads((directory / "writer_revision_request.json").read_text(encoding="utf-8"))
        assert consumed_request == request

        # 4. 06 completa round_01 SIN FileExistsError.
        complete_round_revision(
            project_dir=project_dir, experiment_id=experiment_id, cycle_id=request["cycle_id"], round_number=1,
            writer_revision_request=consumed_request,
            artifacts={
                "revised_draft.json": {"sections": []},
                "revision_changelog.json": [{"section_id": "S2", "action": "REVISED"}],
                "revision_resolution_matrix.json": [{"issue_id": "issue_c2", "result": "RESOLVED"}],
                "unresolved_issues.json": [],
                "fingerprint.json": {"new_fingerprint": "fp_revised"},
            },
        )

        # 5. Los artefactos de 07 y 06 coexisten en la misma ronda.
        for name in ("writer_revision_request.json", "transition.json", "agent07_result.json"):
            assert (directory / name).is_file(), f"falta artefacto de 07: {name}"
        for name in ("revised_draft.json", "revision_changelog.json", "revision_resolution_matrix.json"):
            assert (directory / name).is_file(), f"falta artefacto de 06: {name}"

        # 6. El estado es REVISION_COMPLETED.
        final_status = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=1)
        assert final_status["status"] == "REVISION_COMPLETED"

        # 7. Un segundo intento de completar la misma ronda falla.
        try:
            complete_round_revision(
                project_dir=project_dir, experiment_id=experiment_id, cycle_id=request["cycle_id"], round_number=1,
                writer_revision_request=consumed_request,
                artifacts={"revised_draft.json": {"sections": []}},
            )
        except RuntimeError as exc:
            assert "DRAFT_REVISION_ROUND_ALREADY_COMPLETED" in str(exc)
        else:
            raise AssertionError("debia lanzar RuntimeError -- la ronda ya estaba completada")


@scenario("T07. complete_round_revision rechaza un writer_revision_request alterado (hash distinto)")
def test_complete_round_revision_rejects_altered_request():
    from src.tools.verification.cycle_round_persistence import complete_round_revision, create_round_awaiting_revision

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        request = _revision_request(1)
        create_round_awaiting_revision(
            project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
            writer_revision_request=request, artifacts={"writer_revision_request.json": request},
        )

        altered_request = dict(request)
        altered_request["summary"] = "resumen alterado, no es el que dejo 07"

        try:
            complete_round_revision(
                project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
                writer_revision_request=altered_request, artifacts={"revised_draft.json": {}},
            )
        except RuntimeError as exc:
            assert "DRAFT_REVISION_REQUEST_HASH_MISMATCH" in str(exc)
        else:
            raise AssertionError("debia rechazar un request alterado")


@scenario("T08. complete_round_revision exige que la ronda ya exista (07 debe crearla primero)")
def test_complete_round_revision_requires_existing_round():
    from src.tools.verification.cycle_round_persistence import complete_round_revision

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        request = _revision_request(1)
        try:
            complete_round_revision(
                project_dir=project_dir, experiment_id="exp1", cycle_id=request["cycle_id"], round_number=1,
                writer_revision_request=request, artifacts={"revised_draft.json": {}},
            )
        except FileNotFoundError as exc:
            assert "DRAFT_REVISION_ROUND_NOT_FOUND" in str(exc)
        else:
            raise AssertionError("debia exigir que la ronda ya exista")



if __name__ == "__main__":
    for fn in (
        test_revision_preserves_unaffected_section,
        test_revision_regenerates_only_affected_section,
        test_revision_never_uses_ground_truth,
        test_verification_runtime_emits_return_with_evidence,
        test_verification_runtime_emits_advance,
        test_verification_runtime_unknown_eligibility_halts,
        test_verification_runtime_no_evidence_halts,
        test_verification_runtime_max_rounds_halts,
        test_apply_return_with_cycle_real,
        test_apply_return_with_cycle_exhausted,
        test_no_07c_in_wiring,
        test_round_persistence_no_overwrite,
        test_round_persistence_multiple_rounds,
        test_revision_mode_persists_round_on_disk,
        test_upstream_fingerprint_propagates_to_evaluation_signature,
        test_resolve_draft_execution_mode_detects_revision,
        test_resolve_draft_execution_mode_no_cycle,
        test_resolve_draft_execution_mode_experiment_mismatch,
        test_resolve_draft_execution_mode_round_mismatch,
        test_round_persistence_failure_leaves_no_partial_state,
        test_seven_and_six_write_same_round_sequentially,
        test_complete_round_revision_rejects_altered_request,
        test_complete_round_revision_requires_existing_round,
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
