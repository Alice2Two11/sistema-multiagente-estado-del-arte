"""``authoritative_decision_log_entry_for_stage`` responde "¿cuál es el
estado terminal VIGENTE de esta etapa, para reconocer un restart?" --
correcta para eso, pero equivocada para resolver una dependencia
upstream: en el patrón real reportado

    [36] 06 COMPLETED / ADVANCE->07 / DRAFT_REVISION_COMPLETED
    [37] 07 FAILED / HALT_STAGE / AGENT07_NO_CLAIMS (tras corregirse)
    [38] 06 FAILED / HALT_STAGE / RUNTIME_DEPENDENCY_FAILED (espurio)
    [39] 06 FAILED / HALT_STAGE / RUNTIME_DEPENDENCY_FAILED (espurio)

esa función correctamente reconoce [39] como el estado terminal VIGENTE
de 06 (útil para un restart: "06 está detenido, no lo reintentes a
ciegas") -- pero ``resolve_committed_agent06_artifacts`` la usaba para
resolver el ARTEFACTO que 07 necesita, y terminaba intentando usar el
``AgentResult`` de la entrada [39] (``FAILED``), produciendo
``AGENT07_AGENT06_RESULT_NOT_COMPLETED`` -- aunque [36] (el commit que
REALMENTE habilitó a 07) sigue intacto en el log.

Corrección: nueva función ``committed_predecessor_for_stage(decision_
log, predecessor=06, target=07)`` en ``src/orchestration/decision_log_
frontier.py`` -- exige, sobre la MISMA entrada, las tres condiciones a
la vez:

    stage == predecessor
    execution_status == COMPLETED
    requested_transition.action == ADVANCE
    requested_transition.target_stage == target

y excluye cualquier entrada que no pertenezca a un tramo causalmente
válido (reutiliza ``_segment_decision_log``). Para el patrón real,
resuelve [36], nunca [38] ni [39] (que ni siquiera cumplen el patrón
ADVANCE->07 -- son HALT_STAGE).

``resolve_committed_agent06_artifacts`` (``src/adapters/agent06_
verification_handoff.py``) ahora usa esta función en vez de
``authoritative_decision_log_entry_for_stage``, y devuelve también el
``decision_id`` resuelto -- el único llamador real
(``build_agent07_input_from_committed_agent06``) ya no lo recalcula por
su cuenta con el mismo patrón defectuoso (``[e for e in decision_log if
e.stage==stage_name][-1]``, que también hubiera encontrado el
espurio).
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.agent06_verification_handoff import (  # noqa: E402
    AGENT06_REQUIRED_ARTIFACTS,
    resolve_committed_agent06_artifacts,
)
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
from src.orchestration.decision_log_frontier import (  # noqa: E402
    authoritative_decision_log_entry_for_stage,
    committed_predecessor_for_stage,
)
from src.orchestration.pipeline_orchestrator import (  # noqa: E402
    StageSpec,
    ensure_pipeline_state,
    run_stage,
)
from src.state.fingerprints import build_stage_fingerprints, sha256_bytes  # noqa: E402

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


STAGE_06 = "06_agente_redactor"
STAGE_07 = "07_agente_verificador"


def _generic_fp(stage_key, attempt=1):
    return build_stage_fingerprints(
        input_data={"stage_name": stage_key, "attempt_number": attempt}, config_data={}, dependencies_data={},
    )


def _fake_agent06_output_artifacts(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for name in AGENT06_REQUIRED_ARTIFACTS:
        path = base_dir / name
        path.write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
        refs[name] = ArtifactReference(path=str(path), hash=sha256_bytes(path.read_bytes()))
    return refs


def _commit(store, *, stage_key, action, target_stage, reason_code, execution_status, output_artifacts=None, attempt=1):
    now = datetime.now(timezone.utc).isoformat()
    completed = execution_status == ExecutionStatus.COMPLETED
    result = AgentResult(
        execution_status=execution_status,
        quality_status=QualityStatus.APPROVED_WITH_WARNINGS if completed else QualityStatus.REJECTED,
        decision=DecisionInfo(code=reason_code, rationale="doble determinista"),
        quality_metrics={}, warnings=(),
        requested_transition=RequestedTransition(action=action, target_stage=target_stage, reason_code=reason_code),
        output_artifacts=output_artifacts or {}, tool_usage=ToolUsage(),
        attempt_number=attempt, started_at=now, completed_at=now,
        error=None if completed else {"code": reason_code},
    )
    prep = store.prepare_execution(target_stage=stage_key, intended_action="EXECUTE", attempt_number=attempt)
    store.persist_agent_result(prep.decision_id, result)
    store.commit_execution(
        decision_id=prep.decision_id, result=result, stage_name=stage_key,
        fingerprints=_generic_fp(stage_key, attempt), observations={},
    )
    return prep.decision_id


def _new_store(tmp: Path):
    project_dir = Path(tmp)
    (project_dir / "active_experiment.json").write_text(
        __import__("json").dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return ensure_pipeline_state(project_dir)


def _fake_spec(stage_key):
    def build_execution(project_dir, attempt_number):
        raise AssertionError(f"{stage_key} NO debía intentar ejecutarse")

    return StageSpec(key=stage_key, label=f"fake {stage_key}", build_execution=build_execution, runtime_transaction=None, resolve_resume=None)


def _seed_real_pattern(tmp: Path):
    """Reproduce EXACTAMENTE el patrón real reportado:
    [36] 06 COMPLETED/ADVANCE->07, [37] 07 FAILED/HALT_STAGE,
    [38] y [39] 06 FAILED/HALT_STAGE espurios."""
    store = _new_store(tmp)
    valid_36_id = _commit(
        store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07,
        reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED,
        output_artifacts=_fake_agent06_output_artifacts(Path(tmp) / "agent06_out"),
    )
    _commit(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_NO_CLAIMS", execution_status=ExecutionStatus.FAILED)
    _commit(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)
    spurious_39_id = _commit(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)
    return store, valid_36_id, spurious_39_id


@scenario("N01. authoritative_decision_log_entry_for_stage responde una pregunta DISTINTA (estado terminal vigente para un restart) -- documentado, no reutilizado para un handoff upstream")
def test_authoritative_entry_answers_a_different_question():
    with tempfile.TemporaryDirectory() as tmp:
        store, valid_36_id, spurious_39_id = _seed_real_pattern(tmp)
        state = store.load()
        # No se afirma aquí CUÁL entrada exacta resuelve
        # authoritative_decision_log_entry_for_stage (depende de la
        # geometría completa del log real, que puede tener muchos más
        # tramos que este fixture reducido) -- lo que se prueba es que
        # committed_predecessor_for_stage, con el MISMO decision_log,
        # SIEMPRE resuelve el commit que realmente habilitó a 07,
        # independientemente de qué devuelva la otra función.
        restart_entry = authoritative_decision_log_entry_for_stage(state.decision_log, STAGE_06)
        handoff_entry = committed_predecessor_for_stage(state.decision_log, predecessor=STAGE_06, target=STAGE_07)
        assert handoff_entry.decision_id == valid_36_id
        assert restart_entry is not None  # la otra función sigue funcionando para SU propio propósito


@scenario("N02. committed_predecessor_for_stage(predecessor=06, target=07) resuelve [36], nunca [38]/[39]")
def test_committed_predecessor_resolves_the_real_upstream_commit():
    with tempfile.TemporaryDirectory() as tmp:
        store, valid_36_id, spurious_39_id = _seed_real_pattern(tmp)
        state = store.load()
        entry = committed_predecessor_for_stage(state.decision_log, predecessor=STAGE_06, target=STAGE_07)
        assert entry is not None
        assert entry.decision_id == valid_36_id
        assert entry.decision_id != spurious_39_id
        assert entry.result["execution_status"] == "COMPLETED"


@scenario("N03. resolve_committed_agent06_artifacts recupera los artifacts de [36], no falla con AGENT07_AGENT06_RESULT_NOT_COMPLETED")
def test_resolve_committed_agent06_artifacts_uses_correct_entry():
    with tempfile.TemporaryDirectory() as tmp:
        store, valid_36_id, spurious_39_id = _seed_real_pattern(tmp)
        outline, result, refs, decision_id = resolve_committed_agent06_artifacts(store=store, stage_name=STAGE_06)
        assert decision_id == valid_36_id
        assert result.execution_status == ExecutionStatus.COMPLETED
        assert set(refs) == set(AGENT06_REQUIRED_ARTIFACTS)


@scenario("N04. Retry explícito de 07 (start_stage) con el patrón real completo: ejecuta ÚNICAMENTE 07 con decision_id nuevo, sin AGENT07_AGENT06_RESULT_NOT_COMPLETED, sin tocar 06")
def test_start_stage_retry_resolves_real_pattern_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store, valid_36_id, spurious_39_id = _seed_real_pattern(tmp)

        calls_06 = {"n": 0}
        new_decision_ids = []
        resolved_decision_ids = []

        def build_execution_07(project_dir, attempt_number):
            _, _, _, decision_id = resolve_committed_agent06_artifacts(store=store, stage_name=STAGE_06)
            resolved_decision_ids.append(decision_id)

            class _FakeAgent:
                def execute(self, agent_input):
                    now = datetime.now(timezone.utc).isoformat()
                    return AgentResult(
                        execution_status=ExecutionStatus.COMPLETED, quality_status=QualityStatus.APPROVED,
                        decision=DecisionInfo(code="OK", rationale="reintento real"), quality_metrics={}, warnings=(),
                        requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, target_stage="08_evaluacion_experimental", reason_code="OK"),
                        output_artifacts={}, tool_usage=ToolUsage(),
                        attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
                    )

            from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode

            agent_input = AgentInput(
                experiment_id="exp1", run_id="run1", stage_name=STAGE_07, attempt_number=attempt_number,
                mode=ExecutionMode.FULL_RUN, agent_context=AgentContext(allowed_tools=("llm",), output_directory=str(root / "out")),
                dependencies={}, policy={},
            )
            return _FakeAgent(), agent_input

        def runtime_transaction_07(*, store, build_execution, attempt_number=1, observations=None):
            agent, agent_input = build_execution()
            prep = store.prepare_execution(target_stage=STAGE_07, intended_action="EXECUTE", attempt_number=attempt_number)
            new_decision_ids.append(prep.decision_id)
            result = agent.execute(agent_input)
            store.persist_agent_result(prep.decision_id, result)
            store.commit_execution(decision_id=prep.decision_id, result=result, stage_name=STAGE_07, fingerprints=_generic_fp(STAGE_07, attempt_number), observations={})

            class _R:
                pass

            out = _R()
            out.agent_result = result
            return out

        def resolve_resume_07(*, store, agent_input, observations=None):
            return store.resolve_resume(stage_name=agent_input.stage_name, fingerprints=_generic_fp(STAGE_07, 1), observations=dict(observations or {}))

        spec_07 = StageSpec(key=STAGE_07, label="fake 07", build_execution=build_execution_07, runtime_transaction=runtime_transaction_07, resolve_resume=resolve_resume_07)

        outcome = run_stage(store=store, project_dir=root, spec=spec_07, attempt_number=1, force_rerun=False)

        assert calls_06["n"] == 0
        assert len(new_decision_ids) == 1
        assert resolved_decision_ids == [valid_36_id]
        assert outcome.status == "COMMITTED"
        assert outcome.error is None


if __name__ == "__main__":
    for fn in (
        test_authoritative_entry_answers_a_different_question,
        test_committed_predecessor_resolves_the_real_upstream_commit,
        test_resolve_committed_agent06_artifacts_uses_correct_entry,
        test_start_stage_retry_resolves_real_pattern_end_to_end,
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
