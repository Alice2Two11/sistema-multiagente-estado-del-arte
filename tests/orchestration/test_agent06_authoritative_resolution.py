"""``resolve_committed_agent06_artifacts`` (src/adapters/
agent06_verification_handoff.py) leía el ``StageState`` VIGENTE de 06
(``state.stages["06_agente_redactor"]``) y la última entrada
CRONOLÓGICA de 06 en ``decision_log`` -- ambas fuentes pueden reflejar
una ejecución ESPURIA posterior a un ``HALT_STAGE`` terminal ya
comprometido en OTRA etapa (07), que sobrescribe el estado de 06 sin
haber sido nunca una continuación causal legítima de nada.

Caso real reportado: ``--start-stage 07_agente_verificador`` (parche
10) fallaba con ``AGENT07_AGENT06_STAGE_NOT_COMMITTED`` a pesar de que
``decision_log`` contiene un commit ``COMPLETED``/``ADVANCE->07``
legítimo de 06 (la entrada 36 real) -- porque una ejecución espuria de
06 POSTERIOR al ``HALT_STAGE`` real de 07 (la entrada 38, producida por
el mismo bug de control de flujo ya corregido en los parches 3-4) había
sobrescrito ``state.stages["06_agente_redactor"]`` a ``FAILED``.

Corrección: se extrajo la reconstrucción causal ya usada por el
orquestador (``_segment_decision_log``/``_reconstruct_authoritative_
frontier``, antes en ``pipeline_orchestrator.py``) a un módulo
compartido, ``src/orchestration/decision_log_frontier.py``, y se agregó
``authoritative_decision_log_entry_for_stage`` -- la misma lógica,
aplicada para encontrar el último commit causalmente válido de UNA
etapa concreta, no solo el frontier terminal global.
``resolve_committed_agent06_artifacts`` ahora usa esta función en vez
de leer ``state.stages``/la última entrada cronológica directamente.
"""

from __future__ import annotations

import json
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
        json.dumps({"active_experiment_id": "exp1", "run_id": "run1"}), encoding="utf-8"
    )
    return ensure_pipeline_state(project_dir)


@scenario("M01. Patrón real: 06 válido -> 07 HALT -> 06 espurio FAILED -- resolve_committed_agent06_artifacts encuentra el 06 causalmente válido, no el espurio")
def test_resolves_causally_valid_06_not_spurious():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        _commit(
            store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07,
            reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED,
            output_artifacts=_fake_agent06_output_artifacts(Path(tmp) / "agent06_out"),
        )
        _commit(
            store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None,
            reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED,
        )
        _commit(
            store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None,
            reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED,
        )

        state = store.load()
        # Confirma que el bug real (StageState sobrescrito) sigue
        # presente en los datos -- si no, la prueba no probaría nada.
        assert state.stages[STAGE_06].execution_status == ExecutionStatus.FAILED

        outline, result, refs, decision_id = resolve_committed_agent06_artifacts(store=store, stage_name=STAGE_06)
        assert result.execution_status == ExecutionStatus.COMPLETED
        assert result.decision.code == "DRAFT_REVISION_COMPLETED"
        assert set(refs) == set(AGENT06_REQUIRED_ARTIFACTS)


@scenario("M02. authoritative_decision_log_entry_for_stage encuentra la entrada 36-equivalente (ADVANCE), no la 38-equivalente (espuria)")
def test_authoritative_entry_lookup_directly():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)

        valid_id = _commit(store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07, reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED)
        _commit(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED)
        spurious_id = _commit(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)

        state = store.load()
        entry = authoritative_decision_log_entry_for_stage(state.decision_log, STAGE_06)
        assert entry.decision_id == valid_id
        assert entry.decision_id != spurious_id


@scenario("M03. Retry explícito de 07 (start_stage) con el patrón real 06-válido->07-HALT->06-espurio: ejecuta ÚNICAMENTE 07 con decision_id nuevo, sin tocar 06 ni fallar por AGENT07_AGENT06_STAGE_NOT_COMMITTED")
def test_start_stage_retry_no_longer_blocked_by_stale_06():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _new_store(tmp)

        _commit(
            store, stage_key=STAGE_06, action=TransitionAction.ADVANCE, target_stage=STAGE_07,
            reason_code="DRAFT_REVISION_COMPLETED", execution_status=ExecutionStatus.COMPLETED,
            output_artifacts=_fake_agent06_output_artifacts(Path(tmp) / "agent06_out"),
        )
        old_07_id = _commit(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED)
        _commit(store, stage_key=STAGE_06, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="RUNTIME_DEPENDENCY_FAILED", execution_status=ExecutionStatus.FAILED)

        calls_06 = {"n": 0}

        def build_execution_06(project_dir, attempt_number):
            calls_06["n"] += 1
            raise AssertionError("06 NO debía ejecutarse en un retry explícito de 07")

        new_decision_ids = []

        def build_execution_07(project_dir, attempt_number):
            # Aquí es donde, en producción, se llamaría a
            # resolve_committed_agent06_artifacts -- lo ejercitamos
            # directamente para confirmar que YA NO lanza
            # AGENT07_AGENT06_STAGE_NOT_COMMITTED con este historial real.
            resolve_committed_agent06_artifacts(store=store, stage_name=STAGE_06)

            class _FakeAgent:
                def execute(self, agent_input):
                    now = datetime.now(timezone.utc).isoformat()
                    return AgentResult(
                        execution_status=ExecutionStatus.FAILED, quality_status=QualityStatus.NEEDS_REVISION,
                        decision=DecisionInfo(code="AGENT07_BLOCKED", rationale="reproducido"), quality_metrics={}, warnings=(),
                        requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED"),
                        output_artifacts={}, tool_usage=ToolUsage(),
                        attempt_number=agent_input.attempt_number, started_at=now, completed_at=now, error={"code": "AGENT07_BLOCKED"},
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
        assert new_decision_ids[0] != old_07_id
        # Ya no falla por AGENT07_AGENT06_STAGE_NOT_COMMITTED -- llegó a
        # ejecutar 07 de verdad y reprodujo su propio bloqueo real.
        assert outcome.error is None or "AGENT07_AGENT06_STAGE_NOT_COMMITTED" not in str(outcome.error)


if __name__ == "__main__":
    for fn in (
        test_resolves_causally_valid_06_not_spurious,
        test_authoritative_entry_lookup_directly,
        test_start_stage_retry_no_longer_blocked_by_stale_06,
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
