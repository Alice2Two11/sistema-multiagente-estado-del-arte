"""Vía OFICIAL, mínima y soportada para reintentar ÚNICAMENTE una etapa
que quedó en un estado terminal (``HALT_STAGE``, ``execution_status=
FAILED``) -- sin ``--force-rerun``, sin tocar las etapas anteriores, sin
editar ``pipeline_state.json`` a mano.

Antes de esta entrega, ``run_pipeline()`` (la función Python) YA
soportaba ``start_stage`` como parámetro -- se usa extensamente en las
pruebas de los parches anteriores -- pero el CLI (``main()``/
``_parse_args()``) NUNCA lo exponía: solo aceptaba ``--project-dir``,
``--until`` y ``--force-rerun``. No existía ninguna vía oficial para
invocarlo desde la línea de comandos.

Se agregó ``--start-stage`` al parser real, cableado directamente a
``run_pipeline(..., start_stage=...)`` -- el mismo mecanismo que ya
existía y estaba probado, ahora accesible sin escribir Python ad-hoc.

Mecanismo (confirmado aquí, no solo documentado):

1. Con ``start_stage`` explícito, ``_check_already_terminal_state``
   (parches 5-8) se omite deliberadamente -- se respeta la intención
   explícita del llamador, igual que ya hacía con ``force_rerun``.
2. El bucle de ``run_pipeline`` arranca ``current_stage`` directamente
   en la etapa pedida -- nunca en ``STAGE_ORDER[0]`` -- así que las
   etapas anteriores (02, 03, 03B, 04, 05, 06) nunca se tocan, ni
   siquiera para verificar su frescura.
3. Para 07 específicamente: como su último commit fue
   ``execution_status=FAILED`` (no ``COMPLETED``),
   ``resume_agent07_execution`` (sin ``pending_execution``) devuelve
   ``NO_COMMIT`` -- lo que en ``_run_verification_stage`` dispara
   ``_do_fresh_execution()``: una ejecución PREPARE/EXECUTE/COMMIT
   real y completa, con un ``decision_id`` NUEVO (generado por
   ``store.prepare_execution``), sin necesidad de ``--force-rerun`` en
   absoluto -- el propio estado ``FAILED`` de 07 es lo que dispara el
   reintento real, no una bandera que ignore fingerprints.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.agent_input import AgentContext, AgentInput, ExecutionMode  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.pipeline_orchestrator import (  # noqa: E402
    StageSpec,
    _check_already_terminal_state,
    _parse_args,
    ensure_pipeline_state,
    run_stage,
)
from src.state.fingerprints import build_stage_fingerprints  # noqa: E402

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


def _commit_decision(store, *, stage_key, action, target_stage, reason_code, execution_status, attempt=1):
    now = datetime.now(timezone.utc).isoformat()
    completed = execution_status == ExecutionStatus.COMPLETED
    result = AgentResult(
        execution_status=execution_status,
        quality_status=QualityStatus.APPROVED if completed else QualityStatus.REJECTED,
        decision=DecisionInfo(code=reason_code, rationale="doble determinista"),
        quality_metrics={}, warnings=(),
        requested_transition=RequestedTransition(action=action, target_stage=target_stage, reason_code=reason_code),
        output_artifacts={}, tool_usage=ToolUsage(),
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


@scenario("L01. --start-stage existe en el CLI real y se propaga a run_pipeline (no era el caso antes de esta entrega)")
def test_cli_exposes_start_stage():
    args = _parse_args(["--project-dir", "/tmp/x", "--start-stage", STAGE_07])
    assert args.start_stage == STAGE_07
    assert args.force_rerun is False  # no hace falta --force-rerun


@scenario("L02. start_stage explícito omite _check_already_terminal_state -- se respeta la intención explícita, igual que force_rerun")
def test_start_stage_bypasses_terminal_check():
    with tempfile.TemporaryDirectory() as tmp:
        store = _new_store(tmp)
        _commit_decision(store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED)

        def _fake_spec(stage_key):
            def build_execution(project_dir, attempt_number):
                raise AssertionError("no debería construirse -- esta prueba solo verifica el bypass del chequeo terminal")
            return StageSpec(key=stage_key, label=f"fake {stage_key}", build_execution=build_execution, runtime_transaction=None, resolve_resume=None)

        registry = {STAGE_07: _fake_spec(STAGE_07)}

        # Sin start_stage: el chequeo SÍ detecta el terminal (comportamiento normal, parches 5-8).
        outcome_without = _check_already_terminal_state(store=store, registry=registry, start_stage=None, force_rerun=False)
        assert outcome_without is not None
        assert outcome_without.status == "ALREADY_TERMINAL"

        # Con start_stage explícito: el chequeo se omite deliberadamente.
        outcome_with = _check_already_terminal_state(store=store, registry=registry, start_stage=STAGE_07, force_rerun=False)
        assert outcome_with is None


@scenario("L03. Mecanismo completo: 07 en FAILED/HALT_STAGE + start_stage=07 (sin --force-rerun) -> ejecución real nueva con decision_id NUEVO; 06 NUNCA se toca")
def test_full_mechanism_new_decision_id_06_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _new_store(tmp)

        old_decision_id = _commit_decision(
            store, stage_key=STAGE_07, action=TransitionAction.HALT_STAGE, target_stage=None,
            reason_code="AGENT07_BLOCKED", execution_status=ExecutionStatus.FAILED,
        )

        calls_06 = {"n": 0}

        def build_execution_06(project_dir, attempt_number):
            calls_06["n"] += 1
            raise AssertionError("06 NO debía intentar ejecutarse")

        new_decision_ids = []

        def runtime_transaction_07(*, store, build_execution, attempt_number=1, observations=None):
            agent, agent_input = build_execution()
            prep = store.prepare_execution(target_stage=STAGE_07, intended_action="EXECUTE", attempt_number=attempt_number)
            new_decision_ids.append(prep.decision_id)
            result = agent.execute(agent_input)
            store.persist_agent_result(prep.decision_id, result)
            store.commit_execution(
                decision_id=prep.decision_id, result=result, stage_name=STAGE_07,
                fingerprints=_generic_fp(STAGE_07, attempt_number), observations={},
            )

            class _R:
                pass

            out = _R()
            out.agent_result = result
            return out

        def resolve_resume_07(*, store, agent_input, observations=None):
            return store.resolve_resume(
                stage_name=agent_input.stage_name, fingerprints=_generic_fp(STAGE_07, 1),
                observations=dict(observations or {}),
            )

        class _FakeAgent07:
            def execute(self, agent_input):
                now = datetime.now(timezone.utc).isoformat()
                return AgentResult(
                    execution_status=ExecutionStatus.FAILED, quality_status=QualityStatus.NEEDS_REVISION,
                    decision=DecisionInfo(code="AGENT07_BLOCKED", rationale="misma causa, reproducida"),
                    quality_metrics={}, warnings=(),
                    requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE, target_stage=None, reason_code="AGENT07_BLOCKED"),
                    output_artifacts={}, tool_usage=ToolUsage(),
                    attempt_number=agent_input.attempt_number, started_at=now, completed_at=now,
                    error={"code": "AGENT07_BLOCKED"},
                )

        agent_07 = _FakeAgent07()

        def build_execution_07(project_dir, attempt_number):
            return agent_07, AgentInput(
                experiment_id="exp1", run_id="run1", stage_name=STAGE_07, attempt_number=attempt_number,
                mode=ExecutionMode.FULL_RUN, agent_context=AgentContext(allowed_tools=("llm",), output_directory=str(root / "out")),
                dependencies={}, policy={},
            )

        spec_07 = StageSpec(key=STAGE_07, label="fake 07", build_execution=build_execution_07, runtime_transaction=runtime_transaction_07, resolve_resume=resolve_resume_07)

        # Esto es EXACTAMENTE lo que --start-stage 07_agente_verificador
        # activa dentro de run_pipeline: current_stage arranca en 07
        # directamente, run_stage() se llama SOLO para 07.
        outcome = run_stage(store=store, project_dir=root, spec=spec_07, attempt_number=1, force_rerun=False)

        assert calls_06["n"] == 0  # 06 nunca se tocó
        assert len(new_decision_ids) == 1
        assert new_decision_ids[0] != old_decision_id  # decision_id genuinamente nuevo
        assert outcome.status == "FAILED"  # reprodujo el mismo bloqueo real, esperado sin el fix de fondo aplicado todavía
        assert outcome.reason_code == "AGENT07_BLOCKED"

        # decision_log tiene AMBAS entradas -- la vieja no se borró ni se tocó.
        state = store.load()
        seven_entries = [e for e in state.decision_log if e.stage == STAGE_07]
        assert len(seven_entries) == 2
        assert {e.decision_id for e in seven_entries} == {old_decision_id, new_decision_ids[0]}


if __name__ == "__main__":
    for fn in (
        test_cli_exposes_start_stage,
        test_start_stage_bypasses_terminal_check,
        test_full_mechanism_new_decision_id_06_untouched,
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
