"""Resolución fail-closed de ``pipeline_outcome`` (``SUCCESS`` |
``PARTIAL_HALT``) que decide si 07 es evaluable por 08 -- y con qué
metadatos.

Dos tipos de entrada científica admitidos, exactamente:

1. 07 ``COMPLETED`` + ``APPROVED``/``APPROVED_WITH_WARNINGS`` +
   ``ADVANCE -> 08`` => ``pipeline_outcome = "SUCCESS"``.
2. 07 ``COMPLETED`` + ``NEEDS_REVISION`` + ``HALT_STAGE`` por
   agotamiento científico/human review (ej. ``WRITER_VERIFIER_MAX_
   ROUNDS_EXHAUSTED``) => ``pipeline_outcome = "PARTIAL_HALT"`` --
   pero SOLO evaluable si se autoriza explícitamente (``allow_
   partial_halt=True``, nunca inferido ni activado por defecto): esto
   es lo que separa "camino explícito de evaluación" de "falsear una
   transición histórica 07->08" -- 07 SIGUE habiendo hecho
   ``HALT_STAGE`` en su propio ``decision_log``, nunca se reescribe ni
   reinterpreta como ``ADVANCE``.

Cualquier otra combinación -- en particular ``execution_status ==
FAILED`` (fallo técnico real de 07: runtime/contract/artifact errors)
-- nunca es evaluable, sin excepción.

Esta función es de SOLO LECTURA sobre ``decision_log`` -- nunca
modifica el estado de 06 ni de 07."""

from __future__ import annotations

from typing import Any

from src.contracts.agent_result import AgentResult, ExecutionStatus, TransitionAction
from src.orchestration.decision_log_frontier import authoritative_decision_log_entry_for_stage
from src.state.state_store import StateStore

AGENT07_STAGE_NAME = "07_agente_verificador"
AGENT08_STAGE_NAME = "08_evaluacion_experimental"

# Whitelist explícita y estrecha -- únicamente reason_codes de HALT_STAGE
# que representan agotamiento científico/human review legítimo (no un
# fallo técnico). Nunca se adivina por texto/similitud: un reason_code
# fuera de esta lista simplemente no es evaluable como PARTIAL_HALT.
SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES = frozenset({
    "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED",
})


def resolve_pipeline_outcome_for_evaluation(
    *, store: StateStore, allow_partial_halt: bool,
) -> dict[str, Any]:
    """Devuelve el dict de metadatos de evaluabilidad para 08, o lanza
    ``ValueError`` (fail-closed, reason code explícito) si 07 no es
    evaluable en absoluto.

    Lee la entrada CAUSALMENTE VÁLIDA de 07 en ``decision_log`` (nunca
    ``state.stages`` directo, que puede reflejar una ejecución espuria
    posterior -- mismo criterio ya usado para 06 en ``agent06_
    verification_handoff.py``)."""

    state = store.load()
    entry = authoritative_decision_log_entry_for_stage(state.decision_log, AGENT07_STAGE_NAME)
    if entry is None:
        raise ValueError("AGENT08_UPSTREAM_07_NOT_COMMITTED")

    result = AgentResult.from_dict(entry.result)
    if result.execution_status != ExecutionStatus.COMPLETED:
        # Fallo técnico real de 07 (runtime/contract/artifact errors) --
        # nunca evaluable, sin excepción.
        raise ValueError("AGENT08_UPSTREAM_07_TECHNICAL_FAILURE")

    transition = result.requested_transition

    if transition.action == TransitionAction.ADVANCE and transition.target_stage == AGENT08_STAGE_NAME:
        return {
            "pipeline_outcome": "SUCCESS",
            "verification_approved": True,
            "autonomous_convergence": True,
            "human_review_required": False,
            "agent07_reason_code": transition.reason_code,
            "agent07_decision_id": entry.decision_id,
        }

    if (
        transition.action == TransitionAction.HALT_STAGE
        and transition.reason_code in SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES
    ):
        if not allow_partial_halt:
            raise ValueError("AGENT08_PARTIAL_HALT_NOT_EXPLICITLY_AUTHORIZED")
        cycle = state.cycles.get("writer_verifier")
        return {
            "pipeline_outcome": "PARTIAL_HALT",
            "verification_approved": False,
            "autonomous_convergence": False,
            "human_review_required": True,
            "rounds_used": cycle.rounds_used if cycle is not None else None,
            "max_rounds": cycle.max_rounds if cycle is not None else None,
            "agent07_reason_code": transition.reason_code,
            "agent07_decision_id": entry.decision_id,
        }

    raise ValueError(
        f"AGENT08_UPSTREAM_07_NOT_EVALUABLE:{transition.action.value}:{transition.reason_code}"
    )
