"""Parte B1+B2 del ciclo correctivo real `06 ↔ 07` — versión endurecida.

Deriva TODO de ``ClaimVerificationResult`` real (``src/agents/
verification_agent.py``, campo ``final_correction_eligibility`` — los 5
valores reales, confirmados en ``src/tools/verification/validation.py``,
``determine_final_correction_eligibility``): ``NO_CORRECTION_NEEDED``,
``MANUAL_REVIEW_REQUIRED``, ``NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE``,
``AUTO_CORRECTION_ELIGIBLE``, ``POTENTIALLY_AUTO_CORRECTABLE``. No se
inventa ninguna categoría nueva.

No usa un LLM para decidir la transición — es una función pura sobre los
resultados de verificación ya calculados.

Correcciones de esta ronda (fail-closed, sin aprobar/retornar sin
evidencia suficiente):

1. Un claim con elegibilidad corregible pero SIN evidencia utilizable
   (``evidence_used`` vacío y sin ``correction_proposal`` con soporte) ya
   NO produce RETURN — produce HALT_STAGE con
   ``AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT``. Un solo claim así bloquea
   todo el lote (mismo criterio que un claim en revisión manual): no se
   envían correcciones parciales mientras exista un problema que 06 no
   puede resolver de forma segura.
2. Cualquier claim con ``final_correction_eligibility`` ausente o fuera de
   los 5 valores reales conocidos produce HALT_STAGE con
   ``AGENT07_UNKNOWN_ELIGIBILITY`` — nunca cae en ADVANCE por omisión.
3. ``severity`` ya no colapsa todo lo que no es ``HIGH`` en ``medium``:
   se conserva el valor real de ``hallucination_risk`` tal cual lo produce
   07 (normalizado a minúsculas únicamente — transformación de formato,
   no de contenido), sin inventar una escala nueva.
4. ``requested_change`` usa la propuesta correctiva real
   (``claim["correction_proposal"]["requested_change"]``) cuando el
   runtime de 07 la provee. El texto genérico queda como fallback
   EXPLÍCITO, marcado con ``"requested_change_is_fallback": True``, y solo
   se usa cuando ya se confirmó evidencia suficiente (nunca sustituye a
   una propuesta real disponible).

Validaciones nuevas (ver ``_validate_claims_shape``/
``build_writer_revision_request``): ``claim_id`` presente en cada claim;
elegibilidad conocida; consistencia entre ``correctable_claim_ids``
devueltos por ``classify_verification_transition`` y los claims recibidos;
``source_draft_path`` obligatorio (no ``None``) en el artefacto final;
``issue_id`` estable (derivado de ``claim_id``, no del orden de
iteración); rechazo explícito si el artefacto quedaría con ``issues``
vacío.

Alcance de esta entrega (documentado, no oculto): estas dos piezas siguen
siendo el núcleo puro, listo para conectarse a ``verification_notebook.py``
(que hoy construye ``RequestedTransition(action=ADVANCE if completed else
HALT_STAGE, target_stage=None, ...)`` — confirmado por lectura directa, sin
ninguna rama RETURN).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Valores reales confirmados de ClaimVerificationResult.final_correction_eligibility
NO_CORRECTION_NEEDED = "NO_CORRECTION_NEEDED"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE = "NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"
AUTO_CORRECTION_ELIGIBLE = "AUTO_CORRECTION_ELIGIBLE"
POTENTIALLY_AUTO_CORRECTABLE = "POTENTIALLY_AUTO_CORRECTABLE"

KNOWN_ELIGIBILITIES = frozenset(
    {
        NO_CORRECTION_NEEDED,
        MANUAL_REVIEW_REQUIRED,
        NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE,
        AUTO_CORRECTION_ELIGIBLE,
        POTENTIALLY_AUTO_CORRECTABLE,
    }
)
CORRECTABLE_ELIGIBILITIES = frozenset({AUTO_CORRECTION_ELIGIBLE, POTENTIALLY_AUTO_CORRECTABLE})
BLOCKING_ELIGIBILITIES = frozenset(
    {MANUAL_REVIEW_REQUIRED, NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE}
)


def _has_usable_correction_support(claim: dict[str, Any]) -> bool:
    """Fail-closed: hay soporte suficiente para pedir una corrección solo
    si existe evidencia real usada (``evidence_used`` no vacío) O una
    propuesta correctiva real con su propio respaldo de evidencia
    (``correction_proposal.supporting_evidence`` no vacío). Ninguno de los
    dos se infiere ni se inventa — se lee tal cual lo entrega 07."""

    if claim.get("evidence_used"):
        return True
    proposal = claim.get("correction_proposal")
    if isinstance(proposal, dict) and proposal.get("supporting_evidence"):
        return True
    return False


def _validate_claims_shape(claims: list[dict[str, Any]]) -> str | None:
    """Devuelve un reason_code de bloqueo si algún claim está malformado
    (sin claim_id) o tiene elegibilidad desconocida; ``None`` si todos son
    válidos."""

    for claim in claims:
        if not claim.get("claim_id"):
            return "AGENT07_MALFORMED_CLAIM"
        if claim.get("final_correction_eligibility") not in KNOWN_ELIGIBILITIES:
            return "AGENT07_UNKNOWN_ELIGIBILITY"
    return None


# -----------------------------------------------------------------------
# B1. Política determinista de transición
# -----------------------------------------------------------------------


def classify_verification_transition(
    *,
    claims: list[dict[str, Any]],
    technical_status: str,
    rounds_used: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Deriva ADVANCE/RETURN/HALT_STAGE exclusivamente de los datos reales
    de cada claim (``claims``: lista de dicts con forma de
    ``ClaimVerificationResult.to_dict()``, más opcionalmente
    ``correction_proposal`` si el runtime de 07 lo produce).

    Prioridad cuando coinciden varias condiciones (mayor a menor):
    1. Fallo técnico (``technical_status != "COMPLETED"``) -> HALT_STAGE.
    2. Artefactos incompletos (``claims`` vacío) -> HALT_STAGE.
    3. Cualquier claim malformado (sin ``claim_id``) o con elegibilidad
       desconocida -> HALT_STAGE. Nunca se aprueba ni se retorna un lote
       con datos que no se pueden interpretar con certeza.
    4. Rondas agotadas -> HALT_STAGE, incluso si aún quedan problemas
       corregibles con evidencia.
    5. Cualquier claim en ``BLOCKING_ELIGIBILITIES`` -> HALT_STAGE.
    6. Cualquier claim con elegibilidad corregible pero SIN soporte de
       corrección utilizable (``_has_usable_correction_support`` es
       False) -> HALT_STAGE con ``AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT``.
       Fail-closed: no se solicita una corrección sin evidencia real.
    7. Al menos un claim corregible CON soporte (y ninguno de 1-6) ->
       RETURN a 06.
    8. Todos los claims en ``NO_CORRECTION_NEEDED`` -> ADVANCE a 08.

    Devuelve ``{"action", "reason_code", "correctable_claim_ids",
    "blocking_claim_ids", "rationale"}``.
    """

    if technical_status != "COMPLETED":
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_TECHNICAL_FAILURE",
            "correctable_claim_ids": (),
            "blocking_claim_ids": (),
            "rationale": f"Fallo técnico de 07: technical_status={technical_status!r}.",
        }

    if not claims:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_NO_CLAIMS",
            "correctable_claim_ids": (),
            "blocking_claim_ids": (),
            "rationale": "No hay claims verificados — artefactos incompletos.",
        }

    malformed_reason = _validate_claims_shape(claims)
    if malformed_reason is not None:
        return {
            "action": "HALT_STAGE",
            "reason_code": malformed_reason,
            "correctable_claim_ids": (),
            "blocking_claim_ids": (),
            "rationale": (
                "Al menos un claim no tiene claim_id"
                if malformed_reason == "AGENT07_MALFORMED_CLAIM"
                else "Al menos un claim tiene final_correction_eligibility ausente o desconocida."
            ),
        }

    if rounds_used >= max_rounds:
        return {
            "action": "HALT_STAGE",
            "reason_code": "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED",
            "correctable_claim_ids": tuple(
                c["claim_id"]
                for c in claims
                if c.get("final_correction_eligibility") in CORRECTABLE_ELIGIBILITIES
            ),
            "blocking_claim_ids": (),
            "rationale": f"Se agotaron las {max_rounds} rondas permitidas ({rounds_used} usadas).",
        }

    blocking_claim_ids = tuple(
        c["claim_id"] for c in claims if c.get("final_correction_eligibility") in BLOCKING_ELIGIBILITIES
    )
    if blocking_claim_ids:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_NON_CORRECTABLE_ISSUE",
            "correctable_claim_ids": (),
            "blocking_claim_ids": blocking_claim_ids,
            "rationale": (
                f"{len(blocking_claim_ids)} claim(s) requieren revisión manual o "
                f"no tienen evidencia disponible para corregirse: {list(blocking_claim_ids)}."
            ),
        }

    correctable_candidates = [
        c for c in claims if c.get("final_correction_eligibility") in CORRECTABLE_ELIGIBILITIES
    ]
    insufficient_evidence_ids = tuple(
        c["claim_id"] for c in correctable_candidates if not _has_usable_correction_support(c)
    )
    if insufficient_evidence_ids:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT",
            "correctable_claim_ids": (),
            "blocking_claim_ids": insufficient_evidence_ids,
            "rationale": (
                f"{len(insufficient_evidence_ids)} claim(s) están marcados como corregibles "
                "pero no tienen evidencia ni propuesta correctiva utilizable: "
                f"{list(insufficient_evidence_ids)}."
            ),
        }

    correctable_claim_ids = tuple(c["claim_id"] for c in correctable_candidates)
    if correctable_claim_ids:
        return {
            "action": "RETURN",
            "reason_code": "AGENT07_CORRECTABLE_ISSUES",
            "correctable_claim_ids": correctable_claim_ids,
            "blocking_claim_ids": (),
            "rationale": f"{len(correctable_claim_ids)} claim(s) corregibles con evidencia: {list(correctable_claim_ids)}.",
        }

    return {
        "action": "ADVANCE",
        "reason_code": "AGENT07_ALL_CLAIMS_APPROVED",
        "correctable_claim_ids": (),
        "blocking_claim_ids": (),
        "rationale": "Todos los claims están en NO_CORRECTION_NEEDED.",
    }


# -----------------------------------------------------------------------
# B2. Artefacto de retroalimentación (writer_revision_request.json)
# -----------------------------------------------------------------------

REVISION_REQUEST_SCHEMA_VERSION = "writer_revision_request_v1"

_PROBLEM_TYPE_BY_ELIGIBILITY = {
    AUTO_CORRECTION_ELIGIBLE: "AUTO_CORRECTABLE",
    POTENTIALLY_AUTO_CORRECTABLE: "POTENTIALLY_CORRECTABLE",
}

_FALLBACK_REQUESTED_CHANGE = (
    "Ajustar el claim para que sea consistente exclusivamente con la "
    "evidencia citada; no introducir información fuera de evidence_used."
)


def _issue_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    eligibility = claim.get("final_correction_eligibility")
    evidence_used = tuple(claim.get("evidence_used") or ())
    citations = tuple(
        f"[{row.get('source_filename')} | {row.get('chunk_id')}]" for row in evidence_used
    )
    proposal = claim.get("correction_proposal")
    proposal_change = (
        proposal.get("requested_change") if isinstance(proposal, dict) else None
    )
    uses_fallback = not proposal_change

    hallucination_risk = claim.get("hallucination_risk")
    severity = str(hallucination_risk).lower() if hallucination_risk else None

    return {
        # issue_id estable: derivado de claim_id, NO del orden de iteración.
        "issue_id": f"issue_{claim['claim_id']}",
        "claim_id": claim["claim_id"],
        "section_id": claim.get("section_id"),
        "claim_text": claim.get("claim_text"),
        "problem_type": _PROBLEM_TYPE_BY_ELIGIBILITY.get(eligibility, eligibility),
        "verdict": claim.get("scientific_verdict"),
        "severity": severity,  # valor real de hallucination_risk, solo normalizado a minúsculas
        "hallucination_risk": hallucination_risk,
        "correction_needed": True,
        "source_filename": evidence_used[0].get("source_filename") if evidence_used else None,
        "chunk_id": evidence_used[0].get("chunk_id") if evidence_used else None,
        "evidence_text": evidence_used[0].get("text") if evidence_used else None,
        "citation": citations[0] if citations else None,
        "requested_change": proposal_change or _FALLBACK_REQUESTED_CHANGE,
        "requested_change_is_fallback": uses_fallback,
        "constraints": (
            "No modificar claims aprobados de otras secciones. "
            "No usar Ground Truth. No usar conocimiento externo al corpus."
        ),
        "correctable": eligibility in CORRECTABLE_ELIGIBILITIES,
    }


def build_writer_revision_request(
    *,
    experiment_id: str,
    cycle_id: str,
    round_number: int,
    source_draft_path: str,
    source_draft_fingerprint: str,
    verification_fingerprint: str,
    claims: list[dict[str, Any]],
    correctable_claim_ids: tuple[str, ...],
    transition_reason: str,
) -> dict[str, Any]:
    """Construye ``writer_revision_request.json`` DERIVADO exclusivamente
    de los claims reales marcados como corregibles CON soporte suficiente
    — nunca desde Ground Truth ni desde conocimiento externo a lo que 07
    ya recuperó. Un ``issue`` por ``claim_id`` corregible, sin duplicados.

    Validaciones (fail-closed, lanzan ``ValueError`` con reason code
    explícito en el mensaje):
    - ``source_draft_path`` obligatorio (no vacío/``None``).
    - Cada id en ``correctable_claim_ids`` debe existir entre ``claims``
      (consistencia entre lo que dijo ``classify_verification_transition``
      y lo que realmente se recibió).
    - Cada claim referenciado debe tener soporte de corrección utilizable
      (mismo criterio fail-closed que B1 — defensa en profundidad si esta
      función se llama directamente, sin pasar por B1).
    - El artefacto final no puede quedar con ``issues`` vacío.
    """

    if not source_draft_path:
        raise ValueError("AGENT07_REVISION_REQUEST_MALFORMED: source_draft_path es obligatorio.")

    claims_by_id = {c.get("claim_id"): c for c in claims if c.get("claim_id")}
    correctable_set = set(correctable_claim_ids)

    missing_ids = correctable_set - set(claims_by_id)
    if missing_ids:
        raise ValueError(
            "AGENT07_REVISION_REQUEST_MALFORMED: correctable_claim_ids "
            f"referencia claim_id inexistentes en claims: {sorted(missing_ids)}."
        )

    issues = []
    for claim_id in correctable_claim_ids:  # orden de correctable_claim_ids, sin reordenar por conveniencia
        claim = claims_by_id[claim_id]
        if not _has_usable_correction_support(claim):
            raise ValueError(
                "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT: "
                f"claim_id={claim_id!r} no tiene evidencia ni propuesta correctiva utilizable."
            )
        issues.append(_issue_from_claim(claim))

    if not issues:
        raise ValueError("AGENT07_REVISION_REQUEST_MALFORMED: no hay issues que enviar a 06.")

    return {
        "schema_version": REVISION_REQUEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "cycle_id": cycle_id,
        "round_number": round_number,
        "source_draft_path": source_draft_path,
        "source_draft_fingerprint": source_draft_fingerprint,
        "verification_fingerprint": verification_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transition_reason": transition_reason,
        "summary": f"{len(issues)} observación(es) corregible(s) de verificación.",
        "issues": issues,
    }
