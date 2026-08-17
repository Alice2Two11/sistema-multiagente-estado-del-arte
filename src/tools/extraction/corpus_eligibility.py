"""Corpus eligibility gate de Stage 03 (Agent03).

Un documento individual no útil o no validable NUNCA debe detener
todo el corpus. Este módulo clasifica cada ficha en uno de tres
estados canónicos, ANTES del quality gate científico estricto de
scientific cards:

- ``INCLUDE``: documento pertinente, usable y permitido. Solo estos
  entran al quality gate científico (``build_revision_plan``) y
  pueden requerir retry por campos faltantes.
- ``EXCLUDE``: review/survey excluido por policy, fuera de scope o
  dominio excluido. Nunca entra al revision_plan, nunca bloquea
  Stage03.
- ``QUARANTINE``: título/metadata irrecuperable, contenido
  insuficiente/corrupto, o relevancia indeterminable. No se usa para
  generación, no entra al quality gate científico, queda auditado
  para revisión humana -- nunca bloquea Stage03 por sí solo.

Sin lógica nueva duplicada: esta cascada combina exclusivamente
señales YA EXISTENTES en otros módulos --

- ``is_review_excluded`` (``review_exclusion.py``): review/survey
  confirmado por tipo documental o título.
- ``relevance_level == "exclude"`` (``relevance_classification.py``,
  producido por el LLM de relevancia a partir de ``topic_profile``/
  ``excluded_domains`` del experimento -- ver ``src/prompts.py``):
  fuera de scope o dominio excluido.
- ``is_bad_card`` (``card_validation.py``): título irrecuperable.
- ``has_valid_classification`` (``relevance_classification.py``):
  relevancia indeterminable (clasificación de relevancia incompleta
  o nunca ejecutada con éxito).
- ``card.get("evidence")``: contenido/evidencia insuficiente.

Ninguna de estas señales depende de un dominio, filename ni
experimento concretos -- multidominio y genérico por construcción,
heredado de los módulos que reutiliza.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .card_validation import is_bad_card
from .relevance_classification import has_valid_classification
from .review_exclusion import is_review_excluded

INCLUDE = "INCLUDE"
EXCLUDE = "EXCLUDE"
QUARANTINE = "QUARANTINE"

# Campo canónico único: fuente de verdad de la elegibilidad de una
# ficha para el resto de Stage03 (revision_plan, KB, summary, quality,
# manifest) -- evita que cada consumidor reimplemente su propia
# cascada y diverja de las demás.
CORPUS_ELIGIBILITY_FIELD = "corpus_eligibility"

QUARANTINE_AUDIT_COLUMNS = [
    "source_filename",
    "reason",
    "title_irrecoverable",
    "content_insufficient",
    "relevance_indeterminate",
    "created_at",
]


def _normalize(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def classify_corpus_eligibility(card: Mapping[str, Any]) -> dict[str, Any]:
    """Clasifica UNA ficha en ``INCLUDE`` | ``EXCLUDE`` | ``QUARANTINE``,
    de forma puramente determinista (sin LLM), reutilizando
    exclusivamente señales ya calculadas por módulos existentes.

    Orden de la cascada (la primera señal que aplica decide):
    1. Ya excluida por policy de reviews (``is_review_excluded``).
    2. Relevancia excluida por el LLM (fuera de scope/dominio
       excluido, ``relevance_level == "exclude"``).
    3. Título irrecuperable (``is_bad_card``) -> QUARANTINE.
    4. Sin evidencia recuperada (contenido insuficiente) ->
       QUARANTINE.
    5. Relevancia indeterminable (``not has_valid_classification``) ->
       QUARANTINE.
    6. Ninguna de las anteriores -> INCLUDE.
    """

    if is_review_excluded(card):
        return {"state": EXCLUDE, "reason": "Excluida por policy de reviews (review_exclusion.py)."}

    if _normalize(card.get("relevance_level")) == "exclude":
        return {"state": EXCLUDE, "reason": "Fuera de scope o dominio excluido (relevance_level=exclude)."}

    title_irrecoverable = is_bad_card(card)
    content_insufficient = not card.get("evidence")
    relevance_indeterminate = not has_valid_classification(card)

    if title_irrecoverable or content_insufficient or relevance_indeterminate:
        reasons = []
        if title_irrecoverable:
            reasons.append("título irrecuperable")
        if content_insufficient:
            reasons.append("contenido/evidencia insuficiente")
        if relevance_indeterminate:
            reasons.append("relevancia indeterminable")
        return {
            "state": QUARANTINE, "reason": "; ".join(reasons),
            "title_irrecoverable": title_irrecoverable,
            "content_insufficient": content_insufficient,
            "relevance_indeterminate": relevance_indeterminate,
        }

    return {"state": INCLUDE, "reason": "Documento pertinente, usable y permitido."}


def is_corpus_include(card: Mapping[str, Any]) -> bool:
    """True si esta ficha YA fue marcada ``INCLUDE`` por el
    eligibility gate completo. Si el gate TODAVÍA no corrió sobre
    esta ficha (``corpus_eligibility`` ausente -- por ejemplo, el
    camino temprano del intento 1, antes de que la clasificación de
    relevancia esté disponible), cae en la única señal de exclusión
    ya calculada en ese punto: ``is_review_excluded`` (Paso 1 de
    ``review_exclusion.py``, que SIEMPRE corre antes de este chequeo
    en ``extraction_agent.py``). Esto evita que una review ya
    excluida se trate como ``INCLUDE`` solo porque el gate completo
    aún no terminó de ejecutarse -- nunca se pierde una decisión de
    exclusión ya tomada."""

    value = card.get(CORPUS_ELIGIBILITY_FIELD)
    if value is not None:
        return value == INCLUDE
    return not is_review_excluded(card)


def is_corpus_quarantined(card: Mapping[str, Any]) -> bool:
    return card.get(CORPUS_ELIGIBILITY_FIELD) == QUARANTINE


def apply_corpus_eligibility_policy(
    cards: Sequence[Mapping[str, Any]], *, created_at: str,
) -> dict[str, Any]:
    """Aplica ``classify_corpus_eligibility`` a todas las fichas,
    persistiendo el estado canónico en ``card["corpus_eligibility"]``
    -- fuente única de verdad para el resto del flujo.

    Para ``QUARANTINE``: además de marcar el campo canónico, se
    setea ``include_in_state_of_art = False`` (reutilizando el MISMO
    contrato que ya consume Stage04/``corpus_filtering.py`` -- ningún
    cambio necesario ahí) para que nunca se use en generación, sin
    inventar un mecanismo de filtrado paralelo.

    ``EXCLUDE`` no se toca aquí -- ``review_exclusion.py`` (o el LLM
    de relevancia) ya dejó ``include_in_state_of_art=False`` puesto
    antes de que esta función se invoque.

    Devuelve ``{"cards": [...], "quarantine_audit_rows": [...],
    "counts": {"include": int, "exclude": int, "quarantine": int}}``.
    """

    updated_cards: list[dict[str, Any]] = []
    quarantine_audit_rows: list[dict[str, Any]] = []
    counts = {"include": 0, "exclude": 0, "quarantine": 0}

    for card in cards:
        classification = classify_corpus_eligibility(card)
        state = classification["state"]
        new_card = dict(card)
        new_card[CORPUS_ELIGIBILITY_FIELD] = state

        if state == QUARANTINE:
            new_card["include_in_state_of_art"] = False
            counts["quarantine"] += 1
            quarantine_audit_rows.append({
                "source_filename": card.get("source_filename"),
                "reason": classification["reason"],
                "title_irrecoverable": classification.get("title_irrecoverable", False),
                "content_insufficient": classification.get("content_insufficient", False),
                "relevance_indeterminate": classification.get("relevance_indeterminate", False),
                "created_at": created_at,
            })
        elif state == EXCLUDE:
            counts["exclude"] += 1
        else:
            counts["include"] += 1

        updated_cards.append(new_card)

    return {
        "cards": updated_cards,
        "quarantine_audit_rows": quarantine_audit_rows,
        "counts": counts,
    }
