"""Exclusión determinista y auditable de reviews en Stage 03 (Agent03).

Sin LLM, sin hardcodear nombres de archivo, dominio o experimento: la
decisión depende exclusivamente de ``paper_type``/``task_type`` de la
propia ficha y de la policy activa (``extraction_policy.exclude_
reviews``). Reutiliza la abstracción YA EXISTENTE de relevancia
(``include_in_state_of_art``/``relevance_level``/``relevance_reason``,
ver ``relevance_classification.py`` y los valores reales que reconoce
``corpus_filtering.py`` en Stage 04 -- ``"exclude"``) en vez de crear
un mecanismo paralelo: una review excluida por esta política queda
marcada exactamente igual que una ficha de baja relevancia excluida
por el clasificador LLM, así que Stage 04/generación no necesitan
ningún cambio para respetarla.

Fail-closed: si la señal de tipo es incierta o contradictoria (un
campo dice "review" y el otro indica explícitamente algo distinto, ni
review ni vacío/no-especificado), NUNCA se excluye automáticamente --
la ficha sigue el camino normal de validación de campos críticos.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

REVIEW_TYPE_VALUES = {"review"}
_EMPTY_OR_UNSPECIFIED_VALUES = {"", "no especificado", "none", "nan", "n/a"}

EXCLUSION_POLICY_RULE = "exclude_reviews"
EXCLUDED_RELEVANCE_LEVEL = "exclude"

# Marcador propio, además de include_in_state_of_art/relevance_level:
# distingue "excluida por esta regla determinista de reviews" de
# "excluida por el clasificador LLM de relevancia" (que corre después
# y sobre un criterio distinto) -- necesario para que build_revision_
# plan y el cálculo de cobertura de campos críticos sepan exactamente
# cuáles fichas saltar sin depender de heurísticas sobre el texto de
# relevance_reason.
EXCLUDED_BY_RULE_FIELD = "excluded_by_policy_rule"

EXCLUSION_AUDIT_COLUMNS = [
    "source_filename",
    "detected_paper_type",
    "detected_task_type",
    "action",
    "reason",
    "policy_rule",
    "created_at",
]


def _normalize(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def classify_review_exclusion(
    card: Mapping[str, Any], *, exclude_reviews: bool,
) -> dict[str, Any]:
    """Clasifica UNA ficha, de forma puramente determinista, en
    ``"EXCLUDE"`` | ``"KEEP"`` | ``"UNCERTAIN"``.

    - ``exclude_reviews=False`` -> siempre ``"KEEP"`` (la policy no
      pide excluir nada; ninguna ficha se toca).
    - ``paper_type`` o ``task_type`` == ``"review"`` (case-insensitive,
      espacios recortados), y el OTRO campo es ese mismo valor o está
      vacío/"no especificado" (sin contradicción) -> ``"EXCLUDE"``.
    - Un campo dice ``"review"`` y el otro indica explícitamente algo
      DISTINTO de review y de vacío -> ``"UNCERTAIN"`` (contradictorio;
      fail-closed, nunca se excluye a ciegas).
    - Ningún campo indica review -> ``"KEEP"``.
    """

    if not exclude_reviews:
        return {
            "action": "KEEP", "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"),
            "reason": "Policy no solicita excluir reviews (exclude_reviews=False).",
        }

    paper_type = _normalize(card.get("paper_type"))
    task_type = _normalize(card.get("task_type"))

    paper_is_review = paper_type in REVIEW_TYPE_VALUES
    task_is_review = task_type in REVIEW_TYPE_VALUES
    paper_is_empty = paper_type in _EMPTY_OR_UNSPECIFIED_VALUES
    task_is_empty = task_type in _EMPTY_OR_UNSPECIFIED_VALUES

    if paper_is_review and not task_is_review and not task_is_empty:
        return {
            "action": "UNCERTAIN", "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"),
            "reason": (
                f"paper_type='review' pero task_type='{card.get('task_type')}' "
                "indica algo distinto -- clasificación contradictoria, no se excluye automáticamente."
            ),
        }
    if task_is_review and not paper_is_review and not paper_is_empty:
        return {
            "action": "UNCERTAIN", "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"),
            "reason": (
                f"task_type='review' pero paper_type='{card.get('paper_type')}' "
                "indica algo distinto -- clasificación contradictoria, no se excluye automáticamente."
            ),
        }

    if paper_is_review or task_is_review:
        return {
            "action": "EXCLUDE", "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"),
            "reason": "Clasificada de forma confiable como review (paper_type/task_type); excluida por policy.exclude_reviews.",
        }

    return {
        "action": "KEEP", "detected_paper_type": card.get("paper_type"),
        "detected_task_type": card.get("task_type"),
        "reason": "Ningún campo indica review.",
    }


def is_review_excluded(card: Mapping[str, Any]) -> bool:
    """True si esta ficha YA fue marcada como excluida por esta regla
    determinista concreta (nunca confunde con exclusión por baja
    relevancia del clasificador LLM, que usa el mismo ``relevance_
    level`` pero no este marcador)."""

    return card.get(EXCLUDED_BY_RULE_FIELD) == EXCLUSION_POLICY_RULE


def apply_review_exclusion_policy(
    cards: Sequence[Mapping[str, Any]], *, exclude_reviews: bool, created_at: str,
) -> dict[str, Any]:
    """Aplica ``classify_review_exclusion`` a todas las fichas.

    Para cada ficha con ``action == "EXCLUDE"``: reutiliza el contrato
    YA EXISTENTE de relevancia -- ``include_in_state_of_art = False``,
    ``relevance_level = "exclude"`` (mismo valor literal que ya
    reconoce ``corpus_filtering.py``), y agrega ``relevance_reason``
    (sin sobrescribir una razón previa no vacía) más el marcador
    ``excluded_by_policy_rule``. NUNCA rellena ni inventa
    ``methods_or_models``/``evaluation_metrics``/``main_results`` --
    esos campos quedan exactamente como la extracción los produjo.

    Fichas ``"UNCERTAIN"``/``"KEEP"`` no se tocan en absoluto.

    Devuelve ``{"cards": [...], "audit_rows": [...], "num_excluded":
    int, "num_uncertain": int}`` -- ``audit_rows`` sigue el shape de
    ``EXCLUSION_AUDIT_COLUMNS``, con una fila por ficha EXCLUDE o
    UNCERTAIN (nunca por KEEP -- evita inflar el CSV de auditoría con
    fichas sin ninguna señal de review)."""

    updated_cards: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    num_excluded = 0
    num_uncertain = 0

    for card in cards:
        classification = classify_review_exclusion(card, exclude_reviews=exclude_reviews)
        action = classification["action"]
        new_card = dict(card)

        if action == "EXCLUDE":
            new_card["include_in_state_of_art"] = False
            new_card["relevance_level"] = EXCLUDED_RELEVANCE_LEVEL
            existing_reason = str(new_card.get("relevance_reason") or "").strip()
            if not existing_reason or existing_reason.casefold() in _EMPTY_OR_UNSPECIFIED_VALUES:
                new_card["relevance_reason"] = classification["reason"]
            new_card[EXCLUDED_BY_RULE_FIELD] = EXCLUSION_POLICY_RULE
            num_excluded += 1
            audit_rows.append({
                "source_filename": card.get("source_filename"),
                "detected_paper_type": classification["detected_paper_type"],
                "detected_task_type": classification["detected_task_type"],
                "action": action,
                "reason": classification["reason"],
                "policy_rule": EXCLUSION_POLICY_RULE,
                "created_at": created_at,
            })
        elif action == "UNCERTAIN":
            num_uncertain += 1
            audit_rows.append({
                "source_filename": card.get("source_filename"),
                "detected_paper_type": classification["detected_paper_type"],
                "detected_task_type": classification["detected_task_type"],
                "action": action,
                "reason": classification["reason"],
                "policy_rule": EXCLUSION_POLICY_RULE,
                "created_at": created_at,
            })

        updated_cards.append(new_card)

    return {
        "cards": updated_cards,
        "audit_rows": audit_rows,
        "num_excluded": num_excluded,
        "num_uncertain": num_uncertain,
    }
