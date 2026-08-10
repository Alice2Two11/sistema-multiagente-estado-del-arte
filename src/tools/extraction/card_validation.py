"""Card validation and deterministic row-building helpers from notebook 03.

This module preserves the characterized behavior of cells 7, 8, and 9.
It performs no LLM calls, no repair, no relevance classification, and no I/O.
"""

from __future__ import annotations

from typing import Any


CARD_LIST_FIELDS = [
    "methods_or_models",
    "method_families",
    "input_variables_or_data_sources",
    "evaluation_metrics",
    "evidence",
]

CARD_REQUIRED_FIELDS = [
    "source_filename",
    "title",
    "research_problem",
    "objective",
    "task_type",
    "target_domain",
    "methods_or_models",
    "main_results",
    "evidence",
]

SUMMARY_COLUMNS = [
    "source_filename",
    "title",
    "paper_type",
    "research_problem",
    "objective",
    "task_type",
    "target_domain",
    "target_variable_or_object",
    "temporal_horizon_or_scope",
    "methods_or_models",
    "method_families",
    "datasets_or_case_study",
    "input_variables_or_data_sources",
    "evaluation_metrics",
    "main_results",
    "reported_best_method_or_model",
    "limitations_or_gaps",
    "contribution",
    "relevance_for_state_of_art",
    "domain_specific_notes",
    "retrieved_chunk_ids",
    "num_evidence_items",
]

QUALITY_COLUMNS = [
    "source_filename",
    "title",
    "missing_fields",
    "num_missing_fields",
    "num_evidence_items",
    "num_retrieved_chunks",
    "methods_or_models",
    "method_families",
    "evaluation_metrics",
    "reported_best_method_or_model",
]


def is_bad_card(card: dict[str, Any]) -> bool:
    title = str(card.get("title", "")).strip().lower()
    return title in ["", "error", "no especificado", "nan"]


def normalize_card(
    card: dict[str, Any],
    source_filename: str,
    *,
    card_list_fields: list[str] = CARD_LIST_FIELDS,
) -> dict[str, Any]:
    card["source_filename"] = source_filename

    for field in card_list_fields:
        value = card.get(field)

        if value is None:
            card[field] = []

        elif isinstance(value, str):
            if value.strip().lower() in [
                "",
                "no especificado",
                "none",
                "nan",
            ]:
                card[field] = []
            else:
                card[field] = [value]

        elif not isinstance(value, list):
            card[field] = [value]

    return card


def list_to_str(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(
            str(item)
            for item in value
        )

    return str(value or "")


def first_available(
    card: dict[str, Any],
    *keys: str,
    default: Any = "",
) -> Any:
    for key in keys:
        value = card.get(key)

        if value not in [
            None,
            "",
            [],
            {},
        ]:
            return value

    return default


def build_summary_row(
    card: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_filename": card.get(
            "source_filename"
        ),
        "title": card.get("title"),
        "paper_type": card.get("paper_type"),
        "research_problem": card.get(
            "research_problem"
        ),
        "objective": card.get("objective"),
        "task_type": card.get("task_type"),
        "target_domain": card.get(
            "target_domain"
        ),
        "target_variable_or_object": card.get(
            "target_variable_or_object"
        ),
        "temporal_horizon_or_scope": card.get(
            "temporal_horizon_or_scope"
        ),
        "methods_or_models": list_to_str(
            card.get("methods_or_models")
        ),
        "method_families": list_to_str(
            first_available(
                card,
                "method_families",
                "method_family",
                default=[],
            )
        ),
        "datasets_or_case_study": (
            first_available(
                card,
                "datasets_or_case_study",
                "dataset_or_case_study",
            )
        ),
        "input_variables_or_data_sources": (
            list_to_str(
                first_available(
                    card,
                    "input_variables_or_data_sources",
                    "input_variables",
                    default=[],
                )
            )
        ),
        "evaluation_metrics": list_to_str(
            card.get("evaluation_metrics")
        ),
        "main_results": card.get(
            "main_results"
        ),
        "reported_best_method_or_model": (
            first_available(
                card,
                "reported_best_method_or_model",
                "reported_best_model",
            )
        ),
        "limitations_or_gaps": card.get(
            "limitations_or_gaps"
        ),
        "contribution": card.get(
            "contribution"
        ),
        "relevance_for_state_of_art": card.get(
            "relevance_for_state_of_art"
        ),
        "domain_specific_notes": card.get(
            "domain_specific_notes"
        ),
        "retrieved_chunk_ids": list_to_str(
            card.get(
                "retrieved_chunk_ids",
                [],
            )
        ),
        "num_evidence_items": (
            len(card.get("evidence", []))
            if isinstance(
                card.get("evidence"),
                list,
            )
            else 0
        ),
    }


def build_quality_row(
    card: dict[str, Any],
    *,
    required_fields: list[str] = CARD_REQUIRED_FIELDS,
) -> dict[str, Any]:
    missing_fields = []

    for field in required_fields:
        value = card.get(field)

        if value is None:
            missing_fields.append(field)

        elif (
            isinstance(value, str)
            and value.strip().lower()
            in ["", "no especificado"]
        ):
            missing_fields.append(field)

        elif (
            isinstance(value, list)
            and len(value) == 0
        ):
            missing_fields.append(field)

    evidence_count = (
        len(card.get("evidence", []))
        if isinstance(
            card.get("evidence"),
            list,
        )
        else 0
    )

    return {
        "source_filename": card.get(
            "source_filename"
        ),
        "title": card.get("title"),
        "missing_fields": "; ".join(
            missing_fields
        ),
        "num_missing_fields": len(
            missing_fields
        ),
        "num_evidence_items": (
            evidence_count
        ),
        "num_retrieved_chunks": len(
            card.get(
                "retrieved_chunk_ids",
                [],
            )
        ),
        "methods_or_models": list_to_str(
            card.get("methods_or_models")
        ),
        "method_families": list_to_str(
            card.get("method_families")
        ),
        "evaluation_metrics": list_to_str(
            card.get("evaluation_metrics")
        ),
        "reported_best_method_or_model": (
            card.get(
                "reported_best_method_or_model"
            )
        ),
    }
