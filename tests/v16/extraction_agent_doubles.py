"""Test doubles for the dependency-injected extraction agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FakeResponse:
    content: str


class FakeMessage:
    def __init__(self, content: Any):
        self.content = content


class RecordingPrinter:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


class DeterministicClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return (
            "2026-07-15T12:00:"
            f"{self.calls:02d}"
        )


class FakeCollection:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.calls = []

    def query(
        self,
        *,
        query_texts,
        n_results,
        where,
    ):
        self.calls.append({
            "query_texts": list(
                query_texts
            ),
            "n_results": n_results,
            "where": dict(where),
        })
        source = where["source_filename"]
        rows = (
            self.dataframe[
                self.dataframe[
                    "source_filename"
                ]
                == source
            ]
            .sort_values("chunk_index")
            .head(n_results)
        )

        return {
            "documents": [
                [
                    str(row["text"])
                    for _, row
                    in rows.iterrows()
                ]
            ],
            "metadatas": [
                [
                    {
                        "chunk_id": str(
                            row["chunk_id"]
                        ),
                        "chunk_index": int(
                            row["chunk_index"]
                        ),
                        "source_pdf_path": str(
                            row[
                                "source_pdf_path"
                            ]
                        ),
                    }
                    for _, row
                    in rows.iterrows()
                ]
            ],
            "distances": [
                [
                    0.1
                    + index * 0.01
                    for index in range(
                        len(rows)
                    )
                ]
            ],
        }


def complete_card(
    source_filename: str,
    *,
    title: Any = None,
) -> dict[str, Any]:
    resolved_title = (
        f"Title {source_filename}"
        if title is None
        else title
    )
    return {
        "source_filename": source_filename,
        "title": resolved_title,
        "paper_type": "research",
        "research_problem": "problem",
        "objective": "objective",
        "task_type": "task",
        "target_domain": "domain",
        "target_variable_or_object": (
            "target"
        ),
        "temporal_horizon_or_scope": (
            "scope"
        ),
        "methods_or_models": ["model"],
        "method_families": ["family"],
        "datasets_or_case_study": (
            "dataset"
        ),
        "input_variables_or_data_sources": [
            "input"
        ],
        "evaluation_metrics": ["metric"],
        "main_results": "result",
        "reported_best_method_or_model": (
            "model"
        ),
        "limitations_or_gaps": (
            "limitation"
        ),
        "contribution": "contribution",
        "relevance_for_state_of_art": (
            "relevant"
        ),
        "domain_specific_notes": "notes",
        "evidence": [
            {
                "chunk_id": (
                    f"{source_filename}-0"
                )
            }
        ],
    }


class RoutedLLM:
    def __init__(
        self,
        *,
        extraction_cards=None,
        classification=None,
        extraction_errors=None,
        classification_errors=None,
        repair_cards=None,
        repaired_titles=None,
        repair_errors=None,
        title_errors=None,
    ):
        self.extraction_cards = dict(
            extraction_cards or {}
        )
        self.classification = dict(
            classification or {}
        )
        self.extraction_errors = dict(
            extraction_errors or {}
        )
        self.classification_errors = dict(
            classification_errors or {}
        )
        self.repair_cards = dict(
            repair_cards or {}
        )
        self.repaired_titles = dict(
            repaired_titles or {}
        )
        self.repair_errors = dict(
            repair_errors or {}
        )
        self.title_errors = dict(
            title_errors or {}
        )
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        prompt = messages[0].content

        if prompt.startswith(
            "EXTRACT::"
        ):
            source = prompt.split(
                "::",
                2,
            )[1]
            is_repair = (
                "IMPORTANTE FINAL:"
                in prompt
            )

            if is_repair:
                if source in self.repair_errors:
                    raise self.repair_errors[
                        source
                    ]
                card = self.repair_cards.get(
                    source,
                    complete_card(source),
                )
            else:
                if source in self.extraction_errors:
                    raise self.extraction_errors[
                        source
                    ]
                card = (
                    self.extraction_cards.get(
                        source,
                        complete_card(source),
                    )
                )

            return FakeResponse(
                json.dumps(
                    card,
                    ensure_ascii=False,
                )
            )

        if prompt.startswith(
            "RELEVANCE::"
        ):
            source = prompt.split(
                "::",
                1,
            )[1]

            if (
                source
                in self.classification_errors
            ):
                raise self.classification_errors[
                    source
                ]

            data = self.classification.get(
                source,
                {
                    "task_type": "classified",
                    "target_domain": "domain",
                    "method_families": [
                        "family"
                    ],
                    "relevance_level": "high",
                    "include_in_state_of_art": (
                        True
                    ),
                    "relevance_reason": (
                        "relevant"
                    ),
                },
            )
            return FakeResponse(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
            )

        if (
            "Extrae el título exacto"
            in prompt
        ):
            marker = "Archivo:\n"
            source = prompt.split(
                marker,
                1,
            )[1].split(
                "\n",
                1,
            )[0]

            if source in self.title_errors:
                raise self.title_errors[
                    source
                ]

            return FakeResponse(
                json.dumps({
                    "title": (
                        self.repaired_titles.get(
                            source,
                            f"Repaired {source}",
                        )
                    )
                })
            )

        raise AssertionError(
            f"Prompt no reconocido: {prompt}"
        )


class FakeRawWriter:
    def __init__(self):
        self.writes = []

    def write(
        self,
        path,
        content,
        *,
        encoding="utf-8",
    ):
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_text(
            content,
            encoding=encoding,
        )
        self.writes.append({
            "path": destination,
            "content": content,
            "encoding": encoding,
        })


def extraction_prompt_builder(
    *,
    source_filename,
    context,
    experiment_profile,
):
    return (
        f"EXTRACT::{source_filename}::"
        f"{context}"
    )


def relevance_prompt_builder(
    *,
    card,
    experiment_profile,
):
    return (
        "RELEVANCE::"
        + str(
            card["source_filename"]
        )
    )
