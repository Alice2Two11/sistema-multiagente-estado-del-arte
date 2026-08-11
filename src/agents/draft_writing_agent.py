from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import (
    AgentResult,
    AgentWarning,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
    WarningSeverity,
)
from src.state.fingerprints import sha256_file
from src.tools.draft_writing.artifacts import (
    NAMES,
    write_draft_artifacts,
    write_partial_validation,
    write_raw_section_output,
    write_raw_section_rag_trace,
    write_raw_section_validation,
)
from src.tools.draft_writing.hybrid_retrieval import retrieve_section_evidence_hybrid
from src.tools.draft_writing.input_validation import validate_draft_dependencies
from src.tools.draft_writing.normalization import (
    normalize_generated_section,
    split_sentences_preserving_citations,
)
from src.tools.draft_writing.prompting import (
    assign_section_budgets,
    build_section_prompt,
    build_source_free_organizational_section,
)
from src.tools.draft_writing.quantitative_augmentation import (
    augment_evidence_with_quantitative_chunks_greedy,
)
from src.tools.draft_writing.retrieval import (
    build_section_query,
    retrieve_section_evidence,
)
from src.tools.draft_writing.source_aware_budgets import (
    assign_source_aware_section_budgets,
)
from src.tools.draft_writing.validation import (
    CITATION_RE,
    build_draft_reports,
    count_words,
    validate_draft_global,
    validate_generated_section,
    section_allows_no_sources,
)


LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted"
HYBRID_RETRIEVAL_STRATEGY = "hybrid_chroma_csv_rrf_balanced"

LEGACY_VERSIONS = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    "validation_version": "legacy_notebook06_validation_v1",
}
HYBRID_VERSIONS = {
    "stage_version": "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE",
    "rag_version": "hybrid_chroma_csv_rrf_balanced_v1",
    "quantitative_selection_version": "confirmed_literal_greedy_coverage_v1",
    "budget_version": "source_aware_exact_total_v1",
    "validation_version": "legacy_notebook06_validation_v1",
}


class DraftWritingAgent:
    """Contractual Agent 06 with explicit legacy and V17 hybrid branches."""

    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def _section_sources(section: Mapping[str, Any]) -> list[str]:
        sources: list[str] = []
        for paper in section.get("papers_to_use") or []:
            if not isinstance(paper, Mapping):
                continue
            source = str(paper.get("source_filename", "")).strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _valid_source_chunk_pairs(chunks: pd.DataFrame) -> set[tuple[str, str]]:
        if chunks.empty or not {"source_filename", "chunk_id"}.issubset(chunks.columns):
            return set()
        return {
            (str(row["source_filename"]).strip(), str(row["chunk_id"]).strip())
            for _, row in chunks.iterrows()
            if str(row["source_filename"]).strip() and str(row["chunk_id"]).strip()
        }

    def _quant_context(
        self,
        section: Mapping[str, Any],
        bundle: Mapping[str, Any],
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        sources = set(self._section_sources(section))
        quantitative = bundle["quantitative"]
        dataset_summary = bundle["dataset_summary"]
        quantitative_rows = (
            quantitative[
                quantitative["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not quantitative.empty and "source_filename" in quantitative.columns
            else []
        )
        dataset_rows = (
            dataset_summary[
                dataset_summary["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not dataset_summary.empty and "source_filename" in dataset_summary.columns
            else []
        )
        return {
            "quantitative_results": quantitative_rows,
            "dataset_technique_summary": dataset_rows,
        }

    @staticmethod
    def _strategy(policy: Mapping[str, Any]) -> str:
        strategy = str(
            policy.get("retrieval_strategy", LEGACY_RETRIEVAL_STRATEGY)
        ).strip()
        if strategy not in {LEGACY_RETRIEVAL_STRATEGY, HYBRID_RETRIEVAL_STRATEGY}:
            raise ValueError(f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy}")
        return strategy

    @staticmethod
    def _effective_versions(
        policy: Mapping[str, Any], strategy: str
    ) -> dict[str, str]:
        del policy
        if strategy == HYBRID_RETRIEVAL_STRATEGY:
            return dict(HYBRID_VERSIONS)
        return dict(LEGACY_VERSIONS)

    @staticmethod
    def _section_budgets(
        sections: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        strategy: str,
    ) -> dict[str, dict[str, Any]]:
        target_total_words = int(policy.get("target_total_words", 1000))
        if strategy == HYBRID_RETRIEVAL_STRATEGY:
            return assign_source_aware_section_budgets(
                sections,
                target_total_words,
                policy=policy,
            )
        return assign_section_budgets(sections, target_total_words)

    def _retrieve_section_evidence(
        self,
        section: Mapping[str, Any],
        bundle: Mapping[str, Any],
        policy: Mapping[str, Any],
        strategy: str,
        quantitative_context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        chunks = bundle["chunks"]
        top_k = int(policy.get("top_k_evidence_per_section", 8))
        max_chars = int(policy.get("max_evidence_chars", 18000))
        if strategy == LEGACY_RETRIEVAL_STRATEGY:
            return retrieve_section_evidence(
                section,
                self.runtime.collection,
                chunks,
                top_k,
                max_chars,
            )

        hybrid_evidence = retrieve_section_evidence_hybrid(
            section,
            self.runtime.collection,
            chunks,
            candidate_multiplier=int(policy["candidate_multiplier"]),
            chroma_quota=int(policy["chroma_quota"]),
            csv_quota=int(policy["csv_quota"]),
            rrf_quota=int(policy["rrf_quota"]),
            rrf_k=int(policy["rrf_k"]),
            top_k_evidence_per_section=top_k,
            max_evidence_chars=max_chars,
            max_candidates_per_source=int(policy["max_candidates_per_source"]),
        )
        return augment_evidence_with_quantitative_chunks_greedy(
            hybrid_evidence,
            chunks,
            quantitative_context,
            allowed_papers=self._section_sources(section),
            top_k_evidence_per_section=top_k,
            quantitative_evidence_quota=int(
                policy.get("quantitative_evidence_quota", 0)
            ),
            max_evidence_chars=max_chars,
            max_candidates_per_source=int(policy["max_candidates_per_source"]),
            valid_source_chunk_pairs=self._valid_source_chunk_pairs(chunks),
            max_quantitative_rows_per_section=int(
                policy.get("max_quantitative_rows_per_section", 12)
            ),
        )

    @staticmethod
    def _trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
        trace_fields = (
            "source_filename",
            "chunk_id",
            "text",
            "score",
            "retrieval_method",
            "retrieval_source",
            "retrieval_sources",
            "chroma_rank",
            "csv_rank",
            "rrf_score",
            "selection_bucket",
            "selection_order",
            "quantitative_values",
            "quantitative_coverage_keys",
            "quantitative_marginal_gain",
            "quantitative_row_ids",
            "verification_statuses",
        )
        return {field: row.get(field) for field in trace_fields if field in row}

    @staticmethod
    def _unique_validation_items(items: Sequence[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @classmethod
    def _combine_section_validations(
        cls,
        original_validation: Mapping[str, Any],
        normalized_validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        combined: dict[str, Any] = dict(normalized_validation)
        for field in ("errors", "citation_errors", "claim_errors", "numeric_errors"):
            combined[field] = cls._unique_validation_items(
                list(original_validation.get(field) or [])
                + list(normalized_validation.get(field) or [])
            )
        combined["validation_ok"] = bool(
            original_validation.get("validation_ok")
            and normalized_validation.get("validation_ok")
        )
        combined["original_validation_ok"] = bool(
            original_validation.get("validation_ok")
        )
        combined["normalized_validation_ok"] = bool(
            normalized_validation.get("validation_ok")
        )
        return combined

    # ------------------------------------------------------------------
    # Conservative deterministic numeric salvage
    # ------------------------------------------------------------------

    @staticmethod
    def _unsupported_numeric_values(
        validation: Mapping[str, Any],
    ) -> list[str]:
        """
        Return unsupported numeric literals only when the section has no
        non-numeric validation failures.

        This is intentionally fail-closed: citation, claim, structural, or
        mixed numeric errors prevent salvage.
        """
        if list(validation.get("errors") or []):
            return []
        if list(validation.get("citation_errors") or []):
            return []
        if list(validation.get("claim_errors") or []):
            return []

        numeric_errors = list(validation.get("numeric_errors") or [])
        if not numeric_errors:
            return []

        prefix = "UNSUPPORTED_NUMERIC_VALUE:"
        values: list[str] = []

        for item in numeric_errors:
            text = str(item).strip()
            if not text.startswith(prefix):
                return []

            value = text[len(prefix):].strip()
            if not value:
                return []

            values.append(value)

        return values

    @staticmethod
    def _numeric_literal_pattern(raw_value: str) -> re.Pattern[str] | None:
        """
        Build an exact numeric literal matcher.

        Examples:
        40%   -> matches 40% or 40 %
        4.5   -> matches 4.5 (and text normalized from 4,5)
        20    -> does not match 120 or 20.5
        """
        value = str(raw_value or "").strip().replace(",", ".")
        if not value:
            return None

        has_percent = value.endswith("%")
        numeric = value[:-1].strip() if has_percent else value

        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
            return None

        suffix = r"\s*%" if has_percent else ""
        return re.compile(
            rf"(?<![\d.]){re.escape(numeric)}{suffix}(?![\d.])"
        )

    @classmethod
    def _contains_unsupported_numeric(
        cls,
        text: str,
        values: Sequence[str],
    ) -> bool:
        normalized_text = str(text or "").replace(",", ".")
        for value in values:
            pattern = cls._numeric_literal_pattern(value)
            if pattern is not None and pattern.search(normalized_text):
                return True
        return False

    @classmethod
    def _salvage_numeric_only_section(
        cls,
        generated: Mapping[str, Any],
        validation: Mapping[str, Any],
        allowed_pairs: set[tuple[str, str]],
    ) -> tuple[dict[str, Any], list[str], list[str]] | None:
        """
        Conservatively delete entire sentences/claims containing unsupported
        numeric literals after LLM revision attempts are exhausted.

        No value is replaced, inferred, rounded, or invented.
        """
        values = cls._unsupported_numeric_values(validation)
        if not values:
            return None

        draft_text = str(generated.get("draft_text", "") or "")
        sentences = split_sentences_preserving_citations(draft_text)
        if not sentences:
            return None

        kept_sentences: list[str] = []
        removed_sentences: list[str] = []

        for sentence in sentences:
            if cls._contains_unsupported_numeric(sentence, values):
                removed_sentences.append(sentence)
            else:
                kept_sentences.append(sentence)

        # Fail closed if no offending sentence was localized, or salvage would
        # erase the whole section.
        if not removed_sentences or not kept_sentences:
            return None

        kept_claims: list[dict[str, Any]] = []
        for claim in generated.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim", "") or "")
            if cls._contains_unsupported_numeric(claim_text, values):
                continue
            kept_claims.append(dict(claim))

        candidate = dict(generated)
        candidate["draft_text"] = " ".join(kept_sentences)
        candidate["claims"] = kept_claims

        # Re-run canonical normalization so citations and claims preserve the
        # same closed allowed-pair contract used for ordinary generation.
        candidate = normalize_generated_section(candidate, allowed_pairs)
        candidate["generation_attempt"] = generated.get("generation_attempt")

        return candidate, removed_sentences, values

    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0
        retrieval_rounds = 0
        validation_calls = 0
        out = Path(agent_input.agent_context.output_directory)
        raw_dir = out / "raw_section_outputs"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            bundle = validate_draft_dependencies(agent_input)
            policy = dict(agent_input.policy)
            strategy = self._strategy(policy)
            versions = self._effective_versions(policy, strategy)
            policy.update(versions)
            manifest_path = out / "draft_generation_manifest.json"
            reuse = False
            required_reuse = (
                "state_of_art_draft.json",
                "state_of_art_draft.md",
                "draft_sections.csv",
                "draft_rag_evidence.csv",
                "draft_quality_check.csv",
                "draft_length_check.csv",
                "draft_claim_evidence.csv",
                "numeric_hallucination_check.csv",
                "draft_validation_report.json",
                "draft_generation_manifest.json",
            )
            if manifest_path.exists() and not policy.get("force_rebuild", False):
                try:
                    old = json.loads(manifest_path.read_text())
                    report = json.loads((out / "draft_validation_report.json").read_text())
                    reuse = (
                        old.get("fingerprint") == policy.get("current_fingerprint")
                        and report.get("validation_ok") is True
                        and all((out / name).exists() for name in required_reuse)
                    )
                except Exception:
                    reuse = False

            if reuse:
                artifacts = {
                    name: ArtifactReference(str(out / name), sha256_file(out / name))
                    for name in NAMES
                    if (out / name).exists()
                }
                artifacts["raw_section_outputs"] = ArtifactReference(
                    str(raw_dir), "DIRECTORY"
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.APPROVED,
                    decision=DecisionInfo(
                        code="DRAFT_REUSED",
                        rationale="Borrador válido reutilizado con fingerprint vigente.",
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": True, "reused": True},
                    },
                    warnings=(),
                    failure_reason_codes=(),
                    requested_transition=RequestedTransition(
                        action=TransitionAction.ADVANCE,
                        target_stage=None,
                        reason_code="APPROVED",
                        requires_human_confirmation=False,
                    ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=0, llm_calls=0, validation_calls=0
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            sections = bundle["outline"].get("sections") or []
            if not isinstance(sections, list) or not sections:
                raise ValueError("INVALID_OUTLINE_SCHEMA")
            policy["outline_sections"] = sections
            policy["section_budgets"] = self._section_budgets(
                sections, policy, strategy
            )

            generated: list[dict[str, Any]] = []
            all_evidence: list[dict[str, Any]] = []
            attempt_logs: dict[str, list[dict[str, Any]]] = {}

            for section in sections:
                sid = str(section.get("section_id", "")).strip()
                section_query = build_section_query(section)
                quant_context = self._quant_context(
                    section,
                    bundle,
                    int(policy.get("max_quantitative_rows_per_section", 12)),
                )
                if (
                    strategy == HYBRID_RETRIEVAL_STRATEGY
                    and not self._section_sources(section)
                    and section_allows_no_sources(section)
                ):
                    evidence = []
                else:
                    evidence = self._retrieve_section_evidence(
                        section,
                        bundle,
                        policy,
                        strategy,
                        quant_context,
                    )
                if section.get("papers_to_use"):
                    retrieval_rounds += 1
                all_evidence.extend({"section_id": sid, **row} for row in evidence)

                if not evidence:
                    if not section_allows_no_sources(section):
                        raise ValueError(f"MISSING_SECTION_EVIDENCE:{sid}")
                    generated_section = build_source_free_organizational_section(
                        section, policy.get("output_language", "español")
                    )
                    attempt_logs[sid] = [
                        {
                            "attempt": 0,
                            "mode": "deterministic_source_free_organizational_section",
                            "validation": generated_section["section_validation"],
                        }
                    ]
                    generated.append(generated_section)
                    continue

                previous_errors: list[Any] = []
                logs: list[dict[str, Any]] = []
                accepted = None
                normalized: dict[str, Any] | None = None
                allowed: set[tuple[str, str]] = set()

                for generation_attempt in range(
                    1, int(policy.get("max_section_revision_attempts", 2)) + 2
                ):
                    prompt = build_section_prompt(
                        section,
                        evidence,
                        quant_context,
                        previous_errors,
                        policy,
                    )
                    raw = self.runtime.invoke(prompt)
                    llm_calls += 1
                    raw_path = write_raw_section_output(
                        raw_dir, sid, generation_attempt, raw
                    )
                    parsed = self.runtime.parse(raw)
                    allowed = {
                        (row["source_filename"], row["chunk_id"]) for row in evidence
                    }

                    original_validation = None
                    if strategy == HYBRID_RETRIEVAL_STRATEGY:
                        original_validation = validate_generated_section(
                            parsed, section, evidence
                        )
                        validation_calls += 1

                    normalized = normalize_generated_section(parsed, allowed)
                    normalized["generation_attempt"] = generation_attempt
                    normalized_validation = validate_generated_section(
                        normalized, section, evidence
                    )
                    validation_calls += 1

                    if original_validation is None:
                        validation = normalized_validation
                    else:
                        validation = self._combine_section_validations(
                            original_validation,
                            normalized_validation,
                        )

                    normalized["section_validation"] = validation
                    citation_errors = list(validation.get("citation_errors") or [])
                    claim_errors = list(validation.get("claim_errors") or [])
                    numeric_errors = list(validation.get("numeric_errors") or [])

                    def reason(item: Any) -> str:
                        return (
                            str(item.get("reason", ""))
                            if isinstance(item, dict)
                            else str(item)
                        )

                    validation_errors = self._unique_validation_items(
                        list(validation.get("errors") or [])
                        + citation_errors
                        + claim_errors
                        + numeric_errors
                    )
                    attempt_validation = {
                        "section_id": sid,
                        "generation_attempt": generation_attempt,
                        "validation_ok": bool(validation.get("validation_ok")),
                        "validation_errors": validation_errors,
                        "invalid_citations": [
                            item
                            for item in citation_errors
                            if reason(item)
                            in {
                                "invalid_citation",
                                "citation_not_in_section_evidence",
                                "citation_in_source_free_section",
                            }
                        ],
                        "unsupported_claims": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "missing_claim_for_sentence",
                                "claim_without_supporting_citations",
                                "claim_citation_not_in_section_evidence",
                                "claim_not_exact_sentence",
                                "substantive_sentence_missing_from_claims",
                            }
                        ],
                        "substantive_sentences_without_claim": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "missing_claim_for_sentence",
                                "substantive_sentence_missing_from_claims",
                            }
                        ],
                        "substantive_sentences_without_citation": [
                            item
                            for item in citation_errors
                            if reason(item)
                            in {
                                "uncited_substantive_sentence",
                                "substantive_sentence_without_citation",
                                "section_without_citations",
                            }
                        ],
                        "claim_sentence_mismatches": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "claim_citation_mismatch",
                                "claim_sentence_citation_mismatch",
                                "claim_not_exact_sentence",
                            }
                        ],
                        "numeric_support_errors": numeric_errors,
                        "word_count": count_words(normalized.get("draft_text", "")),
                        "citation_count": len(
                            CITATION_RE.findall(str(normalized.get("draft_text", "")))
                        ),
                        "raw_output_path": str(raw_path),
                    }
                    if strategy == HYBRID_RETRIEVAL_STRATEGY:
                        attempt_validation["original_validation"] = original_validation
                        attempt_validation["normalized_validation"] = normalized_validation
                    validation_path = write_raw_section_validation(
                        raw_dir, sid, generation_attempt, attempt_validation
                    )
                    raw_draft_text = (
                        str(parsed.get("draft_text", ""))
                        if isinstance(parsed, dict)
                        else ""
                    )
                    normalized_draft_text = str(normalized.get("draft_text", ""))
                    rag_trace = {
                        "section_id": sid,
                        "generation_attempt": generation_attempt,
                        "retrieval_strategy": strategy,
                        "query": section_query,
                        "retrieved_chunks": [self._trace_row(row) for row in evidence],
                        "allowed_citations": [
                            f"[{row.get('source_filename', '')} | {row.get('chunk_id', '')}]"
                            for row in evidence
                        ],
                        "llm_citations": CITATION_RE.findall(raw_draft_text),
                        "normalized_citations": CITATION_RE.findall(
                            normalized_draft_text
                        ),
                    }
                    rag_trace_path = write_raw_section_rag_trace(
                        raw_dir, sid, generation_attempt, rag_trace
                    )
                    logs.append(
                        {
                            "attempt": generation_attempt,
                            "validation": validation,
                            "attempt_validation_path": str(validation_path),
                            "rag_trace_path": str(rag_trace_path),
                        }
                    )
                    if validation["validation_ok"]:
                        accepted = normalized
                        break
                    previous_errors = (
                        list(validation.get("errors") or [])
                        + citation_errors
                        + claim_errors
                        + numeric_errors
                    )

                attempt_logs[sid] = logs

                # ----------------------------------------------------------
                # Final deterministic salvage:
                # only after all LLM revision attempts have been exhausted,
                # and only when the remaining failures are exclusively
                # UNSUPPORTED_NUMERIC_VALUE.
                # ----------------------------------------------------------
                if accepted is None and normalized is not None and logs:
                    last_validation = logs[-1].get("validation") or {}
                    salvage = self._salvage_numeric_only_section(
                        normalized,
                        last_validation,
                        allowed,
                    )

                    if salvage is not None:
                        salvaged_section, removed_sentences, removed_values = salvage
                        salvage_validation = validate_generated_section(
                            salvaged_section,
                            section,
                            evidence,
                        )
                        validation_calls += 1
                        salvaged_section["section_validation"] = salvage_validation

                        salvage_tag = "numeric_salvage"
                        salvage_raw_path = write_raw_section_output(
                            raw_dir,
                            sid,
                            salvage_tag,
                            json.dumps(
                                salvaged_section,
                                ensure_ascii=False,
                                indent=2,
                            ),
                        )
                        salvage_validation_payload = {
                            "section_id": sid,
                            "generation_attempt": salvage_tag,
                            "mode": "deterministic_numeric_sentence_salvage",
                            "validation_ok": bool(
                                salvage_validation.get("validation_ok")
                            ),
                            "validation_errors": self._unique_validation_items(
                                list(salvage_validation.get("errors") or [])
                                + list(
                                    salvage_validation.get("citation_errors") or []
                                )
                                + list(
                                    salvage_validation.get("claim_errors") or []
                                )
                                + list(
                                    salvage_validation.get("numeric_errors") or []
                                )
                            ),
                            "numeric_support_errors": list(
                                salvage_validation.get("numeric_errors") or []
                            ),
                            "removed_unsupported_numeric_values": removed_values,
                            "removed_sentences": removed_sentences,
                            "word_count": count_words(
                                salvaged_section.get("draft_text", "")
                            ),
                            "citation_count": len(
                                CITATION_RE.findall(
                                    str(salvaged_section.get("draft_text", ""))
                                )
                            ),
                            "raw_output_path": str(salvage_raw_path),
                        }
                        salvage_validation_path = write_raw_section_validation(
                            raw_dir,
                            sid,
                            salvage_tag,
                            salvage_validation_payload,
                        )

                        logs.append(
                            {
                                "attempt": salvage_tag,
                                "mode": "deterministic_numeric_sentence_salvage",
                                "validation": salvage_validation,
                                "attempt_validation_path": str(
                                    salvage_validation_path
                                ),
                                "removed_unsupported_numeric_values": removed_values,
                                "removed_sentences": removed_sentences,
                            }
                        )
                        attempt_logs[sid] = logs

                        if salvage_validation.get("validation_ok") is True:
                            accepted = salvaged_section

                if accepted is None:
                    last_validation = (
                        (logs[-1].get("validation") or {}) if logs else {}
                    )
                    partial_validation = {
                        "stage": "06_agente_redactor",
                        "experiment_id": agent_input.experiment_id,
                        "validation_version": policy.get("validation_version"),
                        "validation_ok": False,
                        "failed_section": sid,
                        "section_attempts": len(logs),
                        "last_attempt_errors": list(
                            last_validation.get("errors") or []
                        )
                        + list(last_validation.get("citation_errors") or [])
                        + list(last_validation.get("claim_errors") or [])
                        + list(last_validation.get("numeric_errors") or []),
                        "generation_attempts": attempt_logs,
                        "raw_section_outputs_directory": str(raw_dir),
                        "published_draft": False,
                    }
                    report_path = write_partial_validation(out, partial_validation)
                    artifacts = {
                        "draft_validation_report.json": ArtifactReference(
                            str(report_path), sha256_file(report_path)
                        ),
                        "raw_section_outputs": ArtifactReference(
                            str(raw_dir), "DIRECTORY"
                        ),
                    }
                    action = (
                        TransitionAction.RETRY
                        if agent_input.attempt_number == 1
                        else TransitionAction.HALT_STAGE
                    )
                    return AgentResult(
                        execution_status=ExecutionStatus.COMPLETED,
                        quality_status=QualityStatus.NEEDS_REVISION,
                        decision=DecisionInfo(
                            code="SECTION_VALIDATION_FAILED",
                            rationale=(
                                f"La sección {sid} agotó sus reintentos internos; "
                                "se preservaron salidas y validaciones por intento."
                            ),
                        ),
                        quality_metrics={
                            "scientific": {},
                            "technical": {
                                "validation_ok": False,
                                "reused": False,
                                "failed_section": sid,
                                "section_attempts": len(logs),
                            },
                        },
                        warnings=(
                            AgentWarning(
                                code="SECTION_VALIDATION_FAILED",
                                severity=WarningSeverity.ERROR,
                                blocking=True,
                                message=(
                                    f"La sección {sid} no superó la validación "
                                    f"tras {len(logs)} intentos."
                                ),
                            ),
                        ),
                        failure_reason_codes=("SECTION_VALIDATION_FAILED",),
                        requested_transition=RequestedTransition(
                            action=action,
                            target_stage=None,
                            reason_code="NEEDS_REVISION",
                            requires_human_confirmation=False,
                        ),
                        output_artifacts=artifacts,
                        tool_usage=ToolUsage(
                            retrieval_rounds=retrieval_rounds,
                            llm_calls=llm_calls,
                            validation_calls=validation_calls,
                        ),
                        attempt_number=agent_input.attempt_number,
                        started_at=start,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )

                generated.append(accepted)

            evidence_map: dict[str, list[dict[str, Any]]] = {}
            for row in all_evidence:
                evidence_map.setdefault(row["section_id"], []).append(
                    {key: value for key, value in row.items() if key != "section_id"}
                )
            _, quality_rows, section_rows, claim_rows, numeric_rows = (
                build_draft_reports(generated, sections, evidence_map, policy)
            )
            validation = validate_draft_global(
                generated, sections, evidence_map, policy
            )
            validation.update(
                {
                    "stage": "06_agente_redactor",
                    "experiment_id": agent_input.experiment_id,
                    "validation_version": policy.get("validation_version"),
                    "generation_attempts": attempt_logs,
                }
            )
            validation_calls += 1
            if not validation["validation_ok"]:
                path = write_partial_validation(out, validation)
                artifacts = {
                    "draft_validation_report.json": ArtifactReference(
                        str(path), sha256_file(path)
                    ),
                    "raw_section_outputs": ArtifactReference(
                        str(raw_dir), "DIRECTORY"
                    ),
                }
                action = (
                    TransitionAction.RETRY
                    if agent_input.attempt_number == 1
                    else TransitionAction.HALT_STAGE
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.NEEDS_REVISION,
                    decision=DecisionInfo(
                        code="DRAFT_VALIDATION_FAILED",
                        rationale=(
                            "El borrador no superó la validación global; "
                            "no se publicaron salidas finales."
                        ),
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": False, "reused": False},
                    },
                    warnings=(
                        AgentWarning(
                            code="INVALID_DRAFT",
                            severity=WarningSeverity.ERROR,
                            blocking=True,
                            message="La validación global fue negativa.",
                        ),
                    ),
                    failure_reason_codes=("INVALID_DRAFT",),
                    requested_transition=RequestedTransition(
                        action=action,
                        target_stage=None,
                        reason_code="NEEDS_REVISION",
                        requires_human_confirmation=False,
                    ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=retrieval_rounds,
                        llm_calls=llm_calls,
                        validation_calls=validation_calls,
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            draft = {
                "title": bundle["outline"].get(
                    "title", "Borrador del estado del arte"
                ),
                "topic": bundle["outline"].get("topic", ""),
                "status": "draft_validated_for_verification",
                "sections": generated,
                "generation_summary": {
                    "experiment_id": agent_input.experiment_id,
                    "section_count": len(generated),
                    "ground_truth_used": False,
                    "open_search_used": False,
                    "citation_format": "[source_filename | chunk_id]",
                    "retrieval_strategy": strategy,
                    **versions,
                },
            }
            manifest_versions = {
                "stage": versions["stage_version"],
                "prompt": policy.get("prompt_version"),
                "rag": versions["rag_version"],
                "validation": versions["validation_version"],
            }
            if strategy == HYBRID_RETRIEVAL_STRATEGY:
                manifest_versions.update(
                    {
                        "quantitative_selection": versions[
                            "quantitative_selection_version"
                        ],
                        "budget": versions["budget_version"],
                    }
                )
            manifest = {
                "stage": agent_input.stage_name,
                "experiment_id": agent_input.experiment_id,
                "run_id": agent_input.run_id,
                "attempt_number": agent_input.attempt_number,
                "fingerprint": policy.get("current_fingerprint"),
                "retrieval_strategy": strategy,
                "validation_ok": True,
                "safety_policy": {
                    "uses_ground_truth": False,
                    "uses_external_knowledge": False,
                    "open_search_used": False,
                },
                "counts": {
                    "sections": len(generated),
                    "llm_calls": llm_calls,
                    "retrieval_rounds": retrieval_rounds,
                },
                "versions": manifest_versions,
            }
            artifacts = write_draft_artifacts(
                out,
                draft,
                all_evidence,
                validation,
                bundle["quantitative"],
                bundle["dataset_summary"],
                manifest,
                quality_rows,
                section_rows,
                claim_rows,
                numeric_rows,
            )
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=QualityStatus.APPROVED,
                decision=DecisionInfo(
                    code="DRAFT_APPROVED",
                    rationale=(
                        "Borrador generado por secciones y validado con "
                        "evidencia restringida."
                    ),
                ),
                quality_metrics={
                    "scientific": {},
                    "technical": {"validation_ok": True, "reused": False},
                },
                warnings=(),
                failure_reason_codes=(),
                requested_transition=RequestedTransition(
                    action=TransitionAction.ADVANCE,
                    target_stage=None,
                    reason_code="APPROVED",
                    requires_human_confirmation=False,
                ),
                output_artifacts=artifacts,
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            message = str(exc)
            known = (
                "DRAFT_INPUT_NOT_FOUND",
                "OUTLINE_NOT_APPROVED",
                "OUTLINE_MANIFEST_MISMATCH",
                "GROUND_TRUTH_POLICY_VIOLATION",
                "INVALID_DRAFT_KB_SCHEMA",
                "INVALID_CHUNKS_SCHEMA",
                "INVALID_QUANTITATIVE_CONTEXT",
                "THEMATIC_NOT_APPROVED",
                "OUTLINE_MANIFEST_NOT_APPROVED",
                "THEMATIC_MANIFEST_NOT_APPROVED",
                "OUTLINE_SOURCES_NOT_VALIDATED",
                "OUTLINE_TITLES_NOT_VALIDATED",
                "CHROMA_COLLECTION_MISMATCH",
                "CHROMA_EMBEDDING_MODEL_MISMATCH",
                "UNSAFE_CHROMA_INDEX",
                "DUPLICATE_KB_SOURCE",
                "DUPLICATE_CHUNK_ID",
                "UNSAFE_CHUNKS",
                "CHROMA_CHUNK_COUNT_MISMATCH",
                "INVALID_OUTLINE_SECTION_IDS",
                "INVALID_OUTLINE_MAPPING_SCHEMA",
                "OUTLINE_MAPPING_INCONSISTENT",
                "QUANTITATIVE_MANIFEST_MISMATCH",
                "INVALID_OUTLINE_SCHEMA",
                "MISSING_SECTION_EVIDENCE",
                "SECTION_VALIDATION_FAILED",
                "INVALID_LLM_OUTPUT",
                "CREDENTIAL_NOT_FOUND",
                "ATOMIC_WRITE_FAILED",
                "UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY",
            )
            code = next((item for item in known if item in message), "RUNTIME_DEPENDENCY_FAILED")
            return AgentResult(
                execution_status=ExecutionStatus.FAILED,
                quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(
                    code="DRAFT_WRITING_FAILED",
                    rationale="Falló la ejecución del Agente Redactor.",
                ),
                quality_metrics={"scientific": {}, "technical": {}},
                warnings=(
                    AgentWarning(
                        code=code,
                        severity=WarningSeverity.ERROR,
                        blocking=True,
                        message=message,
                    ),
                ),
                failure_reason_codes=(code,),
                requested_transition=RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code=code,
                    requires_human_confirmation=False,
                ),
                output_artifacts={},
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={
                    "type": type(exc).__name__,
                    "message": message,
                    "stage": agent_input.stage_name,
                },
            )
