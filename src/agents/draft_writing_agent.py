from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
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
from src.tools.draft_writing.normalization import normalize_generated_section
from src.tools.draft_writing.claim_identity import (
    ClaimIdentityDeclaration,
    ClaimIdentityRecord,
    check_no_claim_uid_collisions,
    default_mint_claim_uid,
    enforce_forced_claim_uid_continuations,
    resolve_claim_identity,
)
from src.tools.draft_writing.prompting import (
    assign_section_budgets,
    build_section_prompt,
    build_section_revision_prompt,
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
from src.tools.verification.corrections import fingerprint_text
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
        """Return algorithm identity derived only from executed strategy.

        Policy version fields remain available for fingerprinting and audit,
        but cannot override the effective identity published by the agent.
        """
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
        """Return a deterministic union while preserving the first occurrence."""
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
        """Use the canonical V17 validation as the effective decision.

        The raw LLM response is still validated and preserved for auditability,
        but it must not reject a section after ``normalize_generated_section``
        has produced the canonical representation that passes every validation.
        """
        combined: dict[str, Any] = dict(normalized_validation)

        # The effective blocking errors belong to the canonical representation.
        # Raw-response errors are retained separately for diagnosis and tracing.
        for field in ("errors", "citation_errors", "claim_errors", "numeric_errors"):
            combined[field] = cls._unique_validation_items(
                list(normalized_validation.get(field) or [])
            )
            combined[f"original_{field}"] = cls._unique_validation_items(
                list(original_validation.get(field) or [])
            )

        combined["validation_ok"] = bool(
            normalized_validation.get("validation_ok")
        )
        combined["original_validation_ok"] = bool(
            original_validation.get("validation_ok")
        )
        combined["normalized_validation_ok"] = bool(
            normalized_validation.get("validation_ok")
        )
        combined["normalization_repaired_output"] = bool(
            not original_validation.get("validation_ok")
            and normalized_validation.get("validation_ok")
        )
        return combined

    def _resolve_section_claim_identities(
        self, section: dict[str, Any], *, previous_claims_by_uid: dict[str, ClaimIdentityRecord],
        round_number: int, forced_parent_uids: tuple[str, ...] = (),
    ) -> list[ClaimIdentityRecord]:
        """Resuelve claim_uid/claim_version/parent_claim_uids/etc. (ver
        ``src/tools/draft_writing/claim_identity.py``) para cada claim
        de ``section["claims"]``, en el orden en que aparecen -- el
        ``claim_id`` posicional (``f"{sid}_C{idx}"``) se sigue
        calculando igual, pero deja de ser la fuente de identidad.

        Muta ``section["claims"]`` in-place, agregando los campos de
        ``ClaimIdentityRecord`` a cada claim. Devuelve la lista de
        records resueltos, para que el llamador acumule el chequeo de
        colisión (``check_no_claim_uid_collisions``) y de continuación
        forzada (``enforce_forced_claim_uid_continuations``) a nivel de
        TODO el borrador -- nunca solo de esta sección, porque un
        claim_uid debe ser único en todo el documento, no solo dentro
        de su propia sección."""

        section_id = str(section.get("section_id", "")).strip()
        claims = section.get("claims") or []
        resolved: list[ClaimIdentityRecord] = []
        for idx, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                continue
            claim_id = f"{section_id}_C{idx}"
            claim_text = str(claim.get("claim") or "")
            action_raw = claim.get("identity_action")
            parents_raw = tuple(claim.get("parent_claim_uids") or ())
            if action_raw is None:
                # El LLM no declaró identidad (ej. sección INITIAL_DRAFT
                # sin contrato de identidad todavía, o una respuesta
                # incompleta) -- nunca se asume en silencio: se trata
                # como NEW explícito solo cuando no había NINGÚN claim
                # previo posible (INITIAL_DRAFT real); si había
                # previous_claims_by_uid disponibles y el LLM omitió el
                # campo, es un fallo de contrato real.
                if previous_claims_by_uid:
                    raise ValueError(
                        f"CLAIM_IDENTITY_MISSING_DECLARATION:{section_id}:{claim_id}"
                    )
                action_raw = "NEW"
            declaration = ClaimIdentityDeclaration(action=action_raw, parent_claim_uids=parents_raw)
            record = resolve_claim_identity(
                declaration=declaration, claim_text=claim_text, claim_id=claim_id,
                previous_claims_by_uid=previous_claims_by_uid, forced_parent_uid=None,
                round_number=round_number, text_fingerprint=fingerprint_text,
                mint_uid=default_mint_claim_uid,
            )
            claim.update(record.to_dict())
            resolved.append(record)

        enforce_forced_claim_uid_continuations(
            resolved_claims=resolved, forced_parent_uids=forced_parent_uids,
        )
        return resolved

    def execute(self, agent_input):
        policy_mode = dict(agent_input.policy).get("mode", "INITIAL_DRAFT")
        if policy_mode == "REVISION":
            return self._execute_revision(agent_input)
        return self._execute_initial_draft(agent_input)

    def _execute_revision(self, agent_input):
        """Modo REVISION (mínimo funcional): reutiliza el procesamiento
        por sección existente. Recibe en ``agent_input.policy``:
        ``writer_revision_request`` (dict real de ``build_writer_revision_
        request``), ``previous_draft`` (dict con ``sections``, el borrador
        comprometido anterior), ``round_number``. Regenera SOLO las
        secciones con issues; preserva el resto sin llamar al LLM. No usa
        Ground Truth (nunca aparece en ``policy``/``bundle`` — mismos
        insumos que el modo inicial). No regenera todo el documento
        silenciosamente: una sección sin issues nunca se reenvía al LLM.

        Limitación documentada de esta ronda: no reutiliza
        ``write_draft_artifacts``/``NAMES`` (el set completo de 10
        artefactos del modo inicial, pensado para un borrador nuevo) —
        escribe un subconjunto mínimo propio (``revised_draft.json``,
        ``revision_changelog.json``, ``revision_resolution_matrix.json``)
        en el mismo ``output_directory``. Fingerprints/PREPARE/COMMIT/
        RESUME del ciclo completo de 06-revisión NO están integrados con
        ``StateStore`` todavía — ver limitaciones del informe.
        """

        start = datetime.now(timezone.utc).isoformat()
        out = Path(agent_input.agent_context.output_directory)
        out.mkdir(parents=True, exist_ok=True)
        raw_dir = out / "raw_section_outputs"
        raw_dir.mkdir(parents=True, exist_ok=True)

        policy = dict(agent_input.policy)
        revision_request = policy["writer_revision_request"]
        previous_draft = policy["previous_draft"]
        round_number = int(policy.get("round_number", revision_request.get("round_number", 1)))

        bundle = validate_draft_dependencies(agent_input)
        strategy = self._strategy(policy)
        versions = self._effective_versions(policy, strategy)
        policy.update(versions)
        policy["round_number"] = round_number

        outline_sections = bundle["outline"].get("sections") or []
        outline_by_id = {str(s.get("section_id", "")).strip(): s for s in outline_sections}
        policy["outline_sections"] = outline_sections
        policy["section_budgets"] = self._section_budgets(outline_sections, policy, strategy)

        issues_by_section: dict[str, list[Any]] = {}
        for issue in revision_request.get("issues", ()):
            issues_by_section.setdefault(str(issue.get("section_id", "")).strip(), []).append(issue)

        previous_sections_by_id = {
            str(s.get("section_id", "")).strip(): s for s in previous_draft.get("sections", ())
        }

        generated: list[dict[str, Any]] = []
        changelog: list[dict[str, Any]] = []
        resolution_matrix: list[dict[str, Any]] = []
        unresolved_issue_ids: list[str] = []
        llm_calls = 0
        retrieval_rounds = 0

        for section_id, previous_section in previous_sections_by_id.items():
            section_issues = issues_by_section.get(section_id)

            if not section_issues:
                generated.append(dict(previous_section))
                changelog.append(
                    {"section_id": section_id, "action": "PRESERVED", "issue_ids": []}
                )
                continue

            issue_ids = [str(i.get("issue_id", "")) for i in section_issues]
            outline_section = outline_by_id.get(section_id)
            if outline_section is None:
                # La sección ya no existe en el esquema vigente -- no hay
                # forma segura de regenerarla; se preserva y se registran
                # sus issues como no resueltos, sin inventar nada.
                generated.append(dict(previous_section))
                changelog.append(
                    {"section_id": section_id, "action": "SKIPPED_NO_OUTLINE_SECTION", "issue_ids": issue_ids}
                )
                unresolved_issue_ids.extend(issue_ids)
                continue

            quant_context = self._quant_context(
                outline_section, bundle, int(policy.get("max_quantitative_rows_per_section", 12))
            )
            evidence = self._retrieve_section_evidence(
                outline_section, bundle, policy, strategy, quant_context
            )
            retrieval_rounds += 1

            if not evidence:
                generated.append(dict(previous_section))
                changelog.append(
                    {"section_id": section_id, "action": "SKIPPED_NO_EVIDENCE", "issue_ids": issue_ids}
                )
                unresolved_issue_ids.extend(issue_ids)
                continue

            # Identidad estable de claims (ver claim_identity.py):
            # solo se ofrecen como "continuables" los claims previos que
            # YA tienen claim_uid -- una sección de un experimento
            # anterior a este contrato ("legacy", requisito explícito
            # de no reconstruir identidades retroactivamente por
            # similitud) simplemente no tiene ningún claim_uid previo
            # que continuar: todo lo que el LLM genere para ella será
            # necesariamente NEW, sin intentar adivinar de dónde viene.
            previous_claims_by_uid = {
                c["claim_uid"]: ClaimIdentityRecord(
                    claim_uid=c["claim_uid"], claim_version=c["claim_version"], claim_id=c.get("claim_id", ""),
                    parent_claim_uids=tuple(c.get("parent_claim_uids") or ()),
                    claim_text_fingerprint=c.get("claim_text_fingerprint", ""),
                    created_round=c.get("created_round", 1), updated_round=c.get("updated_round", 1),
                )
                for c in (previous_section.get("claims") or [])
                if isinstance(c, dict) and c.get("claim_uid")
            }
            previous_claims_for_identity = [
                {"claim_uid": c["claim_uid"], "claim_text": str(c.get("claim", ""))}
                for c in (previous_section.get("claims") or [])
                if isinstance(c, dict) and c.get("claim_uid")
            ]
            forced_parent_uids = tuple(
                str(issue["claim_uid"]) for issue in section_issues if issue.get("claim_uid")
            )

            prompt = build_section_revision_prompt(
                outline_section,
                evidence,
                quant_context,
                policy,
                previous_section_draft_text=str(previous_section.get("draft_text", "")),
                issues=section_issues,
                previous_claims_for_identity=previous_claims_for_identity,
            )
            raw = self.runtime.invoke(prompt)
            llm_calls += 1
            write_raw_section_output(raw_dir, section_id, round_number, raw)
            parsed = self.runtime.parse(raw)

            allowed = {(row["source_filename"], row["chunk_id"]) for row in evidence}
            normalized = normalize_generated_section(parsed, allowed)
            normalized["generation_attempt"] = round_number
            validation = validate_generated_section(normalized, outline_section, evidence)
            normalized["section_validation"] = validation
            self._resolve_section_claim_identities(
                normalized, previous_claims_by_uid=previous_claims_by_uid,
                round_number=round_number, forced_parent_uids=forced_parent_uids,
            )
            generated.append(normalized)

            has_errors = bool(
                validation.get("citation_errors")
                or validation.get("claim_errors")
                or validation.get("numeric_errors")
            )
            action = "REVISED" if not has_errors else "REVISED_WITH_REMAINING_ISSUES"
            changelog.append({"section_id": section_id, "action": action, "issue_ids": issue_ids})
            evidence_refs = [f"{row['source_filename']}|{row['chunk_id']}" for row in evidence]
            claim_uid_by_issue_id = {
                str(issue.get("issue_id")): str(issue.get("claim_uid") or "")
                for issue in section_issues
            }
            for issue in section_issues:
                resolution_matrix.append(
                    {
                        "issue_id": issue.get("issue_id"),
                        "claim_uid": claim_uid_by_issue_id.get(str(issue.get("issue_id")), ""),
                        "action_taken": "SECTION_REGENERATED_LOCALIZED",
                        "section_id": section_id,
                        "evidence_used": evidence_refs,
                        "result": "RESOLVED" if not has_errors else "UNRESOLVED",
                    }
                )
            if has_errors:
                unresolved_issue_ids.extend(issue_ids)

        check_no_claim_uid_collisions(
            [
                ClaimIdentityRecord(
                    claim_uid=c["claim_uid"], claim_version=c["claim_version"], claim_id=c.get("claim_id", ""),
                    parent_claim_uids=tuple(c.get("parent_claim_uids") or ()),
                    claim_text_fingerprint=c.get("claim_text_fingerprint", ""),
                    created_round=c.get("created_round", round_number), updated_round=c.get("updated_round", round_number),
                )
                for section in generated for c in (section.get("claims") or []) if isinstance(c, dict) and c.get("claim_uid")
            ]
        )

        revised_draft = {**previous_draft, "sections": generated, "revision_round": round_number}
        revised_draft_path = out / "revised_draft.json"
        revised_draft_path.write_text(
            json.dumps(revised_draft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        changelog_path = out / "revision_changelog.json"
        changelog_path.write_text(json.dumps(changelog, ensure_ascii=False, indent=2), encoding="utf-8")
        matrix_path = out / "revision_resolution_matrix.json"
        matrix_path.write_text(
            json.dumps(resolution_matrix, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Nuevo fingerprint: borrador anterior + revision request + ronda +
        # configuración + evidencia usada (vía el propio contenido del
        # borrador revisado, que ya incorpora qué evidencia se citó).
        from src.state.fingerprints import fingerprint_mapping

        new_fingerprint = fingerprint_mapping(
            {
                "previous_draft_fingerprint": previous_draft.get("source_draft_fingerprint")
                or previous_draft.get("draft_fingerprint"),
                "writer_revision_request_fingerprint": fingerprint_mapping(revision_request),
                "round_number": round_number,
                "policy": {k: v for k, v in policy.items() if k not in {"outline_sections"}},
                "revised_draft": revised_draft,
            }
        )

        unresolved = sorted(set(unresolved_issue_ids))
        all_issue_ids = [str(i.get("issue_id", "")) for i in revision_request.get("issues", ())]
        resolved = sorted(set(all_issue_ids) - set(unresolved))

        # Persistencia por ronda (punto 1), lado de 06: opcional, activada
        # solo si policy trae project_dir/experiment_id (el llamador real
        # los conoce; las pruebas unitarias que no los pasan simplemente no
        # persisten por ronda, sin romper). No sobrescribe rondas previas.
        # Punto 4: sin captura de excepciones -- si la persistencia falla
        # (FileExistsError, OSError, error de serialización), se propaga
        # tal cual y el resultado de 06 NO se produce (ver más abajo esta
        # llamada ya no está envuelta en try/except).
        cycle_project_dir = policy.get("cycle_project_dir")
        cycle_experiment_id = policy.get("experiment_id")
        if cycle_project_dir and cycle_experiment_id:
            from src.tools.verification.cycle_round_persistence import complete_round_revision

            complete_round_revision(
                project_dir=cycle_project_dir,
                experiment_id=cycle_experiment_id,
                cycle_id=revision_request["cycle_id"],
                round_number=round_number,
                writer_revision_request=revision_request,
                artifacts={
                    "revised_draft.json": revised_draft,
                    "revision_changelog.json": changelog,
                    "revision_resolution_matrix.json": resolution_matrix,
                    "unresolved_issues.json": unresolved,
                    "fingerprint.json": {
                        "new_fingerprint": new_fingerprint,
                        "previous_draft_fingerprint": previous_draft.get("source_draft_fingerprint"),
                        "writer_revision_request_fingerprint": fingerprint_mapping(revision_request),
                        "round_number": round_number,
                    },
                },
            )

        artifacts = {
            "revised_draft.json": ArtifactReference(str(revised_draft_path), sha256_file(revised_draft_path)),
            "revision_changelog.json": ArtifactReference(str(changelog_path), sha256_file(changelog_path)),
            "revision_resolution_matrix.json": ArtifactReference(str(matrix_path), sha256_file(matrix_path)),
        }

        quality_status = (
            QualityStatus.APPROVED_WITH_WARNINGS if unresolved else QualityStatus.APPROVED
        )
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED,
            quality_status=quality_status,
            decision=DecisionInfo(
                code="DRAFT_REVISION_COMPLETED",
                rationale=f"Ronda {round_number}: {len(resolved)} issue(s) resuelto(s), {len(unresolved)} pendiente(s).",
            ),
            quality_metrics={
                "scientific": {"resolved_issues": len(resolved), "unresolved_issues": len(unresolved)},
                "technical": {
                    "revision_fingerprint": new_fingerprint,
                    "sections_regenerated": sum(1 for c in changelog if c["action"].startswith("REVISED")),
                    "sections_preserved": sum(1 for c in changelog if c["action"] == "PRESERVED"),
                },
            },
            warnings=(
                tuple(
                    AgentWarning(
                        code="UNRESOLVED_REVISION_ISSUE",
                        severity=WarningSeverity.WARNING,
                        blocking=False,
                        message=f"Issue no resuelto: {issue_id}",
                    )
                    for issue_id in unresolved
                )
            ),
            failure_reason_codes=(),
            requested_transition=RequestedTransition(
                action=TransitionAction.ADVANCE,
                target_stage="07_agente_verificador",
                reason_code="DRAFT_REVISION_COMPLETED",
                requires_human_confirmation=False,
            ),
            output_artifacts=artifacts,
            tool_usage=ToolUsage(
                retrieval_rounds=retrieval_rounds, llm_calls=llm_calls, validation_calls=len(generated)
            ),
            attempt_number=agent_input.attempt_number,
            started_at=start,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _execute_initial_draft(self, agent_input):
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
                self._resolve_section_claim_identities(
                    accepted, previous_claims_by_uid={}, round_number=1,
                )
                generated.append(accepted)

            # Colisión de claim_uid a nivel de TODO el borrador inicial
            # -- nunca solo por sección (ver check_no_claim_uid_collisions).
            check_no_claim_uid_collisions(
                [
                    ClaimIdentityRecord(
                        claim_uid=c["claim_uid"], claim_version=c["claim_version"], claim_id=c["claim_id"],
                        parent_claim_uids=tuple(c.get("parent_claim_uids") or ()),
                        claim_text_fingerprint=c["claim_text_fingerprint"],
                        created_round=c["created_round"], updated_round=c["updated_round"],
                    )
                    for section in generated for c in (section.get("claims") or []) if "claim_uid" in c
                ]
            )

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
