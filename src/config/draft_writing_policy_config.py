from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted"
PLANNED_HYBRID_RETRIEVAL_STRATEGY = "hybrid_chroma_csv_rrf_balanced"


DEFAULT_DRAFT_WRITING_POLICY: dict[str, Any] = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "prompt_version": "legacy_notebook06_section_prompt_v1",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    "validation_version": "legacy_notebook06_validation_v1",
    "temperature": 0.0,
    "auto_rebuild": True,
    "force_rebuild": False,
    "max_section_revision_attempts": 2,
    # Compatibility default: the contractual Agent 06 still executes the
    # behavior-preserving retrieval. The hybrid strategy is activated only in
    # the later integration phase that promotes the new agent implementation.
    "retrieval_strategy": LEGACY_RETRIEVAL_STRATEGY,
    # Hybrid-policy parameters are declared now, but remain inactive while the
    # retrieval_strategy is the legacy strategy.
    "candidate_multiplier": 3,
    "top_k_evidence_per_section": 8,
    "chroma_quota": 3,
    "csv_quota": 3,
    "rrf_quota": 2,
    "rrf_k": 60,
    "max_evidence_chars": 18000,
    "max_candidates_per_source": 24,
    "quantitative_evidence_quota": 2,
    "max_quantitative_rows_per_section": 12,
    "organizational_target_words": 40,
    "organizational_minimum_words": 1,
    "organizational_maximum_words": 80,
    "substantive_minimum_ratio": 0.65,
    "substantive_maximum_ratio": 1.40,
    "allow_open_search_outside_outline_sources": False,
    "validate_citations_against_section_evidence": True,
    "validate_numeric_values_against_source_chunks": True,
    "fail_on_invalid_draft": True,
}

_QUOTA_KEYS = ("chroma_quota", "csv_quota", "rrf_quota")
_DEFAULT_QUOTA_WEIGHTS = (3, 3, 2)
_ALLOWED_RETRIEVAL_STRATEGIES = {
    LEGACY_RETRIEVAL_STRATEGY,
    PLANNED_HYBRID_RETRIEVAL_STRATEGY,
}


def _require_integer(policy: Mapping[str, Any], key: str) -> int:
    value = policy.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"DRAFT_POLICY_INVALID_TYPE:{key}:expected_integer")
    return value


def _require_number(policy: Mapping[str, Any], key: str) -> float:
    value = policy.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"DRAFT_POLICY_INVALID_TYPE:{key}:expected_number")
    return float(value)


def _require_nonempty_string(policy: Mapping[str, Any], key: str) -> str:
    value = policy.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DRAFT_POLICY_INVALID_TYPE:{key}:expected_nonempty_string")
    return value.strip()


def _derive_retrieval_quotas(top_k: int) -> dict[str, int]:
    if top_k <= 0:
        return {key: 0 for key in _QUOTA_KEYS}

    total_weight = sum(_DEFAULT_QUOTA_WEIGHTS)
    raw_values = [top_k * weight / total_weight for weight in _DEFAULT_QUOTA_WEIGHTS]
    quotas = [int(value) for value in raw_values]
    remaining = top_k - sum(quotas)
    order = sorted(
        range(len(raw_values)),
        key=lambda index: (-(raw_values[index] - quotas[index]), index),
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return dict(zip(_QUOTA_KEYS, quotas, strict=True))


def validate_draft_writing_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Agent 06 policy with deterministic, explicit errors."""
    if not isinstance(policy, Mapping):
        raise ValueError("DRAFT_POLICY_INVALID_TYPE:policy:expected_mapping")

    validated = dict(policy)
    retrieval_strategy = _require_nonempty_string(validated, "retrieval_strategy")
    if retrieval_strategy not in _ALLOWED_RETRIEVAL_STRATEGIES:
        raise ValueError(
            "DRAFT_POLICY_INVALID:retrieval_strategy:unsupported_strategy"
        )

    candidate_multiplier = _require_integer(validated, "candidate_multiplier")
    top_k = _require_integer(validated, "top_k_evidence_per_section")
    chroma_quota = _require_integer(validated, "chroma_quota")
    csv_quota = _require_integer(validated, "csv_quota")
    rrf_quota = _require_integer(validated, "rrf_quota")
    rrf_k = _require_integer(validated, "rrf_k")
    quantitative_quota = _require_integer(validated, "quantitative_evidence_quota")

    if candidate_multiplier < 1:
        raise ValueError(
            "DRAFT_POLICY_INVALID:candidate_multiplier:must_be_greater_than_or_equal_to_1"
        )
    if top_k <= 0:
        raise ValueError(
            "DRAFT_POLICY_INVALID:top_k_evidence_per_section:must_be_greater_than_0"
        )
    if rrf_k <= 0:
        raise ValueError("DRAFT_POLICY_INVALID:rrf_k:must_be_greater_than_0")
    if quantitative_quota < 0 or quantitative_quota > top_k:
        raise ValueError(
            "DRAFT_POLICY_INVALID:quantitative_evidence_quota:must_be_between_0_and_top_k_evidence_per_section"
        )

    for key, value in (
        ("chroma_quota", chroma_quota),
        ("csv_quota", csv_quota),
        ("rrf_quota", rrf_quota),
    ):
        if value < 0:
            raise ValueError(
                f"DRAFT_POLICY_INVALID:{key}:must_be_greater_than_or_equal_to_0"
            )

    if chroma_quota + csv_quota + rrf_quota != top_k:
        raise ValueError(
            "DRAFT_POLICY_INVALID:retrieval_quotas:chroma_quota_plus_csv_quota_plus_rrf_quota_must_equal_top_k_evidence_per_section"
        )

    for key in (
        "max_evidence_chars",
        "max_candidates_per_source",
        "max_quantitative_rows_per_section",
        "organizational_target_words",
        "organizational_minimum_words",
        "organizational_maximum_words",
    ):
        value = _require_integer(validated, key)
        if value < 0:
            raise ValueError(
                f"DRAFT_POLICY_INVALID:{key}:must_be_greater_than_or_equal_to_0"
            )

    minimum_ratio = _require_number(validated, "substantive_minimum_ratio")
    maximum_ratio = _require_number(validated, "substantive_maximum_ratio")
    if minimum_ratio < 0:
        raise ValueError(
            "DRAFT_POLICY_INVALID:substantive_minimum_ratio:must_be_greater_than_or_equal_to_0"
        )
    if maximum_ratio < minimum_ratio:
        raise ValueError(
            "DRAFT_POLICY_INVALID:substantive_maximum_ratio:must_be_greater_than_or_equal_to_substantive_minimum_ratio"
        )

    if validated["organizational_minimum_words"] > validated["organizational_target_words"]:
        raise ValueError(
            "DRAFT_POLICY_INVALID:organizational_minimum_words:must_not_exceed_organizational_target_words"
        )
    if validated["organizational_target_words"] > validated["organizational_maximum_words"]:
        raise ValueError(
            "DRAFT_POLICY_INVALID:organizational_maximum_words:must_be_greater_than_or_equal_to_organizational_target_words"
        )

    return validated


def get_draft_writing_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_DRAFT_WRITING_POLICY)
    if overrides is None:
        return validate_draft_writing_policy(policy)
    if not isinstance(overrides, Mapping):
        raise ValueError("DRAFT_POLICY_INVALID_TYPE:overrides:expected_mapping")

    override_values = dict(overrides)
    provided_quotas = [key for key in _QUOTA_KEYS if key in override_values]
    if "top_k_evidence_per_section" in override_values and not provided_quotas:
        # Validate the override type before quota derivation. Do not coerce
        # strings, floats or booleans with int(...), because that would produce
        # native or ambiguous errors instead of the contractual error.
        top_k_override = _require_integer(
            override_values,
            "top_k_evidence_per_section",
        )
        policy.update(_derive_retrieval_quotas(top_k_override))
    elif provided_quotas and len(provided_quotas) != len(_QUOTA_KEYS):
        raise ValueError(
            "DRAFT_POLICY_INVALID:retrieval_quotas:all_quota_overrides_must_be_provided_together"
        )

    policy.update(override_values)
    return validate_draft_writing_policy(policy)

# Perfil V17 promovido para el Agente 06.
# Se mantiene fuera del notebook para evitar una segunda fuente de verdad.
V17_HYBRID_DRAFT_POLICY_OVERRIDES: dict[str, Any] = {
    "retrieval_strategy": PLANNED_HYBRID_RETRIEVAL_STRATEGY,
    "candidate_multiplier": 3,
    "top_k_evidence_per_section": 8,
    "chroma_quota": 3,
    "csv_quota": 3,
    "rrf_quota": 2,
    "rrf_k": 60,
    "max_evidence_chars": 18000,
    "max_candidates_per_source": 24,
    "quantitative_evidence_quota": 2,
    "organizational_target_words": 40,
    "organizational_minimum_words": 25,
    "organizational_maximum_words": 70,
    "substantive_minimum_ratio": 0.65,
    "substantive_maximum_ratio": 1.40,
}


def get_v17_hybrid_draft_policy_overrides() -> dict[str, Any]:
    """Devuelve una copia del perfil V17 para evitar mutaciones globales."""
    return dict(V17_HYBRID_DRAFT_POLICY_OVERRIDES)

