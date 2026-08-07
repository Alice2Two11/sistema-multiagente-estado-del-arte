"""Políticas exclusivas del Agente 04. Umbrales iniciales diagnósticos."""
from __future__ import annotations
POLICY_STATUS = "PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED"
DEFAULT_THEMATIC_ANALYSIS_POLICY = {
    "temperature": 0.1,
    "max_field_chars": 3500,
    "max_attempts": 2,
    "allow_quantitative_context": True,
    "require_quantitative_manifest_if_files_exist": True,
    "validate_titles": True,
    "require_gap_supporting_sources": True,
    "require_comparative_sources": True,
    "auto_rebuild": True,
    "force_rebuild": False,
    "thresholds_status": POLICY_STATUS,
    "diagnostic_thresholds": {
        "paper_coverage": None,
        "supported_theme_rate": None,
        "supported_gap_rate": None,
        "comparative_dimension_support_rate": None,
        "valid_reference_rate": None,
    },
    "manual_review_policy": {"allowed": True},
}

def get_thematic_analysis_policy(overrides=None):
    policy={**DEFAULT_THEMATIC_ANALYSIS_POLICY}
    policy["diagnostic_thresholds"]=dict(DEFAULT_THEMATIC_ANALYSIS_POLICY["diagnostic_thresholds"])
    policy["manual_review_policy"]=dict(DEFAULT_THEMATIC_ANALYSIS_POLICY["manual_review_policy"])
    if overrides: policy.update(dict(overrides))
    if int(policy.get("max_attempts",2)) != 2: raise ValueError("04 admite exactamente dos intentos contractuales.")
    return policy
