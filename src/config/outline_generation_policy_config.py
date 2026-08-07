from __future__ import annotations
OUTLINE_STAGE_NAME='05_generador_esquema'
OUTLINE_STAGE_VERSION='05_CORREGIDO_v1_manifest_auto_generation_profile'
OUTLINE_PROMPT_VERSION='v2_spanish_generation_profile_valid_sources'
OUTLINE_SCHEMA_VERSION='v2_sections_paper_mapping_traceability'
OUTLINE_VALIDATION_VERSION='v2_repair_then_validate'
DEFAULT_POLICY={
 'force_rebuild':False,
 'max_attempts':2,
 'max_field_chars':1800,
 'title_match_cutoff':0.55,
 'stage_version':OUTLINE_STAGE_VERSION,
 'prompt_version':OUTLINE_PROMPT_VERSION,
 'schema_version':OUTLINE_SCHEMA_VERSION,
 'validation_version':OUTLINE_VALIDATION_VERSION,
}
def get_outline_generation_policy(overrides=None):
 p=dict(DEFAULT_POLICY); p.update(dict(overrides or {})); return p
