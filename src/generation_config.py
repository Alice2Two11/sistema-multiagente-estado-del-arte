# ============================================================
# CONFIGURACIÓN DE GENERACIÓN DEL ESTADO DEL ARTE
# ============================================================

from config import (
    GENERATION_PROFILE,
    RAG_POLICY,
)

required_generation_keys = {
    "output_language",
    "output_language_label",
    "length_profile",
    "length_profile_label",
    "writing_mode",
    "writing_mode_label",
    "focus_mode",
    "focus_mode_label",
    "citation_style",
    "embedding_model",
}

missing_generation_keys = sorted(
    required_generation_keys
    - set(GENERATION_PROFILE)
)

if missing_generation_keys:
    raise ValueError(
        "GENERATION_PROFILE está incompleto. "
        f"Faltan: {missing_generation_keys}"
    )

OUTPUT_LANGUAGE = str(
    GENERATION_PROFILE[
        "output_language"
    ]
).strip()
OUTPUT_LANGUAGE_LABEL = str(
    GENERATION_PROFILE[
        "output_language_label"
    ]
).strip()
LENGTH_PROFILE = str(
    GENERATION_PROFILE[
        "length_profile"
    ]
).strip()
LENGTH_PROFILE_LABEL = str(
    GENERATION_PROFILE[
        "length_profile_label"
    ]
).strip()
WRITING_MODE = str(
    GENERATION_PROFILE[
        "writing_mode"
    ]
).strip()
WRITING_MODE_LABEL = str(
    GENERATION_PROFILE[
        "writing_mode_label"
    ]
).strip()
FOCUS_MODE = str(
    GENERATION_PROFILE[
        "focus_mode"
    ]
).strip()
FOCUS_MODE_LABEL = str(
    GENERATION_PROFILE[
        "focus_mode_label"
    ]
).strip()
CITATION_STYLE = str(
    GENERATION_PROFILE[
        "citation_style"
    ]
).strip()
EMBEDDING_MODEL = str(
    GENERATION_PROFILE[
        "embedding_model"
    ]
).strip()

if not all([
    OUTPUT_LANGUAGE,
    OUTPUT_LANGUAGE_LABEL,
    LENGTH_PROFILE,
    LENGTH_PROFILE_LABEL,
    WRITING_MODE,
    WRITING_MODE_LABEL,
    FOCUS_MODE,
    FOCUS_MODE_LABEL,
    CITATION_STYLE,
    EMBEDDING_MODEL,
]):
    raise ValueError(
        "GENERATION_PROFILE contiene valores vacíos."
    )

LENGTH_PROFILES = {
    "small": {
        "label": "pequeña",
        "min_sections": 4,
        "max_sections": 5,
        "target_total_words": 900,
        "min_total_words": 700,
        "max_total_words": 1200,
    },
    "medium": {
        "label": "mediana",
        "min_sections": 6,
        "max_sections": 7,
        "target_total_words": 1600,
        "min_total_words": 1300,
        "max_total_words": 2200,
    },
    "large": {
        "label": "grande",
        "min_sections": 8,
        "max_sections": 9,
        "target_total_words": 2800,
        "min_total_words": 2400,
        "max_total_words": 4000,
    },
}

if LENGTH_PROFILE not in LENGTH_PROFILES:
    raise ValueError(
        "length_profile desconocido: "
        f"{LENGTH_PROFILE!r}."
    )

rag_generation_config = (
    RAG_POLICY.get(
        "generation"
    )
)

if not isinstance(
    rag_generation_config,
    dict,
):
    raise ValueError(
        "RAG_POLICY['generation'] debe ser un diccionario."
    )

required_rag_generation_keys = {
    "temperature",
    "answer_max_words",
}

missing_rag_generation_keys = sorted(
    required_rag_generation_keys
    - set(rag_generation_config)
)

if missing_rag_generation_keys:
    raise ValueError(
        "RAG_POLICY['generation'] está incompleta. "
        f"Faltan: {missing_rag_generation_keys}"
    )

RAG_TEMPERATURE = float(
    rag_generation_config[
        "temperature"
    ]
)
RAG_ANSWER_MAX_WORDS = int(
    rag_generation_config[
        "answer_max_words"
    ]
)

if not 0.0 <= RAG_TEMPERATURE <= 2.0:
    raise ValueError(
        "RAG_TEMPERATURE debe estar entre 0 y 2."
    )

if RAG_ANSWER_MAX_WORDS <= 0:
    raise ValueError(
        "RAG_ANSWER_MAX_WORDS debe ser mayor que cero."
    )

ACTIVE_LENGTH_CONFIG = (
    LENGTH_PROFILES[
        LENGTH_PROFILE
    ]
)

MIN_SECTIONS = (
    ACTIVE_LENGTH_CONFIG[
        "min_sections"
    ]
)
MAX_SECTIONS = (
    ACTIVE_LENGTH_CONFIG[
        "max_sections"
    ]
)
TARGET_TOTAL_WORDS = (
    ACTIVE_LENGTH_CONFIG[
        "target_total_words"
    ]
)
MIN_TOTAL_WORDS = (
    ACTIVE_LENGTH_CONFIG[
        "min_total_words"
    ]
)
MAX_TOTAL_WORDS = (
    ACTIVE_LENGTH_CONFIG[
        "max_total_words"
    ]
)

TRACEABILITY_CITATION_STYLE = (
    "source_chunk"
)


def get_generation_profile():
    return {
        "output_language": OUTPUT_LANGUAGE,
        "output_language_label": (
            OUTPUT_LANGUAGE_LABEL
        ),
        "length_profile": LENGTH_PROFILE,
        "length_profile_label": (
            LENGTH_PROFILE_LABEL
        ),
        "writing_mode": WRITING_MODE,
        "writing_mode_label": (
            WRITING_MODE_LABEL
        ),
        "focus_mode": FOCUS_MODE,
        "focus_mode_label": (
            FOCUS_MODE_LABEL
        ),
        "citation_style": (
            CITATION_STYLE
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "min_sections": (
            MIN_SECTIONS
        ),
        "max_sections": (
            MAX_SECTIONS
        ),
        "target_total_words": (
            TARGET_TOTAL_WORDS
        ),
        "min_total_words": (
            MIN_TOTAL_WORDS
        ),
        "max_total_words": (
            MAX_TOTAL_WORDS
        ),
        "traceability_citation_style": (
            TRACEABILITY_CITATION_STYLE
        ),
        "rag_temperature": (
            RAG_TEMPERATURE
        ),
        "rag_answer_max_words": (
            RAG_ANSWER_MAX_WORDS
        ),
    }
