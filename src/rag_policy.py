# ============================================================
# POLÍTICA RAG Y GROUND TRUTH
# ============================================================

from copy import deepcopy
from config import RAG_POLICY as ACTIVE_RAG_POLICY

if not isinstance(
    ACTIVE_RAG_POLICY,
    dict,
) or not ACTIVE_RAG_POLICY:
    raise ValueError(
        "RAG_POLICY debe ser un diccionario no vacío."
    )

required_policy_keys = {
    "exclude_review_sections_from_reference_papers",
    "excluded_reference_section_types",
    "ground_truth_usage",
    "use_ground_truth_for_generation",
    "use_ground_truth_for_rag",
    "use_ground_truth_for_verification",
    "use_ground_truth_for_evaluation",
    "retrieval_profiles",
    "indexing",
    "generation",
}

missing_policy_keys = sorted(
    required_policy_keys
    - set(ACTIVE_RAG_POLICY)
)

if missing_policy_keys:
    raise ValueError(
        "RAG_POLICY está incompleta. "
        f"Faltan: {missing_policy_keys}"
    )

GROUND_TRUTH_POLICY = {
    "use_ground_truth_for_generation": bool(
        ACTIVE_RAG_POLICY[
            "use_ground_truth_for_generation"
        ]
    ),
    "use_ground_truth_for_rag": bool(
        ACTIVE_RAG_POLICY[
            "use_ground_truth_for_rag"
        ]
    ),
    "use_ground_truth_for_verification": bool(
        ACTIVE_RAG_POLICY[
            "use_ground_truth_for_verification"
        ]
    ),
    "use_ground_truth_for_evaluation": bool(
        ACTIVE_RAG_POLICY[
            "use_ground_truth_for_evaluation"
        ]
    ),
}

if any([
    GROUND_TRUTH_POLICY[
        "use_ground_truth_for_generation"
    ],
    GROUND_TRUTH_POLICY[
        "use_ground_truth_for_rag"
    ],
    GROUND_TRUTH_POLICY[
        "use_ground_truth_for_verification"
    ],
]):
    raise ValueError(
        "El Ground Truth solo puede utilizarse para evaluación."
    )

if not GROUND_TRUTH_POLICY[
    "use_ground_truth_for_evaluation"
]:
    raise ValueError(
        "El Ground Truth debe estar habilitado para evaluación."
    )

EXCLUDE_REVIEW_SECTIONS_FROM_REFERENCE_PAPERS = bool(
    ACTIVE_RAG_POLICY[
        "exclude_review_sections_from_reference_papers"
    ]
)

if not EXCLUDE_REVIEW_SECTIONS_FROM_REFERENCE_PAPERS:
    raise ValueError(
        "La política metodológica exige excluir "
        "secciones de revisión del RAG."
    )

REVIEW_SECTION_TYPES = set(
    ACTIVE_RAG_POLICY[
        "excluded_reference_section_types"
    ]
)

if not REVIEW_SECTION_TYPES:
    raise ValueError(
        "excluded_reference_section_types no puede estar vacío."
    )

REVIEW_SECTION_LABELS_ES = {
    "related_work": "trabajos relacionados",
    "literature_review": "revisión de literatura",
    "state_of_the_art": "estado del arte",
    "background": "antecedentes",
    "theoretical_background": (
        "marco teórico / antecedentes teóricos"
    ),
    "previous_work": "trabajo previo",
    "prior_work": "trabajo anterior",
}

REVIEW_SECTION_PATTERNS = [
    r"\brelated\s+work\b",
    r"\bliterature\s+review\b",
    r"\bstate\s+of\s+the\s+art\b",
    r"\bbackground\b",
    r"\btheoretical\s+background\b",
    r"\bprevious\s+work\b",
    r"\bprior\s+work\b",
    r"\btrabajos?\s+relacionados?\b",
    r"\brevisión\s+de\s+literatura\b",
    r"\brevision\s+de\s+literatura\b",
    r"\brevisión\s+bibliográfica\b",
    r"\brevision\s+bibliografica\b",
    r"\bestado\s+del\s+arte\b",
    r"\bantecedentes\b",
    r"\bmarco\s+teórico\b",
    r"\bmarco\s+teorico\b",
    r"\btrabajos?\s+previos?\b",
]

RAG_ALLOWED_CONTENT_POLICY = (
    "Solo se indexan fragmentos de papers de referencia "
    "que no pertenezcan a secciones de revisión, antecedentes, "
    "trabajos relacionados o bibliografía. "
    "El Ground Truth se reserva exclusivamente para evaluación."
)

RETRIEVAL_PROFILES = deepcopy(
    ACTIVE_RAG_POLICY[
        "retrieval_profiles"
    ]
)
INDEXING_CONFIG = deepcopy(
    ACTIVE_RAG_POLICY[
        "indexing"
    ]
)
RAG_GENERATION_CONFIG = deepcopy(
    ACTIVE_RAG_POLICY[
        "generation"
    ]
)

if not isinstance(
    RETRIEVAL_PROFILES,
    dict,
) or not RETRIEVAL_PROFILES:
    raise ValueError(
        "retrieval_profiles debe ser un diccionario no vacío."
    )

if not isinstance(
    INDEXING_CONFIG,
    dict,
):
    raise ValueError(
        "indexing debe ser un diccionario."
    )

if not isinstance(
    RAG_GENERATION_CONFIG,
    dict,
):
    raise ValueError(
        "generation debe ser un diccionario."
    )

if "batch_size" not in INDEXING_CONFIG:
    raise ValueError(
        "indexing requiere batch_size."
    )

for key in [
    "temperature",
    "answer_max_words",
]:
    if key not in RAG_GENERATION_CONFIG:
        raise ValueError(
            f"generation requiere {key!r}."
        )

INDEX_BATCH_SIZE = int(
    INDEXING_CONFIG[
        "batch_size"
    ]
)
RAG_TEMPERATURE = float(
    RAG_GENERATION_CONFIG[
        "temperature"
    ]
)
RAG_ANSWER_MAX_WORDS = int(
    RAG_GENERATION_CONFIG[
        "answer_max_words"
    ]
)

if INDEX_BATCH_SIZE <= 0:
    raise ValueError(
        "INDEX_BATCH_SIZE debe ser mayor que cero."
    )

if not 0.0 <= RAG_TEMPERATURE <= 2.0:
    raise ValueError(
        "RAG_TEMPERATURE debe estar entre 0 y 2."
    )

if RAG_ANSWER_MAX_WORDS <= 0:
    raise ValueError(
        "RAG_ANSWER_MAX_WORDS debe ser mayor que cero."
    )


def get_retrieval_profile(
    profile_name="default",
):
    if profile_name not in RETRIEVAL_PROFILES:
        valid = ", ".join(
            sorted(
                RETRIEVAL_PROFILES
            )
        )

        raise KeyError(
            f"Perfil RAG desconocido: {profile_name!r}. "
            f"Perfiles válidos: {valid}"
        )

    profile = deepcopy(
        RETRIEVAL_PROFILES[
            profile_name
        ]
    )

    required = {
        "top_k",
        "fetch_k",
        "max_per_source",
    }

    missing = sorted(
        required - set(profile)
    )

    if missing:
        raise ValueError(
            f"El perfil RAG {profile_name!r} está incompleto. "
            f"Faltan: {missing}"
        )

    profile["top_k"] = int(
        profile["top_k"]
    )
    profile["fetch_k"] = int(
        profile["fetch_k"]
    )
    profile["max_per_source"] = int(
        profile["max_per_source"]
    )

    if profile["top_k"] <= 0:
        raise ValueError(
            "top_k debe ser mayor que cero."
        )

    if (
        profile["fetch_k"]
        < profile["top_k"]
    ):
        raise ValueError(
            "fetch_k debe ser mayor o igual que top_k."
        )

    if profile[
        "max_per_source"
    ] <= 0:
        raise ValueError(
            "max_per_source debe ser mayor que cero."
        )

    return profile


def get_rag_policy():
    return {
        "ground_truth_policy": deepcopy(
            GROUND_TRUTH_POLICY
        ),
        "exclude_review_sections_from_reference_papers": (
            EXCLUDE_REVIEW_SECTIONS_FROM_REFERENCE_PAPERS
        ),
        "review_section_types": sorted(
            REVIEW_SECTION_TYPES
        ),
        "review_section_labels_es": deepcopy(
            REVIEW_SECTION_LABELS_ES
        ),
        "review_section_patterns": list(
            REVIEW_SECTION_PATTERNS
        ),
        "rag_allowed_content_policy": (
            RAG_ALLOWED_CONTENT_POLICY
        ),
        "retrieval_profiles": deepcopy(
            RETRIEVAL_PROFILES
        ),
        "indexing": deepcopy(
            INDEXING_CONFIG
        ),
        "generation": deepcopy(
            RAG_GENERATION_CONFIG
        ),
        "index_batch_size": (
            INDEX_BATCH_SIZE
        ),
        "rag_temperature": (
            RAG_TEMPERATURE
        ),
        "rag_answer_max_words": (
            RAG_ANSWER_MAX_WORDS
        ),
    }
