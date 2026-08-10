# ============================================================
# CONFIGURACIÓN TEMÁTICA DEL EXPERIMENTO
# ============================================================

from config import (
    EXPERIMENT_ID,
    TOPIC_PROFILE,
)
from generation_config import (
    OUTPUT_LANGUAGE,
    OUTPUT_LANGUAGE_LABEL,
)

if not isinstance(
    TOPIC_PROFILE,
    dict,
) or not TOPIC_PROFILE:
    raise RuntimeError(
        "No existe un TOPIC_PROFILE válido. "
        "Ejecuta 00_setup_config y configura el tema."
    )

required_topic_fields = {
    "topic_name",
    "research_scope",
    "domain_terms",
    "method_dimensions",
    "analysis_dimensions",
    "relevance_rules",
    "excluded_domains",
    "relevance_levels_included",
    "rag_test_query",
    "rag_tool_test_question",
    "rag_synthesis_question",
    "rag_test_queries",
}

missing_topic_fields = sorted(
    required_topic_fields
    - set(TOPIC_PROFILE)
)

if missing_topic_fields:
    raise ValueError(
        "TOPIC_PROFILE está incompleto. "
        f"Faltan: {missing_topic_fields}"
    )

for key in [
    "topic_name",
    "research_scope",
    "relevance_rules",
    "rag_test_query",
    "rag_tool_test_question",
    "rag_synthesis_question",
]:
    if not str(
        TOPIC_PROFILE[
            key
        ]
    ).strip():
        raise ValueError(
            f"TOPIC_PROFILE[{key!r}] no puede estar vacío."
        )

for key in [
    "domain_terms",
    "method_dimensions",
    "analysis_dimensions",
    "relevance_levels_included",
    "rag_test_queries",
]:
    value = TOPIC_PROFILE[
        key
    ]

    if not isinstance(
        value,
        list,
    ) or not value:
        raise ValueError(
            f"TOPIC_PROFILE[{key!r}] debe ser una lista no vacía."
        )

if not isinstance(
    TOPIC_PROFILE[
        "excluded_domains"
    ],
    list,
):
    raise TypeError(
        "TOPIC_PROFILE['excluded_domains'] debe ser una lista."
    )

if OUTPUT_LANGUAGE == "es":
    OUTPUT_LANGUAGE_ACADEMIC = (
        "español académico"
    )
elif OUTPUT_LANGUAGE == "en":
    OUTPUT_LANGUAGE_ACADEMIC = (
        "academic English"
    )
else:
    OUTPUT_LANGUAGE_ACADEMIC = (
        OUTPUT_LANGUAGE_LABEL
    )

CARD_SCHEMA = {
    "source_filename": (
        "Nombre del archivo PDF de origen"
    ),
    "title": "Título del paper",
    "paper_type": "Tipo de paper",
    "research_problem": (
        "Problema de investigación"
    ),
    "objective": (
        "Objetivo principal"
    ),
    "task_type": (
        "Tipo de tarea científica"
    ),
    "target_domain": (
        "Dominio objetivo"
    ),
    "target_variable_or_object": (
        "Variable, fenómeno u objeto analizado"
    ),
    "temporal_horizon_or_scope": (
        "Horizonte temporal o alcance del estudio"
    ),
    "methods_or_models": (
        "Métodos, modelos o algoritmos"
    ),
    "method_families": (
        "Familias metodológicas"
    ),
    "datasets_or_case_study": (
        "Datasets, población o casos de estudio"
    ),
    "input_variables_or_data_sources": (
        "Variables de entrada o fuentes de datos"
    ),
    "evaluation_metrics": (
        "Métricas o criterios de evaluación"
    ),
    "main_results": (
        "Resultados principales"
    ),
    "reported_best_method_or_model": (
        "Mejor método o modelo reportado"
    ),
    "limitations_or_gaps": (
        "Limitaciones o vacíos identificados"
    ),
    "contribution": (
        "Contribución principal"
    ),
    "relevance_for_state_of_art": (
        "Relevancia para el estado del arte"
    ),
    "domain_specific_notes": (
        "Notas específicas del dominio"
    ),
    "evidence": (
        "Lista de claim, supporting_quote y chunk_id"
    ),
}

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
    "evaluation_metrics",
    "main_results",
    "evidence",
]

CLASSIFICATION_FIELDS = [
    "task_type",
    "target_domain",
    "method_families",
    "relevance_level",
    "include_in_state_of_art",
    "relevance_reason",
]

EXPERIMENT_PROFILE = {
    "experiment_id": EXPERIMENT_ID,
    "topic_name": (
        TOPIC_PROFILE[
            "topic_name"
        ]
    ),
    "research_scope": (
        TOPIC_PROFILE[
            "research_scope"
        ]
    ),
    "domain_terms": list(
        TOPIC_PROFILE[
            "domain_terms"
        ]
    ),
    "method_dimensions": list(
        TOPIC_PROFILE[
            "method_dimensions"
        ]
    ),
    "analysis_dimensions": list(
        TOPIC_PROFILE[
            "analysis_dimensions"
        ]
    ),
    "relevance_rules": (
        TOPIC_PROFILE[
            "relevance_rules"
        ]
    ),
    "excluded_domains": list(
        TOPIC_PROFILE[
            "excluded_domains"
        ]
    ),
    "relevance_levels_included": list(
        TOPIC_PROFILE[
            "relevance_levels_included"
        ]
    ),
    "rag_test_query": (
        TOPIC_PROFILE[
            "rag_test_query"
        ]
    ),
    "rag_tool_test_question": (
        TOPIC_PROFILE[
            "rag_tool_test_question"
        ]
    ),
    "rag_synthesis_question": (
        TOPIC_PROFILE[
            "rag_synthesis_question"
        ]
    ),
    "rag_test_queries": list(
        TOPIC_PROFILE[
            "rag_test_queries"
        ]
    ),
    "output_language": (
        OUTPUT_LANGUAGE_ACADEMIC
    ),
}


def get_card_fields():
    return list(
        CARD_SCHEMA.keys()
    )


def get_profile(
    key,
    default=None,
):
    return EXPERIMENT_PROFILE.get(
        key,
        default,
    )


def get_rag_test_query():
    return EXPERIMENT_PROFILE[
        "rag_test_query"
    ]


def get_rag_tool_test_question():
    return EXPERIMENT_PROFILE[
        "rag_tool_test_question"
    ]


def get_rag_synthesis_question():
    return EXPERIMENT_PROFILE[
        "rag_synthesis_question"
    ]


def get_rag_test_queries():
    return list(
        EXPERIMENT_PROFILE[
            "rag_test_queries"
        ]
    )
