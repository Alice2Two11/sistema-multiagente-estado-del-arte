# ============================================================
# PROMPTS PARAMETRIZABLES DEL SISTEMA MULTIAGENTE
# La temática y el esquema se controlan desde experiment_config.py.
# ============================================================

import json


# ------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------

def _as_json(obj):
    """Serializa a JSON legible para insertar en un prompt."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _profile_block(experiment_profile, include_methods=True, include_excluded=True):
    """
    Construye el bloque de contexto común a todos los prompts.
    """
    parts = [
        f"Tema del experimento:\n{experiment_profile.get('topic_name', '')}",
        f"\nAlcance científico:\n{experiment_profile.get('research_scope', '')}",
        f"\nTérminos o conceptos relevantes del dominio:\n"
        f"{_as_json(experiment_profile.get('domain_terms', []))}",
    ]

    if include_methods:
        parts.append(
            f"\nDimensiones metodológicas a considerar:\n"
            f"{_as_json(experiment_profile.get('method_dimensions', []))}"
        )

    if include_excluded:
        parts.append(
            f"\nDominios o enfoques que deben excluirse si aparecen:\n"
            f"{_as_json(experiment_profile.get('excluded_domains', []))}"
        )

    return "\n".join(parts)


# ------------------------------------------------------------
# 1) EXTRACCIÓN DE FICHA CIENTÍFICA
# ------------------------------------------------------------

def build_scientific_extraction_prompt(source_filename, context, experiment_profile):
    """
    Prompt para extraer una ficha científica estructurada desde un paper.
    """
    output_language = experiment_profile.get("output_language", "español académico")
    analysis_dimensions = experiment_profile.get("analysis_dimensions", [])

    profile_ctx = _profile_block(
        experiment_profile,
        include_methods=True,
        include_excluded=False
    )

    output_schema = {
        "source_filename": source_filename,
        "title": "",
        "paper_type": "",
        "research_problem": "",
        "objective": "",
        "task_type": "forecasting | prediction | estimation | classification | review | comparative_study | methodological_proposal | other",
        "target_domain": "",
        "target_variable_or_object": "",
        "temporal_horizon_or_scope": "",
        "methods_or_models": [],
        "method_families": [],
        "datasets_or_case_study": "",
        "input_variables_or_data_sources": [],
        "evaluation_metrics": [],
        "main_results": "",
        "reported_best_method_or_model": "",
        "limitations_or_gaps": "",
        "contribution": "",
        "relevance_for_state_of_art": "",
        "domain_specific_notes": "",
        "evidence": [
            {
                "claim": "",
                "supporting_quote": "",
                "chunk_id": ""
            }
        ],
    }

    prompt = f"""
Eres un agente de extracción científica para un sistema multiagente que genera estados del arte.

Tu tarea es extraer una ficha científica estructurada desde un paper, usando SOLO el contexto dado.

{profile_ctx}

Dimensiones de análisis esperadas:
{_as_json(analysis_dimensions)}

Reglas obligatorias:
- Usa SOLO la información del contexto. No inventes nada.
- Si un dato no aparece, escribe "no especificado" o lista vacía [] si es campo de lista.
- Distingue el tipo de tarea: predicción, clasificación, estimación, revisión, propuesta metodológica o estudio comparativo.
- En "method_families", clasifica usando SOLO las dimensiones metodológicas listadas arriba.
- En "evidence", incluye 2 a 5 ítems con cita textual breve y su chunk_id exacto del contexto.
- Redacta los textos en {output_language}.
- Devuelve SOLO JSON válido. No uses Markdown ni texto fuera del JSON.

Paper:
{source_filename}

Contexto:
{context}

Devuelve exactamente un JSON con esta estructura:

{_as_json(output_schema)}
"""
    return prompt


# ------------------------------------------------------------
# 2) CLASIFICACIÓN DE RELEVANCIA
# ------------------------------------------------------------

def build_relevance_classification_prompt(card, experiment_profile):
    """
    Prompt para clasificar la relevancia de una ficha respecto al tema configurado.
    """
    profile_ctx = _profile_block(
        experiment_profile,
        include_methods=True,
        include_excluded=True
    )
    relevance_rules = experiment_profile.get("relevance_rules", "")

    output_schema = {
        "task_type": "",
        "target_domain": "",
        "method_families": [],
        "relevance_level": "high | medium | low | exclude",
        "include_in_state_of_art": True,
        "relevance_reason": "",
    }

    prompt = f"""
Eres un clasificador académico para un sistema que genera estados del arte científicos.

Debes clasificar una ficha científica de acuerdo con la temática configurada.

{profile_ctx}

Criterios de relevancia:
{relevance_rules}

Reglas:
- Usa SOLO la ficha proporcionada. No inventes información.
- Si el paper NO pertenece al alcance científico, marca include_in_state_of_art = false y relevance_level = "exclude".
- Si pertenece directamente al tema, relevance_level = "high".
- Si es complementario, revisión o aproximación indirecta, relevance_level = "medium".
- Si aporta poco al tema, relevance_level = "low".
- En "method_families", usa SOLO las dimensiones metodológicas configuradas.
- Devuelve SOLO JSON válido.

Ficha:
{_as_json(card)}

Formato obligatorio:
{_as_json(output_schema)}
"""
    return prompt


# ------------------------------------------------------------
# 3) ANÁLISIS TEMÁTICO
# ------------------------------------------------------------

def build_thematic_analysis_prompt(compact_kb, experiment_profile):
    """
    Prompt para el Agente de Análisis Temático.
    Agrupa la KB en temas, detecta patrones, vacíos y propone estructura.
    """
    output_language = experiment_profile.get("output_language", "español académico")

    profile_ctx = _profile_block(
        experiment_profile,
        include_methods=True,
        include_excluded=True
    )
    analysis_dimensions = experiment_profile.get("analysis_dimensions", [])
    relevance_rules = experiment_profile.get("relevance_rules", "")

    output_schema = {
        "corpus_summary": {
            "included_papers": 0,
            "main_domains": [],
            "main_method_families": [],
            "general_observation": "",
        },
        "themes": [
            {
                "theme_id": "T1",
                "theme_name": "",
                "description": "",
                "representative_papers": [
                    {
                        "source_filename": "",
                        "title": "",
                        "reason": ""
                    }
                ],
                "main_methods": [],
                "typical_targets": [],
                "typical_metrics": [],
                "reported_strengths": [],
                "reported_limitations": [],
            }
        ],
        "method_evolution": "",
        "research_gaps": [],
        "suggested_state_of_art_structure": [
            {
                "section_title": "",
                "purpose": "",
                "themes_to_use": []
            }
        ],
    }

    prompt = f"""
Eres un agente de análisis temático para un sistema multiagente que genera estados del arte científicos.

Tu tarea es analizar una Knowledge Base de papers científicos y organizarla temáticamente.

{profile_ctx}

Dimensiones de análisis esperadas:
{_as_json(analysis_dimensions)}

Criterios de relevancia:
{relevance_rules}

Reglas obligatorias:
- Usa SOLO la Knowledge Base proporcionada. No inventes papers, resultados, métodos ni métricas.
- El perfil del experimento sirve como guía, pero NO debes forzar categorías que no aparezcan en la Knowledge Base.
- Agrupa los papers en temas científicos coherentes, idealmente de 3 a 7 temas.
- Cada tema debe citar papers representativos por su source_filename exacto.
- Identifica patrones metodológicos, fortalezas, limitaciones y vacíos de investigación.
- En "method_evolution", describe cómo evolucionan los métodos de lo simple a lo complejo.
- En "research_gaps", lista vacíos concretos y accionables.
- Propón una estructura lógica para redactar el estado del arte.
- Redacta en {output_language}.
- Devuelve SOLO JSON válido. No uses Markdown.

Knowledge Base:
{_as_json(compact_kb)}

Devuelve exactamente este JSON:

{_as_json(output_schema)}
"""
    return prompt
