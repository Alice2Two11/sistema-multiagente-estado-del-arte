"""LLM Judge (celda 21): rúbrica, prompt, validación estricta, reintentos.

Mismo patrón que ``translation.py`` (Bloque 3): recibe ``llm_factory``
(construye un cliente NUEVO por intento, igual que ``get_llm(...)`` dentro
del bucle real) en vez de construir el LLM internamente.

No incluye el cacheo por fingerprint (``judge_cache_valid``,
``LLM_JUDGE_MANIFEST_PATH``) ni la escritura de
``attempt_{n}.txt``/``llm_judge_evaluation.json``/``llm_judge_scores.csv``
— eso es persistencia, migrada aparte.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from src.tools.evaluation.text_normalization import safe_str
from src.utils.json_parsing import parse_json_safely as _parse_json_safely

PROMPT_VERSION = "v5_rubric_reference_comparison_strict_json"

JUDGE_CRITERIA = {
    "coherence": "Continuidad lógica entre ideas, secciones y transiciones.",
    "organization": "Estructura clara, progresión temática y distribución adecuada.",
    "critical_depth": "Comparación, contraste, limitaciones, vacíos y análisis crítico.",
    "synthesis_quality": "Integración de múltiples trabajos en una narrativa propia.",
    "argumentative_clarity": (
        "Claridad de la tesis, precisión de afirmaciones y legibilidad académica."
    ),
}


def balanced_excerpt(text: Any, maximum_chars: int) -> tuple[str, bool]:
    value = safe_str(text)
    if len(value) <= maximum_chars:
        return value, False

    part = maximum_chars // 3
    middle_start = max(0, len(value) // 2 - part // 2)
    excerpt = (
        value[:part]
        + "\n\n[...]\n\n"
        + value[middle_start : middle_start + part]
        + "\n\n[...]\n\n"
        + value[-part:]
    )
    return excerpt, True


def build_judge_prompt(
    *,
    topic_name: str,
    source_stage: str,
    automatic_metrics: dict[str, float],
    factual_metrics: dict[str, Any],
    generated_judge_text: str,
    ground_truth_judge_text: str,
    previous_errors: list[str] | None = None,
) -> str:
    error_instruction = ""
    if previous_errors:
        error_instruction = (
            "\nLa respuesta anterior fue inválida. Corrige estos errores:\n"
            + json.dumps(previous_errors, ensure_ascii=False, indent=2)
        )

    return f"""
Eres un evaluador académico. Compara un estado del arte generado
contra una revisión de literatura real publicada.

TEMA:
{topic_name}

REGLAS:
1. Evalúa únicamente calidad académica y cobertura comparativa.
2. No sustituyas la auditoría factual del pipeline; esa dimensión procede
   del upstream seleccionado ({source_stage}) y se reporta por separado.
3. Usa la rúbrica 1-5:
   1 = muy deficiente,
   2 = deficiente,
   3 = aceptable,
   4 = buena,
   5 = excelente.
4. Cada puntuación debe incluir una justificación concreta.
5. evidence_from_generated puede incluir hasta tres fragmentos breves,
   cada uno de máximo 20 palabras.
6. missing_topics_or_omissions debe basarse en contenido visible
   en el Ground Truth y ausente o débil en el texto generado.
7. No inventes papers, autores ni resultados.
8. Devuelve únicamente JSON válido.

CRITERIOS:
{json.dumps(JUDGE_CRITERIA, ensure_ascii=False, indent=2)}

MÉTRICAS AUTOMÁTICAS:
{json.dumps(automatic_metrics, ensure_ascii=False, indent=2)}

MÉTRICAS FACTUALES:
{json.dumps(factual_metrics, ensure_ascii=False, indent=2)}

ESTADO DEL ARTE GENERADO:
{generated_judge_text}

GROUND TRUTH — REVISIÓN DE LITERATURA:
{ground_truth_judge_text}

FORMATO:
{{
  "scores": {{
    "coherence": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "organization": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "critical_depth": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "synthesis_quality": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "argumentative_clarity": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }}
  }},
  "strengths": [],
  "organization_differences": [],
  "missing_topics_or_omissions": [
    {{
      "topic": "",
      "ground_truth_basis": "",
      "importance": "",
      "search_keywords": []
    }}
  ],
  "overall_assessment": ""
}}
{error_instruction}
""".strip()


def validate_judge_result(result: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(result, dict):
        return ["judge_result_not_object"]

    scores = result.get("scores")
    if not isinstance(scores, dict):
        errors.append("scores_not_object")
        return errors

    missing_criteria = sorted(set(JUDGE_CRITERIA) - set(scores))
    unknown_criteria = sorted(set(scores) - set(JUDGE_CRITERIA))
    if missing_criteria:
        errors.append("missing_criteria:" + ",".join(missing_criteria))
    if unknown_criteria:
        errors.append("unknown_criteria:" + ",".join(unknown_criteria))

    for criterion in JUDGE_CRITERIA:
        item = scores.get(criterion)
        if not isinstance(item, dict):
            errors.append(f"{criterion}:not_object")
            continue

        try:
            score = float(item.get("score"))
            if score < 1 or score > 5 or score != int(score):
                errors.append(f"{criterion}:score_not_integer_1_to_5")
        except Exception:
            errors.append(f"{criterion}:invalid_score")

        if not safe_str(item.get("justification")):
            errors.append(f"{criterion}:empty_justification")

        evidence = item.get("evidence_from_generated")
        if not isinstance(evidence, list):
            errors.append(f"{criterion}:evidence_not_list")
        else:
            if len(evidence) > 3:
                errors.append(f"{criterion}:too_many_evidence_items")
            for evidence_item in evidence:
                if len(safe_str(evidence_item).split()) > 20:
                    errors.append(f"{criterion}:evidence_item_over_20_words")

    for list_field in ["strengths", "organization_differences", "missing_topics_or_omissions"]:
        if not isinstance(result.get(list_field), list):
            errors.append(f"{list_field}:not_list")

    omissions = result.get("missing_topics_or_omissions", [])
    if isinstance(omissions, list):
        for index, omission in enumerate(omissions, start=1):
            if not isinstance(omission, dict):
                errors.append(f"omission_{index}:not_object")
                continue
            for key in ["topic", "ground_truth_basis", "importance", "search_keywords"]:
                if key not in omission:
                    errors.append(f"omission_{index}:missing_{key}")
            if "search_keywords" in omission and not isinstance(
                omission["search_keywords"], list
            ):
                errors.append(f"omission_{index}:search_keywords_not_list")

    if not safe_str(result.get("overall_assessment")):
        errors.append("empty_overall_assessment")

    return errors


parse_json_safely = _parse_json_safely
"""Extrae JSON aunque la respuesta venga envuelta en un bloque Markdown.

Reexportada tal cual desde ``src.utils.json_parsing`` (utilidad
neutral, sin dependencia hacia ningún dominio) -- es la MISMA función,
no una copia ni un wrapper, para que no exista ninguna duplicación de
lógica. Ahora también reutilizada por
``src/adapters/draft_writing_runtime.py`` (Agent06) sin que eso cree
una dependencia arquitectónica hacia ``src.tools.evaluation``.
Cualquier código que la importe desde aquí (``from src.tools.
evaluation.llm_judge import parse_json_safely``) sigue funcionando sin
cambios -- semántica y comportamiento idénticos, copia literal de
notebook 08, celda 1."""


def run_llm_judge(
    *,
    topic_name: str,
    source_stage: str,
    automatic_metrics: dict[str, float],
    factual_metrics: dict[str, Any],
    generated_plain_text: str,
    ground_truth_plain_text: str,
    max_generated_chars: int,
    max_ground_truth_chars: int,
    max_attempts: int,
    llm_factory: Callable[[], Any],
    parse_json_safely_fn: Callable[[str], Any] = parse_json_safely,
) -> dict[str, Any]:
    """Reproduce el bucle de reintentos real (celda 21, rama ``else`` —
    sin cacheo). Un intento fallido NO se reintenta con el mismo cliente:
    se construye una instancia NUEVA por intento, igual que
    ``get_llm(...)`` dentro del bucle real. Ningún error se silencia:
    ``json_parse_error``/errores de validación se acumulan en
    ``previous_errors`` y se inyectan en el siguiente prompt, tal cual.

    Devuelve ``{"result": dict, "raw_attempts": list[str], "judge_mode": "new"}``
    (persistencia de ``attempt_{n}.txt``/JSON/manifest queda fuera de este
    módulo).
    """

    generated_judge_text, generated_truncated = balanced_excerpt(
        generated_plain_text, max_generated_chars
    )
    ground_truth_judge_text, ground_truth_truncated = balanced_excerpt(
        ground_truth_plain_text, max_ground_truth_chars
    )

    previous_errors: list[str] = []
    llm_judge_result = None
    raw_attempts: list[str] = []

    for _attempt in range(1, max_attempts + 1):
        prompt = build_judge_prompt(
            topic_name=topic_name,
            source_stage=source_stage,
            automatic_metrics=automatic_metrics,
            factual_metrics=factual_metrics,
            generated_judge_text=generated_judge_text,
            ground_truth_judge_text=ground_truth_judge_text,
            previous_errors=previous_errors,
        )

        llm = llm_factory()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = safe_str(response.content)
        raw_attempts.append(raw_text)

        try:
            parsed = parse_json_safely_fn(raw_text)
        except Exception as error:
            previous_errors = [f"json_parse_error:{error}"]
            continue

        errors = validate_judge_result(parsed)
        if not errors:
            llm_judge_result = parsed
            break
        previous_errors = errors

    if llm_judge_result is None:
        raise ValueError(
            "El LLM Judge no produjo un resultado válido después de "
            f"{max_attempts} intentos. Errores: {previous_errors}"
        )

    return {
        "result": llm_judge_result,
        "raw_attempts": raw_attempts,
        "judge_mode": "new",
        "generated_excerpt_truncated": generated_truncated,
        "ground_truth_excerpt_truncated": ground_truth_truncated,
    }


def build_judge_score_rows(llm_judge_result: dict[str, Any]) -> list[dict[str, Any]]:
    judge_score_rows = []
    for criterion in JUDGE_CRITERIA:
        item = llm_judge_result["scores"][criterion]
        judge_score_rows.append(
            {
                "metric": criterion,
                "score_1_to_5": int(item["score"]),
                "justification": safe_str(item["justification"]),
                "evidence_from_generated": json.dumps(
                    item["evidence_from_generated"], ensure_ascii=False
                ),
            }
        )
    return judge_score_rows
