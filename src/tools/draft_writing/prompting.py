from __future__ import annotations
import json
from .retrieval import safe_str


def language_instruction(output_language):
    normalized = safe_str(output_language).casefold()
    if normalized in {"es", "español", "espanol", "spanish"}:
        return "Redacta en español académico."
    if normalized in {"en", "inglés", "ingles", "english"}:
        return "Write in academic English."
    return f"Redacta todos los campos en {output_language}."


def assign_section_budgets(outline_sections, target_total_words):
    section_count = max(len(outline_sections), 1)
    base_target = max(80, int(int(target_total_words) / section_count))
    budgets = {}
    for section in outline_sections:
        section_id = safe_str(section.get("section_id"))
        budgets[section_id] = {
            "target_words": base_target,
            "minimum_words": max(50, int(base_target * 0.65)),
            "maximum_words": max(90, int(base_target * 1.40)),
        }
    return budgets


def build_section_prompt(section, evidence, quantitative_context, previous_errors, policy, *, previous_claims_for_identity=None):
    section_id = safe_str(section.get("section_id"))
    allowed_citations = [f"[{row['source_filename']} | {row['chunk_id']}]" for row in evidence]
    budgets = policy.get("section_budgets") or assign_section_budgets(
        policy.get("outline_sections") or [section],
        policy.get("target_total_words", 1000),
    )
    budget = budgets[section_id]
    no_sources = not evidence
    if previous_claims_for_identity is None:
        identity_rules = (
            "17. IDENTIDAD DE CLAIMS: esta sección se redacta por primera vez -- "
            "cada elemento de claims debe llevar \"identity_action\": \"NEW\" y "
            "\"parent_claim_uids\": []."
        )
    else:
        previous_claims_block = json.dumps(
            [{"claim_uid": c["claim_uid"], "claim_text": c["claim_text"]} for c in previous_claims_for_identity],
            ensure_ascii=False, indent=2,
        )
        identity_rules = f"""
17. IDENTIDAD DE CLAIMS -- esta sección ya existía; a continuación se
    listan sus claims previos, cada uno con su claim_uid real. Para
    CADA claim que devuelvas en "claims", declara "identity_action" y
    "parent_claim_uids" según cuál de estos cuatro casos corresponde:

    - CONTINUE: este claim es una versión revisada de UN claim previo
      (aunque cambie de posición o se reescriba). "parent_claim_uids"
      debe tener EXACTAMENTE ese claim_uid, ej. ["<uid>"].
    - NEW: este claim es una afirmación genuinamente nueva, que no
      existía antes en esta sección. "parent_claim_uids": [].
    - SPLIT_CHILD: este claim es UNA de varias partes en que se dividió
      un claim previo. "parent_claim_uids" debe tener EXACTAMENTE el
      claim_uid de ese claim previo, ej. ["<uid>"] (igual forma que
      CONTINUE, pero se está dividiendo en más de un claim nuevo).
    - MERGE: este claim fusiona DOS O MÁS claims previos en uno solo.
      "parent_claim_uids" debe tener los claim_uid de TODOS los claims
      previos que se fusionaron, ej. ["<uid1>", "<uid2>"].

    Nunca inventes un claim_uid que no aparezca en la lista de claims
    previos de abajo. Si un claim previo no corresponde a ningún claim
    de tu respuesta, simplemente no lo continúes -- no hace falta
    declarar nada sobre los claims previos que decidiste eliminar.

CLAIMS PREVIOS DE ESTA SECCIÓN (con su claim_uid real):
{previous_claims_block}
""".strip()
    special_rule = (
        "Esta sección no tiene fuentes asignadas. Redacta únicamente "
        "una apertura o cierre organizativo, sin datos, resultados, "
        "comparaciones ni afirmaciones factuales. Devuelve claims=[] "
        "y no insertes citas."
        if no_sources
        else
        "Cada oración sustantiva debe terminar con una o más citas "
        "exactas tomadas de allowed_citations. Las citas deben aparecer "
        "también dentro de draft_text, no solo en claims. Debes copiar "
        "todo el texto de cada oración sustantiva sin sus citas, incluidos "
        "conectores como 'For instance', 'Moreover', 'Similarly' o sus "
        "equivalentes en el idioma de salida. Omite cualquier oración "
        "sustantiva que no tenga evidencia documental."
    )
    return f"""
Eres el agente redactor de un sistema multiagente para estados del arte científicos.

REGLAS:
1. Usa exclusivamente la evidencia proporcionada.
2. No uses conocimiento externo ni Ground Truth.
3. No cites papers o chunks fuera de allowed_citations.
4. No inventes autores, años, datasets, métricas, valores ni resultados.
5. No sustituyas una cita por otra.
6. Las citas trazables siempre usan [source_filename | chunk_id].
7. El estilo bibliográfico {policy.get('citation_style', '')} no autoriza inventar autores o años.
8. {language_instruction(policy.get('output_language', 'español académico'))}
9. Modo de escritura: {policy.get('writing_mode', '')}. Enfoque: {policy.get('focus_mode', '')}.
10. {special_rule}
11. Cada elemento de claims debe tener:
    - claim: copia literal completa de una oración sustantiva sin sus citas,
      conservando conectores discursivos iniciales;
    - supporting_citations: exactamente las citas que aparecen en esa oración.
12. Nunca pongas citas únicamente en supporting_citations: deben aparecer
    primero en la oración correspondiente dentro de draft_text.
13. Un valor numérico solo puede escribirse si aparece literalmente en uno
    de los chunks citados por esa misma oración.
14. No cierres la sección con una inferencia sin cita. Si una transición
    no está respaldada, omítela.
15. Extensión objetivo: {budget['target_words']} palabras;
    rango orientativo: {budget['minimum_words']}-{budget['maximum_words']}.
16. Devuelve únicamente JSON válido.
{identity_rules}

FORMATO:
{{
  "section_id": "{section_id}",
  "section_title": {json.dumps(safe_str(section.get('section_title')), ensure_ascii=False)},
  "draft_text": "",
  "claims": [
    {{
      "claim": "",
      "supporting_citations": [
        "[source_filename | chunk_id]"
      ],
      "identity_action": "NEW",
      "parent_claim_uids": []
    }}
  ]
}}

SECCIÓN DEL ESQUEMA:
{json.dumps(section, ensure_ascii=False, indent=2)}

ALLOWED_CITATIONS:
{json.dumps(allowed_citations, ensure_ascii=False, indent=2)}

EVIDENCIA:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

CONTEXTO CUANTITATIVO CONFIRMADO:
{json.dumps(quantitative_context, ensure_ascii=False, indent=2)}

ERRORES DE UN INTENTO ANTERIOR:
{json.dumps(previous_errors or [], ensure_ascii=False, indent=2)}
""".strip()


def build_source_free_organizational_section(section, output_language="español"):
    section_id = safe_str(section.get("section_id"))
    section_title = safe_str(section.get("section_title"))
    normalized_language = safe_str(output_language).casefold()
    if normalized_language in {"es", "español", "espanol", "spanish", "español académico"}:
        text = (
            "Esta sección presenta el alcance y la organización de la revisión. "
            "Su función es orientar la lectura y establecer la transición hacia "
            "el análisis de la evidencia científica desarrollado en las secciones siguientes."
        )
    elif normalized_language in {"en", "inglés", "ingles", "english", "academic english"}:
        text = (
            "This section presents the scope and organization of the review. "
            "Its purpose is to guide the reader and establish the transition toward "
            "the evidence-based analysis developed in the following sections."
        )
    else:
        raise ValueError(f"No existe una plantilla organizativa segura para el idioma de salida {output_language!r}.")
    return {
        "section_id": section_id,
        "section_title": section_title,
        "draft_text": text,
        "claims": [],
        "generation_attempt": 0,
        "section_validation": {
            "validation_ok": True,
            "errors": [],
            "citation_errors": [],
            "claim_errors": [],
            "numeric_errors": [],
            "valid_citation_count": 0,
            "substantive_sentence_count": 0,
            "source_free_organizational_section": True,
        },
        "deterministic_normalization": {
            "applied": True,
            "normalization_version": "v3_source_free_organizational_template",
            "source_free_organizational_section": True,
            "reason": "No evidence assigned by outline and section type permits an organizational introduction or conclusion.",
        },
    }


def build_section_revision_prompt(
    section, evidence, quantitative_context, policy, *, previous_section_draft_text, issues, previous_claims_for_identity
):
    """Modo REVISION: reutiliza ``build_section_prompt`` completo (mismas
    reglas científicas, mismo formato JSON, mismas restricciones de citas)
    y AÑADE un bloque correctivo al final — no reescribe ni duplica las
    reglas originales.

    ``previous_claims_for_identity``: lista de ``{"claim_uid", "claim_text"}``
    de los claims previos de ESTA sección (ver ``resolve_claim_identity`` --
    la identidad se declara explícitamente por 06/el LLM en la respuesta,
    nunca se infiere después por similitud de texto)."""

    base_prompt = build_section_prompt(
        section, evidence, quantitative_context, [], policy,
        previous_claims_for_identity=previous_claims_for_identity,
    )
    issues_block = json.dumps(issues, ensure_ascii=False, indent=2)
    forced_uid_issues = [i for i in issues if i.get("claim_uid")]
    forced_block = ""
    if forced_uid_issues:
        forced_lines = "\n".join(
            f'    - El claim con claim_uid="{i["claim_uid"]}" DEBE seguir siendo ESE '
            f'mismo claim_uid en tu respuesta (identity_action="CONTINUE", '
            f'parent_claim_uids=["{i["claim_uid"]}"]) -- no es una decisión libre '
            "para este claim en particular; la observación de verificación ya lo "
            "identificó explícitamente."
            for i in forced_uid_issues
        )
        forced_block = f"""

IMPORTANTE -- IDENTIDAD FORZADA para los claims señalados abajo:
{forced_lines}
"""
    return (
        base_prompt
        + f"""

MODO REVISIÓN — RONDA {policy.get('round_number', '?')}:
Esta sección ya fue redactada antes. A continuación se listan observaciones
REALES de verificación (Agente 07) sobre claims de ESTA sección que deben
corregirse. No modifiques nada que no esté relacionado con estas
observaciones. No uses Ground Truth ni conocimiento externo al corpus. Usa
únicamente la evidencia ya provista arriba en EVIDENCIA.

BORRADOR ANTERIOR DE ESTA SECCIÓN:
{previous_section_draft_text}

OBSERVACIONES A RESOLVER:
{issues_block}
{forced_block}
""".strip()
    )
