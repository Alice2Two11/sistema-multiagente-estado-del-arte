from __future__ import annotations
import json

PROMPT_VERSION = "v16_thematic_agent_01"


def build_thematic_prompt(context, valid_sources, title_map, repair_plan=None):
    repair = ""
    if repair_plan:
        repair = "\nREPARACIÓN DIRIGIDA:\n" + json.dumps(
            repair_plan,
            ensure_ascii=False,
        )
    return (
        "Analiza exclusivamente la KB proporcionada. No uses Ground Truth, "
        "bibliografía ni conocimiento externo.\n"
        "Devuelve un objeto JSON con corpus_summary, themes, research_gaps, "
        "suggested_state_of_art_structure y comparative_dimensions.\n"
        "Cada tema debe tener representative_papers con source_filename y title exactos. "
        "Cada gap y dimensión debe tener fuentes válidas.\n"
        f"FUENTES VÁLIDAS: {json.dumps(valid_sources, ensure_ascii=False)}\n"
        f"TÍTULOS: {json.dumps(title_map, ensure_ascii=False)}\n"
        f"CORPUS: {json.dumps(context, ensure_ascii=False)}{repair}"
    )
