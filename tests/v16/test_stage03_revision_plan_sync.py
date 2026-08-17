"""Sincronización de ``scientific_cards_revision_plan.csv`` con las
cards finales de Stage 03.

Causa raíz: el revision plan se escribía UNA SOLA VEZ, al comienzo del
intento 2 (``extraction_agent.py``, para decidir qué reparar --
title_sources, etc.), usando las cards tal como llegaron del intento 1
-- ANTES de title repair del intento 2, de la reclasificación de
relevancia y de la exclusión de reviews. El archivo en disco quedaba
desactualizado: una ficha ya excluida como review seguía apareciendo
como "inválida" bloqueante, y con el esquema de columnas viejo (sin
``underlying_reason_code``), aunque la DECISIÓN real del agente
(``quality_status``, vía ``_scientific_reason_codes``) siempre
recalculó correctamente sobre las cards finales en memoria -- el bug
era exclusivamente de sincronización del artefacto persistido, nunca
de la decisión en sí.

Fix: el mismo bloque que ya reconstruye ``scientific_cards_summary.
csv``/``scientific_cards_quality_check.csv`` al final del flujo (tras
title repair, reclasificación de relevancia y exclusión de reviews)
ahora también reconstruye y sobrescribe ``scientific_cards_revision_
plan.csv`` con las cards finales -- sin borrar archivos manualmente,
sin ``--force-rerun``, sin tocar Stage 04-08.

Multidominio y genérico: ningún test usa contenido, dominio, filename
ni experimento real."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

import pandas as pd  # noqa: E402

from agent_environment import ExtractionAgentEnvironment  # noqa: E402
from extraction_agent_doubles import complete_card  # noqa: E402

from src.agents.extraction_agent import ExtractionAgent  # noqa: E402
from src.contracts.agent_input import AgentInput  # noqa: E402
from src.contracts.agent_result import QualityStatus  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


def _run_two_attempts(*, extraction_cards, repair_cards=None, repaired_titles=None):
    """Ejecuta intento 1 y luego intento 2 (mismo project_dir, mismo
    agent_input salvo attempt_number) sobre el mismo entorno real --
    devuelve (env, result1, result2)."""

    env = ExtractionAgentEnvironment(
        extraction_cards=extraction_cards,
        repair_cards=repair_cards or {},
        repaired_titles=repaired_titles or {},
    )
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input1 = AgentInput.from_dict(payload)
    result1 = ExtractionAgent(env.dependencies).execute(agent_input1)

    payload2 = dict(payload)
    payload2["attempt_number"] = 2
    agent_input2 = AgentInput.from_dict(payload2)
    result2 = ExtractionAgent(env.dependencies).execute(agent_input2)
    return env, result1, result2


@scenario("SYNC-01. El revision plan final refleja las cards YA sincronizadas: una review excluida por título nunca vuelve a aparecer, aunque tuviera campos faltantes")
def test_sync_01_excluded_review_never_reappears_in_final_plan():
    survey = complete_card("a.pdf")
    survey.update({
        "title": "Convolutional Neural Networks: A Survey",
        "paper_type": "no especificado", "task_type": "classification",
        "methods_or_models": [], "evaluation_metrics": [], "main_results": "no especificado",
    })
    valid_second = complete_card("b.pdf")

    env, result1, result2 = _run_two_attempts(extraction_cards={"a.pdf": survey, "b.pdf": valid_second})

    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert "a.pdf" not in plan["source_filename"].tolist()


@scenario("SYNC-02. UNKNOWN documental + ficha inválida que persiste tras el intento 2 -> aparece en el plan FINAL con DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID y underlying_reason_code conservado")
def test_sync_02_unknown_invalid_persists_with_explicit_reason_code():
    survey = complete_card("a.pdf")
    survey.update({
        "title": "Convolutional Neural Networks: A Survey",
        "paper_type": "no especificado", "task_type": "classification",
        "methods_or_models": [], "evaluation_metrics": [], "main_results": "no especificado",
    })
    unknown_invalid = complete_card("b.pdf")
    unknown_invalid.update({
        "title": "no especificado", "paper_type": "no especificado", "task_type": "classification",
        "main_results": "no especificado",
    })
    # El repair del intento 2 sigue sin poder completar main_results/título.
    still_invalid = dict(unknown_invalid)

    env, result1, result2 = _run_two_attempts(
        extraction_cards={"a.pdf": survey, "b.pdf": unknown_invalid},
        repair_cards={"b.pdf": still_invalid}, repaired_titles={},
    )

    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert plan["source_filename"].tolist() == ["b.pdf"]
    row = plan[plan["source_filename"] == "b.pdf"].iloc[0]
    assert row["primary_reason_code"] == "DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID"
    assert row["underlying_reason_code"] == "MISSING_CRITICAL_FIELDS"


@scenario("SYNC-03. El esquema de columnas del CSV final siempre incluye underlying_reason_code -- nunca el esquema viejo")
def test_sync_03_final_csv_schema_includes_underlying_reason_code():
    card = complete_card("a.pdf")
    env, result1, result2 = _run_two_attempts(extraction_cards={"a.pdf": card})
    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert "underlying_reason_code" in plan.columns
    assert "primary_reason_code" in plan.columns


@scenario("SYNC-04. El revision plan se sobrescribe (mtime cambia) en la misma fase en que se regeneran summary/quality/manifest, sin borrar archivos manualmente")
def test_sync_04_revision_plan_mtime_updates_alongside_summary_quality():
    card = complete_card("a.pdf")
    env = ExtractionAgentEnvironment(extraction_cards={"a.pdf": card})
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input1 = AgentInput.from_dict(payload)
    ExtractionAgent(env.dependencies).execute(agent_input1)

    revision_path = env.paths["CARDS_REVISION_PLAN_CSV_PATH"]
    quality_path = env.paths["CARDS_QUALITY_CSV_PATH"]
    mtime_revision_before = revision_path.stat().st_mtime_ns if revision_path.exists() else 0
    mtime_quality_before = quality_path.stat().st_mtime_ns
    time.sleep(0.05)

    payload2 = dict(payload)
    payload2["attempt_number"] = 2
    agent_input2 = AgentInput.from_dict(payload2)
    ExtractionAgent(env.dependencies).execute(agent_input2)

    assert revision_path.stat().st_mtime_ns > mtime_revision_before
    assert quality_path.stat().st_mtime_ns > mtime_quality_before


@scenario("SYNC-05. Multidominio: distintas combinaciones review/UNKNOWN/válida en el mismo lote -> el plan final refleja exactamente la política, sin nombres de dominio concretos")
def test_sync_05_multidomain_mixed_batch_final_plan_matches_policy():
    review_card = complete_card("a.pdf")
    review_card.update({
        "title": "A Systematic Review of Signal Processing Techniques",
        "paper_type": "no especificado", "task_type": "forecasting",
        "methods_or_models": [], "main_results": "no especificado",
    })
    valid_primary = complete_card("b.pdf")
    valid_primary.update({"paper_type": "empirical", "task_type": "segmentation"})

    env, result1, result2 = _run_two_attempts(extraction_cards={"a.pdf": review_card, "b.pdf": valid_primary})

    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    # Ninguna de las dos fichas debería quedar bloqueante: la review se
    # excluye, la otra es válida desde el inicio.
    assert plan.empty or "a.pdf" not in plan["source_filename"].tolist()


if __name__ == "__main__":
    for fn in (
        test_sync_01_excluded_review_never_reappears_in_final_plan,
        test_sync_02_unknown_invalid_persists_with_explicit_reason_code,
        test_sync_03_final_csv_schema_includes_underlying_reason_code,
        test_sync_04_revision_plan_mtime_updates_alongside_summary_quality,
        test_sync_05_multidomain_mixed_batch_final_plan_matches_policy,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    raise SystemExit(1 if failed else 0)
