"""Stage 03 -- dos correcciones genéricas, sin hardcodear ningún
experimento, dominio ni nombre de paper:

1. Exclusión determinista y auditable de reviews (review_exclusion.py):
   una ficha clasificada de forma confiable como paper_type=review o
   task_type=review, bajo policy.extraction_policy.exclude_reviews,
   nunca bloquea Stage 03 por carecer de methods_or_models/evaluation_
   metrics/main_results que nunca tuvo. Reutiliza include_in_state_of_
   art/relevance_level (relevance_level="exclude", ya reconocido por
   Stage 04 en corpus_filtering.py) en vez de un mecanismo paralelo.
   Fail-closed: clasificación incierta o contradictoria nunca excluye
   automáticamente.

2. scientific_cards_quality_check.csv se reconstruye al FINAL del
   flujo, después de TODAS las mutaciones (repair, title repair,
   relevancia, exclusión) -- nunca queda desfasado respecto al
   scientific_cards.jsonl final committed.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import sys
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


def _run_with_exclude_reviews(cards_by_source, *, exclude_reviews):
    """Ejecuta ExtractionAgent real con extraction_policy.exclude_
    reviews explícito -- devuelve (env, result)."""

    env = ExtractionAgentEnvironment(extraction_cards=cards_by_source)
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {
        "auto_rebuild": True, "exclude_reviews": exclude_reviews,
    }
    agent_input = AgentInput.from_dict(payload)
    result = ExtractionAgent(env.dependencies).execute(agent_input)
    return env, result


def _review_card(source="a.pdf"):
    card = complete_card(source)
    card.update({
        "paper_type": "review", "task_type": "review",
        "methods_or_models": [], "evaluation_metrics": [], "main_results": "no especificado",
    })
    return card


@scenario("R1. review + exclude_reviews=True -> excluida, no bloquea Stage 03")
def test_r1_review_excluded_does_not_block():
    review = _review_card("a.pdf")
    env, result = _run_with_exclude_reviews({"a.pdf": review}, exclude_reviews=True)

    assert result.quality_status == QualityStatus.APPROVED
    scientific = result.quality_metrics["scientific"]
    assert scientific["papers_excluded"] == 1
    assert scientific["papers_excluded_by_review_policy"] == 1
    assert scientific["papers_included"] == 1  # solo b.pdf, el resto del fixture estándar

    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    excluded = next(c for c in cards if c["source_filename"] == "a.pdf")
    assert excluded["include_in_state_of_art"] is False
    assert excluded["relevance_level"] == "exclude"
    # Nunca se inventa ni se rellena methods_or_models/evaluation_metrics/main_results.
    assert excluded["methods_or_models"] == []
    assert excluded["evaluation_metrics"] == []
    assert excluded["main_results"] == "no especificado"

    audit_df = pd.read_csv(env.paths["CARDS_REVIEW_EXCLUSION_AUDIT_CSV_PATH"])
    row = audit_df[audit_df["source_filename"] == "a.pdf"].iloc[0]
    assert row["action"] == "EXCLUDE"
    assert row["policy_rule"] == "exclude_reviews"
    assert "review" in row["reason"].lower()


@scenario("R2. review + exclude_reviews=False -> comportamiento normal (histórico), sigue bloqueando")
def test_r2_review_with_policy_off_blocks_as_before():
    review = _review_card("a.pdf")
    env, result = _run_with_exclude_reviews({"a.pdf": review}, exclude_reviews=False)
    assert result.quality_status == QualityStatus.NEEDS_REVISION

    # Clave ausente por completo -> usa el DEFAULT CANÓNICO del sistema
    # (exclude_reviews=True, ver _DEFAULT_EXTRACTION_POLICY en
    # generation_policy_config.py), NO el viejo default False -- una
    # review sin exclude_reviews explícito ahora sí se excluye,
    # exactamente como cualquier experimento nuevo sin overrides.
    env2 = ExtractionAgentEnvironment(extraction_cards={"a.pdf": _review_card("a.pdf")})
    result2 = ExtractionAgent(env2.dependencies).execute(env2.agent_input)
    assert result2.quality_status == QualityStatus.APPROVED


@scenario("R3. Paper empírico con campos críticos faltantes -> sigue bloqueando/reintentando, exclude_reviews=True no lo afecta")
def test_r3_empirical_paper_missing_fields_still_blocks():
    empirical = complete_card("a.pdf")
    empirical.update({
        "paper_type": "empirical", "task_type": "classification",
        "methods_or_models": [], "main_results": "no especificado",
    })
    env, result = _run_with_exclude_reviews({"a.pdf": empirical}, exclude_reviews=True)
    assert result.quality_status == QualityStatus.NEEDS_REVISION


@scenario("R4. Clasificación incierta/contradictoria (paper_type EXPLÍCITO distinto de review, título con marcador) -> NO se excluye automáticamente, fail-closed")
def test_r4_uncertain_classification_never_auto_excludes():
    # task_type nunca contradice (ver review_exclusion.py) -- la
    # única contradicción real es entre el tipo documental
    # estructurado (paper_type/document_type) y el título.
    contradictory = complete_card("a.pdf")
    contradictory.update({
        "title": "A Systematic Review of Existing Methods Motivates a New Hybrid Approach",
        "paper_type": "empirical", "task_type": "classification",
        "methods_or_models": [], "main_results": "no especificado",
    })
    env, result = _run_with_exclude_reviews({"a.pdf": contradictory}, exclude_reviews=True)
    # Sigue bloqueando -- nunca se excluyó a ciegas.
    assert result.quality_status == QualityStatus.NEEDS_REVISION


@scenario("R5. Ground Truth sigue sin entrar en generación/RAG/verificación -- este módulo no lo menciona en absoluto")
def test_r5_ground_truth_never_referenced():
    import inspect

    from src.tools.extraction import review_exclusion as module

    source = inspect.getsource(module)
    assert "ground_truth" not in source.lower()
    assert "gt" not in {tok.strip(",.()[]").lower() for tok in source.split()}


@scenario("Q1. Título reparado en el intento aparece también reparado en scientific_cards_quality_check.csv (no desfasado)")
def test_q1_repaired_title_reflected_in_quality_csv():
    original = complete_card("b.pdf", title="no especificado")
    env = ExtractionAgentEnvironment(
        extraction_cards={"b.pdf": original},
        repaired_titles={"b.pdf": "Exact title"},
    )
    result = ExtractionAgent(env.dependencies).execute(env.agent_input)
    assert result.quality_status in {QualityStatus.APPROVED, QualityStatus.APPROVED_WITH_WARNINGS}

    quality_df = pd.read_csv(env.paths["CARDS_QUALITY_CSV_PATH"])
    row = quality_df[quality_df["source_filename"] == "b.pdf"].iloc[0]
    assert row["title"] == "Exact title"
    assert row["title"] != "no especificado"

    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    jsonl_title = next(c["title"] for c in cards if c["source_filename"] == "b.pdf")
    assert jsonl_title == "Exact title"


@scenario("Q2. Consistencia por source_filename entre scientific_cards.jsonl y scientific_cards_quality_check.csv (mismos títulos, mismo conjunto de fichas)")
def test_q2_jsonl_and_quality_csv_consistency():
    original = complete_card("b.pdf", title="no especificado")
    review = _review_card("a.pdf")
    env = ExtractionAgentEnvironment(
        extraction_cards={"a.pdf": review, "b.pdf": original},
        repaired_titles={"b.pdf": "Exact title"},
    )
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input = AgentInput.from_dict(payload)
    ExtractionAgent(env.dependencies).execute(agent_input)

    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    quality_df = pd.read_csv(env.paths["CARDS_QUALITY_CSV_PATH"])

    jsonl_by_source = {c["source_filename"]: c["title"] for c in cards}
    quality_by_source = dict(zip(quality_df["source_filename"], quality_df["title"]))

    assert set(jsonl_by_source.keys()) == set(quality_by_source.keys())
    for source, title in jsonl_by_source.items():
        assert quality_by_source[source] == title, source


if __name__ == "__main__":
    for fn in (
        test_r1_review_excluded_does_not_block,
        test_r2_review_with_policy_off_blocks_as_before,
        test_r3_empirical_paper_missing_fields_still_blocks,
        test_r4_uncertain_classification_never_auto_excludes,
        test_r5_ground_truth_never_referenced,
        test_q1_repaired_title_reflected_in_quality_csv,
        test_q2_jsonl_and_quality_csv_consistency,
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
