"""Cascada de detección de reviews (Stage 03, ``review_exclusion.py``):
señal explícita de TIPO DOCUMENTAL (``document_type``/``paper_type``)
-> marcadores bibliográficos inequívocos del título (survey, review,
systematic/literature/scoping review, meta-analysis) -> fail-closed
(UNCERTAIN) cuando las señales de tipo documental se contradicen.

``task_type`` (tarea científica: classification, forecasting,
segmentation, detection, regression, generation, etc.) NUNCA participa
en esta detección -- ni como señal positiva de review ni como
contradicción. Es una dimensión semántica distinta a la de tipo
documental: un survey puede estudiar perfectamente "classification" o
cualquier otra tarea sin dejar de ser un survey. En estos tests,
``task_type`` solo aparece como contexto informativo del registro,
nunca como parte de lo que se está verificando que decida la
clasificación.

Causa raíz cerrada (v1 -> v2): la exclusión dependía exclusivamente de
``paper_type`` == ``"review"`` (un único literal, sin consultar el
título) -- y, en un primer intento de corrección, trataba
incorrectamente ``task_type`` como una segunda señal capaz de
contradecir un marcador de tipo documental, lo que producía
``UNCERTAIN`` en vez de ``EXCLUDE`` para un survey real cuyo
``task_type`` describía una tarea científica cualquiera (ver CASC-01/
CASC-11/CASC-12, registros reales exactos). La versión final consulta
``document_type``/``paper_type`` (tipo documental) y el título -- 
nunca ``task_type`` -- para decidir si un documento es review/survey.

Multidominio y genérico: ningún test usa contenido, dominio ni
experimento real -- los títulos son sintéticos, deliberadamente de
temas distintos entre sí."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

from agent_environment import ExtractionAgentEnvironment  # noqa: E402
from extraction_agent_doubles import complete_card  # noqa: E402

from src.agents.extraction_agent import ExtractionAgent  # noqa: E402
from src.contracts.agent_input import AgentInput  # noqa: E402
from src.contracts.agent_result import QualityStatus  # noqa: E402
from src.tools.extraction.review_exclusion import classify_review_exclusion  # noqa: E402

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


@scenario("CASC-01. Título con 'A Survey' explícito + paper_type sin clasificar (registro REAL exacto, incluido task_type='classification') -> EXCLUDE por marcador de título")
def test_casc_01_title_survey_unclassified_type():
    card = {"title": "Convolutional Neural Networks: A Survey", "paper_type": "no especificado", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"
    assert r["detected_title_marker"] == "survey"


@scenario("CASC-02. paper_type='survey' (sinónimo, antes no reconocido) -> EXCLUDE")
def test_casc_02_paper_type_survey_synonym():
    card = {"title": "no especificado", "paper_type": "survey", "task_type": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"


@scenario("CASC-03. 'Systematic Review'/'Literature Review'/'Scoping Review'/'Meta-Analysis' en título -> todos EXCLUDE")
def test_casc_03_all_title_markers_detected():
    titles = [
        "A Systematic Review of Deep Learning Approaches",
        "Machine Learning in Healthcare: A Literature Review",
        "Wearable Sensors: A Scoping Review",
        "Efficacy of Treatment X: A Meta-Analysis",
        "Efficacy of Treatment X: A Meta Analysis",
    ]
    for title in titles:
        card = {"title": title, "paper_type": "no especificado", "task_type": "no especificado"}
        r = classify_review_exclusion(card, exclude_reviews=True)
        assert r["action"] == "EXCLUDE", title


@scenario("CASC-04. Título sin ninguna señal + tipos sin clasificar -> KEEP (caso real: título irrecuperable, fail-closed correcto)")
def test_casc_04_no_signal_at_all_keeps():
    card = {"title": "no especificado", "paper_type": "no especificado", "task_type": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"
    assert r["classification"] == "UNKNOWN"


@scenario("CASC-05. Paper primario normal (sin marcador en título) -> KEEP, nunca se excluye por accidente")
def test_casc_05_normal_primary_paper_kept():
    card = {"title": "A Novel Deep Learning Method for Signal Classification", "paper_type": "empirical", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"


@scenario("CASC-06. Marcador de título presente PERO paper_type (tipo documental) contradice explícitamente -> UNCERTAIN, fail-closed, nunca EXCLUDE a ciegas (task_type presente en el registro, pero irrelevante para esta decisión)")
def test_casc_06_title_marker_contradicted_by_explicit_type_is_uncertain():
    card = {
        "title": "A Systematic Review of Existing Methods Motivates a New Hybrid Approach",
        "paper_type": "empirical", "task_type": "classification",
    }
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "UNCERTAIN"
    assert r["detected_title_marker"] == "systematic review"


@scenario("CASC-07. 'Overview' no matchea 'review' -- ningún falso positivo por substring")
def test_casc_07_overview_does_not_match_review_substring():
    card = {"title": "An Overview of Recent Advances in Signal Processing", "paper_type": "no especificado", "task_type": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"
    assert r["detected_title_marker"] is None


@scenario("CASC-08. Título vacío/no-especificado nunca se evalúa por marcador -- solo tipo estructurado, incluso si contuviera texto ambiguo por accidente")
def test_casc_08_unspecified_title_never_evaluated_for_markers():
    card = {"title": "no especificado", "paper_type": "empirical", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"
    assert r["detected_title_marker"] is None
    assert r["classification"] is None  # paper_type conocido (empirical), nunca UNKNOWN


@scenario("CASC-09. Integración real end-to-end: 2 papers (1 survey por título, 1 primario) -> Stage 03 ya no bloquea, la review queda excluida y auditada")
def test_casc_09_full_real_scenario_end_to_end():
    survey = complete_card("a.pdf")
    survey.update({
        "title": "Convolutional Neural Networks: A Survey",
        "paper_type": "no especificado", "task_type": "no especificado",
        "methods_or_models": [], "evaluation_metrics": [], "main_results": "no especificado",
    })
    primary = complete_card("b.pdf")
    primary.update({"paper_type": "empirical", "task_type": "classification"})

    env = ExtractionAgentEnvironment(extraction_cards={"a.pdf": survey, "b.pdf": primary})
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input = AgentInput.from_dict(payload)
    result = ExtractionAgent(env.dependencies).execute(agent_input)

    assert result.quality_status == QualityStatus.APPROVED
    scientific = result.quality_metrics["scientific"]
    assert scientific["papers_excluded_by_review_policy"] == 1

    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    survey_card = next(c for c in cards if c["source_filename"] == "a.pdf")
    assert survey_card["include_in_state_of_art"] is False
    assert survey_card["relevance_level"] == "exclude"
    # Nunca inventa methods_or_models/evaluation_metrics/main_results.
    assert survey_card["methods_or_models"] == []
    assert survey_card["main_results"] == "no especificado"

    import pandas as pd
    audit_df = pd.read_csv(env.paths["CARDS_REVIEW_EXCLUSION_AUDIT_CSV_PATH"])
    row = audit_df[audit_df["source_filename"] == "a.pdf"].iloc[0]
    assert row["detected_title_marker"] == "survey"


@scenario("CASC-10. exclude_reviews=False -> comportamiento histórico preservado (compatibilidad con experimentos anteriores)")
def test_casc_10_backward_compatible_when_policy_off():
    card = {"title": "Convolutional Neural Networks: A Survey", "paper_type": "no especificado", "task_type": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=False)
    assert r["action"] == "KEEP"


@scenario("CASC-11. Registro REAL exacto (computers-12-00151.pdf): title='...A Survey', paper_type='no especificado', task_type='classification' -> EXCLUDE (bug reportado: task_type NO debe contradecir)")
def test_casc_11_real_record_survey_with_classification_task():
    card = {"title": "Convolutional Neural Networks: A Survey", "paper_type": "no especificado", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"
    assert r["detected_title_marker"] == "survey"
    assert r["detected_task_type"] == "classification"  # reportado, pero nunca usado en la decisión


@scenario("CASC-12. task_type nunca contradice, para ninguna tarea científica -- forecasting/segmentation/detection/regression/generation")
def test_casc_12_task_type_never_contradicts_any_scientific_task():
    for task in ("classification", "forecasting", "segmentation", "detection", "regression", "generation"):
        card = {"title": "A Systematic Review of Methods", "paper_type": "no especificado", "task_type": task}
        r = classify_review_exclusion(card, exclude_reviews=True)
        assert r["action"] == "EXCLUDE", task


@scenario("CASC-13. Registro REAL exacto (2004.02806v1.pdf): title/paper_type irrecuperables, task_type='classification' -> KEEP/UNKNOWN, nunca EXCLUDE, nunca se asume review")
def test_casc_13_real_record_unrecoverable_title_is_unknown_not_review():
    card = {"title": "no especificado", "paper_type": "no especificado", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"
    assert r["classification"] == "UNKNOWN"


@scenario("CASC-14. Solo paper_type/document_type (tipo documental) puede contradecir un marcador de título -- nunca task_type")
def test_casc_14_only_document_type_field_can_contradict():
    # paper_type explícito y distinto de review + marcador de título -> UNCERTAIN
    card_contradicts = {"title": "A Systematic Review of X", "paper_type": "research article", "task_type": "classification"}
    r1 = classify_review_exclusion(card_contradicts, exclude_reviews=True)
    assert r1["action"] == "UNCERTAIN"

    # MISMO título, paper_type vacío, CUALQUIER task_type -> EXCLUDE (task_type nunca es la fuente de contradicción)
    card_no_contradiction = {"title": "A Systematic Review of X", "paper_type": "no especificado", "task_type": "research article"}
    r2 = classify_review_exclusion(card_no_contradiction, exclude_reviews=True)
    assert r2["action"] == "EXCLUDE"


@scenario("CASC-15. document_type tiene prioridad sobre paper_type cuando ambos existen (cascada, no votación)")
def test_casc_15_document_type_takes_priority_over_paper_type():
    card = {"title": "no especificado", "document_type": "survey", "paper_type": "empirical", "task_type": "classification"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"


@scenario("CASC-16. Regresión multidominio: papers primarios de distintos dominios y distintas task_type, sin marcador en título -> siempre KEEP")
def test_casc_16_multidomain_primary_papers_regression():
    domains = [
        ("A Deep Learning Approach for Medical Image Segmentation", "empirical", "segmentation"),
        ("Predicting Stock Prices Using Recurrent Neural Networks", "empirical", "forecasting"),
        ("A Hybrid Model for Anomaly Detection in Network Traffic", "research article", "detection"),
        ("Text Generation with Transformer Architectures", "empirical", "generation"),
        ("Regression Analysis of Climate Variables Using Ensemble Methods", "empirical", "regression"),
    ]
    for title, paper_type, task_type in domains:
        card = {"title": title, "paper_type": paper_type, "task_type": task_type}
        r = classify_review_exclusion(card, exclude_reviews=True)
        assert r["action"] == "KEEP", (title, r)
        assert r["classification"] is None  # tipo conocido (empirical/research article), no UNKNOWN


@scenario("CASC-17. Política EXCLUDE/KEEP/UNKNOWN x válida/inválida: EXCLUDE (review confirmada) nunca bloquea Stage03 por campos faltantes, sin importar cuántos falten")
def test_casc_17_excluded_review_never_blocks_regardless_of_missing_fields():
    from src.tools.extraction.review_exclusion import apply_review_exclusion_policy
    from src.tools.extraction.revision_strategy import build_revision_plan

    card = {
        "source_filename": "review.pdf", "title": "A Systematic Review of X",
        "paper_type": "no especificado", "task_type": "no especificado",
        "methods_or_models": [], "main_results": "no especificado", "evidence": [],
    }
    result = apply_review_exclusion_policy([card], exclude_reviews=True, created_at="t")
    assert result["cards"][0]["include_in_state_of_art"] is False
    rows = build_revision_plan(result["cards"], [], [])
    assert rows == []


@scenario("CASC-18. KEEP con tipo documental CONOCIDO (paper_type explícito) + ficha inválida -> retry/halt normal, reason_code histórico sin cambios")
def test_casc_18_known_type_invalid_card_uses_historical_reason_code():
    from src.tools.extraction.review_exclusion import apply_review_exclusion_policy
    from src.tools.extraction.revision_strategy import build_revision_plan

    card = {
        "source_filename": "primary.pdf", "title": "no especificado", "research_problem": "p", "objective": "o",
        "paper_type": "empirical", "task_type": "classification", "target_domain": "d",
        "methods_or_models": ["m"], "main_results": "no especificado", "evidence": ["e1"],
    }
    result = apply_review_exclusion_policy([card], exclude_reviews=True, created_at="t")
    rows = build_revision_plan(result["cards"], [], [])
    assert len(rows) == 1
    assert rows[0]["primary_reason_code"] == rows[0]["underlying_reason_code"]
    assert rows[0]["primary_reason_code"] != "DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID"


@scenario("CASC-19. UNKNOWN documental + ficha VÁLIDA -> continúa sin bloquear (nunca entra al revision_plan)")
def test_casc_19_unknown_type_valid_card_continues():
    from src.tools.extraction.review_exclusion import apply_review_exclusion_policy
    from src.tools.extraction.revision_strategy import build_revision_plan

    card = {
        "source_filename": "unknown_valid.pdf", "title": "Un título completamente válido y recuperado",
        "research_problem": "p", "objective": "o", "paper_type": "no especificado", "task_type": "regression",
        "target_domain": "d", "methods_or_models": ["m"], "main_results": "resultado real reportado", "evidence": ["e1"],
    }
    result = apply_review_exclusion_policy([card], exclude_reviews=True, created_at="t")
    rows = build_revision_plan(result["cards"], [], [])
    assert rows == []


@scenario("CASC-20. UNKNOWN documental + ficha INVÁLIDA -> DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID explícito, sigue bloqueando (nunca excluye, nunca pasa silenciosamente), causa raíz original conservada")
def test_casc_20_unknown_type_invalid_card_gets_explicit_reason_code():
    from src.tools.extraction.review_exclusion import apply_review_exclusion_policy
    from src.tools.extraction.revision_strategy import build_revision_plan

    card = {
        "source_filename": "unknown_invalid.pdf", "title": "no especificado", "research_problem": "p", "objective": "o",
        "paper_type": "no especificado", "task_type": "classification", "target_domain": "d",
        "methods_or_models": ["m"], "main_results": "no especificado", "evidence": ["e1"],
    }
    result = apply_review_exclusion_policy([card], exclude_reviews=True, created_at="t")
    # Nunca se excluye -- sigue siendo include_in_state_of_art por defecto (no tocado por UNKNOWN).
    assert "include_in_state_of_art" not in result["cards"][0] or result["cards"][0].get("include_in_state_of_art") is not False
    rows = build_revision_plan(result["cards"], [], [])
    assert len(rows) == 1
    assert rows[0]["primary_reason_code"] == "DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID"
    assert rows[0]["underlying_reason_code"] in ("MISSING_OR_INVALID_TITLE", "MISSING_CRITICAL_FIELDS")


@scenario("CASC-21. Compatibilidad: exclude_reviews=False -> nunca se calcula UNKNOWN, comportamiento histórico exacto preservado")
def test_casc_21_backward_compatible_no_unknown_when_policy_off():
    from src.tools.extraction.review_exclusion import apply_review_exclusion_policy
    from src.tools.extraction.revision_strategy import build_revision_plan

    card = {
        "source_filename": "x.pdf", "title": "no especificado", "research_problem": "p", "objective": "o",
        "paper_type": "no especificado", "task_type": "classification", "target_domain": "d",
        "methods_or_models": ["m"], "main_results": "no especificado", "evidence": ["e1"],
    }
    result = apply_review_exclusion_policy([card], exclude_reviews=False, created_at="t")
    rows = build_revision_plan(result["cards"], [], [])
    assert rows[0]["primary_reason_code"] == "MISSING_CRITICAL_FIELDS"
    assert rows[0]["primary_reason_code"] == rows[0]["underlying_reason_code"]


@scenario("CASC-22 (contrato). document_type='no especificado' (string truthy pero NO informativo) + paper_type='survey' informativo, título sin marcador -> EXCLUDE (cascada por informatividad, no por 'primer string no vacío')")
def test_casc_22_document_type_unspecified_never_masks_informative_paper_type():
    card = {"document_type": "no especificado", "paper_type": "survey", "title": "Un título cualquiera sin marcador"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"


@scenario("CASC-23 (contrato). document_type='' + paper_type='systematic review' -> EXCLUDE (mismo caso con string vacío en vez de 'no especificado')")
def test_casc_23_document_type_empty_string_never_masks_informative_paper_type():
    card = {"document_type": "", "paper_type": "systematic review", "title": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"


@scenario("CASC-24 (contrato). document_type='research article' (informativo) + paper_type='survey' -> prioridad ABSOLUTA de document_type, nunca votación entre ambos campos")
def test_casc_24_document_type_priority_is_not_a_vote():
    card = {"document_type": "research article", "paper_type": "survey", "title": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    # document_type es informativo y NO es review -> gana la cascada,
    # paper_type='survey' se ignora por completo (nunca se promedia ni
    # se vota entre los dos campos estructurados).
    assert r["action"] == "KEEP"
    assert r["detected_document_type"] == "research article"


@scenario("CASC-25 (contrato). document_type sin informar + paper_type='research article' (informativo, no-review) + título con marcador -> UNCERTAIN, porque paper_type SÍ es señal estructurada informativa que contradice el título")
def test_casc_25_informative_paper_type_contradicts_title_marker():
    card = {"document_type": "no especificado", "paper_type": "research article", "title": "A Survey of Methods"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "UNCERTAIN"
    assert r["detected_title_marker"] == "survey"


@scenario("CASC-26 (contrato). Ambos campos estructurados sin informar + título con marcador -> EXCLUDE por señal de título (único caso donde el título decide)")
def test_casc_26_both_structured_fields_uninformative_title_decides():
    card = {"document_type": "no especificado", "paper_type": "no especificado", "title": "A Systematic Review of X"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "EXCLUDE"


@scenario("CASC-27 (contrato). Ambos campos estructurados sin informar + título sin informar -> KEEP + UNKNOWN, nunca se asume review")
def test_casc_27_no_signal_anywhere_is_unknown():
    card = {"document_type": "no especificado", "paper_type": "no especificado", "title": "no especificado"}
    r = classify_review_exclusion(card, exclude_reviews=True)
    assert r["action"] == "KEEP"
    assert r["classification"] == "UNKNOWN"


if __name__ == "__main__":
    for fn in (
        test_casc_01_title_survey_unclassified_type,
        test_casc_02_paper_type_survey_synonym,
        test_casc_03_all_title_markers_detected,
        test_casc_04_no_signal_at_all_keeps,
        test_casc_05_normal_primary_paper_kept,
        test_casc_06_title_marker_contradicted_by_explicit_type_is_uncertain,
        test_casc_07_overview_does_not_match_review_substring,
        test_casc_08_unspecified_title_never_evaluated_for_markers,
        test_casc_09_full_real_scenario_end_to_end,
        test_casc_10_backward_compatible_when_policy_off,
        test_casc_11_real_record_survey_with_classification_task,
        test_casc_12_task_type_never_contradicts_any_scientific_task,
        test_casc_13_real_record_unrecoverable_title_is_unknown_not_review,
        test_casc_14_only_document_type_field_can_contradict,
        test_casc_15_document_type_takes_priority_over_paper_type,
        test_casc_16_multidomain_primary_papers_regression,
        test_casc_17_excluded_review_never_blocks_regardless_of_missing_fields,
        test_casc_18_known_type_invalid_card_uses_historical_reason_code,
        test_casc_19_unknown_type_valid_card_continues,
        test_casc_20_unknown_type_invalid_card_gets_explicit_reason_code,
        test_casc_21_backward_compatible_no_unknown_when_policy_off,
        test_casc_22_document_type_unspecified_never_masks_informative_paper_type,
        test_casc_23_document_type_empty_string_never_masks_informative_paper_type,
        test_casc_24_document_type_priority_is_not_a_vote,
        test_casc_25_informative_paper_type_contradicts_title_marker,
        test_casc_26_both_structured_fields_uninformative_title_decides,
        test_casc_27_no_signal_anywhere_is_unknown,
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
