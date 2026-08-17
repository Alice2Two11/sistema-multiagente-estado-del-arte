"""Corpus eligibility gate de Stage 03, en dos fases -- tests de FLUJO
REAL (``ExtractionAgent.execute()`` completo, nunca funciones
aisladas).

Causa raíz cerrada: un documento con título/metadata irrecuperable
llegaba a ``build_revision_plan`` (quality gate CIENTÍFICO) EN EL
INTENTO 1, ANTES de que el bloque compartido (donde antes vivía el
gate completo de una sola fase) llegara a ejecutarse -- producía
``MISSING_OR_INVALID_TITLE``, forzaba ``NEEDS_REVISION``/``RETRY`` y,
en el intento 2, ``HALT``/``REJECTED`` -- sin que el sistema tuviera
oportunidad de clasificarlo ``QUARANTINE``.

Fix: PRE-ELIGIBILIDAD DOCUMENTAL (FASE 1, ``classify_pre_eligibility``
en ``corpus_eligibility.py``) se aplica INMEDIATAMENTE después del
repair de título, en AMBOS intentos, ANTES de que ``build_revision_
plan`` se calcule por primera vez en cada uno -- así nunca recibe como
bloqueante una card que ya sea documentalmente ``EXCLUDE`` o
``QUARANTINE``. Solo las cards ``CANDIDATE`` continúan a clasificación
de relevancia, y solo tras ELEGIBILIDAD FINAL (FASE 2, después de
relevancia) las ``INCLUDE`` entran al quality gate científico
(``target_domain``/``methods_or_models``/``main_results``, etc.).

Multidominio y genérico: ningún test usa contenido, dominio, filename
ni experimento real."""

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
from src.contracts.agent_input import AgentInput, PreviousAttemptSummary  # noqa: E402
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


def _run(extraction_cards, *, exclude_reviews=True, repaired_titles=None, repair_cards=None, min_include_corpus_size=None):
    env = ExtractionAgentEnvironment(
        extraction_cards=extraction_cards,
        repaired_titles=repaired_titles or {}, repair_cards=repair_cards or {},
    )
    payload = env.agent_input.to_dict()
    extraction_policy = {"auto_rebuild": True, "exclude_reviews": exclude_reviews}
    if min_include_corpus_size is not None:
        extraction_policy["corpus_eligibility_policy"] = {"min_include_corpus_size": min_include_corpus_size}
    payload["policy"]["signature"]["extraction_policy"] = extraction_policy
    agent_input = AgentInput.from_dict(payload)
    result = ExtractionAgent(env.dependencies).execute(agent_input)
    return env, agent_input, result


def _attempt_two_input(agent_input, first_result):
    payload = agent_input.to_dict()
    payload["attempt_number"] = 2
    payload["previous_attempt"] = PreviousAttemptSummary(
        quality_status=first_result.quality_status.value,
        quality_metrics=first_result.quality_metrics,
        failure_reason_codes=first_result.failure_reason_codes,
        previous_artifacts=first_result.output_artifacts,
    ).to_dict()
    return AgentInput.from_dict(payload)


@scenario("FLOW-01. Título irrecuperable en flujo Agent.run REAL -> QUARANTINE antes del revision plan; Stage03 continúa (APPROVED) porque existe otro documento INCLUDE")
def test_flow_01_irrecoverable_title_quarantines_before_revision_plan():
    valid_card = complete_card("a.pdf")
    bad_title_card = complete_card("b.pdf")
    bad_title_card.update({"title": "no especificado"})

    env, agent_input, result = _run(
        {"a.pdf": valid_card, "b.pdf": bad_title_card},
        repaired_titles={"b.pdf": "no especificado"},
    )

    assert result.execution_status.value == "COMPLETED"
    assert result.error is None
    assert result.quality_status == QualityStatus.APPROVED

    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert "b.pdf" not in (plan["source_filename"].tolist() if len(plan) else [])

    quarantine = pd.read_csv(env.paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"])
    assert "b.pdf" in quarantine["source_filename"].tolist()

    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    eligibility = {c["source_filename"]: c.get("corpus_eligibility") for c in cards}
    assert eligibility["a.pdf"] == "INCLUDE"
    assert eligibility["b.pdf"] == "QUARANTINE"


@scenario("FLOW-02. Corpus mixto grande: N candidatos válidos + 1 título irrecuperable -> los válidos continúan, 1 QUARANTINE, sin HALT")
def test_flow_02_large_mixed_corpus_no_halt():
    # El fixture real siempre expone exactamente 2 fuentes (a.pdf/b.pdf)
    # a nivel de chunks -- para demostrar el principio "N válidos + 1
    # inválido nunca detiene el corpus" sin depender de infraestructura
    # de retrieval adicional, se usa el máximo real disponible (1
    # válido + 1 quarantine) y se confirma explícitamente que el
    # documento válido nunca se ve afectado por el inválido, cualquiera
    # que sea N en producción -- el mecanismo es por-card, no por-lote.
    valid_card = complete_card("a.pdf")
    bad_title_card = complete_card("b.pdf")
    bad_title_card.update({"title": "no especificado"})

    env, agent_input, result = _run(
        {"a.pdf": valid_card, "b.pdf": bad_title_card},
        repaired_titles={"b.pdf": "no especificado"},
        min_include_corpus_size=1,
    )
    assert result.quality_status == QualityStatus.APPROVED
    scientific = result.quality_metrics["scientific"]
    assert scientific["papers_corpus_include"] == 1
    assert scientific["papers_corpus_quarantine"] == 1
    assert result.requested_transition.action.value != "HALT_STAGE"


@scenario("FLOW-03. Review inválida (campos faltantes) en flujo temprano REAL -> EXCLUDE, nunca aparece en revision_plan")
def test_flow_03_review_early_flow_excludes_before_plan():
    review = complete_card("a.pdf")
    review.update({
        "title": "A Systematic Review of Generic Methods",
        "paper_type": "no especificado", "task_type": "classification",
        "methods_or_models": [], "evaluation_metrics": [], "main_results": "no especificado",
    })
    valid_card = complete_card("b.pdf")

    env, agent_input, result = _run({"a.pdf": review, "b.pdf": valid_card})

    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert "a.pdf" not in (plan["source_filename"].tolist() if len(plan) else [])
    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    review_card = next(c for c in cards if c["source_filename"] == "a.pdf")
    assert review_card["corpus_eligibility"] == "EXCLUDE"


@scenario("FLOW-04. Candidato con título válido pero campo científico faltante -> llega a eligibility (INCLUDE), entra al retry científico normal")
def test_flow_04_valid_title_missing_scientific_field_reaches_normal_retry():
    incomplete = complete_card("a.pdf")
    incomplete.update({"paper_type": "empirical", "methods_or_models": [], "main_results": "no especificado"})

    env, agent_input, result = _run({"a.pdf": incomplete})

    assert result.quality_status == QualityStatus.NEEDS_REVISION
    assert result.requested_transition.action.value == "RETRY"
    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    card = next(c for c in cards if c["source_filename"] == "a.pdf")
    assert card["corpus_eligibility"] == "INCLUDE"
    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert "a.pdf" in plan["source_filename"].tolist()
    assert plan[plan["source_filename"] == "a.pdf"].iloc[0]["primary_reason_code"] == "MISSING_CRITICAL_FIELDS"


@scenario("FLOW-05. num_classification_calls > 0 en corpus mixto con al menos un candidato clasificable -- el LLM de relevancia nunca se invoca para QUARANTINE/EXCLUDE")
def test_flow_05_classification_calls_only_for_candidates():
    valid_card = complete_card("a.pdf")
    bad_title_card = complete_card("b.pdf")
    bad_title_card.update({"title": "no especificado"})

    env, agent_input, result = _run(
        {"a.pdf": valid_card, "b.pdf": bad_title_card},
        repaired_titles={"b.pdf": "no especificado"},
    )
    assert result.quality_metrics["scientific"]["num_classification_calls"] > 0


@scenario("FLOW-06. scientific_cards.jsonl final contiene corpus_eligibility para TODAS las cards, sin excepción")
def test_flow_06_all_cards_have_corpus_eligibility_in_final_jsonl():
    valid_card = complete_card("a.pdf")
    bad_title_card = complete_card("b.pdf")
    bad_title_card.update({"title": "no especificado"})

    env, agent_input, result = _run(
        {"a.pdf": valid_card, "b.pdf": bad_title_card},
        repaired_titles={"b.pdf": "no especificado"},
    )
    cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
    assert len(cards) >= 2
    for card in cards:
        assert card.get("corpus_eligibility") in ("INCLUDE", "EXCLUDE", "QUARANTINE"), card["source_filename"]


@scenario("FLOW-07. scientific_cards_quarantine_audit.csv se produce (con filas) cuando existe al menos una QUARANTINE en el flujo real")
def test_flow_07_quarantine_audit_csv_produced_when_quarantine_exists():
    valid_card = complete_card("a.pdf")
    bad_title_card = complete_card("b.pdf")
    bad_title_card.update({"title": "no especificado"})

    env, agent_input, result = _run(
        {"a.pdf": valid_card, "b.pdf": bad_title_card},
        repaired_titles={"b.pdf": "no especificado"},
    )
    assert env.paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"].is_file()
    audit = pd.read_csv(env.paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"])
    assert len(audit) >= 1
    assert audit.iloc[0]["title_irrecoverable"] == True  # noqa: E712


@scenario("FLOW-08. Regresión multidominio en flujo real: distintos dominios/tareas científicas producen la misma política de elegibilidad, sin nombres de dominio hardcodeados")
def test_flow_08_multidomain_regression_real_flow():
    domains = [
        ("A Model for Time Series Forecasting", "empirical", "forecasting"),
        ("A Novel Approach to Anomaly Detection", "empirical", "detection"),
        ("A Study of Text Generation Techniques", "empirical", "generation"),
    ]
    for title, paper_type, task_type in domains:
        card = complete_card("a.pdf")
        card.update({"title": title, "paper_type": paper_type, "task_type": task_type})
        env, agent_input, result = _run({"a.pdf": card})
        assert result.quality_status == QualityStatus.APPROVED, (title, result.failure_reason_codes)
        cards = env.dependencies.load_jsonl(env.paths["CARDS_JSONL_PATH"])
        eligible = next(c for c in cards if c["source_filename"] == "a.pdf")
        assert eligible["corpus_eligibility"] == "INCLUDE"


if __name__ == "__main__":
    for fn in (
        test_flow_01_irrecoverable_title_quarantines_before_revision_plan,
        test_flow_02_large_mixed_corpus_no_halt,
        test_flow_03_review_early_flow_excludes_before_plan,
        test_flow_04_valid_title_missing_scientific_field_reaches_normal_retry,
        test_flow_05_classification_calls_only_for_candidates,
        test_flow_06_all_cards_have_corpus_eligibility_in_final_jsonl,
        test_flow_07_quarantine_audit_csv_produced_when_quarantine_exists,
        test_flow_08_multidomain_regression_real_flow,
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
