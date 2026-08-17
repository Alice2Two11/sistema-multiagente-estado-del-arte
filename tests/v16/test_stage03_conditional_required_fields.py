"""Contrato de campos críticos de scientific cards, SEMÁNTICAMENTE
CONDICIONAL (``card_validation.py``, ``required_fields_for_card`` /
``is_domain_agnostic_paper``).

Causa raíz cerrada: ``target_domain`` se exigía como campo crítico
UNIVERSAL para cualquier paper, sin distinguir su rol científico. Un
paper metodológico/fundacional/de propósito general (ej.
``paper_type="methodological_proposal"``) no reclama un dominio de
aplicación concreto -- exigirlo fuerza al LLM a inventar un dominio
que el paper nunca declara, o bloquea Stage03 con ``MISSING_CRITICAL_
FIELDS`` para un documento ya confirmado ``INCLUDE`` por el Corpus
Eligibility Gate.

Fix: los campos obligatorios ahora dependen del ROL del paper
(``paper_type``/``task_type``), no de una lista fija universal.
``target_domain`` solo se exige para papers que sí reclaman un dominio
de aplicación (estudios empíricos/domain-specific). Nunca se inventa
ni se rellena -- solo se reconoce como no aplicable cuando corresponde.

Multidominio y genérico: los marcadores de rol (``methodological_
proposal``, ``foundational_method``, etc.) describen el TIPO de
contribución, nunca un dominio concreto -- ningún test usa contenido,
dominio, filename ni experimento real."""

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
from src.tools.extraction.card_validation import is_domain_agnostic_paper, required_fields_for_card  # noqa: E402
from src.tools.extraction.revision_strategy import missing_critical_fields  # noqa: E402

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


def _card(**overrides):
    base = {
        "source_filename": "x.pdf", "title": "T", "research_problem": "p", "objective": "o",
        "task_type": "classification", "target_domain": "no especificado",
        "methods_or_models": ["m"], "main_results": "r", "evidence": ["e"],
    }
    base.update(overrides)
    return base


@scenario("REQ-01. Estudio empírico + target_domain faltante -> sigue siendo inválida/bloqueante")
def test_req_01_empirical_missing_target_domain_still_blocks():
    card = _card(paper_type="empirical", task_type="classification")
    assert is_domain_agnostic_paper(card) is False
    missing = missing_critical_fields(card)
    assert "target_domain" in missing


@scenario("REQ-02. Propuesta metodológica general + target_domain faltante -> válida si el resto está completo")
def test_req_02_methodological_proposal_missing_target_domain_is_valid():
    card = _card(paper_type="methodological_proposal", task_type="other")
    assert is_domain_agnostic_paper(card) is True
    missing = missing_critical_fields(card)
    assert missing == []


@scenario("REQ-03. Paper de algoritmo domain-agnostic -> nunca se inventa target_domain (permanece ausente/no especificado)")
def test_req_03_domain_agnostic_algorithm_never_fabricates_target_domain():
    card = _card(paper_type="general_purpose_method", task_type="other", target_domain="no especificado")
    fields = required_fields_for_card(card)
    assert "target_domain" not in fields
    # El valor del campo NUNCA se toca -- sigue exactamente como llegó.
    assert card["target_domain"] == "no especificado"


@scenario("REQ-04. Paper metodológico relevante + INCLUDE -> flujo real completo no hace HALT solo por target_domain")
def test_req_04_methodological_include_no_halt_end_to_end():
    methodological = complete_card("a.pdf")
    methodological.update({"paper_type": "methodological_proposal", "task_type": "other", "target_domain": "no especificado"})

    env = ExtractionAgentEnvironment(extraction_cards={"a.pdf": methodological})
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input = AgentInput.from_dict(payload)
    result = ExtractionAgent(env.dependencies).execute(agent_input)

    assert result.quality_status == QualityStatus.APPROVED
    assert result.requested_transition.action.value != "HALT_STAGE"
    import pandas as pd
    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert plan.empty


@scenario("REQ-05. Paper metodológico con methodology/main_results faltante -> SÍ entra al revision_plan (campos universales siguen exigidos)")
def test_req_05_methodological_paper_missing_universal_field_still_blocks():
    card = _card(paper_type="methodological_proposal", task_type="other", methods_or_models=[])
    missing = missing_critical_fields(card)
    assert "methods_or_models" in missing
    assert "target_domain" not in missing  # condicional, no aplica -- pero methods_or_models sí es universal

    card2 = _card(paper_type="foundational_method", task_type="other", main_results="no especificado")
    missing2 = missing_critical_fields(card2)
    assert "main_results" in missing2


@scenario("REQ-06. Comportamiento idéntico en energía, medicina, NLP, ciberseguridad y otros dominios -- ninguna regla depende del dominio")
def test_req_06_identical_behavior_across_domains():
    domains_task_types = ["forecasting", "diagnosis_support", "text_classification", "intrusion_detection", "recommendation"]
    for task_type in domains_task_types:
        methodological = _card(paper_type="methodological_proposal", task_type=task_type)
        assert missing_critical_fields(methodological) == [], task_type

        empirical = _card(paper_type="empirical", task_type=task_type)
        assert "target_domain" in missing_critical_fields(empirical), task_type


@scenario("REQ-07. Corrida end-to-end: corpus con papers empíricos + metodológicos generales -> Stage03 continúa si los campos realmente aplicables están completos")
def test_req_07_end_to_end_mixed_corpus_continues():
    empirical_complete = complete_card("a.pdf")
    empirical_complete.update({"paper_type": "empirical", "task_type": "classification", "target_domain": "un dominio cualquiera"})
    methodological = complete_card("b.pdf")
    methodological.update({"paper_type": "methodological_proposal", "task_type": "other", "target_domain": "no especificado"})

    env = ExtractionAgentEnvironment(extraction_cards={"a.pdf": empirical_complete, "b.pdf": methodological})
    payload = env.agent_input.to_dict()
    payload["policy"]["signature"]["extraction_policy"] = {"auto_rebuild": True, "exclude_reviews": True}
    agent_input = AgentInput.from_dict(payload)
    result = ExtractionAgent(env.dependencies).execute(agent_input)

    assert result.quality_status == QualityStatus.APPROVED
    import pandas as pd
    plan = pd.read_csv(env.paths["CARDS_REVISION_PLAN_CSV_PATH"])
    assert plan.empty


if __name__ == "__main__":
    for fn in (
        test_req_01_empirical_missing_target_domain_still_blocks,
        test_req_02_methodological_proposal_missing_target_domain_is_valid,
        test_req_03_domain_agnostic_algorithm_never_fabricates_target_domain,
        test_req_04_methodological_include_no_halt_end_to_end,
        test_req_05_methodological_paper_missing_universal_field_still_blocks,
        test_req_06_identical_behavior_across_domains,
        test_req_07_end_to_end_mixed_corpus_continues,
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
