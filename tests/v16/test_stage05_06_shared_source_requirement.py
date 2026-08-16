"""Contrato compartido Stage 05 (``outline_generation``) / Stage 06
(``draft_writing``) de qué secciones pueden legítimamente carecer de
``papers_to_use``/evidencia -- fuente única de verdad en ``src/tools/
shared/section_source_requirement.py`` (``classify_section_source_
requirement``), importada por ambas etapas.

Causa raíz cerrada: Stage 05 reconocía introducción/discusión/gaps-
vacíos/cierre/conclusión como source-free; Stage 06 (``section_allows_
no_sources``) solo reconocía introducción/conclusión/cierre -- una
sección con ``section_type="gaps"`` (o "discusion") aprobada por 05
terminaba en ``MISSING_SECTION_EVIDENCE`` en 06. Ahora ambas etapas
consumen la misma clasificación.

Multidominio y genérico: ningún test usa contenido de un experimento
real, ningún ``section_id``/título/dominio concreto hardcodeado en la
lógica de producción."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

from test_agent06_v16 import Env  # noqa: E402

from src.adapters.draft_writing_runtime import DraftWritingRuntime  # noqa: E402
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.tools.draft_writing.validation import section_allows_no_sources  # noqa: E402
from src.tools.outline_generation.outline_validation import validate_outline, reason_codes  # noqa: E402
from src.tools.outline_generation.source_repair import section_allows_empty_papers  # noqa: E402
from src.tools.shared.section_source_requirement import (  # noqa: E402
    SOURCE_FREE_ORGANIZATIONAL,
    SOURCE_REQUIRED,
    classify_section_source_requirement,
)

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


def _stage05_outline_with_section(section_type, section_title, papers_to_use=None):
    section = {
        "section_id": "SX", "section_title": section_title, "section_type": section_type,
        "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"],
        "papers_to_use": papers_to_use or [],
    }
    outline = {
        "title": "t", "objective": "o", "narrative_strategy": "n",
        "sections": [section], "paper_coverage_summary": [],
    }
    return outline


def _run_stage06_section(section_type, section_title, papers_to_use=None):
    """Ejecuta DraftWritingAgent real con UNA sección del tipo dado --
    devuelve (result, draft_or_none)."""

    e = Env(attempt=1)
    outline_path = e.inp / "outline.json"
    outline = json.loads(outline_path.read_text())
    outline["sections"] = [{
        "section_id": "SX", "section_title": section_title, "section_type": section_type,
        "purpose": "p", "key_arguments": [], "evidence_needs": [],
        "papers_to_use": papers_to_use or [],
    }]
    outline_path.write_text(json.dumps(outline), encoding="utf-8")
    (e.inp / "mapping.csv").write_text("section_id,source_filename,title\n")

    def invoke(prompt):
        raise AssertionError("no debía invocarse LLM para una sección source-free organizacional")

    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(e.ai)
    draft = None
    if (e.out / "state_of_art_draft.json").exists():
        draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    return result, draft


@scenario("SF-01. Introducción sin papers -> Stage 05 aprueba y Stage 06 permite (source-free)")
def test_sf_01_introduccion():
    outline = _stage05_outline_with_section("introduccion", "Introducción")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert report["validation_ok"] is True
    assert "MISSING_REQUIRED_SECTION_PAPERS" not in reason_codes(report)

    assert section_allows_no_sources({"section_type": "introduccion", "section_title": "Introducción"})
    result, draft = _run_stage06_section("introduccion", "Introducción")
    assert result.quality_status.value == "APPROVED"
    assert draft["sections"][0]["claims"] == []


@scenario("SF-02. Conclusión sin papers -> permite en ambas etapas")
def test_sf_02_conclusion():
    outline = _stage05_outline_with_section("conclusion", "Conclusiones del estudio")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert "MISSING_REQUIRED_SECTION_PAPERS" not in reason_codes(report)

    result, draft = _run_stage06_section("conclusion", "Conclusiones del estudio")
    assert result.quality_status.value == "APPROVED"
    assert draft["sections"][0]["claims"] == []


@scenario("SF-03. Gaps/vacíos sin papers -> permite en ambas etapas (el bug real reportado: MISSING_SECTION_EVIDENCE)")
def test_sf_03_gaps():
    outline = _stage05_outline_with_section("gaps", "Vacíos identificados en la literatura")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert "MISSING_REQUIRED_SECTION_PAPERS" not in reason_codes(report)

    result, draft = _run_stage06_section("gaps", "Vacíos identificados en la literatura")
    assert result.execution_status.value == "COMPLETED"
    assert result.error is None
    assert result.quality_status.value == "APPROVED"
    assert draft["sections"][0]["claims"] == []


@scenario("SF-04. Discusión organizativa sin papers -> permite en ambas etapas (Stage 06 antes NO la reconocía)")
def test_sf_04_discusion():
    outline = _stage05_outline_with_section("discusion", "Discusión general de los hallazgos")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert "MISSING_REQUIRED_SECTION_PAPERS" not in reason_codes(report)

    result, draft = _run_stage06_section("discusion", "Discusión general de los hallazgos")
    assert result.execution_status.value == "COMPLETED"
    assert result.error is None
    assert result.quality_status.value == "APPROVED"


@scenario("SF-05. linea_tematica sin papers -> Stage 05 y Stage 06 rechazan (nunca source-free)")
def test_sf_05_linea_tematica_rejected():
    outline = _stage05_outline_with_section("linea_tematica", "Métodos de aprendizaje profundo")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert report["validation_ok"] is False
    assert "MISSING_REQUIRED_SECTION_PAPERS" in reason_codes(report)

    assert not section_allows_no_sources({"section_type": "linea_tematica", "section_title": "Métodos de aprendizaje profundo"})
    result, draft = _run_stage06_section("linea_tematica", "Métodos de aprendizaje profundo")
    assert result.execution_status.value == "FAILED"
    assert result.error["type"] == "ValueError"
    assert "MISSING_SECTION_EVIDENCE" in result.error["message"]


@scenario("SF-06. Fundamentos sin papers -> rechaza en ambas etapas")
def test_sf_06_fundamentos_rejected():
    outline = _stage05_outline_with_section("fundamentos", "Fundamentos teóricos")
    report = validate_outline(outline, set(), 1, 5, [], [], [], [])
    assert report["validation_ok"] is False

    result, draft = _run_stage06_section("fundamentos", "Fundamentos teóricos")
    assert result.execution_status.value == "FAILED"
    assert "MISSING_SECTION_EVIDENCE" in result.error["message"]


@scenario("SF-07. section_type estructurado tiene prioridad sobre palabras accidentales del título")
def test_sf_07_structured_type_has_priority_over_title_words():
    # section_type NO organizacional, pero el título contiene "conclusiones"
    # por accidente -- debe seguir siendo SOURCE_REQUIRED en ambas etapas.
    section = {"section_type": "linea_tematica", "section_title": "Hacia unas conclusiones preliminares sobre el tema"}
    assert classify_section_source_requirement(section) == SOURCE_REQUIRED
    assert not section_allows_empty_papers(section)
    assert not section_allows_no_sources(section)

    # Sin section_type -- el mismo título SÍ dispara el fallback organizativo.
    section_no_type = {"section_type": "", "section_title": "Conclusiones del estudio"}
    assert classify_section_source_requirement(section_no_type) == SOURCE_FREE_ORGANIZATIONAL


@scenario("SF-08. Sección source-free genera claims=[] y cero citas -- nunca texto científico no sustentado")
def test_sf_08_source_free_produces_no_claims_no_citations():
    result, draft = _run_stage06_section("cierre", "Cierre del estado del arte")
    section = draft["sections"][0]
    assert section["claims"] == []
    assert "[" not in section["draft_text"]  # sin marcadores de cita estructurada
    assert section["section_id"] == "SX"


@scenario("SF-09. MISSING_SECTION_EVIDENCE sigue funcionando (fail-closed) para secciones sustantivas sin evidencia")
def test_sf_09_missing_section_evidence_still_fail_closed():
    for section_type, title in (
        ("linea_tematica", "Comparación de arquitecturas"),
        ("comparacion", "Comparación de resultados"),
        ("metodologia", "Diseño metodológico"),
    ):
        result, draft = _run_stage06_section(section_type, title)
        assert result.execution_status.value == "FAILED", section_type
        assert "MISSING_SECTION_EVIDENCE" in result.error["message"], section_type


@scenario("SF-10. Compatibilidad Stage05 -> Stage06: toda sección que 05 aprueba como source-free, 06 la acepta sin excepción")
def test_sf_10_stage05_stage06_compatibility():
    organizational_cases = [
        ("introduccion", "Introducción"),
        ("discusion", "Discusión"),
        ("gaps", "Vacíos y líneas futuras"),
        ("cierre", "Cierre"),
        ("conclusion", "Conclusiones"),
        ("research_gaps", "Research gaps identificados"),
    ]
    for section_type, title in organizational_cases:
        outline = _stage05_outline_with_section(section_type, title)
        report = validate_outline(outline, set(), 1, 10, [], [], [], [])
        stage05_approved_source_free = "MISSING_REQUIRED_SECTION_PAPERS" not in reason_codes(report)
        assert stage05_approved_source_free, section_type

        result, draft = _run_stage06_section(section_type, title)
        stage06_accepted = result.execution_status.value == "COMPLETED" and result.error is None
        assert stage06_accepted, (section_type, result.error)


if __name__ == "__main__":
    for fn in (
        test_sf_01_introduccion,
        test_sf_02_conclusion,
        test_sf_03_gaps,
        test_sf_04_discusion,
        test_sf_05_linea_tematica_rejected,
        test_sf_06_fundamentos_rejected,
        test_sf_07_structured_type_has_priority_over_title_words,
        test_sf_08_source_free_produces_no_claims_no_citations,
        test_sf_09_missing_section_evidence_still_fail_closed,
        test_sf_10_stage05_stage06_compatibility,
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
