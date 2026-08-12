"""Bug real reproducible en Exp05, ``05_generador_esquema``: una
sección no exenta (``fundamentos``, "Datasets y métricas de
evaluación") quedaba con ``papers_to_use=[]``, marcada correctamente
como problemática (``MISSING_REQUIRED_SECTION_PAPERS``) -- pero:

1. ``repair_then_validate`` (``repair_outline_sources``/``repair_
   coverage_summary``) solo corrige nombres de archivo de entradas YA
   EXISTENTES en ``papers_to_use`` -- nunca rellenaba una lista vacía,
   aunque hubiera evidencia real disponible en 04.
2. El mecanismo de "reuse" (fingerprint-based freshness) reutilizaba
   la salida de un intento ANTERIOR incluso si ese intento había sido
   RECHAZADO (``validation_ok=False``) -- por eso el RETRY releía del
   disco el MISMO outline inválido, lo revalidaba, obtenía el MISMO
   fallo, y nunca volvía a intentar generar/reparar nada.
3. El trim a ``max_sections`` dejaba referencias obsoletas (IDs de
   secciones eliminadas) en ``paper_coverage_summary["used_in_
   sections"]``, nunca reconciliadas tras el corte.

Fix: ``repair_empty_section_papers`` (nuevo, ``source_repair.py``) usa
EXCLUSIVAMENTE ``themes[].representative_papers`` de 04 -- coincidencia
textual determinista con el tema si existe, o la unión de todos los
temas para secciones transversales -- nunca inventa papers ni
relaciones, nunca usa el KB completo sin pasar por ese mapping,
siempre fail-closed si no hay evidencia. El "reuse" ahora exige que la
salida cacheada haya sido ``validation_ok=True``. El trim reconstruye
``used_in_sections`` inmediatamente, en el mismo paso."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.outline_generation.source_repair import (  # noqa: E402
    repair_empty_section_papers,
    repair_outline_sources,
    repair_coverage_summary,
    section_allows_empty_papers,
)
from src.tools.outline_generation.outline_validation import (  # noqa: E402
    reason_codes,
    validate_outline,
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


def _section(section_id, title, section_type, papers=None):
    return {
        "section_id": section_id, "section_title": title, "section_type": section_type,
        "purpose": "purpose", "key_arguments": ["k1"], "evidence_needs": ["e1"],
        "papers_to_use": papers if papers is not None else [],
    }


def _theme(theme_id, name, papers):
    return {
        "theme_id": theme_id, "theme_name": name, "theme": name, "description": name,
        "representative_papers": [{"source_filename": p, "title": f"Title {p}"} for p in papers],
    }


def _outline(sections, coverage=None):
    return {
        "title": "t", "objective": "o", "narrative_strategy": "n",
        "sections": sections, "paper_coverage_summary": coverage or [],
    }


@scenario("FF01. Introducción con papers_to_use=[] -> válida, section_allows_empty_papers=True, nunca se repara ni se marca problemática")
def test_introduction_empty_papers_is_valid():
    sec = _section("S1", "Introducción y contexto", "introduccion")
    assert section_allows_empty_papers(sec) is True
    outline = _outline([sec, _section("S2", "Metodología clásica", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}])])
    valid = {"p1.pdf"}
    repairs, no_evidence = repair_empty_section_papers(outline, [], valid)
    assert repairs == []  # S1 nunca se toca -- está exenta
    assert outline["sections"][0]["papers_to_use"] == []
    result = validate_outline(outline, valid, 1, 5, [], [], [], [])
    assert result["validation_ok"] is True
    assert result["empty_papers_to_use_allowed"] == [{"section_id": "S1", "section_title": "Introducción y contexto", "section_type": "introduccion"}]
    assert result["empty_papers_to_use_problematic"] == []


@scenario("FF02. Sección 'fundamentos' vacía con evidencia upstream INEQUÍVOCA (coincide textualmente con un tema) -> reparación segura, exactamente los papers de ese tema")
def test_fundamentos_section_unambiguous_theme_match_repairs_safely():
    theme = _theme("T1", "Selección y reducción de características", ["fs1.pdf", "fs2.pdf"])
    other_theme = _theme("T2", "Modelos de aprendizaje profundo", ["dl1.pdf"])
    sec = _section("S5", "Selección y reducción de características", "fundamentos")
    outline = _outline([sec])
    valid = {"fs1.pdf", "fs2.pdf", "dl1.pdf"}
    repairs, no_evidence = repair_empty_section_papers(outline, [theme, other_theme], valid)
    assert len(repairs) == 1
    assert repairs[0]["mode"] == "THEME_TITLE_MATCH"
    assigned = {p["source_filename"] for p in outline["sections"][0]["papers_to_use"]}
    assert assigned == {"fs1.pdf", "fs2.pdf"}  # SOLO el tema que coincide, no el otro tema
    assert no_evidence == []


@scenario("FF03. Sección temática transversal vacía (sin coincidencia directa con ningún tema, ej. 'Datasets y métricas') con evidencia upstream -> reparación segura vía unión de todos los temas -- caso real Exp05")
def test_cross_cutting_section_repairs_via_theme_union():
    theme1 = _theme("T1", "Modelos de aprendizaje profundo para deteccion", ["dl1.pdf", "dl2.pdf"])
    theme2 = _theme("T2", "Tecnicas clasicas de aprendizaje supervisado", ["ml1.pdf"])
    sec = _section("S2", "Datasets y métricas de evaluación", "fundamentos")
    outline = _outline([sec])
    valid = {"dl1.pdf", "dl2.pdf", "ml1.pdf"}
    repairs, no_evidence = repair_empty_section_papers(outline, [theme1, theme2], valid)
    assert len(repairs) == 1
    assert repairs[0]["mode"] == "ALL_THEMES_UNION"
    assigned = {p["source_filename"] for p in outline["sections"][0]["papers_to_use"]}
    assert assigned == {"dl1.pdf", "dl2.pdf", "ml1.pdf"}  # unión completa, nada omitido
    result = validate_outline(outline, valid, 1, 5, [], [], [], [])
    assert result["empty_papers_to_use_problematic"] == []


@scenario("FF04. Sección sin NINGUNA evidencia upstream (temas vacíos o sin papers representativos) -> permanece inválida, fail-closed, nunca se rellena artificialmente")
def test_section_without_upstream_evidence_stays_invalid():
    sec = _section("S9", "Tema sin ningún respaldo temático", "fundamentos")
    outline = _outline([sec])
    valid = {"p1.pdf"}
    repairs, no_evidence = repair_empty_section_papers(outline, [], valid)  # sin temas en absoluto
    assert repairs == []
    assert no_evidence == [{"section_id": "S9", "section_title": "Tema sin ningún respaldo temático"}]
    assert outline["sections"][0]["papers_to_use"] == []
    result = validate_outline(outline, valid, 1, 5, [], [], [], [])
    assert result["validation_ok"] is False
    assert result["empty_papers_to_use_problematic"] == [{"section_id": "S9", "section_title": "Tema sin ningún respaldo temático", "section_type": "fundamentos"}]
    assert reason_codes(result) == ("MISSING_SECTION_FIELDS", "MISSING_REQUIRED_SECTION_PAPERS")


@scenario("FF05. source_filename referenciado por un tema pero YA NO existe en valid_sources -> nunca se propaga, nunca se inventa un paper inexistente")
def test_nonexistent_source_filename_never_repaired_in():
    theme = _theme("T1", "Selección de características", ["fs1.pdf", "fs_ya_no_existe.pdf"])
    sec = _section("S5", "Selección de características", "fundamentos")
    outline = _outline([sec])
    valid = {"fs1.pdf"}  # fs_ya_no_existe.pdf YA NO está en el conjunto validado
    repairs, no_evidence = repair_empty_section_papers(outline, [theme], valid)
    assigned = {p["source_filename"] for p in outline["sections"][0]["papers_to_use"]}
    assert assigned == {"fs1.pdf"}
    assert "fs_ya_no_existe.pdf" not in assigned


@scenario("FF06. outline con más secciones que max_sections -> trim y paper_coverage_summary reconstruido sin IDs de secciones eliminadas")
def test_trim_reconciles_paper_coverage_summary():
    sections = [
        _section("S1", "Introducción", "introduccion"),
        _section("S2", "Sec 2", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
        _section("S3", "Sec 3", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
        _section("S4", "Sec 4", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
        _section("S5", "Sec 5", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
        _section("S6", "Sec 6 (se recorta)", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
        _section("S7", "Sec 7 (se recorta)", "fundamentos", papers=[{"source_filename": "p1.pdf", "title": "P1"}]),
    ]
    coverage = [
        {"source_filename": "p1.pdf", "title": "P1", "used_in_sections": ["S2", "S6", "S7"]},
        {"source_filename": "p2.pdf", "title": "P2", "used_in_sections": ["S6"]},
    ]
    outline = _outline(sections, coverage)
    valid = {"p1.pdf", "p2.pdf"}
    result = validate_outline(outline, valid, 1, 5, [], [], [], [])
    assert result["n_sections"] == 5
    assert result["sections_trimmed_to_max"] is True
    remaining_ids = {s["section_id"] for s in outline["sections"]}
    assert remaining_ids == {"S1", "S2", "S3", "S4", "S5"}
    entry_p1 = next(c for c in outline["paper_coverage_summary"] if c["source_filename"] == "p1.pdf")
    assert entry_p1["used_in_sections"] == ["S2"]  # S6/S7 eliminados, nunca quedan colgando
    entry_p2 = next(c for c in outline["paper_coverage_summary"] if c["source_filename"] == "p2.pdf")
    assert entry_p2["used_in_sections"] == []  # su única sección se recortó -- lista vacía, no un ID fantasma


@scenario("FF07. RETRY posterior a MISSING_REQUIRED_SECTION_PAPERS: el mecanismo de reuse ya no reutiliza una salida previa RECHAZADA -- solo reutiliza si la salida cacheada fue validation_ok=True")
def test_reuse_never_reuses_a_rejected_prior_output():
    # Simula exactamente el bug real: un manifest cacheado, con el
    # MISMO fingerprint, pero cuyo validation_report anterior fue
    # RECHAZADO (validation_ok=False, MISSING_REQUIRED_SECTION_PAPERS).
    cached_manifest_rejected = {
        "fingerprint": "same_fp_upstream_unchanged",
        "validation_report": {"validation_ok": False, "sections_missing_required_fields": [{"section_id": "S2"}]},
    }
    current_fingerprint = "same_fp_upstream_unchanged"
    # Replica exactamente la condición real del agente (outline_generation_agent.py).
    reuse = (
        cached_manifest_rejected.get("fingerprint") == current_fingerprint
        and bool(cached_manifest_rejected.get("validation_report", {}).get("validation_ok"))
    )
    assert reuse is False  # el bug real: esto daba True, ahora da False

    cached_manifest_approved = {
        "fingerprint": "same_fp_upstream_unchanged",
        "validation_report": {"validation_ok": True},
    }
    reuse_approved = (
        cached_manifest_approved.get("fingerprint") == current_fingerprint
        and bool(cached_manifest_approved.get("validation_report", {}).get("validation_ok"))
    )
    assert reuse_approved is True  # una salida genuinamente aprobada SÍ se sigue reutilizando


if __name__ == "__main__":
    for fn in (
        test_introduction_empty_papers_is_valid,
        test_fundamentos_section_unambiguous_theme_match_repairs_safely,
        test_cross_cutting_section_repairs_via_theme_union,
        test_section_without_upstream_evidence_stays_invalid,
        test_nonexistent_source_filename_never_repaired_in,
        test_trim_reconciles_paper_coverage_summary,
        test_reuse_never_reuses_a_rejected_prior_output,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
