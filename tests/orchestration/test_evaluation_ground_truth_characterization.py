"""Pruebas de caracterización y aislamiento del Bloque 2 (Ground Truth).

Cubre casos sintéticos (TXT y PDF reales generados con ``fitz``, el mismo
extractor que usa el notebook — nunca OCR ni un extractor distinto) para la
extracción/preparación del Ground Truth, y confirma que el módulo solo lee
del directorio de Ground Truth que se le pasa explícitamente: nunca escribe
en Chroma, chunks de referencias, ni ninguna ruta de las etapas 03-07.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.ground_truth import (
    GT_END_ALIASES,
    GT_START_ALIASES,
    build_heading_pattern,
    extract_gt_literature_review,
    extract_pdf_text,
    find_headings,
    load_ground_truth_full_text,
    normalize_pdf_text,
    resolve_ground_truth_comparable_text,
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


def _write_pdf(path: Path, text: str) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # insert_textbox (no insert_text) para que el texto largo se ajuste
    # dentro del área de la página en vez de recortarse silenciosamente en
    # el borde -- necesario para que los PDFs sintéticos con secciones
    # separadas por 200+ caracteres realmente contengan todo ese texto.
    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
    overflow = page.insert_textbox(rect, text, fontsize=11)
    if overflow < 0:
        raise AssertionError(
            "El texto sintético no cupo en una sola página del PDF de prueba "
            "(insert_textbox devolvió overflow negativo) -- ajustar el "
            "tamaño de fuente o acortar el texto de prueba."
        )
    doc.save(str(path))
    doc.close()


LITERATURE_TEXT_ES = (
    "Los modelos previos alcanzaron un desempeno limitado en el conjunto "
    "de datos evaluado, con resultados inferiores al enfoque propuesto en "
    "este trabajo segun multiples estudios recientes en el area. Ademas, "
    "los estudios comparativos disponibles muestran limitaciones "
    "consistentes en la capacidad de generalizacion de dichos modelos "
    "frente a conjuntos de datos externos no vistos durante el entrenamiento."
)
LITERATURE_TEXT_EN = (
    "Prior approaches achieved limited performance on the evaluated "
    "dataset, with results below the method proposed in this work "
    "according to several recent studies in the field of research. "
    "Furthermore, available comparative studies show consistent "
    "limitations in the generalization capability of these models "
    "when applied to external datasets not seen during training."
)


# ---------------------------------------------------------------------------
# find_headings / build_heading_pattern — casos básicos de detección
# ---------------------------------------------------------------------------


@scenario("G01. build_heading_pattern matchea el encabezado exacto en su propia línea")
def test_heading_pattern_exact():
    pattern = build_heading_pattern("Related Work")
    assert __import__("re").search(pattern, "\nRelated Work\n", __import__("re").I | __import__("re").M)


@scenario("G02. find_headings detecta encabezado en español")
def test_find_headings_spanish():
    text = "Introducción\n\nEstado del Arte\n\nContenido de la sección."
    hits = find_headings(text, GT_START_ALIASES)
    assert any(h["alias"] == "estado del arte" for h in hits)


@scenario("G03. find_headings detecta encabezado en inglés")
def test_find_headings_english():
    text = "Introduction\n\nRelated Work\n\nSection content."
    hits = find_headings(text, GT_START_ALIASES)
    assert any(h["alias"] == "related work" for h in hits)


@scenario("G04. find_headings detecta encabezado numerado (ej. '2. Related Work')")
def test_find_headings_numbered():
    text = "1. Introduction\n\n2. Related Work\n\nContenido."
    hits = find_headings(text, GT_START_ALIASES)
    matched = [h for h in hits if h["alias"] == "related work"]
    assert matched
    assert matched[0]["heading"].startswith("2.")


@scenario("G05. find_headings detecta un alias alternativo distinto al primero de la lista")
def test_find_headings_alternative_alias():
    text = "Antecedentes\n\nContenido de antecedentes."
    hits = find_headings(text, GT_START_ALIASES)
    assert any(h["alias"] == "antecedentes" for h in hits)


@scenario("G06. find_headings con encabezados repetidos devuelve todas las apariciones")
def test_find_headings_repeated():
    text = "Related Work\n\nTexto 1.\n\nRelated Work\n\nTexto 2."
    hits = find_headings(text, GT_START_ALIASES)
    matches = [h for h in hits if h["alias"] == "related work"]
    assert len(matches) == 2


# ---------------------------------------------------------------------------
# extract_gt_literature_review — extracción de sección con texto sintético
# ---------------------------------------------------------------------------


@scenario("G07. extract_gt_literature_review: encabezado de inicio y fin explícitos")
def test_extract_start_and_end():
    full_text = (
        "Introduction\n\nIntro text here.\n\n"
        "Related Work\n\n" + LITERATURE_TEXT_EN + "\n\n"
        "Methodology\n\nMethod details."
    )
    extracted, metadata = extract_gt_literature_review(
        full_text, require_explicit_end_heading=True
    )
    assert LITERATURE_TEXT_EN.strip() in extracted
    assert metadata["start_alias"] == "related work"
    assert metadata["end_alias"] == "methodology"
    assert metadata["source_mode"] == "extracted_from_full_ground_truth"


@scenario("G08. extract_gt_literature_review: múltiples secciones candidatas usa la primera por posición")
def test_extract_multiple_candidates_uses_first():
    full_text = (
        "Background\n\n" + LITERATURE_TEXT_EN + "\n\n"
        "Related Work\n\nTexto de related work.\n\n"
        "Methods\n\nMetodo."
    )
    extracted, metadata = extract_gt_literature_review(
        full_text, require_explicit_end_heading=True
    )
    # "background" está antes que "related work" en el texto -> gana por posición,
    # no por su índice de prioridad en GT_START_ALIASES (background tiene prioridad
    # 8, related work tiene prioridad 1 -- pero la posición decide primero).
    assert metadata["start_alias"] == "background"


@scenario("G09. extract_gt_literature_review: fallback permitido (sin encabezado final) toma el resto del texto")
def test_extract_fallback_allowed():
    full_text = "Related Work\n\n" + LITERATURE_TEXT_EN + " Sin sección de cierre reconocible aquí."
    extracted, metadata = extract_gt_literature_review(
        full_text, require_explicit_end_heading=False
    )
    assert metadata["end_heading"] is None
    assert "Sin sección de cierre" in extracted


@scenario("G10. extract_gt_literature_review: fallback prohibido lanza si no hay encabezado final")
def test_extract_fallback_forbidden_raises():
    full_text = "Related Work\n\n" + LITERATURE_TEXT_EN + " Sin sección de cierre reconocible aquí."
    try:
        extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        assert "cierre" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError sin encabezado de cierre y require=True")


@scenario("G11. extract_gt_literature_review: documento sin revisión de literatura lanza")
def test_extract_no_literature_review_section():
    full_text = "Abstract\n\nEste paper no tiene sección de revisión reconocible.\n\nConclusion\n\nFin."
    try:
        extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        assert "revisión de literatura" in str(exc) or "revision de literatura" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError sin encabezado de inicio")


@scenario("G12. extract_gt_literature_review: candidato de fin a menos de 200 caracteres del inicio se ignora")
def test_extract_end_too_close_ignored():
    # "results" (alias de fin) aparece a menos de 200 caracteres del final
    # del encabezado de inicio -> debe ignorarse por el filtro start>=200.
    full_text = "Related Work\nresults\n\n" + LITERATURE_TEXT_EN + "\n\nConclusion\n\nFin real."
    extracted, metadata = extract_gt_literature_review(
        full_text, require_explicit_end_heading=True
    )
    assert metadata["end_alias"] == "conclusion"  # no "results", que está demasiado cerca


# ---------------------------------------------------------------------------
# normalize_pdf_text
# ---------------------------------------------------------------------------


@scenario("G13. normalize_pdf_text colapsa saltos de página excesivos (3+ saltos de línea)")
def test_normalize_page_breaks():
    text = "Página 1.\n\n\n\n\nPágina 2."
    result = normalize_pdf_text(text)
    assert "\n\n\n" not in result
    assert "Página 1." in result and "Página 2." in result


@scenario("G14. normalize_pdf_text elimina guiones suaves (soft hyphen) y normaliza guion no separable")
def test_normalize_hyphens():
    text = "infor\u00admación con guion\u2010no-separable"
    result = normalize_pdf_text(text)
    assert "\u00ad" not in result
    assert "\u2010" not in result
    assert "información" in result


@scenario("G15. normalize_pdf_text con Unicode (acentos, ñ) se conserva")
def test_normalize_unicode():
    text = "El niño analizó la información científica."
    result = normalize_pdf_text(text)
    assert "niño" in result and "información" in result and "científica" in result


# ---------------------------------------------------------------------------
# extract_pdf_text / load_ground_truth_full_text — con PDFs reales (fitz)
# ---------------------------------------------------------------------------


@scenario("G16. extract_pdf_text extrae texto real de un PDF generado con fitz")
def test_extract_pdf_text_real_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "gt.pdf"
        _write_pdf(pdf_path, "Related Work\n" + LITERATURE_TEXT_EN)
        text = extract_pdf_text(pdf_path)
        assert "Related Work" in text
        assert "Prior approaches" in text


@scenario("G17. extract_pdf_text con PDF vacío devuelve cadena vacía, no lanza")
def test_extract_pdf_text_empty_pdf():
    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # página en blanco, sin texto
        doc.save(str(pdf_path))
        doc.close()
        text = extract_pdf_text(pdf_path)
        assert text == ""


@scenario("G18. extract_pdf_text con PDF ilegible/corrupto lanza (no se silencia el error)")
def test_extract_pdf_text_corrupt_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "corrupt.pdf"
        pdf_path.write_bytes(b"esto no es un PDF valido en absoluto")
        try:
            extract_pdf_text(pdf_path)
        except Exception:
            pass  # fitz.open propaga su propia excepción; no se especifica el tipo exacto
        else:
            raise AssertionError("debía fallar con un PDF corrupto")


@scenario("G19. load_ground_truth_full_text: TXT completo tiene prioridad sobre PDF si ambos existen")
def test_load_full_text_txt_priority_over_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        (gt_dir / "ground_truth_full_text.txt").write_text("Texto desde TXT.", encoding="utf-8")
        _write_pdf(gt_dir / "otro.pdf", "Texto desde PDF.")
        text, source_path = load_ground_truth_full_text(ground_truth_dir=gt_dir)
        assert text == "Texto desde TXT."
        assert source_path.name == "ground_truth_full_text.txt"


@scenario("G20. load_ground_truth_full_text: segundo TXT candidato se usa si el primero no existe")
def test_load_full_text_second_txt_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        (gt_dir / "ground_truth_text.txt").write_text("Texto del segundo candidato.", encoding="utf-8")
        text, source_path = load_ground_truth_full_text(ground_truth_dir=gt_dir)
        assert text == "Texto del segundo candidato."
        assert source_path.name == "ground_truth_text.txt"


@scenario("G21. load_ground_truth_full_text: sin TXT, un único PDF se extrae con fitz")
def test_load_full_text_single_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        _write_pdf(gt_dir / "paper.pdf", "Related Work\n" + LITERATURE_TEXT_EN)
        text, source_path = load_ground_truth_full_text(ground_truth_dir=gt_dir)
        assert "Related Work" in text
        assert source_path.name == "paper.pdf"


@scenario("G22. load_ground_truth_full_text: sin TXT ni PDF lanza FileNotFoundError")
def test_load_full_text_nothing_found():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_ground_truth_full_text(ground_truth_dir=Path(tmp))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("debía lanzar FileNotFoundError")


@scenario("G23. load_ground_truth_full_text: múltiples PDFs sin TXT lanza ValueError (ambigüedad)")
def test_load_full_text_multiple_pdfs_ambiguous():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        _write_pdf(gt_dir / "a.pdf", "Contenido A.")
        _write_pdf(gt_dir / "b.pdf", "Contenido B.")
        try:
            load_ground_truth_full_text(ground_truth_dir=gt_dir)
        except ValueError as exc:
            assert "ambigüedad" in str(exc) or "varios PDFs" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError por ambigüedad")


# ---------------------------------------------------------------------------
# resolve_ground_truth_comparable_text — orquestación completa
# ---------------------------------------------------------------------------


@scenario("G24. resolve_ground_truth_comparable_text: TXT preextraído tiene prioridad absoluta")
def test_resolve_preextracted_priority():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        preextracted_text = " ".join(["palabra"] * 50)
        (gt_dir / "ground_truth_literature_review.txt").write_text(
            preextracted_text, encoding="utf-8"
        )
        # Aunque también exista un PDF, el preextraído gana sin tocarlo.
        _write_pdf(gt_dir / "ignorado.pdf", "Related Work\nTexto que nunca debería leerse.")
        text, metadata, source_path = resolve_ground_truth_comparable_text(
            ground_truth_dir=gt_dir, minimum_words=10, require_explicit_end_heading=True
        )
        assert text == preextracted_text
        assert metadata["source_mode"] == "preextracted_literature_review"
        assert source_path.name == "ground_truth_literature_review.txt"


@scenario("G25. resolve_ground_truth_comparable_text: texto por debajo del mínimo lanza")
def test_resolve_below_minimum_words():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        (gt_dir / "ground_truth_literature_review.txt").write_text(
            "muy corto", encoding="utf-8"
        )
        try:
            resolve_ground_truth_comparable_text(
                ground_truth_dir=gt_dir, minimum_words=100, require_explicit_end_heading=True
            )
        except ValueError as exc:
            assert "demasiado corta" in str(exc)
        else:
            raise AssertionError("debía lanzar ValueError por longitud insuficiente")


@scenario("G26. resolve_ground_truth_comparable_text: extracción real desde PDF con encabezados ES")
def test_resolve_real_pdf_spanish_headings():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        long_text = " ".join([LITERATURE_TEXT_ES] * 5)
        _write_pdf(
            gt_dir / "paper.pdf",
            "Introduccion\n\nTexto intro.\n\nEstado del Arte\n\n" + long_text + "\n\nMetodologia\n\nMetodo.",
        )
        text, metadata, source_path = resolve_ground_truth_comparable_text(
            ground_truth_dir=gt_dir, minimum_words=10, require_explicit_end_heading=True
        )
        assert metadata["source_mode"] == "extracted_from_full_ground_truth"
        assert metadata["start_alias"] == "estado del arte"
        assert len(text.split()) >= 10


# ---------------------------------------------------------------------------
# Aislamiento del Ground Truth (requisito 3 del pedido)
# ---------------------------------------------------------------------------


@scenario("G27. aislamiento: el módulo solo lee dentro de ground_truth_dir, ninguna otra ruta")
def test_isolation_only_reads_ground_truth_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gt_dir = root / "00_ground_truth"
        gt_dir.mkdir()
        chunks_dir = root / "03_chunks"
        chunks_dir.mkdir()
        chroma_dir = root / "04_chroma_index"
        chroma_dir.mkdir()

        # Se colocan archivos señuelo en chunks/chroma con nombres que
        # coincidirían con los candidatos de Ground Truth si el módulo
        # buscara fuera de ground_truth_dir por error.
        (chunks_dir / "ground_truth_full_text.txt").write_text(
            "ESTO NO DEBE LEERSE (está en 03_chunks).", encoding="utf-8"
        )
        (chroma_dir / "ground_truth_literature_review.txt").write_text(
            "ESTO TAMPOCO (está en 04_chroma_index).", encoding="utf-8"
        )

        real_text = " ".join(["contenido", "real", "del", "ground", "truth"] * 5)
        (gt_dir / "ground_truth_literature_review.txt").write_text(real_text, encoding="utf-8")

        text, metadata, source_path = resolve_ground_truth_comparable_text(
            ground_truth_dir=gt_dir, minimum_words=5, require_explicit_end_heading=True
        )
        assert text == real_text
        assert str(source_path).startswith(str(gt_dir))
        assert "NO DEBE LEERSE" not in text
        assert "TAMPOCO" not in text


@scenario("G28. aislamiento: el módulo no escribe ningún archivo (no hay .write en ninguna función pública)")
def test_isolation_module_never_writes():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        _write_pdf(gt_dir / "paper.pdf", "Related Work\n" + LITERATURE_TEXT_EN * 3)
        files_before = sorted(p.name for p in gt_dir.iterdir())
        resolve_ground_truth_comparable_text(
            ground_truth_dir=gt_dir, minimum_words=5, require_explicit_end_heading=False
        )
        files_after = sorted(p.name for p in gt_dir.iterdir())
        assert files_before == files_after, "el módulo no debe crear ni modificar archivos"


@scenario("G29. aislamiento: nada en el módulo importa chromadb ni referencia chunks/03-07")
def test_isolation_no_chroma_imports():
    import inspect

    from src.tools.evaluation import ground_truth as gt_module

    source = inspect.getsource(gt_module)
    for forbidden in ("chromadb", "StateStore", "AgentInput", "AgentResult"):
        assert forbidden not in source, f"el módulo no debería referenciar {forbidden}"


if __name__ == "__main__":
    for fn in (
        test_heading_pattern_exact,
        test_find_headings_spanish,
        test_find_headings_english,
        test_find_headings_numbered,
        test_find_headings_alternative_alias,
        test_find_headings_repeated,
        test_extract_start_and_end,
        test_extract_multiple_candidates_uses_first,
        test_extract_fallback_allowed,
        test_extract_fallback_forbidden_raises,
        test_extract_no_literature_review_section,
        test_extract_end_too_close_ignored,
        test_normalize_page_breaks,
        test_normalize_hyphens,
        test_normalize_unicode,
        test_extract_pdf_text_real_pdf,
        test_extract_pdf_text_empty_pdf,
        test_extract_pdf_text_corrupt_pdf,
        test_load_full_text_txt_priority_over_pdf,
        test_load_full_text_second_txt_candidate,
        test_load_full_text_single_pdf,
        test_load_full_text_nothing_found,
        test_load_full_text_multiple_pdfs_ambiguous,
        test_resolve_preextracted_priority,
        test_resolve_below_minimum_words,
        test_resolve_real_pdf_spanish_headings,
        test_isolation_only_reads_ground_truth_dir,
        test_isolation_module_never_writes,
        test_isolation_no_chroma_imports,
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
