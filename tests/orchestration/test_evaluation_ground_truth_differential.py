"""Pruebas diferenciales del Bloque 2: oráculo reproducido vs. módulo real portado.

Igual metodología que test_evaluation_text_normalization_differential.py: un
oráculo reproducido independientemente (sin compartir código con
``src/tools/evaluation/ground_truth.py``) se compara contra el módulo real,
para: texto extraído, rango seleccionado, encabezado inicial, encabezado
final, estrategia usada (``source_mode``), cantidad de palabras, metadatos
completos, y excepción esperada cuando corresponde.
"""

from __future__ import annotations

import re
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.ground_truth import (
    extract_gt_literature_review,
    load_ground_truth_full_text,
    normalize_pdf_text,
    resolve_ground_truth_comparable_text,
)

# ---------------------------------------------------------------------------
# Oráculo: reproducción independiente de la celda 13 (constantes y funciones
# copiadas por separado, sin importar el módulo bajo prueba).
# ---------------------------------------------------------------------------

_ORACLE_GT_START_ALIASES = [
    "literature review", "related work", "state of the art", "state-of-the-art",
    "previous work", "prior work", "previous studies", "theoretical background",
    "background", "estado del arte", "trabajos relacionados",
    "revisión de literatura", "revision de literatura", "antecedentes",
    "marco teórico", "marco teorico",
]
_ORACLE_GT_END_ALIASES = [
    "materials and methods", "material and methods", "methodology", "methods",
    "proposed method", "proposed model", "system architecture", "dataset",
    "data collection", "experimental setup", "experiments", "results",
    "results and discussion", "discussion", "conclusion", "conclusions",
    "future work", "references", "bibliography", "metodología", "metodologia",
    "métodos", "metodos", "materiales y métodos", "materiales y metodos",
    "modelo propuesto", "arquitectura del sistema", "conjunto de datos",
    "configuración experimental", "configuracion experimental", "resultados",
    "discusión", "discusion", "conclusión", "conclusion", "conclusiones",
    "trabajo futuro", "referencias", "bibliografía", "bibliografia",
]


def _oracle_safe_str(value):
    import json

    import pandas as pd

    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _oracle_normalize_pdf_text(text):
    value = _oracle_safe_str(text)
    value = value.replace("\r", "\n")
    value = value.replace("\u00ad", "")
    value = value.replace("‐", "-")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _oracle_build_heading_pattern(title):
    escaped = re.escape(title).replace(r"\ ", r"\s+")
    return (
        r"(?im)^\s*(?:(?:\d+(?:\.\d+)*\.?)|(?:[IVXLC]+\.?)|(?:[A-Z]\.?))?\s*"
        + escaped
        + r"\s*$"
    )


def _oracle_find_headings(text, aliases):
    candidates = []
    for priority, alias in enumerate(aliases):
        pattern = _oracle_build_heading_pattern(alias)
        for match in re.finditer(pattern, text, flags=(re.IGNORECASE | re.MULTILINE)):
            candidates.append(
                {
                    "alias": alias,
                    "priority": priority,
                    "start": match.start(),
                    "end": match.end(),
                    "heading": match.group(0).strip(),
                }
            )
    return sorted(candidates, key=lambda item: (item["start"], item["priority"]))


def _oracle_extract_gt_literature_review(full_text, *, require_explicit_end_heading):
    text = _oracle_normalize_pdf_text(full_text)
    start_candidates = _oracle_find_headings(text, _ORACLE_GT_START_ALIASES)
    if not start_candidates:
        raise ValueError(
            "No se encontró una sección explícita de revisión "
            "de literatura en el Ground Truth."
        )
    start_info = start_candidates[0]
    content_start = start_info["end"]
    remaining = text[content_start:]
    end_candidates = [
        item for item in _oracle_find_headings(remaining, _ORACLE_GT_END_ALIASES)
        if item["start"] >= 200
    ]
    if not end_candidates:
        if require_explicit_end_heading:
            raise ValueError(
                "Se detectó el inicio de la revisión de literatura, "
                "pero no un encabezado explícito de cierre."
            )
        content_end = len(text)
        end_info = None
    else:
        end_info = end_candidates[0]
        content_end = content_start + end_info["start"]
    extracted = text[content_start:content_end].strip()
    return extracted, {
        "source_mode": "extracted_from_full_ground_truth",
        "start_heading": start_info["heading"],
        "start_alias": start_info["alias"],
        "start_position": start_info["start"],
        "end_heading": end_info["heading"] if end_info else None,
        "end_alias": end_info["alias"] if end_info else None,
        "end_position": content_end,
    }


def _oracle_resolve(*, ground_truth_dir, minimum_words, require_explicit_end_heading):
    ground_truth_dir = Path(ground_truth_dir)
    preextracted_path = ground_truth_dir / "ground_truth_literature_review.txt"
    if preextracted_path.exists():
        text = preextracted_path.read_text(encoding="utf-8", errors="ignore").strip()
        metadata = {
            "source_mode": "preextracted_literature_review",
            "start_heading": "preextracted_literature_review",
            "start_alias": "literature review",
            "start_position": None,
            "end_heading": "preextracted_end",
            "end_alias": None,
            "end_position": None,
        }
        source_path = preextracted_path
    else:
        candidates = [
            ground_truth_dir / "ground_truth_full_text.txt",
            ground_truth_dir / "ground_truth_text.txt",
        ]
        existing = [p for p in candidates if p.exists()]
        if existing:
            full_text = existing[0].read_text(encoding="utf-8", errors="ignore")
            source_path = existing[0]
        else:
            pdfs = sorted(ground_truth_dir.glob("*.pdf"))
            if len(pdfs) == 0:
                raise FileNotFoundError("No se encontró el Ground Truth en TXT ni PDF.")
            if len(pdfs) > 1:
                raise ValueError(
                    "Existen varios PDFs en GROUND_TRUTH_DIR. "
                    "Crea ground_truth_literature_review.txt "
                    "o ground_truth_full_text.txt para eliminar la ambigüedad."
                )
            import fitz

            document = fitz.open(str(pdfs[0]))
            pages = [page.get_text("text") for page in document]
            document.close()
            full_text = "\n".join(pages).strip()
            source_path = pdfs[0]
        text, metadata = _oracle_extract_gt_literature_review(
            full_text, require_explicit_end_heading=require_explicit_end_heading
        )
    if len(text.split()) < minimum_words:
        raise ValueError(
            "La revisión de literatura del Ground Truth "
            "es demasiado corta para una evaluación válida."
        )
    return text, metadata, source_path


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


LONG_BODY = (
    "Este es un cuerpo de texto suficientemente largo para superar el "
    "umbral de doscientos caracteres exigido entre el encabezado de inicio "
    "y cualquier candidato de encabezado de cierre detectado en el "
    "documento, evitando así que se descarte por estar demasiado cerca."
)


@scenario("F01. Diferencial: encabezado inicio+fin explícitos, texto y metadata idénticos")
def test_diff_start_and_end():
    full_text = f"Related Work\n\n{LONG_BODY}\n\nConclusion\n\nFin."
    real = extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    oracle = _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    assert real == oracle


@scenario("F02. Diferencial: fallback permitido, mismo texto y mismo metadata (end=None)")
def test_diff_fallback_allowed():
    full_text = f"Related Work\n\n{LONG_BODY} sin cierre reconocible."
    real = extract_gt_literature_review(full_text, require_explicit_end_heading=False)
    oracle = _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=False)
    assert real == oracle
    assert real[1]["end_heading"] is None


@scenario("F03. Diferencial: fallback prohibido -> misma excepción en ambos caminos")
def test_diff_fallback_forbidden_exception():
    full_text = f"Related Work\n\n{LONG_BODY} sin cierre reconocible."
    real_exc = oracle_exc = None
    try:
        extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        real_exc = str(exc)
    try:
        _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        oracle_exc = str(exc)
    assert real_exc is not None and real_exc == oracle_exc


@scenario("F04. Diferencial: sin sección de revisión -> misma excepción")
def test_diff_no_section_exception():
    full_text = "Abstract\n\nSin sección reconocible.\n\nConclusion\n\nFin."
    real_exc = oracle_exc = None
    try:
        extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        real_exc = str(exc)
    try:
        _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    except ValueError as exc:
        oracle_exc = str(exc)
    assert real_exc is not None and real_exc == oracle_exc


@scenario("F05. Diferencial: encabezado numerado, misma posición y mismo heading detectado")
def test_diff_numbered_heading():
    full_text = f"1. Introduction\n\n2. Related Work\n\n{LONG_BODY}\n\n3. Methods\n\nFin."
    real = extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    oracle = _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    assert real == oracle
    assert real[1]["start_heading"].startswith("2.")


@scenario("F06. Diferencial: normalize_pdf_text produce texto idéntico en ambos caminos")
def test_diff_normalize_pdf_text():
    text = "Texto\u00adcon\u00adguiones\u00adsuaves y\n\n\n\n\nmuchos saltos.\r\nCRLF también."
    assert normalize_pdf_text(text) == _oracle_normalize_pdf_text(text)


@scenario("F07. Diferencial: load_ground_truth_full_text con TXT, mismo texto y misma ruta")
def test_diff_load_full_text_txt():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        (gt_dir / "ground_truth_full_text.txt").write_text("Contenido de prueba.", encoding="utf-8")
        real = load_ground_truth_full_text(ground_truth_dir=gt_dir)
        expected_path = gt_dir / "ground_truth_full_text.txt"
        expected_text = expected_path.read_text(encoding="utf-8")
        assert real == (expected_text, expected_path)


@scenario("F08. Diferencial: resolve_ground_truth_comparable_text con PDF real, mismo resultado completo")
def test_diff_resolve_with_pdf():
    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        content = f"Related Work\n\n{LONG_BODY} {LONG_BODY}\n\nConclusion\n\nFin del documento."
        doc = fitz.open()
        page = doc.new_page()
        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, content, fontsize=11)
        pdf_path = gt_dir / "paper.pdf"
        doc.save(str(pdf_path))
        doc.close()

        real = resolve_ground_truth_comparable_text(
            ground_truth_dir=gt_dir, minimum_words=5, require_explicit_end_heading=True
        )
        oracle = _oracle_resolve(
            ground_truth_dir=gt_dir, minimum_words=5, require_explicit_end_heading=True
        )
        assert real == oracle


@scenario("F09. Diferencial: texto bajo el mínimo -> misma excepción en ambos caminos")
def test_diff_below_minimum():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp)
        (gt_dir / "ground_truth_literature_review.txt").write_text("corto", encoding="utf-8")
        real_exc = oracle_exc = None
        try:
            resolve_ground_truth_comparable_text(
                ground_truth_dir=gt_dir, minimum_words=50, require_explicit_end_heading=True
            )
        except ValueError as exc:
            real_exc = str(exc)
        try:
            _oracle_resolve(
                ground_truth_dir=gt_dir, minimum_words=50, require_explicit_end_heading=True
            )
        except ValueError as exc:
            oracle_exc = str(exc)
        assert real_exc is not None and real_exc == oracle_exc


@scenario("F10. Diferencial: cantidad de palabras coincide exactamente")
def test_diff_word_count():
    full_text = f"Related Work\n\n{LONG_BODY}\n\nConclusion\n\nFin."
    real_text, _ = extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    oracle_text, _ = _oracle_extract_gt_literature_review(full_text, require_explicit_end_heading=True)
    assert len(real_text.split()) == len(oracle_text.split())


if __name__ == "__main__":
    for fn in (
        test_diff_start_and_end,
        test_diff_fallback_allowed,
        test_diff_fallback_forbidden_exception,
        test_diff_no_section_exception,
        test_diff_numbered_heading,
        test_diff_normalize_pdf_text,
        test_diff_load_full_text_txt,
        test_diff_resolve_with_pdf,
        test_diff_below_minimum,
        test_diff_word_count,
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
