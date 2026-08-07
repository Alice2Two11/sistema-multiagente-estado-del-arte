"""Bloque 2 de la migración de la etapa 08: extracción y preparación del Ground Truth.

Las funciones y constantes de aquí abajo son copias LITERALES de
``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``, celda 13
("EXTRAER LA REVISIÓN DE LITERATURA DEL GROUND TRUTH"), extraídas el mismo
día que este archivo. No se simplificó ningún alias, expresión regular,
regla de inicio/fin, numeración, fallback, orden de preferencia, límite de
palabras, metadato ni mensaje de error.

Único cambio respecto al notebook (mecánico, no funcional): las rutas y
umbrales que en el notebook son variables GLOBALES fijadas por celdas
anteriores (``GROUND_TRUTH_DIR``, ``MINIMUM_GT_WORDS``,
``REQUIRE_EXPLICIT_GT_END_HEADING``) se reciben aquí como parámetros
explícitos de ``resolve_ground_truth_comparable_text`` y
``load_ground_truth_full_text``/``extract_gt_literature_review``, para que
importar este módulo no dependa de que ``config.py`` ya esté cargado — es
la misma sustitución mecánica ya aplicada al Bloque 1 (parametrizar en vez
de leer un global), no una relajación de ninguna regla de negocio.

Mapa función → celda original (todas en la celda 13, salvo lo indicado)
--------------------------------------------------------------------------
| Función/constante | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``GT_START_ALIASES`` (16 alias, ES/EN) | 13 | ninguna | — | lista constante | — | — |
| ``GT_END_ALIASES`` (40 alias, ES/EN) | 13 | ninguna | -- | lista constante | -- | -- |
| ``normalize_pdf_text`` | 13 | ``safe_str`` (celda 1, Bloque 1) | ``text: Any`` | ``str`` normalizado | ninguna | ninguno |
| ``build_heading_pattern`` | 13 | ``re`` | ``title: str`` | patrón regex compilable (str) | ninguna | ninguno |
| ``find_headings`` | 13 | ``build_heading_pattern`` | ``text: str``, ``aliases: list[str]`` | lista de candidatos ordenados por ``(start, priority)`` | ninguna | ninguno |
| ``extract_pdf_text`` | 13 | ``fitz`` (PyMuPDF, mismo extractor que 01/02) | ``path`` (str o Path) | texto plano concatenado de todas las páginas | las que ``fitz.open`` propague (p.ej. PDF ilegible/corrupto) | LEE el PDF en ``path`` |
| ``load_ground_truth_full_text`` | 13 | ``extract_pdf_text`` | ``ground_truth_dir: Path`` (parametrizado; el notebook usa el global ``GROUND_TRUTH_DIR``) | ``(texto, ruta_fuente)`` | ``FileNotFoundError`` si no hay TXT ni PDF; ``ValueError`` si hay >1 PDF ambiguo | LEE archivos en ``ground_truth_dir`` (TXT candidatos o el único PDF) |
| ``extract_gt_literature_review`` | 13 | ``normalize_pdf_text``, ``find_headings`` | ``full_text: str``, ``require_explicit_end_heading: bool`` (parametrizado; el notebook usa el global derivado de ``EVALUATION_POLICY["require_explicit_ground_truth_end_heading"]``) | ``(texto_extraído, metadata)`` | ``ValueError`` si no hay encabezado de inicio; ``ValueError`` si no hay encabezado de fin Y ``require_explicit_end_heading`` es ``True`` | ninguno |
| ``resolve_ground_truth_comparable_text`` (orquestación, no es una función nombrada en el notebook — ver nota abajo) | 13 | todas las anteriores | ``ground_truth_dir``, ``minimum_words``, ``require_explicit_end_heading`` | ``(texto, metadata, ruta_fuente)`` | ``ValueError`` si el texto queda bajo ``minimum_words`` (+ las de las funciones internas) | LEE ``ground_truth_literature_review.txt`` o cae a ``load_ground_truth_full_text`` |

Nota sobre ``resolve_ground_truth_comparable_text``: en el notebook real
esta lógica NO es una función — es código de módulo (celda 13, después de
las definiciones) que decide entre el TXT preextraído
(``ground_truth_literature_review.txt``) y la extracción real, y aplica el
chequeo de longitud mínima. Se envolvió aquí en una función porque en un
módulo de ``src/`` no puede quedar como código a nivel de import (eso SÍ
sería una mejora funcional no pedida — código de import con efectos
secundarios). El cuerpo y el orden de las comprobaciones son literales.

Explícitamente FUERA de este bloque (no migrado aquí)
--------------------------------------------------------
- La comparación ``sha256_text(ground_truth) == sha256_text(generado)``
  (celda 13, después del chequeo de longitud mínima): depende de
  ``generated_content_text``, que no es parte de la preparación del Ground
  Truth en sí — pertenece al bloque que carga el texto generado (no
  migrado). No se incluye en ``resolve_ground_truth_comparable_text``.
- El cálculo de ``source_sha256``/``comparable_text_sha256``/
  ``word_count``/``character_count`` que la celda 13 agrega al metadata
  ANTES de guardarlo, y la escritura de
  ``GROUND_TRUTH_COMPARABLE_TEXT_PATH``/
  ``GROUND_TRUTH_EXTRACTION_METADATA_PATH``: son persistencia, no
  preparación — quedan para el bloque de runtime transaccional (Bloque 6).
  ``sha256_file``/``sha256_text`` ya existen como utilidades generales en
  ``src/state/fingerprints.py`` — no son específicas de Ground Truth y no
  se duplican aquí.
- ``split_sentences``/``chunk_text_by_sentences`` (celda 17): pese a que el
  pedido de este bloque los menciona, **no forman parte de la preparación
  del Ground Truth** — se usan exclusivamente dentro de
  ``translate_text_to_language`` (celda 17, bloque de métricas automáticas,
  para partir el texto antes de traducirlo para ROUGE). No hay ninguna
  llamada a estas funciones en la celda 13. Quedan para el Bloque 3.
- ``detect_language_code`` (celda 15): tampoco es parte de la extracción
  del Ground Truth — se aplica DESPUÉS, en el bloque de
  "fingerprint/backup/preprocesamiento" (celda 15), tanto al texto generado
  como al Ground Truth ya extraído, para decidir si hace falta traducir
  antes de ROUGE. No es inseparable de este bloque: la extracción del GT
  (celda 13) termina antes de que se invoque ``detect_language_code``.

Importar este módulo no carga modelos, no abre PDFs, no busca Ground Truth,
no crea directorios, no lee ``config.py`` ni llama a OpenAI — todo el I/O
ocurre solo cuando se invoca explícitamente una de las funciones.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.tools.evaluation.text_normalization import safe_str

# ---------------------------------------------------------------------------
# Alias de encabezados — notebook 08, celda 13, listas literales
# ---------------------------------------------------------------------------

GT_START_ALIASES = [
    "literature review",
    "related work",
    "state of the art",
    "state-of-the-art",
    "previous work",
    "prior work",
    "previous studies",
    "theoretical background",
    "background",
    "estado del arte",
    "trabajos relacionados",
    "revisión de literatura",
    "revision de literatura",
    "antecedentes",
    "marco teórico",
    "marco teorico",
]

GT_END_ALIASES = [
    "materials and methods",
    "material and methods",
    "methodology",
    "methods",
    "proposed method",
    "proposed model",
    "system architecture",
    "dataset",
    "data collection",
    "experimental setup",
    "experiments",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "metodología",
    "metodologia",
    "métodos",
    "metodos",
    "materiales y métodos",
    "materiales y metodos",
    "modelo propuesto",
    "arquitectura del sistema",
    "conjunto de datos",
    "configuración experimental",
    "configuracion experimental",
    "resultados",
    "discusión",
    "discusion",
    "conclusión",
    "conclusion",
    "conclusiones",
    "trabajo futuro",
    "referencias",
    "bibliografía",
    "bibliografia",
]

# Nombres de archivo literales del notebook (celda 13), relativos a
# GROUND_TRUTH_DIR (ahora un parámetro, no un global).
PREEXTRACTED_GT_FILENAME = "ground_truth_literature_review.txt"
GT_FULL_TEXT_CANDIDATE_FILENAMES = (
    "ground_truth_full_text.txt",
    "ground_truth_text.txt",
)


# ---------------------------------------------------------------------------
# normalize_pdf_text
# ---------------------------------------------------------------------------


def normalize_pdf_text(text: Any) -> str:
    value = safe_str(text)
    value = value.replace("\r", "\n")
    value = value.replace("\u00ad", "")
    value = value.replace("‐", "-")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


# ---------------------------------------------------------------------------
# build_heading_pattern / find_headings
# ---------------------------------------------------------------------------


def build_heading_pattern(title: str) -> str:
    escaped = re.escape(title).replace(r"\ ", r"\s+")
    return (
        r"(?im)^"
        r"\s*"
        r"(?:"
        r"(?:\d+(?:\.\d+)*\.?)|"
        r"(?:[IVXLC]+\.?)|"
        r"(?:[A-Z]\.?)"
        r")?"
        r"\s*"
        + escaped
        + r"\s*$"
    )


def find_headings(text: str, aliases: list[str]) -> list[dict]:
    candidates = []
    for priority, alias in enumerate(aliases):
        pattern = build_heading_pattern(alias)
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


# ---------------------------------------------------------------------------
# extract_pdf_text — mismo extractor real que el notebook (fitz/PyMuPDF)
# ---------------------------------------------------------------------------


def extract_pdf_text(path: str | Path) -> str:
    import fitz  # PyMuPDF — mismo extractor que usa el notebook; import

    # diferido para no cargarlo solo por importar este módulo (requisito 7).

    document = fitz.open(str(path))
    pages = [page.get_text("text") for page in document]
    document.close()
    return "\n".join(pages).strip()


# ---------------------------------------------------------------------------
# load_ground_truth_full_text — parametrizado por ground_truth_dir
# ---------------------------------------------------------------------------


def load_ground_truth_full_text(*, ground_truth_dir: str | Path) -> tuple[str, Path]:
    ground_truth_dir = Path(ground_truth_dir)
    full_text_candidates = [
        ground_truth_dir / name for name in GT_FULL_TEXT_CANDIDATE_FILENAMES
    ]

    existing_texts = [path for path in full_text_candidates if path.exists()]
    if existing_texts:
        selected = existing_texts[0]
        return (
            selected.read_text(encoding="utf-8", errors="ignore"),
            selected,
        )

    pdf_candidates = sorted(ground_truth_dir.glob("*.pdf"))

    if len(pdf_candidates) == 0:
        raise FileNotFoundError("No se encontró el Ground Truth en TXT ni PDF.")

    if len(pdf_candidates) > 1:
        raise ValueError(
            "Existen varios PDFs en GROUND_TRUTH_DIR. "
            "Crea ground_truth_literature_review.txt "
            "o ground_truth_full_text.txt para eliminar la ambigüedad."
        )

    selected = pdf_candidates[0]
    return (extract_pdf_text(selected), selected)


# ---------------------------------------------------------------------------
# extract_gt_literature_review — parametrizado por require_explicit_end_heading
# ---------------------------------------------------------------------------


def extract_gt_literature_review(
    full_text: str, *, require_explicit_end_heading: bool
) -> tuple[str, dict]:
    text = normalize_pdf_text(full_text)
    start_candidates = find_headings(text, GT_START_ALIASES)

    if not start_candidates:
        raise ValueError(
            "No se encontró una sección explícita de revisión "
            "de literatura en el Ground Truth."
        )

    start_info = start_candidates[0]
    content_start = start_info["end"]
    remaining = text[content_start:]

    end_candidates = [
        item for item in find_headings(remaining, GT_END_ALIASES) if item["start"] >= 200
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


# ---------------------------------------------------------------------------
# Orquestación de la celda 13 (código de módulo en el notebook, envuelto
# aquí en una función — ver nota en el docstring del módulo)
# ---------------------------------------------------------------------------


def resolve_ground_truth_comparable_text(
    *,
    ground_truth_dir: str | Path,
    minimum_words: int,
    require_explicit_end_heading: bool,
) -> tuple[str, dict, Path]:
    ground_truth_dir = Path(ground_truth_dir)
    preextracted_path = ground_truth_dir / PREEXTRACTED_GT_FILENAME

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
        full_text, source_path = load_ground_truth_full_text(ground_truth_dir=ground_truth_dir)
        text, metadata = extract_gt_literature_review(
            full_text, require_explicit_end_heading=require_explicit_end_heading
        )

    if len(text.split()) < minimum_words:
        raise ValueError(
            "La revisión de literatura del Ground Truth "
            "es demasiado corta para una evaluación válida."
        )

    return text, metadata, source_path
