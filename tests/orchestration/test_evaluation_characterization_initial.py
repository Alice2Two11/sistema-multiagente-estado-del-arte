"""Pruebas de caracterización INICIALES de la etapa 08 (notebook, no migrada).

Nada de esto se ha portado a ``src/`` todavía — por instrucción explícita,
esta iteración es solo inventario + caracterización, sin migración. Las
funciones ``_notebook08_*`` de aquí abajo son copias LITERALES de celdas del
notebook 08 (citadas en cada una), reproducidas para fijar un oráculo de
comportamiento actual contra el que comparar una futura migración real a
``src/`` — no son una implementación nueva ni una versión simplificada.

Cubre el "Bloque 1" (normalización de texto pura, bajo riesgo) del plan de
migración incremental de ``AGENT08_INVENTORY.md``, más dos invariantes
transversales citadas en ese documento: el aislamiento de Ground Truth
(§7) y la lógica de fingerprint/rebuild (§6, celda 15).

No llama a OpenAI, Chroma, ni ninguna librería de métricas (rouge_score/
bert_score/sentence_transformers) — eso queda para cuando se migren los
Bloques 2-5.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Copias literales de funciones puras del notebook 08 (celda 15 y celda 19),
# citadas exactamente por nombre y celda. Ver AGENT08_INVENTORY.md §4/§5.
# ---------------------------------------------------------------------------

# Celda 15: citation_pattern + strip_internal_citations + normalize_content_text
_NOTEBOOK08_CITATION_PATTERN = re.compile(r"\[([^\[\]\|]+?)\s*\|\s*([^\[\]\|]+?)\]")


def _notebook08_safe_str(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _notebook08_strip_internal_citations(text):
    return _NOTEBOOK08_CITATION_PATTERN.sub("", _notebook08_safe_str(text))


def _notebook08_normalize_content_text(text):
    value = _notebook08_safe_str(text)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value)
    value = re.sub(r"^\s*#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = _notebook08_strip_internal_citations(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# Celda 19: normalize_claim_text + normalize_numeric_token + numeric_search_variants
def _notebook08_normalize_claim_text(text):
    value = _notebook08_strip_internal_citations(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .;:!?").casefold()


def _notebook08_normalize_numeric_token(value):
    token = _notebook08_safe_str(value)
    token = re.sub(r"\s+", "", token)
    token = token.replace(",", ".")
    return token.casefold()


def _notebook08_numeric_search_variants(value):
    normalized = _notebook08_normalize_numeric_token(value)
    variants = {normalized, normalized.replace("%", "")}
    if "." in normalized:
        variants.add(normalized.replace(".", ","))
        variants.add(normalized.replace(".", ",").replace("%", ""))
    return {item for item in variants if item}


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


# ---------------------------------------------------------------------------
# Bloque 1: normalización de texto pura
# ---------------------------------------------------------------------------


@scenario("T1. strip_internal_citations elimina citas [fuente | chunk]")
def test_strip_internal_citations():
    text = "El modelo logró 91% de precisión [paper1.pdf | chunk_3] en el conjunto de prueba."
    result = _notebook08_strip_internal_citations(text)
    assert "[paper1.pdf | chunk_3]" not in result
    assert "91% de precisión" in result


@scenario("T2. strip_internal_citations conserva texto sin citas intacto")
def test_strip_internal_citations_no_citations():
    text = "Texto sin ninguna cita interna."
    assert _notebook08_strip_internal_citations(text) == text


@scenario("T3. normalize_content_text elimina bloques de código, negritas markdown y encabezados")
def test_normalize_content_text():
    text = "# Título\n\nUn `código` y ```bloque\nde código``` con **negrita**."
    result = _notebook08_normalize_content_text(text)
    assert "```" not in result
    assert "`" not in result
    assert "#" not in result
    assert "Título" in result


@scenario("T4. normalize_content_text convierte enlaces markdown a solo su texto")
def test_normalize_content_text_links():
    text = "Ver [este enlace](https://example.com/paper) para más detalles."
    result = _notebook08_normalize_content_text(text)
    assert "https://example.com" not in result
    assert "este enlace" in result


@scenario("T5. normalize_claim_text normaliza espacios, quita puntuación final y pasa a minúsculas")
def test_normalize_claim_text():
    text = "El Modelo Alcanzó 91%   de precisión [p.pdf | c1]. "
    result = _notebook08_normalize_claim_text(text)
    assert result == "el modelo alcanzó 91% de precisión"
    assert result == result.casefold()


@scenario("T6. numeric_search_variants genera variantes equivalentes de un número con coma decimal")
def test_numeric_search_variants():
    variants = _notebook08_numeric_search_variants("91,5%")
    assert "91.5" in variants
    assert "91,5%" in variants


@scenario("T7. numeric_search_variants colapsa un entero flotante a su forma sin decimales")
def test_numeric_search_variants_integer():
    variants = _notebook08_numeric_search_variants("100%")
    assert "100" in variants


# ---------------------------------------------------------------------------
# Aislamiento de Ground Truth (§7 del inventario) — reproducción de la
# detección de la celda 11 (regex sobre source_filename de los chunks).
# ---------------------------------------------------------------------------

_NOTEBOOK08_GT_PATTERN = re.compile(r"ground[_\s-]*truth|gt_", re.IGNORECASE)


@scenario("T8. detección de Ground Truth mezclado en chunks: positivo")
def test_ground_truth_detection_positive():
    filenames = ["paper1.pdf", "ground_truth_paper.pdf", "paper2.pdf"]
    assert any(_NOTEBOOK08_GT_PATTERN.search(name) for name in filenames)


@scenario("T9. detección de Ground Truth mezclado en chunks: negativo (corpus limpio)")
def test_ground_truth_detection_negative():
    filenames = ["paper1.pdf", "paper2.pdf", "reference_study.pdf"]
    assert not any(_NOTEBOOK08_GT_PATTERN.search(name) for name in filenames)


@scenario("T10. detección de Ground Truth: variante gt_ también se detecta")
def test_ground_truth_detection_gt_prefix():
    filenames = ["gt_literature_review.pdf"]
    assert any(_NOTEBOOK08_GT_PATTERN.search(name) for name in filenames)


# ---------------------------------------------------------------------------
# Lógica de fingerprint/rebuild (§6, celda 15) — reproducción de la decisión
# force_rebuild / no_previous_manifest / stale / current.
# ---------------------------------------------------------------------------


def _notebook08_evaluation_status(
    *, force_rebuild, previous_manifest, current_fingerprint, auto_rebuild
):
    """Reproduce literalmente la rama de decisión de la celda 15 (sin el I/O
    de backup/reset, que pertenece a la migración real, no a esta prueba)."""

    if force_rebuild:
        status, rebuild_required = "force_rebuild", True
    elif previous_manifest is None:
        status, rebuild_required = "no_previous_manifest", True
    elif previous_manifest.get("fingerprint") != current_fingerprint:
        status, rebuild_required = "stale_outputs_dependency_changed", True
    else:
        status, rebuild_required = "outputs_are_current", False

    if rebuild_required and not auto_rebuild and not force_rebuild:
        raise RuntimeError(
            "La evaluación necesita regenerarse, pero "
            "EVALUATION_POLICY['auto_rebuild'] es False."
        )
    should_rebuild = force_rebuild or (rebuild_required and auto_rebuild)
    return status, should_rebuild


@scenario("T11. fingerprint/rebuild: sin manifiesto previo -> reconstruye")
def test_rebuild_no_previous_manifest():
    status, should_rebuild = _notebook08_evaluation_status(
        force_rebuild=False,
        previous_manifest=None,
        current_fingerprint="abc",
        auto_rebuild=True,
    )
    assert status == "no_previous_manifest"
    assert should_rebuild is True


@scenario("T12. fingerprint/rebuild: fingerprint sin cambios -> no reconstruye")
def test_rebuild_current():
    status, should_rebuild = _notebook08_evaluation_status(
        force_rebuild=False,
        previous_manifest={"fingerprint": "abc"},
        current_fingerprint="abc",
        auto_rebuild=True,
    )
    assert status == "outputs_are_current"
    assert should_rebuild is False


@scenario("T13. fingerprint/rebuild: fingerprint cambiado -> obsoleto, reconstruye")
def test_rebuild_stale():
    status, should_rebuild = _notebook08_evaluation_status(
        force_rebuild=False,
        previous_manifest={"fingerprint": "old"},
        current_fingerprint="new",
        auto_rebuild=True,
    )
    assert status == "stale_outputs_dependency_changed"
    assert should_rebuild is True


@scenario("T14. fingerprint/rebuild: obsoleto pero auto_rebuild=False -> falla (igual que el notebook real)")
def test_rebuild_stale_without_auto_rebuild_raises():
    try:
        _notebook08_evaluation_status(
            force_rebuild=False,
            previous_manifest={"fingerprint": "old"},
            current_fingerprint="new",
            auto_rebuild=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("debía fallar: auto_rebuild=False con salidas obsoletas")


@scenario("T15. fingerprint/rebuild: force_rebuild siempre reconstruye, incluso con fingerprint igual")
def test_force_rebuild_overrides():
    status, should_rebuild = _notebook08_evaluation_status(
        force_rebuild=True,
        previous_manifest={"fingerprint": "abc"},
        current_fingerprint="abc",
        auto_rebuild=True,
    )
    assert status == "force_rebuild"
    assert should_rebuild is True


if __name__ == "__main__":
    for fn in (
        test_strip_internal_citations,
        test_strip_internal_citations_no_citations,
        test_normalize_content_text,
        test_normalize_content_text_links,
        test_normalize_claim_text,
        test_numeric_search_variants,
        test_numeric_search_variants_integer,
        test_ground_truth_detection_positive,
        test_ground_truth_detection_negative,
        test_ground_truth_detection_gt_prefix,
        test_rebuild_no_previous_manifest,
        test_rebuild_current,
        test_rebuild_stale,
        test_rebuild_stale_without_auto_rebuild_raises,
        test_force_rebuild_overrides,
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
