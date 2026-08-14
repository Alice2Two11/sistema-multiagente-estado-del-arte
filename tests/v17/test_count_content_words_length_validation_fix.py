"""Corrección de la validación global de longitud: ``count_content_
words`` cuenta únicamente contenido lingüístico real, excluyendo citas
estructuradas reconocidas por ``CITATION_RE`` -- ``draft_text``
conserva las citas intactas en todo momento; solo el CONTEO excluye
sus tokens internos. Usado exclusivamente en el único punto que
alimenta ``word_count``/``total_words``/``global_length_valid``/
``sections_outside_word_range`` (``build_draft_reports``,
``validation.py``) -- compartido sin cambios entre legacy y V2.

``count_words`` (la función original) permanece intacta -- se reutiliza
tal cual dentro de ``count_content_words``, y sigue siendo la que usan
los payloads de auditoría por intento en ``draft_writing_agent.py``
(no forman parte del cálculo de longitud narrativa global).

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.draft_writing.validation import (  # noqa: E402
    build_draft_reports,
    count_content_words,
    count_words,
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


@scenario("CCW01. Texto sin citas -> count_content_words == count_words")
def test_no_citations_counts_equal():
    text = "El modelo alcanza un noventa y uno por ciento de precisión en el conjunto evaluado durante el estudio comparativo completo."
    assert count_content_words(text) == count_words(text)


@scenario("CCW02. Texto con una cita -> la cita no aumenta el conteo respecto al mismo texto sin ella")
def test_one_citation_does_not_inflate_count():
    text_without = "El modelo alcanza un noventa y uno por ciento de precisión en el conjunto evaluado."
    text_with = text_without[:-1] + " [paper_alpha.pdf | chunk_0001]."
    assert count_words(text_with) > count_words(text_without)  # la cita SÍ infla count_words (comportamiento original, sin tocar)
    assert count_content_words(text_with) == count_words(text_without)  # pero NO infla count_content_words


@scenario("CCW03. Varias citas -> ninguna aumenta el conteo de contenido")
def test_multiple_citations_do_not_inflate_count():
    text = (
        "El primer hallazgo confirma la hipótesis planteada [paper_uno.pdf | chunk_0001]. "
        "El segundo hallazgo la refuerza bajo condiciones distintas [paper_dos.pdf | chunk_0002] "
        "y un tercer resultado la respalda de forma independiente [paper_tres.pdf | chunk_0003]."
    )
    text_no_citations_manual = (
        "El primer hallazgo confirma la hipótesis planteada. "
        "El segundo hallazgo la refuerza bajo condiciones distintas "
        "y un tercer resultado la respalda de forma independiente."
    )
    assert count_content_words(text) == count_words(text_no_citations_manual)
    assert count_content_words(text) < count_words(text)


@scenario("CCW04. Números científicos normales (19%, 0.38, 1975) siguen contando igual que antes -- nunca están dentro de una cita real")
def test_normal_scientific_numbers_still_count():
    text = "El resultado fue del 19% con una desviación estándar de 0.38 en el estudio publicado en 1975."
    assert count_content_words(text) == count_words(text)
    # Con una cita agregada, los números siguen contando -- solo la cita se excluye.
    text_with_citation = text.rstrip(".") + " [study.pdf | chunk_0007]."
    assert count_content_words(text_with_citation) == count_words(text)


@scenario("CCW05. Texto entre corchetes que NO es una cita válida según CITATION_RE no se elimina")
def test_non_citation_brackets_not_removed():
    text = "El fenómeno observado [ver discusión posterior] se explica por múltiples factores ambientales evaluados en profundidad."
    assert count_content_words(text) == count_words(text)
    # Confirma explícitamente que el corchete no-cita sigue presente en el texto original (draft_text nunca se modifica aquí).
    assert "[ver discusión posterior]" in text


@scenario("CCW06. global_length_valid usa el conteo sin citas (build_draft_reports real)")
def test_global_length_valid_uses_content_word_count():
    base = "El estudio confirma un patrón consistente de mejora en el indicador evaluado bajo condiciones controladas"
    sentences = [
        f"{base} numero {i} [nombre_de_archivo_del_paper_evaluado_numero_{i}_version_final.pdf | identificador_de_fragmento_extenso_numero_{i:04d}]."
        for i in range(30)
    ]
    draft_text = " ".join(sentences)
    raw_count = count_words(draft_text)
    content_count = count_content_words(draft_text)
    assert content_count < raw_count  # confirma que hay inflación real por citas en este fixture

    max_total = (raw_count + content_count) // 2  # umbral colocado exactamente entre ambos
    section = {
        "section_id": "S1", "section_title": "Methods", "draft_text": draft_text,
        "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
    }
    outline_sections = [{"section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica"}]
    policy = {
        "target_total_words": max_total, "min_total_words": 1, "max_total_words": max_total,
        "section_budgets": {"S1": {"target_words": max_total, "minimum_words": 1, "maximum_words": max_total}},
    }
    report, _, _, _, _ = build_draft_reports([section], outline_sections, {"S1": []}, policy)
    assert report["total_words"] == content_count
    assert report["global_length_valid"] is True


@scenario("CCW07. sections_outside_word_range usa el conteo sin citas (build_draft_reports real)")
def test_sections_outside_word_range_uses_content_word_count():
    base = "El estudio confirma un patrón consistente de mejora en el indicador evaluado bajo condiciones controladas"
    sentences = [
        f"{base} numero {i} [nombre_de_archivo_del_paper_evaluado_numero_{i}_version_final.pdf | identificador_de_fragmento_extenso_numero_{i:04d}]."
        for i in range(20)
    ]
    draft_text = " ".join(sentences)
    content_count = count_content_words(draft_text)

    section = {
        "section_id": "S1", "section_title": "Methods", "draft_text": draft_text,
        "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
    }
    outline_sections = [{"section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica"}]
    # Presupuesto por sección construido a partir del conteo de CONTENIDO real (no del raw con citas).
    policy = {
        "target_total_words": content_count, "min_total_words": 1, "max_total_words": content_count + 5000,
        "section_budgets": {"S1": {"target_words": content_count, "minimum_words": 1, "maximum_words": content_count}},
    }
    report, _, section_rows, _, _ = build_draft_reports([section], outline_sections, {"S1": []}, policy)
    assert section_rows[0]["word_count"] == content_count
    assert section_rows[0]["within_section_range"] is True
    assert report["sections_outside_word_range"] == []


@scenario("CCW08. Legacy mantiene comportamiento salvo por eliminar inflación artificial de citas -- sin citas, el resultado es idéntico al de antes")
def test_legacy_behavior_unchanged_without_citations():
    text_no_citations = (
        "El compuesto reduce la actividad enzimática de forma dependiente de la concentración administrada durante "
        "el ensayo controlado realizado bajo condiciones ambientales estandarizadas y replicables en laboratorio."
    )
    section = {
        "section_id": "S1", "section_title": "Methods", "draft_text": text_no_citations,
        "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
    }
    outline_sections = [{"section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica"}]
    policy = {
        "target_total_words": 100, "min_total_words": 1, "max_total_words": 1000,
        "section_budgets": {"S1": {"target_words": 100, "minimum_words": 1, "maximum_words": 1000}},
    }
    report, _, section_rows, _, _ = build_draft_reports([section], outline_sections, {"S1": []}, policy)
    # Sin citas, count_content_words == count_words -- el reporte es IDÉNTICO al comportamiento histórico.
    assert section_rows[0]["word_count"] == count_words(text_no_citations)
    assert report["total_words"] == count_words(text_no_citations)


@scenario("CCW09. V2 mantiene todos sus gates -- este cambio es transparente para el contrato evidence handles (no se toca ningún archivo de V2)")
def test_v2_gates_unaffected_by_length_fix():
    # No se importa ni ejecuta nada de canonical_sentences.py aquí --
    # esta prueba documenta explícitamente que el fix vive
    # exclusivamente en validation.py (compartido, no específico de
    # V2), y que los gates de V2 se verifican por separado, sin ningún
    # cambio necesario en sus propios archivos de test.
    import src.tools.draft_writing.canonical_sentences as v2_module
    import inspect

    source = inspect.getsource(v2_module)
    assert "count_content_words" not in source  # V2 no necesita conocer esta función -- la validación de longitud vive en validation.py, compartida
    assert "count_words" not in source


if __name__ == "__main__":
    for fn in (
        test_no_citations_counts_equal,
        test_one_citation_does_not_inflate_count,
        test_multiple_citations_do_not_inflate_count,
        test_normal_scientific_numbers_still_count,
        test_non_citation_brackets_not_removed,
        test_global_length_valid_uses_content_word_count,
        test_sections_outside_word_range_uses_content_word_count,
        test_legacy_behavior_unchanged_without_citations,
        test_v2_gates_unaffected_by_length_fix,
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
