"""Pruebas diferenciales del Bloque 1: oráculo reproducido vs. módulo real portado.

Compara, para cada caso (incluidos los bordes pedidos), el resultado de una
reproducción independiente del notebook (los ``_oracle_*`` de aquí abajo,
copiados por separado de la celda original, sin compartir código con
``src/tools/evaluation/text_normalization.py``) contra el resultado de
importar y llamar al módulo real ya portado. Si algún caso difiere, es un
defecto de la migración — no debería haber ninguno.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.tools.evaluation.text_normalization import (
    normalize_claim_text,
    normalize_content_text,
    normalize_numeric_token,
    numeric_search_variants,
    safe_str,
    strip_internal_citations,
)

# ---------------------------------------------------------------------------
# Oráculo: reproducción independiente (celdas 1/15/19), sin importar el
# módulo bajo prueba, para que la comparación sea genuinamente diferencial.
# ---------------------------------------------------------------------------


def _oracle_safe_str(value):
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


_oracle_citation_pattern = re.compile(r"\[([^\[\]\|]+?)\s*\|\s*([^\[\]\|]+?)\]")


def _oracle_strip_internal_citations(text):
    return _oracle_citation_pattern.sub("", _oracle_safe_str(text))


def _oracle_normalize_content_text(text):
    value = _oracle_safe_str(text)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value)
    value = re.sub(r"^\s*#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = _oracle_strip_internal_citations(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _oracle_normalize_claim_text(text):
    value = _oracle_strip_internal_citations(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .;:!?").casefold()


def _oracle_normalize_numeric_token(value):
    token = _oracle_safe_str(value)
    token = re.sub(r"\s+", "", token)
    token = token.replace(",", ".")
    return token.casefold()


def _oracle_numeric_search_variants(value):
    normalized = _oracle_normalize_numeric_token(value)
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
# safe_str
# ---------------------------------------------------------------------------


@scenario("D01. safe_str(None) == '' en ambos caminos")
def test_safe_str_none():
    assert safe_str(None) == _oracle_safe_str(None) == ""


@scenario("D02. safe_str('') == '' en ambos caminos")
def test_safe_str_empty():
    assert safe_str("") == _oracle_safe_str("") == ""


@scenario("D03. safe_str recorta espacios (comportamiento real, no un passthrough)")
def test_safe_str_strips():
    assert safe_str("  hola  ") == _oracle_safe_str("  hola  ") == "hola"


@scenario("D04. safe_str serializa listas/dicts como JSON")
def test_safe_str_list_dict():
    assert safe_str([1, 2]) == _oracle_safe_str([1, 2]) == "[1, 2]"
    assert safe_str({"a": 1}) == _oracle_safe_str({"a": 1}) == '{"a": 1}'


@scenario("D05. safe_str(NaN) == '' (pd.isna real, no una verificación propia)")
def test_safe_str_nan():
    import math

    assert safe_str(float("nan")) == _oracle_safe_str(float("nan")) == ""
    assert safe_str(math.nan) == _oracle_safe_str(math.nan) == ""


@scenario("D06. safe_str con Unicode se conserva sin alterar (salvo strip)")
def test_safe_str_unicode():
    text = "  café con ñandú y 中文  "
    assert safe_str(text) == _oracle_safe_str(text) == "café con ñandú y 中文"


# ---------------------------------------------------------------------------
# strip_internal_citations
# ---------------------------------------------------------------------------


@scenario("D07. strip_internal_citations con None")
def test_strip_citations_none():
    assert strip_internal_citations(None) == _oracle_strip_internal_citations(None) == ""


@scenario("D08. strip_internal_citations con cadena vacía")
def test_strip_citations_empty():
    assert strip_internal_citations("") == _oracle_strip_internal_citations("") == ""


@scenario("D09. strip_internal_citations con múltiples citas")
def test_strip_citations_multiple():
    text = "Dato A [p1.pdf | c1] y dato B [p2.pdf | c2] confirman esto."
    expected = "Dato A  y dato B  confirman esto."
    assert strip_internal_citations(text) == _oracle_strip_internal_citations(text) == expected


@scenario("D10. strip_internal_citations con cita malformada (sin barra) no la elimina")
def test_strip_citations_malformed():
    text = "Esto tiene una [cita malformada sin barra] que no matchea el patrón."
    result = strip_internal_citations(text)
    oracle = _oracle_strip_internal_citations(text)
    assert result == oracle
    assert "[cita malformada sin barra]" in result  # el patrón exige "|", no la borra


# ---------------------------------------------------------------------------
# normalize_content_text
# ---------------------------------------------------------------------------


@scenario("D11. normalize_content_text con None")
def test_normalize_content_none():
    assert normalize_content_text(None) == _oracle_normalize_content_text(None) == ""


@scenario("D12. normalize_content_text con cadena vacía")
def test_normalize_content_empty():
    assert normalize_content_text("") == _oracle_normalize_content_text("") == ""


@scenario("D13. normalize_content_text con enlace markdown")
def test_normalize_content_link():
    text = "Ver [este enlace](https://example.com/x) para más info."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "https://example.com" not in result
    assert "este enlace" in result


@scenario("D14. normalize_content_text con imagen markdown (se reemplaza por espacio, no por su texto alt)")
def test_normalize_content_image():
    text = "Antes ![alt de la imagen](https://example.com/img.png) después."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "alt de la imagen" not in result
    assert "img.png" not in result


@scenario("D15. normalize_content_text con bloque de código multilínea")
def test_normalize_content_code_block():
    text = "Texto antes.\n```python\ndef f():\n    return 1\n```\nTexto después."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "def f()" not in result
    assert "Texto antes." in result and "Texto después." in result


@scenario("D16. normalize_content_text con código inline")
def test_normalize_content_inline_code():
    text = "Usa la función `mi_funcion()` para procesar."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "`" not in result
    assert "mi_funcion()" in result  # el contenido inline SÍ se conserva


@scenario("D17. normalize_content_text con encabezados markdown de distintos niveles")
def test_normalize_content_headings():
    text = "# Título 1\n## Título 2\n###### Título 6\nTexto normal."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "#" not in result


@scenario("D18. normalize_content_text con espacios repetidos y saltos de línea múltiples")
def test_normalize_content_whitespace():
    text = "Palabra1    Palabra2\n\n\n\nPalabra3"
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "  " not in result


@scenario("D19. normalize_content_text con Unicode (acentos, ñ, CJK) se conserva")
def test_normalize_content_unicode():
    text = "# Título\nEl niño comió papá y 日本語のテスト."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "niño" in result and "papá" in result and "日本語" in result


@scenario("D20. normalize_content_text combina cita interna + markdown en el mismo texto")
def test_normalize_content_combined():
    text = "# Resultado\nEl valor fue 91% [p1.pdf | c1] según `el código` de referencia."
    result = normalize_content_text(text)
    assert result == _oracle_normalize_content_text(text)
    assert "[p1.pdf | c1]" not in result
    assert "#" not in result
    assert "`" not in result


# ---------------------------------------------------------------------------
# normalize_claim_text
# ---------------------------------------------------------------------------


@scenario("D21. normalize_claim_text con None")
def test_normalize_claim_none():
    assert normalize_claim_text(None) == _oracle_normalize_claim_text(None) == ""


@scenario("D22. normalize_claim_text con puntuación final múltiple")
def test_normalize_claim_multi_punctuation():
    text = "El resultado fue significativo!!! ..."
    result = normalize_claim_text(text)
    assert result == _oracle_normalize_claim_text(text)
    assert not result.endswith((".", "!", ";", ":", " "))


@scenario("D23. normalize_claim_text con mayúsculas/minúsculas mixtas (casefold, no solo lower)")
def test_normalize_claim_casefold():
    text = "El MODELO ES SUPERIOR"
    result = normalize_claim_text(text)
    assert result == _oracle_normalize_claim_text(text)
    assert result == "el modelo es superior"


@scenario("D24. normalize_claim_text con Unicode y cita interna combinados")
def test_normalize_claim_unicode_and_citation():
    text = "El análisis reveló una mejora de 91% [paper.pdf | c3].  "
    result = normalize_claim_text(text)
    assert result == _oracle_normalize_claim_text(text)
    assert "[" not in result


# ---------------------------------------------------------------------------
# normalize_numeric_token / numeric_search_variants
# ---------------------------------------------------------------------------


@scenario("D25. normalize_numeric_token con None")
def test_normalize_numeric_none():
    assert normalize_numeric_token(None) == _oracle_normalize_numeric_token(None) == ""


@scenario("D26. normalize_numeric_token con coma decimal")
def test_normalize_numeric_comma():
    assert normalize_numeric_token("91,5") == _oracle_normalize_numeric_token("91,5") == "91.5"


@scenario("D27. normalize_numeric_token con punto decimal (ya normalizado)")
def test_normalize_numeric_dot():
    assert normalize_numeric_token("91.5") == _oracle_normalize_numeric_token("91.5") == "91.5"


@scenario("D28. normalize_numeric_token con porcentaje se conserva el símbolo")
def test_normalize_numeric_percent():
    result = normalize_numeric_token("91,5%")
    assert result == _oracle_normalize_numeric_token("91,5%")
    assert result == "91.5%"  # el % NO se elimina en normalize_numeric_token


@scenario("D29. normalize_numeric_token con entero")
def test_normalize_numeric_integer():
    assert normalize_numeric_token("100") == _oracle_normalize_numeric_token("100") == "100"


@scenario("D30. normalize_numeric_token con valor no numérico (no lanza, solo normaliza el texto)")
def test_normalize_numeric_non_numeric():
    result = normalize_numeric_token("no es un número")
    oracle = _oracle_normalize_numeric_token("no es un número")
    assert result == oracle == "noesunnúmero"  # sin espacios, casefold — SIN validar que sea numérico


@scenario("D31. normalize_numeric_token con espacios repetidos internos")
def test_normalize_numeric_whitespace():
    result = normalize_numeric_token("  91   5  ")
    oracle = _oracle_normalize_numeric_token("  91   5  ")
    assert result == oracle == "915"  # \s+ se elimina por completo, no se colapsa a un espacio


@scenario("D32. numeric_search_variants con coma decimal genera ambas formas")
def test_variants_comma():
    result = numeric_search_variants("91,5%")
    oracle = _oracle_numeric_search_variants("91,5%")
    assert result == oracle
    assert result == {"91.5%", "91.5", "91,5%", "91,5"}


@scenario("D33. numeric_search_variants con entero sin decimales no agrega variante de coma")
def test_variants_integer():
    result = numeric_search_variants("100%")
    oracle = _oracle_numeric_search_variants("100%")
    assert result == oracle
    assert result == {"100%", "100"}


@scenario("D34. numeric_search_variants con valor no numérico devuelve variantes de texto, sin lanzar")
def test_variants_non_numeric():
    result = numeric_search_variants("no numérico")
    oracle = _oracle_numeric_search_variants("no numérico")
    assert result == oracle


@scenario("D35. numeric_search_variants con cadena vacía no incluye cadenas vacías")
def test_variants_empty():
    result = numeric_search_variants("")
    oracle = _oracle_numeric_search_variants("")
    assert result == oracle == set()


@scenario("D36. numeric_search_variants con None no lanza")
def test_variants_none():
    result = numeric_search_variants(None)
    oracle = _oracle_numeric_search_variants(None)
    assert result == oracle == set()


if __name__ == "__main__":
    for fn in (
        test_safe_str_none,
        test_safe_str_empty,
        test_safe_str_strips,
        test_safe_str_list_dict,
        test_safe_str_nan,
        test_safe_str_unicode,
        test_strip_citations_none,
        test_strip_citations_empty,
        test_strip_citations_multiple,
        test_strip_citations_malformed,
        test_normalize_content_none,
        test_normalize_content_empty,
        test_normalize_content_link,
        test_normalize_content_image,
        test_normalize_content_code_block,
        test_normalize_content_inline_code,
        test_normalize_content_headings,
        test_normalize_content_whitespace,
        test_normalize_content_unicode,
        test_normalize_content_combined,
        test_normalize_claim_none,
        test_normalize_claim_multi_punctuation,
        test_normalize_claim_casefold,
        test_normalize_claim_unicode_and_citation,
        test_normalize_numeric_none,
        test_normalize_numeric_comma,
        test_normalize_numeric_dot,
        test_normalize_numeric_percent,
        test_normalize_numeric_integer,
        test_normalize_numeric_non_numeric,
        test_normalize_numeric_whitespace,
        test_variants_comma,
        test_variants_integer,
        test_variants_non_numeric,
        test_variants_empty,
        test_variants_none,
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
