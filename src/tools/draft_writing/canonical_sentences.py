"""Contrato canónico de representación de secciones -- ``sentences[]``
como única fuente textual, materialización determinista de ``draft_text``
+ ``claims[]``.

FASE 1 (esta entrega): solo la interfaz mínima necesaria para demostrar
el aislamiento del camino legacy detrás de la policy ``draft_
representation_contract``. Ninguna función de este módulo implementa
lógica real todavía -- todas lanzan ``NotImplementedError`` de forma
explícita. Importar este módulo NO tiene efectos secundarios: no
registra nada, no modifica ningún estado global, no ejecuta código al
cargarse -- solo define nombres.

La lógica real (segmentación con validación de atomicidad, resolución
fail-closed de citas, materialización de draft_text/claims, los tres
niveles de identidad) se implementa en una fase posterior, solo tras
que los tests LEGACY01-08 confirmen el aislamiento del camino legacy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def generate_section_canonical_v2(
    *,
    section: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    quant_context: Any,
    previous_errors: list[Any],
    policy: Mapping[str, Any],
    previous_claims_for_identity: Any = None,
    runtime: Any = None,
    raw_dir: Any = None,
    sid: str = "",
) -> Mapping[str, Any]:
    """Punto de entrada único del camino V2 -- solo se invoca cuando la
    policy declara explícitamente ``draft_representation_contract ==
    "canonical_sentences_v2"``. Fase 1: no implementado."""

    raise NotImplementedError(
        "canonical_sentences_v2: lógica funcional pendiente de "
        "autorización de fase 2. Este stub confirma que el módulo "
        "puede importarse sin efectos secundarios y que la función "
        "solo se invoca cuando la policy selecciona V2 explícitamente."
    )
