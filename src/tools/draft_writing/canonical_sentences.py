"""Contrato canónico de representación de secciones -- ``sentences[]``
como única fuente textual, materialización determinista de ``draft_text``
+ ``claims[]``.

FASE 1 (entrega anterior): solo la interfaz mínima para demostrar el
aislamiento del camino legacy detrás de la policy ``draft_
representation_contract`` -- ``generate_section_canonical_v2`` sigue
lanzando ``NotImplementedError`` explícitamente, sin cambios en esta
fase (ver más abajo).

FASE 2A (esta entrega): schema/parsing real de ``sentences[]`` --
estructura, atomicidad, citas obligatorias/válidas, asignación
determinista de ``sentence_id``. Deliberadamente NO incluye todavía:

  - identity_action/parent_claim_uids (el contrato LLM de generación
    inicial no los pide -- el sistema los asignará como NEW más
    adelante, durante la materialización, que tampoco es parte de esta
    fase);
  - materialización de draft_text/claims[] (state_of_art_draft.json);
  - conexión con execute() de Agent06 (generate_section_canonical_v2
    sigue siendo un stub -- esta fase NO lo toca).

Toda la lógica nueva reutiliza, por import de solo lectura, las
funciones puras ya existentes y ya probadas en ``normalization.py``
(``split_sentences_preserving_citations``, ``is_substantive_sentence``,
``extract_claim_pairs``, ``resolve_allowed_pair``, ``citation_string``)
-- ese archivo no se modifica en absoluto en esta fase."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Mapping, Sequence

from .normalization import (
    CITATION_RE,
    citation_string,
    extract_claim_pairs,
    is_substantive_sentence,
    resolve_allowed_pair,
    split_sentences_preserving_citations,
)
from .claim_identity import (
    ClaimIdentityDeclaration,
    default_mint_claim_uid,
    resolve_claim_identity,
)

# Vocabulario de errores de esta fase -- códigos EXACTOS pedidos,
# usados como prefijo de cada entrada de "errors" (algunas llevan un
# sufijo ":<índice>" o ":<cita>" para dar contexto específico, pero el
# código base es siempre uno de estos, verificable con .startswith()).
INVALID_SENTENCES_STRUCTURE = "INVALID_SENTENCES_STRUCTURE"
EMPTY_SENTENCE_ITEM = "EMPTY_SENTENCE_ITEM"
SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES = "SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES"
MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE = "MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE"
INVALID_CITATION = "INVALID_CITATION"
INLINE_CITATION_NOT_ALLOWED = "INLINE_CITATION_NOT_ALLOWED"
SECTION_ID_MISMATCH = "SECTION_ID_MISMATCH"
UNEXPECTED_SENTENCE_FIELD = "UNEXPECTED_SENTENCE_FIELD"

# Fase 2A, contrato de GENERACIÓN INICIAL únicamente: cada elemento de
# sentences[] puede contener EXCLUSIVAMENTE estos dos campos. Cualquier
# otro -- incluidos identity_action/parent_claim_uids/claim_uid/claim/
# claim_id/sentence_id, que pertenecen a etapas POSTERIORES
# (materialización, hecha por el sistema, nunca declarada por el LLM en
# esta fase) -- se rechaza explícitamente. Si en el futuro se necesita
# ampliar este conjunto (ej. para revisión dirigida, donde sí haría
# falta un target_claim_uid autoritativo), debe justificarse y ampliarse
# aquí de forma explícita -- nunca de forma implícita ignorando un
# campo desconocido.
ALLOWED_SENTENCE_ITEM_FIELDS = frozenset({"text", "supporting_citations"})

# Fase 2B: sufijo de puntuación final COMPLETO, cualquier combinación
# de "." "?" "!" al final del texto (".", "?", "!", "?!", "!!", "...",
# etc.) -- extraído tal cual, nunca normalizado ni recortado a un solo
# carácter, para preservar exactamente la puntuación original al
# reinsertar las citas después de ella.
_TRAILING_PUNCTUATION_RE = re.compile(r"[.?!]+$")


def validate_sentence_item_fields(item: Mapping[str, Any], index: int) -> str | None:
    """Fail-closed sobre el schema exacto de cada elemento de
    sentences[]: solo ``text``/``supporting_citations`` están
    permitidos en generación inicial. Cualquier otro campo -- en
    particular los de identidad (identity_action, parent_claim_uids,
    claim_uid) o los que el SISTEMA asigna después (claim, claim_id,
    sentence_id) -- se rechaza explícitamente, nunca se ignora en
    silencio."""

    for field in item.keys():
        if field not in ALLOWED_SENTENCE_ITEM_FIELDS:
            return f"{UNEXPECTED_SENTENCE_FIELD}:{index}:{field}"
    return None


def validate_sentences_payload_structure(
    payload: Any, expected_section_id: str | None = None
) -> str | None:
    """V2A01 -- estructura básica del payload completo (antes de mirar
    el contenido de cada oración). Devuelve el código de error, o
    ``None`` si la estructura es válida en este nivel.

    Fail-closed: cualquier forma inesperada (no es dict, "sentences"
    ausente/no-lista/vacía, un elemento que no es dict, ``text``
    ausente/no-string, ``section_id`` ausente/no-string/vacío) se
    rechaza aquí -- nunca se intenta reparar ni completar con un valor
    por defecto. ``section_id`` nunca se infiere ni se sobrescribe: si
    ``expected_section_id`` se proporciona y no coincide EXACTAMENTE
    con ``payload["section_id"]``, se rechaza con un código explícito
    que muestra ambos valores."""

    if not isinstance(payload, Mapping):
        return INVALID_SENTENCES_STRUCTURE
    section_id = payload.get("section_id")
    if not isinstance(section_id, str) or not section_id.strip():
        return INVALID_SENTENCES_STRUCTURE
    if expected_section_id is not None and section_id != expected_section_id:
        return f"{SECTION_ID_MISMATCH}:{expected_section_id}:{section_id}"
    sentences = payload.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return INVALID_SENTENCES_STRUCTURE
    for item in sentences:
        if not isinstance(item, Mapping):
            return INVALID_SENTENCES_STRUCTURE
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return EMPTY_SENTENCE_ITEM
    return None


def validate_sentence_atomicity(text: str) -> str | None:
    """V2A02 -- cada ``sentences[i].text`` debe contener EXACTAMENTE
    una oración, según la función determinista ya existente y ya
    probada ``split_sentences_preserving_citations`` (sin cambios, sin
    reimplementación paralela). Nunca divide ni repara en silencio: si
    el texto contiene 0 o más de 1 oración, se rechaza con el código
    correspondiente."""

    segments = split_sentences_preserving_citations(text)
    if len(segments) == 0:
        return EMPTY_SENTENCE_ITEM
    if len(segments) > 1:
        return SENTENCE_ITEM_CONTAINS_MULTIPLE_SENTENCES
    return None


def validate_inline_citation_absence(text: str, index: int) -> str | None:
    """El contrato V2 exige que ``text`` contenga ÚNICAMENTE la
    oración -- las citas viven exclusivamente en ``supporting_
    citations``. Si aparece cualquier patrón ``[source | chunk]``
    dentro de ``text``, se rechaza explícitamente -- nunca se elimina
    ni se mueve automáticamente a ``supporting_citations`` (eso sería
    una reparación silenciosa, prohibida por el contrato fail-closed).
    """

    if CITATION_RE.search(text):
        return f"{INLINE_CITATION_NOT_ALLOWED}:{index}"
    return None


def validate_raw_supporting_citations(item: Mapping[str, Any], index: int) -> list[str]:
    """Valida la estructura CRUDA de ``supporting_citations`` -- antes
    de convertir nada a pares con ``extract_claim_pairs``, porque esa
    función IGNORA en silencio cualquier elemento que no haga
    ``fullmatch`` con ``CITATION_RE`` (comportamiento heredado y
    correcto para su propio uso en el camino legacy, pero insuficiente
    aquí: usarla directamente para decidir "todas las citas
    declaradas fueron válidas" dejaría desaparecer sin rastro una cita
    malformada).

    Devuelve la lista de errores encontrados (vacía si todo es
    válido). Cada elemento crudo inválido -- ausente de list, no-string,
    string vacío, o que no matchea el formato exacto -- se reporta
    individualmente con su valor original visible, nunca se descarta
    sin dejar rastro."""

    raw = item.get("supporting_citations")
    if raw is None:
        return []
    if not isinstance(raw, list):
        # supporting_citations presente pero no es una lista -- toda la
        # estructura declarada es inválida, se reporta con su valor
        # crudo completo.
        return [f"{INVALID_CITATION}:{index}:{raw!r}"]

    errors: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{INVALID_CITATION}:{index}:{value!r}")
            continue
        if not CITATION_RE.fullmatch(value.strip()):
            # Malformada (no tiene la forma exacta "[source | chunk]")
            # -- nunca se filtra en silencio como haría extract_claim_
            # pairs; se reporta con el texto crudo tal cual se recibió.
            errors.append(f"{INVALID_CITATION}:{index}:{value}")
    return errors


def validate_and_parse_sentences_v2(
    payload: Mapping[str, Any],
    allowed_pairs: Sequence[tuple[str, str]] | set[tuple[str, str]],
    *,
    expected_section_id: str | None = None,
) -> dict[str, Any]:
    """Punto de entrada de fase 2A: valida y parsea ``sentences[]``
    contra el schema del contrato LLM inicial (sin identity_action ni
    parent_claim_uids -- generación inicial únicamente).

    Devuelve::

        {
          "validation_ok": bool,
          "errors": [<código>, ...],           # nunca vacío si no ok
          "sentences": [                        # None si no ok
            {"sentence_id": 0, "text": ..., "supporting_citations": [...]},
            ...
          ],
        }

    Reglas fail-closed, sin excepción:
      - Cualquier error (estructura, sección, atomicidad, cita
        faltante, cita inválida, campo inesperado) invalida la SECCIÓN
        COMPLETA -- nunca se devuelve una lista parcial de oraciones
        "las que sí pasaron".
      - Ninguna cita inválida se descarta en silencio: se reporta como
        error explícito (INVALID_CITATION); nunca se filtra fuera del
        resultado sin dejar rastro.
      - ``section_id`` nunca se infiere ni se sobrescribe: si
        ``expected_section_id`` se pasa y no coincide, se rechaza con
        SECTION_ID_MISMATCH mostrando ambos valores.
      - Cada elemento de sentences[] solo puede tener ``text``/
        ``supporting_citations`` -- cualquier otro campo (incluidos
        los de identidad, o los que el sistema asigna después como
        sentence_id) se rechaza explícitamente, nunca se ignora.
      - ``sentence_id`` es puramente posicional (el índice dentro de
        ``sentences[]`` tal como llegó, 0/1/2/...) -- nunca un UUID, y
        nunca se infiere de contenido/similitud.
    """

    allowed = set(allowed_pairs)
    structure_error = validate_sentences_payload_structure(payload, expected_section_id)
    if structure_error is not None:
        return {"validation_ok": False, "errors": [structure_error], "sentences": None}

    raw_sentences = payload["sentences"]
    errors: list[str] = []
    parsed: list[dict[str, Any]] = []

    for index, item in enumerate(raw_sentences):
        # 0. Schema cerrado por elemento -- solo text/supporting_
        # citations están permitidos en generación inicial. Se
        # comprueba ANTES que cualquier otra validación de contenido:
        # un campo inesperado (identity_action, parent_claim_uids,
        # claim_uid, claim, claim_id, sentence_id, o cualquier otro) es
        # un problema de forma, no de contenido.
        field_error = validate_sentence_item_fields(item, index)
        if field_error is not None:
            errors.append(field_error)
            continue

        text = str(item.get("text", "")).strip()

        # 1. Ninguna cita embebida en text -- comprobado ANTES de
        # atomicidad/materialización, tal como exige el contrato: las
        # citas viven exclusivamente en supporting_citations.
        inline_error = validate_inline_citation_absence(text, index)
        if inline_error is not None:
            errors.append(inline_error)
            continue

        # 2. Estructura CRUDA de supporting_citations -- nunca se usa
        # extract_claim_pairs para decidir "todas las citas declaradas
        # son válidas", porque esa función filtra en silencio
        # cualquier elemento malformado. Se valida ANTES, sobre los
        # valores crudos tal como llegaron, para toda oración (incluso
        # las no sustantivas -- ninguna cita declarada puede
        # desaparecer sin error).
        raw_citation_errors = validate_raw_supporting_citations(item, index)
        if raw_citation_errors:
            errors.extend(raw_citation_errors)
            continue

        atomicity_error = validate_sentence_atomicity(text)
        if atomicity_error is not None:
            errors.append(f"{atomicity_error}:{index}")
            continue

        # A partir de aquí, supporting_citations ya se validó
        # estructuralmente completo (paso 2) -- extract_claim_pairs es
        # seguro de usar, ya no puede haber descartado nada malformado
        # sin que ya se haya reportado como error arriba.
        pairs = extract_claim_pairs(item)
        substantive = is_substantive_sentence(text)

        if substantive and not pairs:
            errors.append(f"{MISSING_CITATIONS_FOR_SUBSTANTIVE_SENTENCE}:{index}")
            continue

        sentence_invalid = False
        for pair in pairs:
            if resolve_allowed_pair(pair, allowed) is None:
                # Fail-closed: la cita inválida se REPORTA, nunca se
                # descarta en silencio ni se filtra fuera del par.
                errors.append(f"{INVALID_CITATION}:{index}:{citation_string(pair)}")
                sentence_invalid = True
        if sentence_invalid:
            continue

        parsed.append({
            "sentence_id": index,
            "text": text,
            "supporting_citations": [citation_string(p) for p in pairs],
        })

    if errors:
        # Fail-closed total: un solo error en cualquier oración
        # invalida la sección COMPLETA -- nunca se materializa un
        # subconjunto "parcialmente válido".
        return {"validation_ok": False, "errors": errors, "sentences": None}

    return {"validation_ok": True, "errors": [], "sentences": parsed}


def _fingerprint_claim_text(text: str) -> str:
    """Misma primitiva exacta que ``fingerprint_text`` (``src/tools/
    verification/corrections.py``, ``sha256(text.encode("utf-8")).
    hexdigest()``) -- redefinida localmente en vez de importada, porque
    importar ``corrections.py`` arrastraría todo el módulo de
    verificación (``validation.py``, masivo) como efecto colateral,
    violando el aislamiento de importación liviana que V2A10 ya
    garantiza. No hay ningún algoritmo de negocio que duplicar aquí --
    es una llamada directa a una primitiva estándar de ``hashlib``,
    idéntica en ambos lugares."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _materialize_sentence_fragment(text: str, citations: Sequence[str]) -> str:
    """Inserta las citas ya resueltas al final de la oración, con
    EXACTAMENTE el mismo formato de espaciado que usa hoy el camino
    legacy (``normalize_generated_section``, ``normalization.py``, sin
    modificar): el texto base, seguido de las citas separadas por
    espacio, seguido del sufijo de puntuación final ORIGINAL COMPLETO
    -- nunca solo un caracter.

    A diferencia del camino legacy (que solo reconoce un signo de
    cierre simple), esta funcion extrae el sufijo de puntuacion final
    COMPLETO (cualquier combinacion de ".", "?", "!" al final del
    texto, ej. "?!", "!!", "...") y lo reinserta tal cual, despues de
    las citas -- nunca lo normaliza ni lo recorta a un solo caracter.
    Nunca reformula ni reordena el texto -- copia literal mas citas al
    final, determinista."""

    stripped = text.rstrip()
    match = _TRAILING_PUNCTUATION_RE.search(stripped)
    if match:
        punct = match.group(0)
        base = stripped[: match.start()].strip()
    else:
        punct = ""
        base = stripped.strip()
    if citations:
        return f"{base} {' '.join(citations)}{punct}"
    return f"{base}{punct}"


def materialize_initial_section_v2(
    sentences: Sequence[Mapping[str, Any]],
    section_id: str,
    *,
    round_number: int = 1,
    mint_uid: Callable[[], str] = default_mint_claim_uid,
    text_fingerprint: Callable[[str], str] = _fingerprint_claim_text,
) -> dict[str, Any]:
    """Fase 2B -- materialización determinista GENERACIÓN INICIAL
    únicamente: ``sentences[]`` (ya validado por ``validate_and_parse_
    sentences_v2``, ``validation_ok=True``) -> ``draft_text`` +
    ``claims[]``, en el shape externo exacto que hoy consumen 06/07/08.

    Principio central: el texto del claim NUNCA se vuelve a escribir --
    es una copia directa de ``sentence["text"]``. No hay LLM, no hay
    fuzzy matching, no hay parafraseo ni corrección de conectores: esta
    función es pura y determinista.

    Cada oración SUSTANTIVA (``is_substantive_sentence``, función real
    ya existente, sin cambios) produce exactamente un claim, con:
      - ``identity_action="NEW"`` y ``parent_claim_uids=()`` --
        asignados por el SISTEMA, nunca declarados por el LLM (el
        contrato de fase 2A ya rechaza que el LLM los envíe);
      - identidad real resuelta vía ``resolve_claim_identity``
        (``claim_identity.py``, sin modificar) -- mismo camino
        productivo que usaría cualquier otro NEW del sistema;
      - ``claim_id = f"{section_id}_C{n}"`` con ``n`` empezando en 1,
        exactamente la convención vigente confirmada en producción
        (``validation.py``, ``enumerate(claims, start=1)``).

    Las oraciones NO sustantivas permanecen en ``draft_text`` -- nunca
    se eliminan -- pero no generan ningún claim ni reciben identidad.

    Precisión sobre determinismo (no todo el output es determinista de
    la misma forma):
      - ``draft_text`` es determinista -- función pura del ``sentences``
        de entrada.
      - Por cada claim, ``claim``/``claim_id``/``supporting_citations``/
        ``claim_text_fingerprint``/``identity_action``/``parent_claim_
        uids``/``claim_version``/``created_round``/``updated_round`` son
        deterministas para el mismo input -- se recalculan igual cada
        vez.
      - ``claim_uid`` de un ``NEW`` NO es determinista por diseño: cada
        llamada mintea una identidad real y única (``default_mint_
        claim_uid``, UUID4 aleatorio, sin cambios) -- dos
        materializaciones del mismo texto producen ``claim_uid``
        distintos, correctamente (nunca deben compartir identidad por
        casualidad de tener el mismo texto).
      - Con un ``mint_uid`` determinista INYECTADO (parámetro de esta
        función), la salida completa -- incluido ``claim_uid`` -- sí es
        reproducible. Esto no cambia el mecanismo real de
        ``claim_identity.py`` ni convierte el UID en un hash del texto
        -- solo permite pruebas deterministas end-to-end.

    Devuelve ``{"draft_text": str, "claims": [...]}``."""

    fragments: list[str] = []
    claims: list[dict[str, Any]] = []
    claim_ordinal = 0

    for sentence in sentences:
        text = str(sentence["text"])
        citations = list(sentence.get("supporting_citations") or [])
        fragments.append(_materialize_sentence_fragment(text, citations))

        if not is_substantive_sentence(text):
            # Permanece en draft_text (ya añadido arriba) -- nunca
            # genera claim, nunca recibe identidad artificial.
            continue

        claim_ordinal += 1
        claim_id = f"{section_id}_C{claim_ordinal}"
        declaration = ClaimIdentityDeclaration(action="NEW", parent_claim_uids=())
        identity = resolve_claim_identity(
            declaration=declaration,
            claim_text=text,
            claim_id=claim_id,
            previous_claims_by_uid={},
            forced_parent_uid=None,
            round_number=round_number,
            text_fingerprint=text_fingerprint,
            mint_uid=mint_uid,
        )
        claims.append({
            "claim_id": claim_id,
            "claim": text,  # copia EXACTA -- nunca re-derivada ni reformulada
            "supporting_citations": citations,
            "identity_action": declaration.action,
            **identity.to_dict(),
        })

    return {"draft_text": " ".join(fragments), "claims": claims}


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
    "canonical_sentences_v2"``.

    Fase 2A/2B: SIN CAMBIOS respecto a fases anteriores -- sigue sin
    implementar. ``validate_and_parse_sentences_v2`` (parsing/
    validación) y ``materialize_initial_section_v2`` (materialización
    determinista de generación inicial) ya existen y están probadas,
    pero deliberadamente NO se conectan aquí todavía: esta fase no
    invoca al runtime ni conecta nada con execute() de Agent06 -- eso
    queda para una fase posterior, autorizada por separado."""

    raise NotImplementedError(
        "canonical_sentences_v2: generación end-to-end pendiente de "
        "autorización de una fase posterior (parsing/validación de "
        "sentences[] -- fase 2A -- y materialización determinista de "
        "generación inicial -- fase 2B -- ya implementadas y probadas "
        "por separado, pero aún no conectadas a la invocación real del "
        "runtime ni a execute() de Agent06)."
    )
