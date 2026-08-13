from __future__ import annotations

import re
from pathlib import PurePosixPath

from .retrieval import safe_str


CITATION_RE = re.compile(r"\[\s*([^\]|]+?)\s*\|\s*([^\]]+?)\s*\]")


def citation_string(pair):
    return f"[{pair[0]} | {pair[1]}]"


def canonicalize_citation_position(text):
    text = safe_str(text)
    return re.sub(
        r"([.!?])\s*(\[[^\]]+\|[^\]]+\])",
        r" \2\1",
        text,
    )


def split_sentences_preserving_citations(text):
    text = canonicalize_citation_position(text)
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+", text)
        if x.strip()
    ]


def is_substantive_sentence(sentence):
    clean = CITATION_RE.sub("", safe_str(sentence))
    return len(re.findall(r"\w+", clean)) >= 8


def normalize_claim_text(text):
    return safe_str(
        CITATION_RE.sub("", safe_str(text))
    ).rstrip(".?!").strip()


def extract_claim_pairs(claim):
    pairs = []

    for c in claim.get("supporting_citations") or []:
        m = CITATION_RE.fullmatch(safe_str(c))

        if m:
            pairs.append(
                (
                    m.group(1).strip(),
                    m.group(2).strip(),
                )
            )

    return pairs


def _source_basename(source_filename):
    """
    Devuelve el basename de una fuente de forma independiente
    del separador de ruta usado por el LLM.
    """
    value = safe_str(source_filename).strip().replace("\\", "/")

    if not value:
        return ""

    return PurePosixPath(value).name


def resolve_allowed_pair(pair, allowed_pairs):
    """
    Resuelve una cita producida por el LLM contra el conjunto cerrado
    de citas permitidas.

    Reglas:
    1. Si (source_filename, chunk_id) coincide exactamente, se conserva.
    2. Si no coincide exactamente, se permite resolver por basename
       SOLO cuando:
         - chunk_id coincide exactamente;
         - basename coincide exactamente;
         - existe UNA ÚNICA coincidencia permitida.
    3. Si no hay coincidencias o hay más de una, se rechaza.

    Nunca crea fuentes ni chunk_ids nuevos.
    """
    if not pair or len(pair) != 2:
        return None

    source_filename = safe_str(pair[0]).strip()
    chunk_id = safe_str(pair[1]).strip()

    candidate = (source_filename, chunk_id)
    allowed = set(allowed_pairs)

    # Coincidencia canónica exacta.
    if candidate in allowed:
        return candidate

    basename = _source_basename(source_filename)

    if not basename or not chunk_id:
        return None

    # Resolver únicamente dentro del conjunto de citas ya permitido.
    matches = [
        allowed_pair
        for allowed_pair in allowed
        if safe_str(allowed_pair[1]).strip() == chunk_id
        and _source_basename(allowed_pair[0]) == basename
    ]

    # Solo se acepta una correspondencia inequívoca.
    if len(matches) == 1:
        return matches[0]

    return None


def _resolve_pairs(pairs, allowed_pairs):
    """
    Resuelve pares de cita y elimina duplicados preservando el orden.
    """
    resolved = []

    for pair in pairs:
        canonical = resolve_allowed_pair(pair, allowed_pairs)

        if canonical is not None and canonical not in resolved:
            resolved.append(canonical)

    return resolved


def normalize_generated_section(section, allowed_pairs):
    allowed = set(allowed_pairs)
    claims = section.get("claims") or []

    by_text = {
        normalize_claim_text(c.get("claim")): extract_claim_pairs(c)
        for c in claims
        if isinstance(c, dict)
    }

    identity_by_text = {
        normalize_claim_text(c.get("claim")): {
            "identity_action": c.get("identity_action"),
            "parent_claim_uids": list(c.get("parent_claim_uids") or ()),
        }
        for c in claims
        if isinstance(c, dict)
    }

    kept = []
    rebuilt = []

    for sent in split_sentences_preserving_citations(
        section.get("draft_text", "")
    ):
        raw_existing = [
            (a.strip(), b.strip())
            for a, b in CITATION_RE.findall(sent)
        ]

        existing = _resolve_pairs(raw_existing, allowed)

        key = normalize_claim_text(sent)

        raw_declared = by_text.get(key, [])
        declared = _resolve_pairs(raw_declared, allowed)

        pairs = existing or declared

        # FAIL-CLOSED, sin heurísticas: si la oración ya tenía citas
        # inline válidas (existing) o coincide TEXTUALMENTE, de forma
        # exacta y determinista (declared -- comparación de igualdad de
        # strings normalizados, nunca fuzzy/semántica), con un claim que
        # ya trae sus propias citas, se hereda esa correspondencia --
        # nunca altera el contenido de la afirmación, solo recupera
        # citas que el propio LLM ya asoció a ese texto exacto.
        #
        # Si NO existe esa correspondencia segura (pairs vacío), la
        # oración NUNCA se borra ni recibe una cita inventada: se
        # preserva tal cual en draft_text, sin claim_entry asociado, para
        # que validate_generated_section() reporte la causa real
        # (uncited_substantive_sentence / missing_claim_for_sentence /
        # claim_citation_mismatch, según corresponda) y Agent06 pueda
        # pedir revisión -- nunca se convierte en EMPTY_DRAFT_TEXT por
        # una normalización que no pudo asociar la oración a un claim.
        preserved_without_claim = is_substantive_sentence(sent) and not pairs

        base = safe_str(
            CITATION_RE.sub("", sent)
        ).rstrip(".?!").strip()

        punct = (
            "."
            if sent.rstrip().endswith(".")
            else (
                "?"
                if sent.rstrip().endswith("?")
                else (
                    "!"
                    if sent.rstrip().endswith("!")
                    else ""
                )
            )
        )

        normalized = (
            base
            + (
                " "
                + " ".join(
                    citation_string(p)
                    for p in pairs
                )
                if pairs
                else ""
            )
            + punct
        )

        kept.append(normalized)

        if preserved_without_claim:
            # Sin correspondencia segura -- se preserva el texto, pero
            # nunca se inventa un claim_entry ni una cita para él. La
            # oración sigue en draft_text (arriba) para que la
            # validación posterior detecte y reporte el problema real.
            continue

        if is_substantive_sentence(normalized):
            claim_entry = {
                "claim": base,
                "supporting_citations": [
                    citation_string(p)
                    for p in pairs
                ],
            }

            # Preservar identity_action/parent_claim_uids:
            # nunca se inventan si el LLM no los declaró.
            identity = (
                identity_by_text.get(normalize_claim_text(base))
                or identity_by_text.get(key)
            )

            if identity is not None:
                claim_entry["identity_action"] = (
                    identity["identity_action"]
                )
                claim_entry["parent_claim_uids"] = (
                    identity["parent_claim_uids"]
                )

            rebuilt.append(claim_entry)

    out = dict(section)
    out["draft_text"] = " ".join(kept)
    out["claims"] = rebuilt
    return out
