from __future__ import annotations
import re
from .retrieval import safe_str
CITATION_RE=re.compile(r"\[\s*([^\]|]+?)\s*\|\s*([^\]]+?)\s*\]")

def citation_string(pair):return f"[{pair[0]} | {pair[1]}]"
def canonicalize_citation_position(text):
    text=safe_str(text)
    return re.sub(r"([.!?])\s*(\[[^\]]+\|[^\]]+\])",r" \2\1",text)
def split_sentences_preserving_citations(text):
    text=canonicalize_citation_position(text)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+",text) if x.strip()]
def is_substantive_sentence(sentence):
    clean=CITATION_RE.sub("",safe_str(sentence));return len(re.findall(r"\w+",clean))>=8
def normalize_claim_text(text):
    return safe_str(CITATION_RE.sub("",safe_str(text))).rstrip(".?!").strip()
def extract_claim_pairs(claim):
    pairs=[]
    for c in claim.get("supporting_citations") or []:
        m=CITATION_RE.fullmatch(safe_str(c))
        if m:pairs.append((m.group(1).strip(),m.group(2).strip()))
    return pairs

def normalize_generated_section(section, allowed_pairs):
    allowed=set(allowed_pairs); claims=section.get("claims") or []; by_text={normalize_claim_text(c.get("claim")):extract_claim_pairs(c) for c in claims if isinstance(c,dict)}
    identity_by_text={
        normalize_claim_text(c.get("claim")): {
            "identity_action": c.get("identity_action"),
            "parent_claim_uids": list(c.get("parent_claim_uids") or ()),
        }
        for c in claims if isinstance(c, dict)
    }
    kept=[]; rebuilt=[]
    for sent in split_sentences_preserving_citations(section.get("draft_text","")):
        existing=[(a.strip(),b.strip()) for a,b in CITATION_RE.findall(sent) if (a.strip(),b.strip()) in allowed]
        key=normalize_claim_text(sent); declared=[p for p in by_text.get(key,[]) if p in allowed]
        pairs=existing or declared
        if is_substantive_sentence(sent) and not pairs:continue
        base=safe_str(CITATION_RE.sub("",sent)).rstrip(".?!").strip()
        punct="." if sent.rstrip().endswith(".") else ("?" if sent.rstrip().endswith("?") else ("!" if sent.rstrip().endswith("!") else ""))
        normalized=base+(" "+" ".join(citation_string(p) for p in pairs) if pairs else "")+punct
        kept.append(normalized)
        if is_substantive_sentence(normalized):
            claim_entry = {"claim": base, "supporting_citations": [citation_string(p) for p in pairs]}
            # Preservar identity_action/parent_claim_uids (ver
            # claim_identity.py) -- se buscan por el MISMO texto
            # normalizado que ya usa este emparejamiento para citas;
            # nunca se inventan si el LLM no los declaró (quedan
            # ausentes, y quien resuelve identidad decide qué hacer).
            identity = identity_by_text.get(normalize_claim_text(base)) or identity_by_text.get(key)
            if identity is not None:
                claim_entry["identity_action"] = identity["identity_action"]
                claim_entry["parent_claim_uids"] = identity["parent_claim_uids"]
            rebuilt.append(claim_entry)
    out=dict(section);out["draft_text"]=" ".join(kept);out["claims"]=rebuilt;return out
