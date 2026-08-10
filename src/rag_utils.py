# ============================================================
# UTILIDADES RAG / CHROMA
# ============================================================

from collections import defaultdict
import chromadb
from chromadb.utils import embedding_functions

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
)

if not str(EMBEDDING_MODEL_NAME).strip():
    raise ValueError(
        "EMBEDDING_MODEL_NAME no puede estar vacío."
    )

if not str(CHROMA_COLLECTION_NAME).strip():
    raise ValueError(
        "CHROMA_COLLECTION_NAME no puede estar vacío."
    )


def load_chroma_collection(
    chroma_path,
    collection_name=None,
    model_name=None
):
    """
    Carga o crea una colección Chroma usando los valores configurados.
    Por defecto usa reference_papers_chunks para mantener compatibilidad
    con los notebooks actuales.
    """
    collection_name = collection_name or CHROMA_COLLECTION_NAME
    model_name = model_name or EMBEDDING_MODEL_NAME

    client = chromadb.PersistentClient(path=str(chroma_path))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed_fn
    )

    return collection


def retrieve_raw(collection, query, top_k=8, fetch_k=30, max_per_source=2):
    res = collection.query(
        query_texts=[query],
        n_results=fetch_k
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    candidates = []

    for doc, meta, dist in zip(docs, metas, dists):
        candidates.append({
            "score": 1 - float(dist),
            "text": doc,
            "metadata": meta
        })

    grouped = defaultdict(list)

    for item in candidates:
        source = item["metadata"].get("source_filename", "unknown")
        grouped[source].append(item)

    balanced = []

    for source, items in grouped.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        balanced.extend(items[:max_per_source])

    balanced = sorted(balanced, key=lambda x: x["score"], reverse=True)

    return balanced[:top_k]


def format_evidence_for_prompt(evidence_items, max_chars_per_chunk=1200):
    parts = []

    for i, item in enumerate(evidence_items, start=1):
        meta = item.get("metadata", {})
        source = meta.get("source_filename", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")
        score = item.get("score", None)

        text = str(item.get("text", ""))[:max_chars_per_chunk]

        parts.append(
            f"[EVIDENCE {i}]\\n"
            f"source_filename: {source}\\n"
            f"chunk_id: {chunk_id}\\n"
            f"score: {score}\\n"
            f"text:\\n{text}\\n"
        )

    return "\\n\\n".join(parts)
