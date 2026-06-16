"""
Embedding wrapper. Only queries get the BGE instruction prefix, documents
don't, so I keep that asymmetry in one place.
"""
from __future__ import annotations
from functools import lru_cache

from .. import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()


def embed_query(query: str) -> list[float]:
    prefixed = f"{config.QUERY_INSTRUCTION} {query}"
    return _model().encode([prefixed], normalize_embeddings=True)[0].tolist()