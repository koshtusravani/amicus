"""
Cross-encoder reranking. The bi-encoder is fast but coarse; the cross-encoder
scores each (query, chunk) pair jointly and is far more precise. Toggling this is
what gives me the RAG vs RAG+rerank column in the eval table.
"""
from __future__ import annotations
from functools import lru_cache

from .. import config
from .retriever import Candidate


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, candidates: list[Candidate], k: int = config.RERANK_K) -> list[Candidate]:
    if not candidates:
        return []
    scores = _reranker().predict([(query, c.text) for c in candidates])
    for c, s in zip(candidates, scores):
        c.score = float(s)
    return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]