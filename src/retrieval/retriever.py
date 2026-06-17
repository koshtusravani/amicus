"""
Dense retrieval: embed the query, ANN search, return candidates with metadata.
I fetch TOP_K here to feed the reranker, which trims to RERANK_K.
"""
from __future__ import annotations
from dataclasses import dataclass

from .. import config
from ..ingestion.embed import embed_query
from ..ingestion.index import get_collection


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict
    score: float


def search(query: str, k: int = config.TOP_K, where: dict | None = None) -> list[Candidate]:
    coll = get_collection()
    qvec = embed_query(query)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out: list[Candidate] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(Candidate(chunk_id=cid, text=doc, metadata=meta, score=1.0 - dist))
    return out