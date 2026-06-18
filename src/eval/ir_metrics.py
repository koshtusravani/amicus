"""
Retrieval metrics. Pure Python so I can unit-test them without any heavy deps.
Each takes an ordered list of retrieved ids (best first) and the set of relevant ids.
"""
from __future__ import annotations
from math import log2


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return sum(1 for r in topk if r in relevant) / len(topk)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(rows: list[dict], ks=(1, 5, 10, 20)) -> dict:
    n = len(rows) or 1
    out: dict[str, float] = {}
    for k in ks:
        out[f"precision_at_{k}"] = sum(precision_at_k(r["retrieved"], set(r["relevant"]), k) for r in rows) / n
        out[f"recall_at_{k}"] = sum(recall_at_k(r["retrieved"], set(r["relevant"]), k) for r in rows) / n
        out[f"ndcg_at_{k}"] = sum(ndcg_at_k(r["retrieved"], set(r["relevant"]), k) for r in rows) / n
    out["mrr"] = sum(reciprocal_rank(r["retrieved"], set(r["relevant"])) for r in rows) / n
    return out