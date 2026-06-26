"""
Retrieval evaluation: vector search only vs vector + rerank, over the labeled
set. Relevance is at the case level, so I map each retrieved chunk to its cluster
id and score the ranked list. No LLM, no API key.
"""
from __future__ import annotations

from .. import config
from ..retrieval.retriever import search
from ..retrieval.rerank import rerank
from .labeled_set import LABELED_SET, case_to_cluster, resolve_relevant
from .ir_metrics import aggregate


def _clusters(chunks) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        cid = c.chunk_id.split("-")[0]
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _rows(use_rerank: bool, mapping: dict[str, str]):
    rows = []
    for q in LABELED_SET:
        if q.should_refuse:
            continue
        hits = search(q.question, k=config.TOP_K)
        if use_rerank:
            hits = rerank(q.question, hits, k=config.TOP_K)
        rows.append({"retrieved": _clusters(hits), "relevant": resolve_relevant(q, mapping)})
    return rows


def _print(name: str, m: dict) -> None:
    print(f"\n{name}")
    for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_20", "mrr", "ndcg_at_10"):
        print(f"  {k:16s}: {m[k]:.3f}")


def main() -> None:
    mapping = case_to_cluster()
    n = sum(1 for q in LABELED_SET if not q.should_refuse)
    print(f"Corpus: {len(mapping)} cases | retrieval queries: {n}")
    _print("Vector search only", aggregate(_rows(False, mapping), ks=(1, 3, 5, 10, 20)))
    _print("Vector search + rerank", aggregate(_rows(True, mapping), ks=(1, 3, 5, 10, 20)))


if __name__ == "__main__":
    main()