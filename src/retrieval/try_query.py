"""
Quick manual retrieval check against the indexed corpus. No LLM, no API key.

    python -m src.retrieval.try_query "your question here"
"""
from __future__ import annotations
import sys

from .retriever import search
from .rerank import rerank


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What must the prosecution disclose to the defense?"
    print(f"Query: {query}\n")

    hits = search(query)
    print("Top 5 by vector search:")
    for h in hits[:5]:
        m = h.metadata
        print(f"  {h.score:.3f}  {m['case_name']} ({m['opinion_type']}) [{h.chunk_id}]")

    top = rerank(query, hits)
    print("\nTop 5 after rerank:")
    for h in top[:5]:
        m = h.metadata
        print(f"  {h.score:.3f}  {m['case_name']} ({m['opinion_type']}) [{h.chunk_id}]")

    print("\nBest chunk preview:")
    print(top[0].text[:400])


if __name__ == "__main__":
    main()