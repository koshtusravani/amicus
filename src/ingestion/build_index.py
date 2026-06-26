"""
Embed all cached chunks and load them into Chroma. Runs off the local cache,
so no API calls. First run downloads the BGE model (a few hundred MB).
"""
from __future__ import annotations
import time

from .chunk import load_all_chunks
from .index import index_chunks, get_collection


def main() -> None:
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding + indexing (first run downloads the model)...")
    t0 = time.perf_counter()
    n = index_chunks(chunks)
    dt = time.perf_counter() - t0
    count = get_collection().count()
    print(f"Indexed {n} chunks in {dt:.1f}s. Collection now holds {count}.")


if __name__ == "__main__":
    main()