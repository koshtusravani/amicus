"""
Upsert embedded chunks into the vector DB. Chroma for local dev; I keep the
interface small so swapping to Qdrant for deployment is a one-file change.
"""
from __future__ import annotations

from .. import config
from .chunk import Chunk
from .embed import embed_documents


def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], batch_size: int = 128) -> int:
    coll = get_collection()
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_documents([c.text for c in batch])
        coll.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[c.to_metadata() for c in batch],
        )
        total += len(batch)
    return total