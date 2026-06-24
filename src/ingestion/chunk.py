"""
Turn cached SCOTUS records into metadata-tagged chunks.

The API already separates majority / concurrence / dissent, so I chunk each
opinion on its own and tag every chunk with enough metadata to reconstruct a
real citation.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict, field

from .. import config


@dataclass
class Chunk:
    text: str
    case_name: str
    citation: str
    year: int | None
    court: str
    opinion_type: str
    paragraph_index: int
    chunk_id: str
    extra: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("extra")
        return {k: v for k, v in d.items() if v is not None}


def recursive_split(text: str, target_tokens: int, overlap: float) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def _ntokens(s: str) -> int:
        return max(1, len(s.split()))

    for para in paragraphs:
        plen = _ntokens(para)
        if buf_len + plen > target_tokens and buf:
            chunks.append("\n\n".join(buf))
            keep = max(1, int(len(buf) * overlap))
            buf = buf[-keep:]
            buf_len = sum(_ntokens(b) for b in buf)
        buf.append(para)
        buf_len += plen

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def chunk_record(record: dict, target_tokens: int = config.CHUNK_TOKENS,
                 overlap: float = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for op in record["opinions"]:
        for i, piece in enumerate(recursive_split(op["text"], target_tokens, overlap)):
            chunks.append(Chunk(
                text=piece,
                case_name=record["case_name"],
                citation=record["citation"],
                year=record["year"],
                court=record["court"],
                opinion_type=op["opinion_type"],
                paragraph_index=i,
                chunk_id=f"{record['cluster_id']}-{op['opinion_id']}-{i}",
            ))
    return chunks


def load_all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(config.RAW_DIR.glob("*.json")):
        chunks.extend(chunk_record(json.loads(path.read_text(encoding="utf-8"))))
    return chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"Built {len(chunks)} chunks from {config.RAW_DIR}")
    if chunks:
        c = chunks[0]
        print(f"{c.case_name} | {c.citation} | {c.opinion_type} | {c.chunk_id}")
        print(c.text[:300])