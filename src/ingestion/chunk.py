"""
Turn raw opinions into metadata-tagged chunks.

I split by opinion type first (so a dissent never blends into the holding), then
recursively within each. Every chunk carries enough metadata to reconstruct a
real citation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import re


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


def split_opinions(raw: dict) -> list[dict]:
    raise NotImplementedError


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


def chunk_record(raw: dict, target_tokens: int, overlap: float) -> list[Chunk]:
    raise NotImplementedError