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


def _ntokens(s: str) -> int:
    return max(1, len(s.split()))


def strip_boilerplate(text: str) -> str:
    text = re.sub(r"\*\d+", " ", text)
    text = re.sub(r"\bNo\.\s+\d+\.\s*", " ", text)
    text = re.sub(r"\b(Argued|Decided|Reargued)\b[^.]*\.", " ", text)
    return text


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _units(text: str) -> list[str]:
    """Split into paragraphs, but fall back to sentences for any oversized block."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:                      # no real paragraph breaks
        paragraphs = _sentences(text)
    units: list[str] = []
    for p in paragraphs:
        if _ntokens(p) > config.CHUNK_TOKENS:     
            units.extend(_sentences(p))
        else:
            units.append(p)
    return units


def recursive_split(text: str, target_tokens: int, overlap: float) -> list[str]:
    units = _units(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for unit in units:
        ulen = _ntokens(unit)
        if ulen > target_tokens:                  
            words = unit.split()
            for i in range(0, len(words), target_tokens):
                chunks.append(" ".join(words[i : i + target_tokens]))
            continue
        if buf_len + ulen > target_tokens and buf:
            chunks.append("\n\n".join(buf))
            keep = max(1, int(len(buf) * overlap))
            buf = buf[-keep:]
            buf_len = sum(_ntokens(b) for b in buf)
        buf.append(unit)
        buf_len += ulen

    if buf:
        chunks.append("\n\n".join(buf))

    # Drop any exact duplicates that overlap can still produce on short texts.
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        key = c.strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def chunk_record(record: dict, target_tokens: int = config.CHUNK_TOKENS,
                 overlap: float = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for op in record["opinions"]:
        cleaned = strip_boilerplate(op["text"])
        for i, piece in enumerate(recursive_split(cleaned, target_tokens, overlap)):
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