"""
Load SCOTUS opinions from the CAP bulk dataset on Hugging Face.
I run this first to confirm the real field names before building on top of them.

"""
from __future__ import annotations
from typing import Iterator
import json

from .. import config


def load_dev_slice(n: int = config.DEV_SLICE) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    records: list[dict] = []
    for row in ds:
        if _is_scotus(row):
            records.append(row)
        if len(records) >= n:
            break
    return records


def _is_scotus(row: dict) -> bool:
    court = row.get("court") or {}
    if isinstance(court, dict):
        name = (court.get("name") or court.get("name_abbreviation") or "").lower()
        return "supreme court of the united states" in name or name == "scotus"
    return False


def stream_corpus() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    for row in ds:
        if _is_scotus(row):
            yield row


if __name__ == "__main__":
    recs = load_dev_slice()
    print(f"Got {len(recs)} records.")
    if recs:
        print(sorted(recs[0].keys()))
        print(json.dumps(recs[0], indent=2, default=str)[:2000])