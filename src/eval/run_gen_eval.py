"""
Generation evaluation: RAG vs the no-retrieval baseline, same Gemini model.

Headline metrics (pure Python, no LLM judge):
  - fabricated-citation rate: of the cases an answer cites, how many are NOT in
    the corpus. Baseline answers from memory and invents cases; RAG should ~0.
  - refusal correctness: on out-of-corpus questions, does the system refuse?
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time

from .. import config
from ..generation.generate import answer
from ..generation.prompt import REFUSAL
from .baseline import baseline_answer
from .labeled_set import LABELED_SET
from .ragas_eval import extract_citations, corpus_citations, _normalize

CACHE = config.DATA_DIR / "gen_eval"
PACE = 5.0   


def _cached(key: str):
    f = CACHE / f"{key}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _save(key: str, data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{key}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_answers():
    selected = [q for q in LABELED_SET if not q.should_refuse][:8] + \
               [q for q in LABELED_SET if q.should_refuse]

    rows = []
    for i, q in enumerate(selected):
        key = f"q{i:02d}"
        row = _cached(key)
        if row is None:
            print(f"  [{i+1}/{len(selected)}] {q.question[:55]}...")
            rag = answer(q.question)
            time.sleep(PACE)
            base = baseline_answer(q.question)
            time.sleep(PACE)
            row = {
                "question": q.question,
                "should_refuse": q.should_refuse,
                "rag": rag.answer,
                "baseline": base,
            }
            _save(key, row)
        rows.append(row)
    return rows

def _fab_rate(text: str, known: set[str]) -> float | None:
    cites = extract_citations(text)
    if not cites:
        return None  
    fake = sum(1 for _, c in cites if _normalize(c) not in known)
    return fake / len(cites)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    known = corpus_citations()
    rows = _get_answers()

    answerable = [r for r in rows if not r["should_refuse"]]
    refusable = [r for r in rows if r["should_refuse"]]

    rag_fab = _mean([_fab_rate(r["rag"], known) for r in answerable])
    base_fab = _mean([_fab_rate(r["baseline"], known) for r in answerable])

    rag_refused = sum(1 for r in refusable if r["rag"].strip().startswith(REFUSAL[:30]))
    base_refused = sum(1 for r in refusable if r["baseline"].strip().startswith(REFUSAL[:30]))

    print("\nGeneration evaluation: RAG vs no-retrieval baseline")
    print(f"answerable questions: {len(answerable)} | out-of-corpus: {len(refusable)}\n")
    print(f"{'metric':28s}{'baseline':>12s}{'RAG':>12s}")
    print(f"{'fabricated-citation rate':28s}{base_fab:>12.3f}{rag_fab:>12.3f}")
    print(f"{'correct refusals':28s}{base_refused:>9d}/{len(refusable):<2d}{rag_refused:>9d}/{len(refusable):<2d}")


if __name__ == "__main__":
    main()