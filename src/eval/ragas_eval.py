"""
Generation-quality eval: RAGAS metrics plus my fabricated-citation rate. The
latter is the headline legal metric: of the cases an answer cites, how many
don't exist in the corpus? The baseline should score badly here; RAG should
approach zero.
"""
from __future__ import annotations
import re

CITATION_RE = re.compile(r"\[([^\]]+?),\s*([^\]]+?)\]")  # [Case Name, Citation]


def extract_citations(answer: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2).strip()) for m in CITATION_RE.finditer(answer)]


def fabricated_citation_rate(answer: str, known_citations: set[str]) -> float:
    cites = extract_citations(answer)
    if not cites:
        return 0.0
    fake = sum(1 for _, cit in cites if _normalize(cit) not in known_citations)
    return fake / len(cites)


def _normalize(cit: str) -> str:
    return re.sub(r"\s+", " ", cit).strip().lower()


def run_ragas(samples: list[dict]) -> dict:
    # samples: {"question", "answer", "contexts": list[str], "ground_truth"}
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)