"""
Smoke tests for the pure-Python pieces. These run without any heavy deps, so I
have a green suite from day one. Run: python -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.ir_metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, aggregate
from src.eval.ragas_eval import extract_citations, fabricated_citation_rate


def test_precision_recall():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.5


def test_mrr_and_ndcg():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0


def test_aggregate_runs():
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a"]},
        {"retrieved": ["c", "d"], "relevant": ["d"]},
    ]
    out = aggregate(rows, ks=(1, 2))
    assert "precision_at_1" in out and "mrr" in out


def test_citation_extraction_and_fabrication():
    ans = "The Court held X [Miranda v. Arizona, 384 U.S. 436]."
    assert extract_citations(ans) == [("Miranda v. Arizona", "384 U.S. 436")]
    assert fabricated_citation_rate(ans, known_citations=set()) == 1.0
    assert fabricated_citation_rate(ans, {"384 u.s. 436"}) == 0.0
    assert fabricated_citation_rate("No answer.", set()) == 0.0