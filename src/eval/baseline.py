"""
No-retrieval baseline: same LLM, same questions, no context. This is the
column everything is measured against. It answers from memory and will invent
cases that don't exist, and capturing that contrast is the point.
"""
from __future__ import annotations

from ..generation.generate import _call_llm

BASELINE_SYSTEM = (
    "You are a legal research assistant. Answer the question about US Supreme "
    "Court opinions. Cite the cases you rely on in the form [Case Name, Citation]."
)


def baseline_answer(question: str) -> str:
    text, _ = _call_llm(BASELINE_SYSTEM, f"QUESTION: {question}")
    return text