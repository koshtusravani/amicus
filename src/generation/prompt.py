"""
The cite-or-refuse prompt. This is the heart of the project: the model answers
only from retrieved sources, cites every claim, and refuses when the sources
don't support an answer. The refusal string is fixed so my eval harness can
detect refusals deterministically, so I don't paraphrase it at call sites.
"""
from __future__ import annotations

from ..retrieval.retriever import Candidate

REFUSAL = "The provided sources do not support an answer to this question."

SYSTEM_PROMPT = f"""You are a legal research assistant answering questions about \
United States Supreme Court opinions. You must follow these rules without exception:

1. Answer ONLY using the SOURCES provided in the user message. Do not use any \
outside knowledge, training data, or recollection of cases.
2. Every factual claim in your answer must be followed by a citation to the \
source it comes from, in the form [Case Name, Citation].
3. If the SOURCES do not contain enough information to answer, respond with \
exactly this sentence and nothing else: "{REFUSAL}"
4. Never invent, guess, approximate, or recall a case name, citation, holding, \
or quotation that does not appear verbatim in the SOURCES.
5. When sources disagree, distinguish the majority holding from any concurrence \
or dissent, and label which is which.
6. Be concise and precise. Do not editorialize or give legal advice.
"""


def format_sources(candidates: list[Candidate]) -> str:
    blocks = []
    for i, c in enumerate(candidates, 1):
        m = c.metadata
        header = (
            f"[Source {i}] {m.get('case_name', '?')}, "
            f"{m.get('citation', '?')} ({m.get('year', '?')}) "
            f"-- {m.get('opinion_type', '?')}"
        )
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_message(question: str, candidates: list[Candidate]) -> str:
    sources = format_sources(candidates) if candidates else "(no sources retrieved)"
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}"