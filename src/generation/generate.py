"""
The full query path, instrumented for cost and latency. Every answer returns
a QueryTrace with per-stage timings and token counts, since I need cost and
latency per query and it's trivial to capture here but annoying to retrofit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time

from .. import config
from ..retrieval.retriever import search, Candidate
from ..retrieval.rerank import rerank
from .prompt import SYSTEM_PROMPT, build_user_message


@dataclass
class QueryTrace:
    question: str
    answer: str
    sources: list[Candidate]
    timings_ms: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


class _Timer:
    def __init__(self, store: dict, key: str):
        self.store, self.key = store, key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] = round((time.perf_counter() - self.t0) * 1000, 1)


def _call_llm(system: str, user: str) -> tuple[str, dict]:
    # Open-source vs closed comparison (a stretch) plugs in here behind LLM_PROVIDER.
    if config.LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        return text, usage
    raise NotImplementedError(f"Provider {config.LLM_PROVIDER} not wired yet.")


def answer(question: str, where: dict | None = None) -> QueryTrace:
    trace = QueryTrace(question=question, answer="", sources=[])

    with _Timer(trace.timings_ms, "retrieve"):
        candidates = search(question, where=where)
    with _Timer(trace.timings_ms, "rerank"):
        top = rerank(question, candidates)
    trace.sources = top

    user_msg = build_user_message(question, top)
    with _Timer(trace.timings_ms, "generate"):
        text, usage = _call_llm(SYSTEM_PROMPT, user_msg)

    trace.answer = text
    trace.tokens = usage
    return trace


if __name__ == "__main__":
    t = answer("What did the Court hold about the right to remain silent?")
    print(t.answer)
    print("latency (ms):", t.timings_ms, "total:", t.total_ms)
    print("tokens:", t.tokens)