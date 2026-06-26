"""
The full query path, instrumented for cost and latency. Every answer returns
a QueryTrace with per-stage timings and token counts, since I need cost and
latency per query and it's trivial to capture here but annoying to retrofit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

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
    if config.LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types, errors

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        last_err = None
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=config.LLM_MODEL, contents=user, config=cfg
                )
                usage = {
                    "input": resp.usage_metadata.prompt_token_count,
                    "output": resp.usage_metadata.candidates_token_count,
                }
                return resp.text, usage
            except errors.ServerError as e:        # 503 / transient: back off and retry
                last_err = e
                time.sleep(2 ** attempt)
            except errors.ClientError as e:        # 429 rate limit: back off and retry
                if e.code == 429:
                    last_err = e
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"Gemini unavailable after retries: {last_err}")

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
        top = rerank(question, candidates[:config.RERANK_INPUT])
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