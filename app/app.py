"""
Gradio demo, deployed free on Hugging Face Spaces for the live link. Free
Spaces sleep when idle, so the first hit is a cold start. This calls a paid LLM
API and a public demo means strangers spend my money, so I cap usage with a
crude rate limit; for anything real I'd put a small proxy in front that holds
the key.
"""
from __future__ import annotations
import time
from collections import deque

import gradio as gr

from src.generation.generate import answer

_WINDOW_S = 60
_MAX_PER_WINDOW = 10
_hits: deque[float] = deque()


def _rate_limited() -> bool:
    now = time.time()
    while _hits and now - _hits[0] > _WINDOW_S:
        _hits.popleft()
    if len(_hits) >= _MAX_PER_WINDOW:
        return True
    _hits.append(now)
    return False


def respond(question: str):
    if not question.strip():
        return "Ask a question about a US Supreme Court opinion.", ""
    if _rate_limited():
        return "Demo is busy (rate limited). Try again in a minute.", ""
    trace = answer(question)
    sources = "\n\n".join(
        f"- {c.metadata.get('case_name','?')}, {c.metadata.get('citation','?')} "
        f"({c.metadata.get('opinion_type','?')})"
        for c in trace.sources
    )
    footer = (
        f"\n\n---\nlatency: {trace.total_ms:.0f} ms | "
        f"tokens in/out: {trace.tokens.get('input','?')}/{trace.tokens.get('output','?')}"
    )
    return trace.answer + footer, sources


with gr.Blocks(title="amicus — SCOTUS Grounded Q&A") as demo:
    gr.Markdown(
        "# amicus\n"
        "Grounded Q&A over US Supreme Court opinions. Answers are retrieved, "
        "cited, and refused when the sources don't support them."
    )
    q = gr.Textbox(label="Question", placeholder="What did the Court hold in ...?")
    btn = gr.Button("Ask")
    ans = gr.Markdown(label="Answer")
    src = gr.Markdown(label="Retrieved sources")
    btn.click(respond, inputs=q, outputs=[ans, src])

if __name__ == "__main__":
    demo.queue().launch()