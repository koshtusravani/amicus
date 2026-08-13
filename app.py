"""
Gradio demo for amicus: a RAG Q&A system over US Supreme Court opinions.
"""
import sys
sys.path.insert(0, ".")

import gradio as gr

from src.generation.generate import answer


def ask(question: str):
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    trace = answer(question)

    # Sources table: case name, citation, relevance score
    sources_md = ""
    if trace.sources:
        sources_md = "| Case | Citation | Relevance |\n|---|---|---|\n"
        for c in trace.sources:
            meta = getattr(c, "metadata", {})
            case_name = meta.get("case_name", "?")
            citation = meta.get("citation", "?")
            score = getattr(c, "score", None)
            score_str = f"{score:.3f}" if score is not None else "?"
            sources_md += f"| {case_name} | {citation} | {score_str} |\n"
    else:
        sources_md = "_No sources retrieved._"

    stats = (
        f"**Latency:** retrieve {trace.timings_ms.get('retrieve', '?')}ms · "
        f"rerank {trace.timings_ms.get('rerank', '?')}ms · "
        f"generate {trace.timings_ms.get('generate', '?')}ms · "
        f"**total {trace.total_ms:.0f}ms**\n\n"
        f"**Tokens:** {trace.tokens.get('input', '?')} in / "
        f"{trace.tokens.get('output', '?')} out"
    )

    return trace.answer, sources_md, stats


EXAMPLES = [
    "What warnings must police give before a custodial interrogation?",
    "When can a police officer stop and frisk someone without a warrant?",
    "What evidence must the prosecution disclose to the defense?",
    "What test decides whether a defendant had ineffective assistance of counsel?",
    "What is the airspeed velocity of an unladen swallow?",  
]

with gr.Blocks(title="amicus — SCOTUS RAG Q&A") as demo:
    gr.Markdown(
        "# amicus\n"
        "Ask a question about US Supreme Court case law. Answers are grounded "
        "in retrieved opinion text — the system cites its sources and refuses "
        "to answer when nothing relevant is found in the corpus."
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. What warnings must police give before a custodial interrogation?",
            lines=2,
        )

    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Markdown(label="Answer")

    with gr.Accordion("Sources", open=True):
        sources_box = gr.Markdown()

    with gr.Accordion("Latency & cost", open=False):
        stats_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=question)

    ask_btn.click(fn=ask, inputs=question, outputs=[answer_box, sources_box, stats_box])
    question.submit(fn=ask, inputs=question, outputs=[answer_box, sources_box, stats_box])


if __name__ == "__main__":
    demo.launch()