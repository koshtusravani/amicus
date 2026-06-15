# amicus

I'm building a retrieval-augmented Q&A system over US Supreme Court opinions. It
retrieves real cases, cites every claim, refuses when the sources don't support
an answer, and measures its own retrieval quality against a no-retrieval baseline.

This is a work in progress. I'm building it layer by layer: ingestion, retrieval,
generation, and evaluation, with a live demo at the end.

## Why retrieval and not a plain LLM

A general LLM answers legal questions from memory and will confidently invent
cases that don't exist. I retrieve real opinions and constrain the model to
answer only from them, with citations I can verify. My evaluation quantifies that
difference directly.

## Scope

v1 covers US Supreme Court opinions only. I'm leaving federal circuit courts as a
stretch feature.

## Stack

- Embeddings: Hugging Face sentence-transformers (`BAAI/bge-large-en-v1.5`)
- Reranking: cross-encoder (`BAAI/bge-reranker-v2-m3`)
- Vector DB: Chroma for local dev, Qdrant for deployment
- Generation: closed LLM API for now, with an open-source comparison planned
- Evaluation: RAGAS plus my own IR metrics
- Demo: Gradio on Hugging Face Spaces

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests\test_smoke.py
```

## Status

Setting up the project structure and scaffolding. More to come.