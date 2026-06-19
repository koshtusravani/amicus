# amicus

Retrieval-augmented Q&A over US Supreme Court opinions. It retrieves real cases,
cites every claim, refuses when the sources don't support an answer, and measures
its own retrieval quality against a no-retrieval baseline.

> v1 scope: US Supreme Court opinions only. Federal circuit courts are a stretch feature.

## Why retrieval and not a plain LLM

A general LLM answers legal questions from memory and will confidently invent
cases that don't exist, the failure mode that has gotten lawyers sanctioned. I
retrieve real opinions and constrain the model to answer only from them, with
citations I can verify. The evaluation quantifies that difference directly.

## Architecture

```
Ingestion (offline):
  load (HF CAP bulk dataset) -> chunk (metadata-tagged) -> embed -> index

Query (online):
  query -> embed -> ANN search -> cross-encoder rerank
  -> cite-or-refuse prompt -> LLM -> answer + sources

Evaluation (offline):
  labeled set -> run RAG and no-retrieval baseline
  -> IR metrics + RAGAS + fabricated-citation rate -> results table
```

## Stack

| Layer        | Choice                                       |
|--------------|----------------------------------------------|
| Corpus       | CAP bulk dataset on Hugging Face (CC0)        |
| Embeddings   | sentence-transformers (`BAAI/bge-large-en-v1.5`) |
| Reranking    | cross-encoder (`BAAI/bge-reranker-v2-m3`)     |
| Vector DB    | Chroma (local dev), Qdrant (deployment)       |
| Generation   | closed LLM API, with an open-source comparison planned |
| Evaluation   | RAGAS plus custom IR metrics                   |
| Demo         | Gradio on Hugging Face Spaces                  |

## Chunking

I split by opinion type first (so a dissent never blends into the holding), then
recursively within each (~700 tokens, ~12% overlap). Every chunk carries
`case_name`, `citation`, `year`, `court`, `opinion_type`, and `paragraph_index`,
which powers filtered retrieval and verifiable citations.

## Evaluation results

Measured against a no-retrieval baseline (same LLM, same questions, no context).

| Metric                       | Target  | Baseline | RAG | RAG + rerank |
|------------------------------|:-------:|:--------:|:---:|:------------:|
| Precision@5                  | >= 0.70 | —        | —   | —            |
| Recall@20                    | >= 0.80 | —        | —   | —            |
| MRR                          | —       | —        | —   | —            |
| nDCG@10                      | —       | —        | —   | —            |
| Faithfulness (RAGAS)         | >= 0.75 | —        | —   | —            |
| Answer relevancy (RAGAS)     | >= 0.80 | —        | —   | —            |
| Context precision (RAGAS)    | >= 0.70 | n/a      | —   | —            |
| Context recall (RAGAS)       | >= 0.80 | n/a      | —   | —            |
| Fabricated-citation rate     | -> 0    | —        | —   | —            |
| Correct-refusal rate         | high    | —        | —   | —            |

I fill these in once the labeled set is built and the pipeline runs end to end.
The fabricated-citation row is the headline: the baseline invents cases, the RAG
system shouldn't.

## Cost and latency per query

| Stage    | Latency (ms) | Tokens | Cost |
|----------|:------------:|:------:|:----:|
| Retrieve | —            | —      | —    |
| Rerank   | —            | —      | —    |
| Generate | —            | in/out | $    |
| Total    | —            | —      | $    |

`QueryTrace` in `src/generation/generate.py` captures per-stage timings and
token counts on every query; I populate this from a representative sample.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

## Repo layout

```
src/ingestion/   load, chunk, embed, index
src/retrieval/   ANN search + cross-encoder rerank
src/generation/  cite-or-refuse prompt + instrumented query path
src/eval/        labeled set, IR metrics, baseline, RAGAS
app/             Gradio demo
tests/           smoke tests
```

## Roadmap

- Hybrid search (dense + BM25) with score fusion
- Open-source vs closed LLM comparison on the eval set
- Citation-graph features via CourtListener
- BGE-M3 long-context vs short-context chunking
- DeepEval in CI
- Expand corpus to federal circuit courts