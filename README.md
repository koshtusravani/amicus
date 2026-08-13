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
  -> IR metrics + fabricated-citation rate -> results table
```

## Stack

| Layer        | Choice                                       |
|--------------|----------------------------------------------|
| Corpus       | CAP bulk dataset on Hugging Face (CC0)        |
| Embeddings   | sentence-transformers (`BAAI/bge-large-en-v1.5`) |
| Reranking    | cross-encoder (`BAAI/bge-reranker-v2-m3`)     |
| Vector DB    | Chroma (local dev), Qdrant (deployment)       |
| Generation   | Gemini 3.5 Flash-Lite, with an open-source comparison planned |
| Evaluation   | Custom IR metrics + fabricated-citation rate  |
| Demo         | Gradio, runs locally                          |

## Chunking

I split by opinion type first (so a dissent never blends into the holding), then
recursively within each (~700 tokens, ~12% overlap). Every chunk carries
`case_name`, `citation`, `year`, `court`, `opinion_type`, and `paragraph_index`,
which powers filtered retrieval and verifiable citations.

## Evaluation results

Measured on a 15-case labeled set, 15 retrieval queries. Generation eval on 10
questions (8 answerable, 2 out-of-corpus negative controls), same LLM for
baseline and RAG.

### Retrieval

| Metric      | Vector search only | Vector + rerank |
|-------------|:-------------------:|:----------------:|
| Recall@1    | 0.700               | 0.767            |
| Recall@3    | 0.833               | 0.967            |
| Recall@5    | 0.833               | 0.967            |
| Recall@20   | 0.967               | 0.967            |
| MRR         | 0.819               | 0.900            |
| nDCG@10     | 0.835               | 0.900            |

Reranking meaningfully improves ranking quality — Recall@3 goes from 0.833 to
0.967 and MRR from 0.819 to 0.900. Recall@20 tops out at 0.967 either way: one
relevant case in the 15-case set isn't reliably surfaced even at a 20-chunk
cutoff, a known retrieval gap rather than a ranking problem.

### Generation

| Metric                    | Baseline | RAG   |
|---------------------------|:--------:|:-----:|
| Fabricated-citation rate  | 0.930    | 0.000 |
| Correct-refusal rate      | 0/2      | 2/2   |

The baseline (same LLM, no retrieved context) invents a citation in nearly
every answer. RAG, constrained to answer only from retrieved opinion text,
fabricated zero citations across all 8 answerable questions and correctly
refused both out-of-corpus questions.

## Cost and latency per query

Measured locally on CPU (no GPU), `RERANK_INPUT=20`, `gemini-3.5-flash-lite`
($0.30/M input tokens, $2.50/M output tokens).

| Stage    | Latency        | Notes                                      |
|----------|:--------------:|---------------------------------------------|
| Retrieve | ~200 ms        | Dense ANN search, bi-encoder query embed     |
| Rerank   | 70–120 s       | Cross-encoder over 20 candidates, CPU-bound  |
| Generate | 1.2–3.3 s      | Gemini 3.5 Flash-Lite                        |
| **Total**| **~75–125 s**  | Dominated entirely by CPU reranking          |

| Metric        | Value (representative query) |
|---------------|:-----------------------------:|
| Input tokens  | ~5,500                        |
| Output tokens | ~175                          |
| Est. cost     | ~$0.002 per query              |

The reranking stage is the bottleneck by a wide margin — the cross-encoder
(`bge-reranker-v2-m3`) scoring 20 (query, chunk) pairs on CPU costs roughly
1,000x more time than retrieval or generation combined. On a GPU, this stage
would drop to low hundreds of milliseconds; the current numbers reflect local
development hardware, not a production deployment target. Generation cost is
negligible either way — under half a cent per query.

`QueryTrace` in `src/generation/generate.py` captures per-stage timings and
token counts on every query.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

## Running the demo

```bash
python app.py
```

Opens a local Gradio UI for asking questions against the corpus, with sources
and per-query latency shown alongside each answer.

## Repo layout

```
src/ingestion/   load, chunk, embed, index
src/retrieval/   ANN search + cross-encoder rerank
src/generation/  cite-or-refuse prompt + instrumented query path
src/eval/        labeled set, IR metrics, baseline, generation eval
app.py           Gradio demo
tests/           smoke tests
```

## Roadmap

- Hybrid search (dense + BM25) with score fusion
- Open-source vs closed LLM comparison on the eval set
- Ground-truth reference answers for context-precision/recall style metrics
- Citation-graph features via CourtListener
- BGE-M3 long-context vs short-context chunking
- DeepEval in CI
- Expand corpus to federal circuit courts