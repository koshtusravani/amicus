from pathlib import Path
import os

# Paths 
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

# Corpus (my v1 scope is US Supreme Court opinions only) 
HF_DATASET = "free-law/Caselaw_Access_Project"  # CC0-licensed bulk corpus
COURT_FILTER = "scotus"
DEV_SLICE = 50  # how many opinions I load while wiring the spine end-to-end

# Embeddings (Hugging Face sentence-transformers) 
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024
# BGE expects me to prefix the QUERY with an instruction, but embed documents
# WITHOUT it. If I get this wrong retrieval quietly degrades, so I centralize it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

# Lighter models I can drop in for fast local iteration, then swap up for final numbers:
#   "BAAI/bge-base-en-v1.5"
#   "sentence-transformers/all-MiniLM-L6-v2"

# Chunking 
CHUNK_TOKENS = 700        # target chunk size
CHUNK_OVERLAP = 0.12      # fraction of overlap between adjacent chunks

# Retrieval
TOP_K = 50                # ANN candidates I fetch before reranking
RERANK_K = 6              # how many I keep for the LLM after reranking
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION_NAME = "scotus_opinions"

# Generation (closed API by default; open-source comparison is my stretch)
# I set LLM_MODEL in my .env to the model string for my provider.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0  # deterministic on purpose: this is grounded QA, not creative writing

# Evaluation targets
TARGETS = {
    "precision_at_5": 0.70,
    "recall_at_20": 0.80,
    "faithfulness": 0.75,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.80,
    "fabricated_citation_rate": 0.0,  # my headline legal metric
}