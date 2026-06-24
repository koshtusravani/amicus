"""
I keep every tunable knob in one place here, so when I want to sweep a
parameter during evaluation I'm not hunting through the whole codebase.
"""
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()  

#Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "scotus"     # cached opinion records land here
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

#Corpus: SCOTUS via the CourtListener v4 API
CL_BASE = "https://www.courtlistener.com/api/rest/v4"
CL_COURT = "scotus"
CL_ORDER_BY = "citeCount desc"   
CL_SLEEP = 3.0                   
CL_MAX_OPINIONS = 1              
DEV_SLICE = 15                   

#Embeddings (Hugging Face sentence-transformers)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

#Chunking
CHUNK_TOKENS = 700
CHUNK_OVERLAP = 0.12

#Retrieval
TOP_K = 50
RERANK_K = 6
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION_NAME = "scotus_opinions"

#Generation
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0

#Evaluation targets
TARGETS = {
    "precision_at_5": 0.70,
    "recall_at_20": 0.80,
    "faithfulness": 0.75,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.80,
    "fabricated_citation_rate": 0.0,
}