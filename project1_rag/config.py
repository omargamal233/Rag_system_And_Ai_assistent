from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-m3")
chunk_size=300
##########
##########
#########
"""
config.py - central settings for Project 1 (Ask Your Documents).

Every other file (ingest.py, rag_chain.py, ask.py) imports from here,
so you change a setting once, in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths (built from this file's location, so they work from any directory)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../project1_rag
PROJECT_ROOT = BASE_DIR.parent                       # .../ai-projects

# Load the shared .env that sits in ai-projects/
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = BASE_DIR / "data"                         # your .txt / .pdf files
CHROMA_DIR = BASE_DIR / "chroma_db"                  # vector store (auto-created)
MODEL_CACHE_DIR = BASE_DIR / "models"                # local embedding model files

# Only these file types are loaded from DATA_DIR
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}

# --------------------------------------------------------------------------
# Chunking (see the chunk-size discussion: ~300 tokens, ~15% overlap)
# --------------------------------------------------------------------------
CHUNK_SIZE = 1200            # characters (~300 tokens)
CHUNK_OVERLAP = 150          # characters (~15%)
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # tried in this order

# --------------------------------------------------------------------------
# Embeddings (local, free, no API key)
# --------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # ~210 MB; use bge-small-en-v1.5 for lighter
EMBEDDING_DEVICE = "cpu"                     # "cuda" if you have an NVIDIA GPU
NORMALIZE_EMBEDDINGS = True                  # needed for cosine similarity

# BGE models work best when QUERIES (not documents) get this prefix.
# Set to "" if you switch to a model that doesn't use one.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
COLLECTION_NAME = "transformers_notes"
DISTANCE_METRIC = "cosine"

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K = 4                    # chunks passed to the LLM (~1,200 tokens of context)
# Optional: ignore weak matches. None = keep everything TOP_K returns.
SCORE_THRESHOLD = None

# --------------------------------------------------------------------------
# LLM (the answer generator). Swap provider/model here, no other file changes.
# The API key itself stays in .env, never in code.
# --------------------------------------------------------------------------
#
############
#############
####################
###########################
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")      # gemini | openai | anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # check your provider for current names
LLM_TEMPERATURE = 0.0        # 0 = factual and repeatable, best for RAG
LLM_MAX_TOKENS = 700

# Which .env variable holds the key for each provider
API_KEY_ENV_VARS = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_api_key() -> str:
    """Return the API key for the chosen provider, or fail with a clear message."""
    var = API_KEY_ENV_VARS.get(LLM_PROVIDER)
    if var is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
                         f"Choose one of: {', '.join(API_KEY_ENV_VARS)}")
    key = os.getenv(var)
    if not key:
        raise EnvironmentError(f"{var} is missing. Add it to {PROJECT_ROOT / '.env'}")
    return key


def ensure_dirs() -> None:
    """Create folders that must exist before running."""
    for folder in (DATA_DIR, CHROMA_DIR, MODEL_CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick sanity check: python config.py
    ensure_dirs()
    print("Data folder   :", DATA_DIR)
    print("Chroma folder :", CHROMA_DIR)
    print("Embedding     :", EMBEDDING_MODEL)
    print("Chunking      :", CHUNK_SIZE, "chars /", CHUNK_OVERLAP, "overlap")
    print("LLM           :", LLM_PROVIDER, "/", LLM_MODEL)
    files = [p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    print("Files found   :", files or "none yet - add files to data/")