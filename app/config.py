"""Central configuration for the RAG slice.

Every tunable knob lives here so experiments can override a single value and
re-measure on the same golden set. Values fall back to environment variables so
nothing secret is committed.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API credentials -------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Providers -------------------------------------------------------------
# Embeddings are pinned to OpenAI text-embedding-3-small per the Week 16 spec.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Generation provider is switchable: "openai" (default, spec) or "anthropic".
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "openai").lower()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001")

# Model used by Ragas / DeepEval as the LLM judge (kept on OpenAI for parity
# with published Ragas defaults; override if you prefer).
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

# --- Retrieval hyperparameters (the experiment surface) --------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "3"))

# --- Paths -----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(ROOT_DIR, "docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(ROOT_DIR, ".chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "admission_policies")

# Phrase the system must emit when the answer is not in the corpus. The
# adversarial / abstention metric checks for this exact sentinel.
ABSTAIN_MESSAGE = "I don't know based on the provided admission policies."


def summary() -> dict:
    """Return the active config (no secrets) for logging into traces/scorecards."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "generation_provider": GENERATION_PROVIDER,
        "chat_model": OPENAI_CHAT_MODEL if GENERATION_PROVIDER == "openai" else ANTHROPIC_CHAT_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }
