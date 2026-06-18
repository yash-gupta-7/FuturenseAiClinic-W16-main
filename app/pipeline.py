"""End-to-end RAG pipeline, traced with Langfuse @observe().

`answer_question` is the single entry point used by the CLI, the Ragas eval
harness, the experiments runner, and the DeepEval gate — so every consumer
measures the same code path.
"""
import time
import logging
from typing import Dict, Any, List, Optional

from langchain_community.vectorstores import Chroma

from app import config
from app.ingest import load_index
from app.retriever import retrieve
from app.generator import generate
from app.observability import observe, update_current_trace, flush

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Approx USD pricing per 1M tokens for the cost estimate logged into traces.
# (gpt-4o-mini list price at time of writing; override if your model differs.)
_PRICE_PER_M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
}


def estimate_cost(model: str, usage: Dict[str, int]) -> float:
    p = _PRICE_PER_M.get(model, {"in": 0.0, "out": 0.0})
    return (
        usage.get("input_tokens", 0) / 1_000_000 * p["in"]
        + usage.get("output_tokens", 0) / 1_000_000 * p["out"]
    )


@observe(name="rag_pipeline")
def answer_question(
    question: str,
    store: Optional[Chroma] = None,
    top_k: int = None,
) -> Dict[str, Any]:
    """Run retrieve -> generate and return a structured result.

    Returned dict (also the shape Ragas consumes):
      question, answer, contexts (list[str]), sources (list[str]),
      latency_s, token_usage, cost_usd
    """
    start = time.perf_counter()
    store = store if store is not None else load_index()
    top_k = top_k or config.TOP_K

    chunks = retrieve(question, store, top_k=top_k)
    answer, usage = generate(question, chunks)

    latency = time.perf_counter() - start
    model = config.OPENAI_CHAT_MODEL if config.GENERATION_PROVIDER == "openai" else config.ANTHROPIC_CHAT_MODEL
    cost = estimate_cost(model, usage)

    result = {
        "question": question,
        "answer": answer,
        "contexts": [c["content"] for c in chunks],
        "sources": [c["source"] for c in chunks],
        "scores": [c["score"] for c in chunks],
        "latency_s": round(latency, 3),
        "token_usage": usage,
        "cost_usd": round(cost, 6),
    }

    # Attach rich metadata to the Langfuse trace (no-op if tracing disabled).
    update_current_trace(
        input={"question": question},
        output={"answer": answer},
        metadata={
            "sources": result["sources"],
            "latency_s": result["latency_s"],
            "cost_usd": result["cost_usd"],
            "token_usage": usage,
            **config.summary(),
        },
    )
    return result


def ask(question: str, top_k: int = None) -> Dict[str, Any]:
    """Convenience wrapper that flushes traces after a single call."""
    result = answer_question(question, top_k=top_k)
    flush()
    return result
