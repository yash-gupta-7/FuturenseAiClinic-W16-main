"""Retrieval step: embed the query and pull the top-k chunks from Chroma."""
import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma

from app import config
from app.observability import observe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@observe(name="retrieve")
def retrieve(query: str, store: Chroma, top_k: int = None) -> List[Dict[str, Any]]:
    """Return the top_k chunks for a query as plain dicts.

    Each dict has: content, source, page, score (lower L2 distance = closer).
    """
    top_k = top_k or config.TOP_K
    if not query or not query.strip():
        return []

    results = store.similarity_search_with_score(query, k=top_k)
    out: List[Dict[str, Any]] = []
    for doc, score in results:
        out.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "score": float(score),
        })
    logger.info("Retrieved %d chunks for query: %s", len(out), query[:60])
    return out
