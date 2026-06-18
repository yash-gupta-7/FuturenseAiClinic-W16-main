"""Ingestion: load -> chunk -> embed -> Chroma.

Supports .md, .txt, and .pdf source documents. Idempotent: re-running rebuilds
the collection so an experiment that changes chunk_size gets a clean index.
"""
import os
import glob
import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_documents(docs_dir: str = None) -> List[Document]:
    """Load every .md/.txt/.pdf file in docs_dir into LangChain Documents."""
    docs_dir = docs_dir or config.DOCS_DIR
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    paths = sorted(
        glob.glob(os.path.join(docs_dir, "*.md"))
        + glob.glob(os.path.join(docs_dir, "*.txt"))
        + glob.glob(os.path.join(docs_dir, "*.pdf"))
    )
    if not paths:
        raise FileNotFoundError(f"No .md/.txt/.pdf documents found in {docs_dir}")

    documents: List[Document] = []
    for path in paths:
        filename = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            loaded = PyPDFLoader(path).load()
        else:
            loaded = TextLoader(path, encoding="utf-8").load()
        # Normalise the source metadata to a bare filename for clean citations.
        for d in loaded:
            d.metadata["source"] = filename
        documents.extend(loaded)

    logger.info("Loaded %d document object(s) from %d file(s).", len(documents), len(paths))
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split documents into overlapping chunks, preserving source metadata."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Created %d chunks (chunk_size=%d, overlap=%d).",
        len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def build_index(
    chunk_size: int = None,
    chunk_overlap: int = None,
    persist_directory: str = None,
    collection_name: str = None,
) -> Chroma:
    """Full ingest pipeline. Returns a populated, persisted Chroma store."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env — embeddings require it "
            "(text-embedding-3-small)."
        )

    persist_directory = persist_directory or config.CHROMA_DIR
    collection_name = collection_name or config.COLLECTION_NAME

    documents = load_documents()
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Rebuild from scratch so a new chunk_size doesn't mix with a stale index.
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    logger.info("Indexed %d chunks into Chroma at %s.", len(chunks), persist_directory)
    return store


def load_index(
    persist_directory: str = None,
    collection_name: str = None,
) -> Chroma:
    """Open an existing persisted Chroma collection for querying."""
    persist_directory = persist_directory or config.CHROMA_DIR
    collection_name = collection_name or config.COLLECTION_NAME
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"No Chroma index at {persist_directory}. Run `python -m app.ingest` first."
        )
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    store = build_index()
    print(f"\nIngestion complete. Config: {config.summary()}")
