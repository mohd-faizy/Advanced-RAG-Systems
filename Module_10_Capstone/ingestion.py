"""
Module 10 — Capstone: Multi-Document Ingestion Pipeline
========================================================
Handles loading, splitting, metadata enrichment, and indexing
into a Chroma vector store.

Stack: HuggingFace Embeddings (all-MiniLM-L6-v2) — fully free, no API key needed.

Usage:
    python ingestion.py --docs_dir ./data --collection my_rag
"""

import os
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 80
EMBED_MODEL   = "all-MiniLM-L6-v2"
PERSIST_DIR   = "./chroma_capstone_db"


# ── Loaders dispatch table ────────────────────────────────────────────────────
def load_file(path: str) -> list[Document]:
    """Load a file based on its extension."""
    ext = Path(path).suffix.lower()
    dispatch = {
        ".pdf" : lambda: PyPDFLoader(path).load(),
        ".txt" : lambda: TextLoader(path).load(),
        ".csv" : lambda: CSVLoader(path).load(),
        ".md"  : lambda: TextLoader(path).load(),
    }
    loader_fn = dispatch.get(ext)
    if loader_fn is None:
        print(f"  ⚠  Unsupported extension: {ext} — skipping {path}")
        return []
    docs = loader_fn()
    print(f"  ✓  {Path(path).name}: {len(docs)} pages/records loaded")
    return docs


def load_url(url: str) -> list[Document]:
    """Load a web page."""
    docs = WebBaseLoader([url]).load()
    print(f"  ✓  {url}: {len(docs)} pages loaded")
    return docs


# ── Metadata enrichment ───────────────────────────────────────────────────────
def enrich_metadata(docs: list[Document], source: str) -> list[Document]:
    """Add tracking metadata to every chunk."""
    enriched = []
    for i, doc in enumerate(docs):
        meta = {
            **doc.metadata,
            "source"      : source,
            "chunk_index" : i,
            "chunk_id"    : hashlib.md5(doc.page_content.encode()).hexdigest()[:10],
            "ingested_at" : datetime.utcnow().isoformat(),
            "char_count"  : len(doc.page_content),
        }
        enriched.append(Document(page_content=doc.page_content, metadata=meta))
    return enriched


# ── Splitting ─────────────────────────────────────────────────────────────────
def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✂  Split into {len(chunks)} chunks")
    return chunks


# ── Vector store ingestion ────────────────────────────────────────────────────
def ingest_to_vectorstore(
    chunks: list[Document],
    collection_name: str,
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"  📦  Indexed {len(chunks)} chunks into '{collection_name}'")
    return vs


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    docs_dir: Optional[str] = None,
    urls: Optional[list[str]] = None,
    collection: str = "rag_capstone",
) -> Chroma:
    all_docs: list[Document] = []

    # Load from directory
    if docs_dir:
        for fpath in Path(docs_dir).rglob("*"):
            if fpath.is_file():
                docs = load_file(str(fpath))
                if docs:
                    all_docs.extend(enrich_metadata(docs, str(fpath)))

    # Load from URLs
    if urls:
        for url in urls:
            docs = load_url(url)
            all_docs.extend(enrich_metadata(docs, url))

    if not all_docs:
        print("⚠  No documents loaded. Check paths or URLs.")
        return None

    print(f"\n📄 Total raw documents: {len(all_docs)}")
    chunks = split_documents(all_docs)

    vs = ingest_to_vectorstore(chunks, collection)
    print(f"\n✅ Ingestion complete: {len(chunks)} chunks in '{collection}'")
    return vs


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--docs_dir",   type=str, default=None, help="Directory of documents")
    parser.add_argument("--urls",       nargs="*", default=[], help="List of URLs to load")
    parser.add_argument("--collection", type=str, default="rag_capstone")
    args = parser.parse_args()

    run_pipeline(
        docs_dir=args.docs_dir,
        urls=args.urls or None,
        collection=args.collection,
    )
