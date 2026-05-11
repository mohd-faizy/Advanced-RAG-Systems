"""Corpus loading, splitting, and local Chroma indexing for Module 11."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import settings
except ImportError:
    from config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local embedding model used across the course."""

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_file(path: str) -> list[Document]:
    """Load one supported document file."""

    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(path).load()
    elif suffix == ".csv":
        docs = CSVLoader(path).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(path, encoding="utf-8").load()
    else:
        return []

    for doc in docs:
        doc.metadata["source"] = path
        doc.metadata["file_name"] = Path(path).name
    return docs


def load_directory(docs_dir: str) -> list[Document]:
    """Load all supported documents from a directory."""

    docs: list[Document] = []
    for path in Path(docs_dir).rglob("*"):
        if path.is_file():
            docs.extend(load_file(str(path)))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents and add stable chunk identifiers."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        chunk.metadata["chunk_id"] = digest
        chunk.metadata["chunk_index"] = index
    return chunks


def build_vectorstore(
    docs: list[Document],
    collection_name: str = settings.collection_name,
) -> tuple[Chroma, list[Document]]:
    """Create a local Chroma vector store from documents."""

    chunks = split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=settings.persist_dir,
    )
    return vectorstore, chunks


def build_vectorstore_from_directory(
    docs_dir: str,
    collection_name: str = settings.collection_name,
) -> tuple[Chroma, list[Document]]:
    """Load a directory and create a Chroma index."""

    docs = load_directory(docs_dir)
    if not docs:
        raise ValueError(f"No supported documents found in {docs_dir}")
    return build_vectorstore(docs, collection_name=collection_name)
