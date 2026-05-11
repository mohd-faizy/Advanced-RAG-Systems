"""Shared settings for Module 11 capstone project 2."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USER_AGENT", "Advanced-RAG-Systems/0.1")


@dataclass(frozen=True)
class Settings:
    """Runtime settings with free-first defaults."""

    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    )
    persist_dir: str = os.getenv("MODULE_11_CHROMA_DIR", "./chroma_module_11_db")
    collection_name: str = os.getenv(
        "MODULE_11_COLLECTION",
        "module_11_research_rag",
    )
    chunk_size: int = int(os.getenv("MODULE_11_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("MODULE_11_CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("MODULE_11_RETRIEVAL_K", "6"))
    rerank_top_n: int = int(os.getenv("MODULE_11_RERANK_TOP_N", "5"))


settings = Settings()

