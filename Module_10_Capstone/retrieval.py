"""
Module 10 — Capstone: Hybrid Retrieval with Re-ranking
=======================================================
Provides a unified retriever that combines:
  - Dense semantic search (Chroma)
  - Sparse keyword search (BM25)
  - Cross-encoder re-ranking (sentence-transformers)
  - Optional contextual compression

Usage:
    from retrieval import build_hybrid_retriever, retrieve_and_rerank
"""

from __future__ import annotations
from typing import Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from sentence_transformers import CrossEncoder


# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = "text-embedding-3-small"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PERSIST_DIR     = "./chroma_db"
DEFAULT_K       = 5
RERANK_TOP_N    = 3


# ── Build hybrid retriever ────────────────────────────────────────────────────
def build_hybrid_retriever(
    collection_name: str,
    all_docs: list[Document],
    dense_weight: float = 0.6,
    k: int = DEFAULT_K,
    compress: bool = False,
) -> EnsembleRetriever | ContextualCompressionRetriever:
    """
    Returns an EnsembleRetriever combining:
      - BM25 sparse retriever (from in-memory docs)
      - Chroma dense retriever

    Args:
        collection_name : Chroma collection name
        all_docs        : Full list of Documents (needed for BM25)
        dense_weight    : Weight for dense retriever (1-dense_weight for BM25)
        k               : Number of docs to retrieve
        compress        : Whether to add EmbeddingsFilter compression
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Dense retriever
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    dense_retriever = vs.as_retriever(search_kwargs={"k": k})

    # Sparse BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)

    # Ensemble
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[1 - dense_weight, dense_weight],
    )

    # Optional: add embedding-based compression
    if compress:
        compressor = EmbeddingsFilter(
            embeddings=embeddings, similarity_threshold=0.75
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

    return retriever


# ── Cross-encoder re-ranker ───────────────────────────────────────────────────
class CrossEncoderReranker:
    """Two-stage retriever: EnsembleRetriever → cross-encoder re-rank."""

    def __init__(
        self,
        base_retriever,
        model_name: str = RERANKER_MODEL,
        top_n: int = RERANK_TOP_N,
    ):
        self.base_retriever = base_retriever
        self.cross_encoder  = CrossEncoder(model_name)
        self.top_n          = top_n

    def invoke(self, query: str) -> list[Document]:
        # Stage 1: broad recall
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # Stage 2: precise re-ranking
        pairs  = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in ranked[: self.top_n]]
        return top_docs


# ── Convenience function ──────────────────────────────────────────────────────
def retrieve_and_rerank(
    query: str,
    collection_name: str,
    all_docs: list[Document],
    k: int = DEFAULT_K,
    top_n: int = RERANK_TOP_N,
    compress: bool = True,
) -> list[Document]:
    """
    One-shot: build hybrid retriever + re-rank results.

    Returns top_n documents after cross-encoder re-ranking.
    """
    hybrid    = build_hybrid_retriever(collection_name, all_docs, k=k, compress=compress)
    reranker  = CrossEncoderReranker(hybrid, top_n=top_n)
    results   = reranker.invoke(query)
    return results


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document

    sample_docs = [
        Document(page_content="LangGraph is used for building agentic workflows with LLMs."),
        Document(page_content="FAISS provides efficient similarity search for dense vectors."),
        Document(page_content="BM25 is a bag-of-words retrieval model based on TF-IDF."),
        Document(page_content="RAG combines retrieval with generation for grounded answers."),
        Document(page_content="Cross-encoders re-rank candidate documents with high accuracy."),
    ]

    # Ingest into Chroma
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    vs  = Chroma.from_documents(
        sample_docs, emb, collection_name="demo", persist_directory=PERSIST_DIR
    )

    results = retrieve_and_rerank(
        "How does retrieval work in RAG?",
        collection_name="demo",
        all_docs=sample_docs,
    )

    print("Re-ranked results:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")
