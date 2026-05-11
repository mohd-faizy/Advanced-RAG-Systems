"""
Module 10 — Capstone: Hybrid Retrieval with Re-ranking
=======================================================
Provides a unified retriever that combines:
  - Dense semantic search (Chroma + HuggingFace)
  - Sparse keyword search (BM25)
  - Cross-encoder re-ranking (sentence-transformers)

Stack: Fully free — no paid API keys required.

Usage:
    from retrieval import retrieve_and_rerank
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL  = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
PERSIST_DIR     = "./chroma_capstone_db"
DEFAULT_K       = 5
RERANK_TOP_N    = 3


# ── Custom Hybrid Retriever (pure Python) ─────────────────────────────────────
def hybrid_retrieve(query: str, bm25_retriever, dense_retriever, k: int = DEFAULT_K) -> list[Document]:
    """
    Combine BM25 sparse + Chroma dense results using simple union dedup.
    This keeps the retrieval blend explicit and easy to inspect.
    """
    sparse_docs = bm25_retriever.invoke(query)
    dense_docs = dense_retriever.invoke(query)

    seen = set()
    combined = []
    for doc in dense_docs + sparse_docs:
        doc_id = doc.page_content[:100]
        if doc_id not in seen:
            seen.add(doc_id)
            combined.append(doc)
    return combined[:k]


# ── Cross-encoder Re-ranker ───────────────────────────────────────────────────
class CrossEncoderReranker:
    """Two-stage retriever: hybrid recall → cross-encoder precision."""

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = RERANK_TOP_N):
        self.cross_encoder = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.top_n]]


# ── Convenience function ──────────────────────────────────────────────────────
def retrieve_and_rerank(
    query: str,
    all_docs: list[Document],
    vectorstore: Chroma,
    k: int = DEFAULT_K,
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    One-shot: build hybrid retriever + re-rank results.
    Returns top_n documents after cross-encoder re-ranking.
    """
    bm25 = BM25Retriever.from_documents(all_docs, k=k)
    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    candidates = hybrid_retrieve(query, bm25, dense, k=k)

    reranker = CrossEncoderReranker(top_n=top_n)
    return reranker.rerank(query, candidates)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_docs = [
        Document(page_content="LangGraph is used for building agentic workflows with LLMs."),
        Document(page_content="FAISS provides efficient similarity search for dense vectors."),
        Document(page_content="BM25 is a bag-of-words retrieval model based on TF-IDF."),
        Document(page_content="RAG combines retrieval with generation for grounded answers."),
        Document(page_content="Cross-encoders re-rank candidate documents with high accuracy."),
    ]

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs  = Chroma.from_documents(sample_docs, emb, collection_name="demo_retrieval")

    results = retrieve_and_rerank(
        "How does retrieval work in RAG?",
        all_docs=sample_docs,
        vectorstore=vs,
    )

    print("Re-ranked results:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")
