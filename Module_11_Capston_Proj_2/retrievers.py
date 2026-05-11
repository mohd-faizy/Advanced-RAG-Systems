"""Adaptive multi-hop retrieval for the Module 11 research RAG capstone."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

try:
    from .config import settings
except ImportError:
    from config import settings


def _document_key(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.page_content[:160])


@dataclass
class RetrievalTrace:
    """Debug trace for one sub-question."""

    sub_question: str
    dense_hits: int
    sparse_hits: int
    returned: int


class AdaptiveMultiHopRetriever:
    """
    Dense + sparse + local reranking retriever for planned multi-hop questions.

    The class accepts a Chroma vector store and the same chunk list used to build
    BM25. It can retrieve for one question or for several sub-questions and
    merge the evidence.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        corpus: list[Document],
        k: int = settings.retrieval_k,
        top_n: int = settings.rerank_top_n,
        reranker_model: str = settings.reranker_model,
    ):
        self.vectorstore = vectorstore
        self.corpus = corpus
        self.k = k
        self.top_n = top_n
        self.dense = vectorstore.as_retriever(search_kwargs={"k": k})
        self.sparse = BM25Retriever.from_documents(corpus, k=k)
        self.reranker = CrossEncoder(reranker_model)
        self.last_trace: list[RetrievalTrace] = []

    def retrieve_one(self, query: str) -> list[Document]:
        """Retrieve and rerank evidence for one query."""

        dense_docs = self.dense.invoke(query)
        sparse_docs = self.sparse.invoke(query)
        merged = self._dedupe(dense_docs + sparse_docs)
        reranked = self._rerank(query, merged)
        self.last_trace.append(
            RetrievalTrace(
                sub_question=query,
                dense_hits=len(dense_docs),
                sparse_hits=len(sparse_docs),
                returned=len(reranked),
            )
        )
        return reranked

    def retrieve_many(self, question: str, sub_questions: list[str]) -> list[Document]:
        """Retrieve evidence for a main question and optional sub-questions."""

        self.last_trace = []
        searches = [question] + [q for q in sub_questions if q.strip()]
        all_hits: list[Document] = []
        for search in searches:
            all_hits.extend(self.retrieve_one(search))
        return self._rerank(question, self._dedupe(all_hits))

    def _dedupe(self, docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            key = _document_key(doc)
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_n]]
