"""Runnable demo for Module 11 capstone project 2."""

from __future__ import annotations

from langchain_core.documents import Document

try:
    from .corpus import build_vectorstore
    from .evaluation import print_research_rag_report
    from .graph import build_research_graph
    from .retrievers import AdaptiveMultiHopRetriever
except ImportError:
    from corpus import build_vectorstore
    from evaluation import print_research_rag_report
    from graph import build_research_graph
    from retrievers import AdaptiveMultiHopRetriever


def sample_corpus() -> list[Document]:
    """Small built-in corpus so the project can run before users add files."""

    return [
        Document(
            page_content=(
                "Corrective RAG checks whether retrieved documents are relevant. "
                "If retrieval quality is poor, the system can rewrite the query, "
                "search again, or route to a fallback source."
            ),
            metadata={"source": "module_11_sample/corrective_rag.txt"},
        ),
        Document(
            page_content=(
                "Self-RAG uses reflection-style checks around retrieval and "
                "generation. It can decide when retrieval is needed and whether "
                "a generated answer is sufficiently supported."
            ),
            metadata={"source": "module_11_sample/self_rag.txt"},
        ),
        Document(
            page_content=(
                "Hybrid retrieval combines dense vector search with sparse keyword "
                "retrieval such as BM25. This improves recall when user wording "
                "does not exactly match document wording."
            ),
            metadata={"source": "module_11_sample/hybrid_retrieval.txt"},
        ),
        Document(
            page_content=(
                "Cross-encoder rerankers score query-document pairs directly. "
                "They are slower than vector search but often improve precision "
                "after an initial recall stage."
            ),
            metadata={"source": "module_11_sample/reranking.txt"},
        ),
        Document(
            page_content=(
                "Advanced RAG systems should log query plans, retrieved evidence, "
                "answer citations, and verification decisions so teams can debug "
                "faithfulness and retrieval failures."
            ),
            metadata={"source": "module_11_sample/observability.txt"},
        ),
    ]


def main() -> None:
    question = (
        "Compare corrective RAG and self-RAG for a production support assistant. "
        "Where do hybrid retrieval and reranking fit?"
    )
    vectorstore, chunks = build_vectorstore(sample_corpus())
    retriever = AdaptiveMultiHopRetriever(vectorstore=vectorstore, corpus=chunks)
    app = build_research_graph(retriever)
    result = app.invoke({"question": question})

    print("Retrieval trace:")
    for trace in retriever.last_trace:
        print(
            f"- {trace.sub_question} "
            f"(dense={trace.dense_hits}, bm25={trace.sparse_hits}, returned={trace.returned})"
        )

    print_research_rag_report(
        question=question,
        answer=result["answer"],
        docs=result.get("documents", []),
    )


if __name__ == "__main__":
    main()
