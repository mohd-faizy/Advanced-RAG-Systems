"""Lightweight evaluation helpers for Module 11."""

from __future__ import annotations

from langchain_core.documents import Document


def citation_coverage(answer: str, docs: list[Document]) -> float:
    """Return the fraction of retrieved contexts cited in the answer."""

    if not docs:
        return 0.0
    cited = 0
    for index in range(1, len(docs) + 1):
        if f"[{index}]" in answer:
            cited += 1
    return cited / len(docs)


def source_diversity(docs: list[Document]) -> int:
    """Count unique sources represented in retrieved context."""

    return len({str(doc.metadata.get("source", "unknown")) for doc in docs})


def print_research_rag_report(question: str, answer: str, docs: list[Document]) -> None:
    """Print a compact report for manual inspection."""

    print("\nMODULE 11 RESEARCH RAG REPORT")
    print("=" * 40)
    print(f"Question: {question}")
    print(f"Retrieved contexts: {len(docs)}")
    print(f"Source diversity: {source_diversity(docs)}")
    print(f"Citation coverage: {citation_coverage(answer, docs):.2f}")
    print("\nAnswer:")
    print(answer)

