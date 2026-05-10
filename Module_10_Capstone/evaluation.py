"""
Module 10 — Capstone: Evaluation Suite
=======================================
Evaluates a deployed RAG pipeline using manual LLM-as-Judge metrics.
Uses ChatGroq (free) instead of paid RAGAS async evaluation
to avoid rate-limit issues on the free tier.

Usage:
    python evaluation.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── LLM Judge ─────────────────────────────────────────────────────────────────
LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ── Manual Metrics (LLM-as-Judge) ─────────────────────────────────────────────

def score_faithfulness(answer: str, context: str) -> float:
    """Is the answer grounded in the context? Returns 0.0 or 1.0."""
    prompt = ChatPromptTemplate.from_template(
        "Is the following answer fully supported by the context?\n"
        "Context: {context}\nAnswer: {answer}\n"
        "Reply ONLY 'yes' or 'no'."
    )
    r = (prompt | LLM | StrOutputParser()).invoke({"context": context, "answer": answer})
    return 1.0 if "yes" in r.lower() else 0.0


def score_relevancy(question: str, answer: str) -> float:
    """Does the answer address the question? Returns 0.0 or 1.0."""
    prompt = ChatPromptTemplate.from_template(
        "Does this answer address the question?\n"
        "Question: {question}\nAnswer: {answer}\n"
        "Reply ONLY 'yes' or 'no'."
    )
    r = (prompt | LLM | StrOutputParser()).invoke({"question": question, "answer": answer})
    return 1.0 if "yes" in r.lower() else 0.0


def score_context_precision(question: str, contexts: list[str]) -> float:
    """What fraction of retrieved contexts are relevant?"""
    if not contexts:
        return 0.0
    prompt = ChatPromptTemplate.from_template(
        "Is this context relevant to the question?\n"
        "Question: {question}\nContext: {context}\n"
        "Reply ONLY 'yes' or 'no'."
    )
    relevant = 0
    for ctx in contexts:
        r = (prompt | LLM | StrOutputParser()).invoke({"question": question, "context": ctx})
        if "yes" in r.lower():
            relevant += 1
    return relevant / len(contexts)


# ── Evaluation Runner ─────────────────────────────────────────────────────────

def evaluate_sample(question: str, answer: str, contexts: list[str]) -> dict:
    """Evaluate a single RAG output."""
    ctx_combined = "\n".join(contexts)
    return {
        "question": question,
        "faithfulness": score_faithfulness(answer, ctx_combined),
        "relevancy": score_relevancy(question, answer),
        "context_precision": score_context_precision(question, contexts),
    }


def print_evaluation_report(results: list[dict]):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 65)
    print("📊 EVALUATION REPORT (LLM-as-Judge)")
    print("=" * 65)

    for r in results:
        print(f"\nQ: {r['question'][:60]}...")
        print(f"  Faithfulness      : {r['faithfulness']:.1f}")
        print(f"  Answer Relevancy  : {r['relevancy']:.1f}")
        print(f"  Context Precision : {r['context_precision']:.2f}")

    # Averages
    n = len(results)
    if n > 0:
        avg_f = sum(r["faithfulness"] for r in results) / n
        avg_r = sum(r["relevancy"] for r in results) / n
        avg_p = sum(r["context_precision"] for r in results) / n

        print("\n── Mean Scores ──")
        for name, val in [("Faithfulness", avg_f), ("Relevancy", avg_r), ("Context Precision", avg_p)]:
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            status = "✅" if val >= 0.7 else "⚠ "
            print(f"  {status} {name:25}: {val:.4f}  [{bar}]")


# ── Production Checklist ──────────────────────────────────────────────────────
PRODUCTION_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════╗
║           Production RAG Best Practices Checklist                ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline Optimization                                           ║
║    • Use async retrieval for parallel chunk fetching             ║
║    • Batch embedding calls (embed_documents vs embed_query)      ║
║    • Pre-filter with metadata before vector search               ║
║  Caching Strategies                                              ║
║    • Cache embedding computations (semantic cache)               ║
║    • Cache frequent query results with TTL                       ║
║  Cost Optimization                                               ║
║    • Use free local embeddings (all-MiniLM-L6-v2)               ║
║    • Use Groq free tier for development (llama-3.1-8b-instant)  ║
║  Monitoring                                                      ║
║    • Enable LangSmith tracing (LANGCHAIN_TRACING_V2=true)        ║
║    • Log retrieval latency and hit rates per query               ║
║  Common Pitfalls                                                 ║
║    • Chunk too small → poor context → use parent doc retriever   ║
║    • Chunk too large → low precision → reduce chunk_size         ║
║    • Embedding drift → re-embed when model version changes       ║
╚══════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(PRODUCTION_CHECKLIST)
