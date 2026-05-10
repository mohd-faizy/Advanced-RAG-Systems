"""
Module 10 — Capstone: Agentic RAG Pipeline (LangGraph)
=======================================================
A production-grade agentic RAG workflow with:
  - Query complexity classification (simple vs complex)
  - Conditional retrieval
  - Relevance grading
  - Query rewriting on poor retrieval
  - Final answer generation with citations

Stack: Groq (llama-3.1-8b-instant) + HuggingFace Embeddings — fully free.

Usage:
    from pipeline import build_rag_pipeline
    pipeline = build_rag_pipeline(retriever_fn)
    result   = pipeline.invoke({"question": "What is RAG?"})
"""

from __future__ import annotations
import os
from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question        : str
    complexity      : str
    documents       : List[Document]
    relevance_grade : str
    rewritten_query : str
    iterations      : int
    answer          : str
    citations       : List[str]


# ── LLM ───────────────────────────────────────────────────────────────────────
LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ── NODE FUNCTIONS ────────────────────────────────────────────────────────────

def classify_query(state: AgentState) -> AgentState:
    """Decide if the query is simple or complex."""
    prompt = ChatPromptTemplate.from_template(
        "Classify this question as 'simple' (general knowledge) "
        "or 'complex' (requires domain documents).\n"
        "Answer only 'simple' or 'complex'.\nQuestion: {question}"
    )
    result = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    complexity = "complex" if "complex" in result.lower() else "simple"
    return {**state, "complexity": complexity}


def retrieve(state: AgentState, retriever_fn) -> AgentState:
    """Retrieve documents using the provided retriever function."""
    query = state.get("rewritten_query") or state["question"]
    docs = retriever_fn(query)
    return {
        **state,
        "documents" : docs,
        "iterations": state.get("iterations", 0) + 1,
    }


def grade_documents(state: AgentState) -> AgentState:
    """Grade each retrieved document for relevance."""
    grade_prompt = ChatPromptTemplate.from_template(
        "Is this document relevant to answering '{question}'?\n"
        "Document: {document}\n"
        "Answer only 'yes' or 'no'."
    )
    grades = []
    for doc in state["documents"]:
        r = (grade_prompt | LLM | StrOutputParser()).invoke(
            {"question": state["question"], "document": doc.page_content[:300]}
        )
        grades.append("yes" in r.lower())

    relevant_docs = [doc for doc, ok in zip(state["documents"], grades) if ok]
    overall = "relevant" if relevant_docs else "irrelevant"
    return {**state, "documents": relevant_docs or state["documents"], "relevance_grade": overall}


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval recall."""
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this query to be more specific for document retrieval.\n"
        "Original: {question}\nRewritten:"
    )
    rewritten = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    return {**state, "rewritten_query": rewritten.strip()}


def direct_answer(state: AgentState) -> AgentState:
    """Answer simple queries directly without retrieval."""
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely.\nQuestion: {question}"
    )
    answer = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    return {**state, "answer": answer, "citations": []}


def generate_with_citations(state: AgentState) -> AgentState:
    """Generate an answer grounded in retrieved documents, with citations."""
    context_parts = []
    citations = []
    for i, doc in enumerate(state["documents"], 1):
        src = doc.metadata.get("source", f"Document {i}")
        context_parts.append(f"[{i}] {doc.page_content}")
        citations.append(src)

    context = "\n\n".join(context_parts)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the numbered context below.\n"
        "Cite source numbers in brackets, e.g. [1].\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = (prompt | LLM | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    return {**state, "answer": answer, "citations": citations}


# ── ROUTING ───────────────────────────────────────────────────────────────────

def route_complexity(state: AgentState) -> str:
    return "direct_answer" if state["complexity"] == "simple" else "retrieve"


def route_grade(state: AgentState) -> str:
    if state["relevance_grade"] == "relevant":
        return "generate"
    if state.get("iterations", 0) >= 2:
        return "generate"
    return "rewrite"


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────

def build_rag_pipeline(retriever_fn):
    """
    Build and compile the agentic RAG LangGraph.

    Args:
        retriever_fn: A callable that takes a query string and returns list[Document]

    Returns:
        Compiled LangGraph runnable
    """
    def _retrieve(state):
        return retrieve(state, retriever_fn)

    graph = StateGraph(AgentState)
    graph.add_node("classify",      classify_query)
    graph.add_node("retrieve",      _retrieve)
    graph.add_node("grade",         grade_documents)
    graph.add_node("rewrite",       rewrite_query)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("generate",      generate_with_citations)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_complexity,
        {"direct_answer": "direct_answer", "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_grade,
        {"generate": "generate", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite",       "retrieve")
    graph.add_edge("direct_answer", END)
    graph.add_edge("generate",      END)

    return graph.compile()
