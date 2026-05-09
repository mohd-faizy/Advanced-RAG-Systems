"""
Module 10 — Capstone: Agentic RAG Pipeline (LangGraph)
=======================================================
A production-grade agentic RAG workflow with:
  - Query complexity classification (simple vs complex)
  - Conditional retrieval
  - Relevance grading
  - Query rewriting on poor retrieval
  - Final answer generation with citations
  - LangSmith tracing support

Usage:
    from pipeline import build_rag_pipeline
    pipeline = build_rag_pipeline(vectorstore)
    result   = pipeline.invoke({"question": "What is RAG?"})
"""

from __future__ import annotations
import os
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Optional: enable LangSmith tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"]    = "RAG-Capstone"

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question        : str
    complexity      : str          # "simple" | "complex"
    documents       : List[Document]
    relevance_grade : str          # "relevant" | "irrelevant"
    rewritten_query : str
    iterations      : int
    answer          : str
    citations       : List[str]


# ── LLM ───────────────────────────────────────────────────────────────────────
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def classify_query(state: AgentState) -> AgentState:
    """Decide if the query is simple (direct answer) or complex (requires retrieval)."""
    prompt = ChatPromptTemplate.from_template(
        "Classify this question as 'simple' (general knowledge, no docs needed) "
        "or 'complex' (requires domain knowledge from documents).\n"
        "Answer only 'simple' or 'complex'.\n\nQuestion: {question}"
    )
    result = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    complexity = "complex" if "complex" in result.lower() else "simple"
    return {**state, "complexity": complexity}


def retrieve(state: AgentState, retriever) -> AgentState:
    """Retrieve documents using the provided retriever."""
    query = state.get("rewritten_query") or state["question"]
    docs  = retriever.invoke(query)
    return {
        **state,
        "documents" : docs,
        "iterations": state.get("iterations", 0) + 1,
    }


def grade_documents(state: AgentState) -> AgentState:
    """Grade each retrieved document for relevance to the question."""
    grade_prompt = ChatPromptTemplate.from_template(
        "Is this document relevant to answering '{question}'?\n"
        "Document: {document}\n"
        "Answer only 'yes' or 'no'."
    )
    grades = []
    for doc in state["documents"]:
        r = (grade_prompt | LLM | StrOutputParser()).invoke(
            {"question": state["question"], "document": doc.page_content[:400]}
        )
        grades.append("yes" in r.lower())

    relevant_docs = [doc for doc, ok in zip(state["documents"], grades) if ok]
    overall       = "relevant" if relevant_docs else "irrelevant"

    return {**state, "documents": relevant_docs, "relevance_grade": overall}


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval recall."""
    prompt = ChatPromptTemplate.from_template(
        "The original query did not return useful results.\n"
        "Rewrite it to be more specific and retrieve better documents.\n"
        "Original: {question}\nRewritten:"
    )
    rewritten = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    return {**state, "rewritten_query": rewritten.strip()}


def direct_answer(state: AgentState) -> AgentState:
    """Answer simple queries directly without retrieval."""
    prompt  = ChatPromptTemplate.from_template(
        "Answer this question concisely and accurately.\nQuestion: {question}"
    )
    answer  = (prompt | LLM | StrOutputParser()).invoke({"question": state["question"]})
    return {**state, "answer": answer, "citations": []}


def generate_with_citations(state: AgentState) -> AgentState:
    """Generate an answer grounded in retrieved documents, with source citations."""
    context_parts = []
    citations     = []

    for i, doc in enumerate(state["documents"], 1):
        src  = doc.metadata.get("source", f"Document {i}")
        context_parts.append(f"[{i}] {doc.page_content}")
        citations.append(src)

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the numbered context below.\n"
        "After each fact you use, cite the source number in brackets, e.g. [1].\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    answer = (prompt | LLM | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    return {**state, "answer": answer, "citations": citations}


# ────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def route_complexity(state: AgentState) -> str:
    return "direct_answer" if state["complexity"] == "simple" else "retrieve"


def route_grade(state: AgentState) -> str:
    if state["relevance_grade"] == "relevant":
        return "generate"
    if state.get("iterations", 0) >= 2:
        # Max retries reached — generate with what we have or fallback
        return "generate"
    return "rewrite"


# ────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ────────────────────────────────────────────────────────────────────────────

def build_rag_pipeline(retriever):
    """
    Build and compile the agentic RAG LangGraph.

    Args:
        retriever: Any LangChain retriever (hybrid, ensemble, etc.)

    Returns:
        Compiled LangGraph runnable
    """
    # Bind retriever into retrieve node via closure
    def _retrieve(state):
        return retrieve(state, retriever)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify",       classify_query)
    graph.add_node("retrieve",       _retrieve)
    graph.add_node("grade",          grade_documents)
    graph.add_node("rewrite",        rewrite_query)
    graph.add_node("direct_answer",  direct_answer)
    graph.add_node("generate",       generate_with_citations)

    # Edges
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_complexity,
        {"direct_answer": "direct_answer", "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve",       "grade")
    graph.add_conditional_edges(
        "grade", route_grade,
        {"generate": "generate", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite",        "retrieve")
    graph.add_edge("direct_answer",  END)
    graph.add_edge("generate",       END)

    return graph.compile()


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document

    demo_docs = [
        Document(page_content="LangGraph is a graph-based framework for agentic LLM workflows.",
                 metadata={"source": "langgraph_docs.txt"}),
        Document(page_content="RAGAS evaluates RAG pipelines using faithfulness and relevance metrics.",
                 metadata={"source": "ragas_docs.txt"}),
        Document(page_content="Hybrid search combines BM25 and dense vector retrieval for better recall.",
                 metadata={"source": "retrieval_guide.txt"}),
    ]

    emb  = OpenAIEmbeddings(model="text-embedding-3-small")
    vs   = Chroma.from_documents(demo_docs, emb, collection_name="capstone_demo")
    retr = vs.as_retriever(search_kwargs={"k": 3})

    pipeline = build_rag_pipeline(retr)

    test_questions = [
        "What is 2 + 2?",                          # simple
        "How does LangGraph support agentic RAG?",  # complex
    ]

    for q in test_questions:
        print(f"\n{'═'*60}")
        print(f"Question: {q}")
        result = pipeline.invoke({
            "question"        : q,
            "complexity"      : "",
            "documents"       : [],
            "relevance_grade" : "",
            "rewritten_query" : "",
            "iterations"      : 0,
            "answer"          : "",
            "citations"       : [],
        })
        print(f"Complexity : {result['complexity']}")
        print(f"Iterations : {result['iterations']}")
        print(f"Citations  : {result['citations']}")
        print(f"Answer     : {result['answer'][:300]}")
