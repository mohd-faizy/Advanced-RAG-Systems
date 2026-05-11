"""LangGraph workflow for the Module 11 adaptive research RAG assistant."""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

try:
    from .config import settings
    from .retrievers import AdaptiveMultiHopRetriever
except ImportError:
    from config import settings
    from retrievers import AdaptiveMultiHopRetriever


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    documents: list[Document]
    answer: str
    verification: str
    attempts: int


_LLM: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Create the Groq client lazily."""

    global _LLM
    if _LLM is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY in .env before running Module 11.")
        _LLM = ChatGroq(model=settings.groq_model, temperature=0)
    return _LLM


def _parse_plan(raw_plan: str) -> list[str]:
    lines = [line.strip(" -0123456789.").strip() for line in raw_plan.splitlines()]
    return [line for line in lines if line][:4]


def plan_query(state: ResearchState) -> ResearchState:
    """Break the user question into retrieval-focused sub-questions."""

    prompt = ChatPromptTemplate.from_template(
        "Create 2 to 4 focused retrieval sub-questions for this advanced RAG "
        "research question. Keep each sub-question short and searchable.\n\n"
        "Question: {question}\n\nSub-questions:"
    )
    raw_plan = (prompt | get_llm() | StrOutputParser()).invoke(
        {"question": state["question"]}
    )
    return {**state, "sub_questions": _parse_plan(raw_plan), "attempts": 0}


def retrieve_evidence(
    state: ResearchState,
    retriever: AdaptiveMultiHopRetriever,
) -> ResearchState:
    """Retrieve evidence for the main question and planned sub-questions."""

    docs = retriever.retrieve_many(
        question=state["question"],
        sub_questions=state.get("sub_questions", []),
    )
    return {**state, "documents": docs}


def generate_answer(state: ResearchState) -> ResearchState:
    """Generate a citation-first answer from retrieved context."""

    context = _format_context(state.get("documents", []))
    prompt = ChatPromptTemplate.from_template(
        "You are an advanced RAG engineer. Answer the question using only the "
        "context below. Cite sources with bracket numbers like [1]. If the "
        "context is insufficient, say what is missing.\n\n"
        "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
    )
    answer = (prompt | get_llm() | StrOutputParser()).invoke(
        {"question": state["question"], "context": context}
    )
    return {**state, "answer": answer}


def verify_grounding(state: ResearchState) -> ResearchState:
    """Ask Groq to judge whether the answer is grounded in the context."""

    context = _format_context(state.get("documents", []))
    prompt = ChatPromptTemplate.from_template(
        "Check if the answer is fully grounded in the context.\n"
        "Reply with exactly one word: grounded or revise.\n\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    )
    verdict = (prompt | get_llm() | StrOutputParser()).invoke(
        {
            "question": state["question"],
            "context": context,
            "answer": state["answer"],
        }
    )
    attempts = state.get("attempts", 0) + 1
    verification = "grounded" if "grounded" in verdict.lower() else "revise"
    return {**state, "verification": verification, "attempts": attempts}


def route_after_verification(state: ResearchState) -> str:
    if state.get("verification") == "grounded":
        return "end"
    if state.get("attempts", 0) >= 2:
        return "end"
    return "generate"


def build_research_graph(retriever: AdaptiveMultiHopRetriever):
    """Build the adaptive multi-hop RAG graph."""

    def _retrieve(state: ResearchState) -> ResearchState:
        return retrieve_evidence(state, retriever)

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_query)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", generate_answer)
    graph.add_node("verify", verify_grounding)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {"generate": "generate", "end": END},
    )
    return graph.compile()


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "No context retrieved."
    parts = []
    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown source")
        parts.append(f"[{index}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)
