# Module 11 - Capstone Project 2

## Adaptive Multi-Hop Research RAG Assistant

This second capstone Project focuses on a production-style RAG pipeline. Module 11 focuses on **advanced research-style RAG** where a single user question is decomposed into smaller retrieval tasks, searched through both dense and sparse retrieval, reranked locally, synthesized with citations, and checked for grounding.

The project stays free-first:

- LLM: Groq through `langchain-groq`
- Embeddings: local Hugging Face/SentenceTransformers through `langchain-huggingface`
- Vector store: local Chroma
- Sparse retrieval: BM25
- Reranking: local SentenceTransformers CrossEncoder
- No paid OpenAI API key required

## What You Build

An adaptive assistant for questions such as:

> Compare corrective RAG and self-RAG. When should each one be used in a production support assistant?

The system will:

1. Create a small retrieval plan with focused sub-questions.
2. Retrieve evidence from a local knowledge base using dense vector search and BM25.
3. Deduplicate and rerank evidence with a local cross-encoder.
4. Generate a grounded answer with source citations using Groq.
5. Check whether the answer is supported by the retrieved context.

## Files

```text
Module_11_Capston_Proj_2/
|-- README.md
|-- How_to_Run_Capstone_Proj_2.ipynb
|-- config.py
|-- corpus.py
|-- retrievers.py
|-- graph.py
|-- demo.py
`-- evaluation.py
```

## Environment

Copy `.env.example` to `.env` at the repository root and set:

```ini
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally. Groq is only needed for query planning and answer generation.

## Run The Demo

From the repository root:

```bash
uv run python Module_11_Capston_Proj_2/demo.py
```

Or open the guided notebook:

```text
Module_11_Capston_Proj_2/How_to_Run_Capstone_Proj_2.ipynb
```

To build an index from your own documents:

```python
from Module_11_Capston_Proj_2.corpus import build_vectorstore_from_directory

vectorstore, chunks = build_vectorstore_from_directory(
    docs_dir="./data",
    collection_name="module_11_research_rag",
)
```

Then wire the retriever and graph:

```python
from Module_11_Capston_Proj_2.graph import build_research_graph
from Module_11_Capston_Proj_2.retrievers import AdaptiveMultiHopRetriever

retriever = AdaptiveMultiHopRetriever(vectorstore=vectorstore, corpus=chunks)
app = build_research_graph(retriever)

result = app.invoke({
    "question": "How should I choose between hybrid search, reranking, and query rewriting?"
})

print(result["answer"])
```

## Project Extensions

- Add metadata filters for document type, owner, freshness, or product area.
- Add query routing between policy docs, support tickets, and API references.
- Store every sub-question, retrieved source, and verification result for debugging.
- Add RAGAS evaluation once you have a small labeled test set.
- Add a Streamlit or FastAPI interface after the core pipeline is working.
