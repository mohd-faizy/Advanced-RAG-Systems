# Module 10 - Capstone Project 1

## Production-Style RAG System

This first capstone brings the earlier course modules together into a practical production-style RAG system. It focuses on the core engineering pieces needed to move from notebook experiments toward a reliable retrieval pipeline: ingestion, chunking, vector indexing, hybrid retrieval, reranking, agentic orchestration, and evaluation.

The project stays free-first:

- LLM: Groq through `langchain-groq`
- Embeddings: local Hugging Face/SentenceTransformers through `langchain-huggingface`
- Vector store: local Chroma
- Sparse retrieval: BM25
- Reranking: local SentenceTransformers CrossEncoder
- Orchestration: LangGraph
- No paid OpenAI API key required

## What You Build

A production-oriented RAG pipeline for questions over your own documents.

The system will:

1. Load documents from local files or URLs.
2. Split documents into retrieval-friendly chunks.
3. Enrich chunks with source, chunk ID, timestamp, and size metadata.
4. Store embeddings in a local Chroma vector database.
5. Retrieve with dense search plus BM25 sparse search.
6. Rerank candidate chunks with a local cross-encoder.
7. Use a LangGraph workflow for classification, retrieval, relevance grading, query rewriting, and cited answer generation.
8. Run lightweight LLM-as-judge checks for faithfulness, answer relevancy, and context precision.

## Files

```text
Module_10_Capston_Proj_1/
|-- README.md
|-- How_to_Run_Capstone.ipynb
|-- Production_RAG_Capstone.ipynb
|-- ingestion.py
|-- retrieval.py
|-- pipeline.py
`-- evaluation.py
```

## Environment

Copy `.env.example` to `.env` at the repository root and set:

```ini
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally. Groq is required only for cells or scripts that call an LLM.

## Run The Project

From the repository root, ingest documents into a local Chroma vector store:

```bash
uv run python Module_10_Capston_Proj_1/ingestion.py --docs_dir ./data --collection rag_capstone
```

Run the retrieval demo:

```bash
uv run python Module_10_Capston_Proj_1/retrieval.py
```

Print the production checklist:

```bash
uv run python Module_10_Capston_Proj_1/evaluation.py
```

Or open the guided notebook:

```text
Module_10_Capston_Proj_1/How_to_Run_Capstone.ipynb
```

The main interactive capstone notebook is:

```text
Module_10_Capston_Proj_1/Production_RAG_Capstone.ipynb
```

## Import The LangGraph Pipeline

```python
from Module_10_Capston_Proj_1.pipeline import build_rag_pipeline

app = build_rag_pipeline(retriever_fn)
result = app.invoke({"question": "What is RAG?"})

print(result["answer"])
```

## Project Extensions

- Add FastAPI or Streamlit as a user-facing interface.
- Persist retrieval traces and answer metadata for debugging.
- Add async ingestion and retrieval for larger document collections.
- Add metadata filters for source type, date, owner, or product area.
- Add Docker and deployment templates.
- Expand evaluation with RAGAS once a labeled test set is available.

