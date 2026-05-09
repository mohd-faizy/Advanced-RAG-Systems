"""
Module 10 — Capstone: RAGAS Evaluation Suite
=============================================
Evaluates a deployed RAG pipeline using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Noise Sensitivity

Usage:
    python evaluation.py --collection rag_capstone
"""

from __future__ import annotations
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ── Sample evaluation dataset ─────────────────────────────────────────────────
DEFAULT_EVAL_SAMPLES = [
    {
        "question": "What is RAG and how does it reduce hallucination?",
        "reference": (
            "RAG (Retrieval-Augmented Generation) reduces hallucination by grounding "
            "LLM responses in retrieved documents rather than relying on parametric memory."
        ),
        "contexts": [
            "RAG retrieves relevant documents and injects them into the LLM prompt.",
            "Hallucination occurs when an LLM generates facts not present in its training data.",
            "By conditioning on retrieved facts, RAG anchors the generation to real sources.",
        ],
        "answer": (
            "RAG reduces hallucination by retrieving relevant documents at query time and "
            "providing them as grounding context to the LLM, so the model generates answers "
            "based on real retrieved facts rather than guessing from parametric memory."
        ),
    },
    {
        "question": "What are the main vector stores used in LangChain?",
        "reference": (
            "Common vector stores in LangChain include Chroma, FAISS, Qdrant, Pinecone, and Milvus."
        ),
        "contexts": [
            "Chroma is an open-source embedding database ideal for prototyping.",
            "FAISS is a high-performance library for similarity search from Meta.",
            "Qdrant provides a production-grade vector store with 1 GB free cloud tier.",
        ],
        "answer": (
            "The main vector stores in LangChain include Chroma (for local prototyping), "
            "FAISS (high-performance), Qdrant (production-grade), and Pinecone (managed)."
        ),
    },
    {
        "question": "What is the difference between MMR and standard similarity search?",
        "reference": (
            "MMR balances relevance and diversity, while standard similarity search returns "
            "the k most similar documents regardless of redundancy."
        ),
        "contexts": [
            "Similarity search returns the top-k documents by cosine similarity to the query.",
            "MMR (Maximal Marginal Relevance) selects documents that are relevant but "
            "also diverse from already-selected documents.",
            "The lambda parameter in MMR controls the trade-off: 1=pure relevance, 0=pure diversity.",
        ],
        "answer": (
            "Standard similarity search retrieves the top-k most similar chunks, which can be redundant. "
            "MMR avoids redundancy by penalising documents similar to already-selected ones, "
            "balancing relevance and diversity via the lambda parameter."
        ),
    },
    {
        "question": "How does a cross-encoder differ from a bi-encoder for re-ranking?",
        "reference": (
            "Bi-encoders embed query and document independently; cross-encoders process "
            "both together for higher accuracy but slower speed."
        ),
        "contexts": [
            "Bi-encoders encode query and document separately, enabling fast pre-computation.",
            "Cross-encoders jointly encode the query-document pair, producing more accurate scores.",
            "Re-ranking pipelines typically use bi-encoders for Stage 1 recall and cross-encoders "
            "for Stage 2 precision.",
        ],
        "answer": (
            "Bi-encoders embed query and document independently and compare via cosine similarity — "
            "fast but less accurate. Cross-encoders process the query-document pair jointly, "
            "producing more precise scores at the cost of higher latency, making them ideal for re-ranking."
        ),
    },
    {
        "question": "What does faithfulness measure in RAGAS?",
        "reference": (
            "Faithfulness measures whether the generated answer is factually consistent "
            "with the retrieved context, detecting hallucinations."
        ),
        "contexts": [
            "Faithfulness in RAGAS checks if every claim in the answer is supported by the context.",
            "A faithfulness score of 1.0 means every statement can be verified in the retrieved chunks.",
            "Low faithfulness indicates hallucination — the model added facts not in the context.",
        ],
        "answer": (
            "Faithfulness measures what fraction of claims in the generated answer are "
            "supported by the retrieved context. A score of 1.0 means fully grounded; "
            "lower scores indicate hallucinated content."
        ),
    },
]


# ── Build EvaluationDataset ───────────────────────────────────────────────────
def build_dataset(samples: list[dict]) -> EvaluationDataset:
    eval_samples = [
        SingleTurnSample(
            user_input         =s["question"],
            response           =s["answer"],
            retrieved_contexts =s["contexts"],
            reference          =s["reference"],
        )
        for s in samples
    ]
    return EvaluationDataset(samples=eval_samples)


# ── Run evaluation ────────────────────────────────────────────────────────────
def run_evaluation(
    samples: Optional[list[dict]] = None,
    output_file: Optional[str]    = None,
) -> dict:
    samples    = samples or DEFAULT_EVAL_SAMPLES
    dataset    = build_dataset(samples)
    llm        = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("🔬 Running RAGAS evaluation...")
    results = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    df      = results.to_pandas()

    # ── Print summary ─────────────────────────────────────────────────────────
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "="*70)
    print("RAGAS EVALUATION RESULTS")
    print("="*70)
    print(df[["user_input"] + metric_cols].to_string(max_colwidth=45))
    print("\n── Mean Scores ──")
    for col in metric_cols:
        mean = df[col].mean()
        bar  = "█" * int(mean * 20) + "░" * (20 - int(mean * 20))
        print(f"  {col:25} : {mean:.4f}  [{bar}]")

    # ── Identify weak spots ───────────────────────────────────────────────────
    print("\n── Samples below 0.7 threshold ──")
    for _, row in df.iterrows():
        weak = [c for c in metric_cols if row[c] < 0.7]
        if weak:
            print(f"  Q: {row['user_input'][:50]}...")
            for m in weak:
                print(f"    ⚠  {m}: {row[m]:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    if output_file:
        report = {
            "timestamp"   : datetime.utcnow().isoformat(),
            "n_samples"   : len(df),
            "mean_scores" : {col: float(df[col].mean()) for col in metric_cols},
            "per_sample"  : df.to_dict(orient="records"),
        }
        Path(output_file).write_text(json.dumps(report, indent=2))
        print(f"\n📄 Report saved to: {output_file}")

    return {col: float(df[col].mean()) for col in metric_cols}


# ── Production RAG Good Practices (Module 10.2) ───────────────────────────────
PRODUCTION_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════╗
║           Module 10.2 — Production RAG Good Practices           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Pipeline Optimization                                           ║
║    • Use async retrieval for parallel chunk fetching             ║
║    • Batch embedding calls (embed_documents vs embed_query)      ║
║    • Pre-filter with metadata before vector search               ║
║                                                                  ║
║  Caching Strategies                                              ║
║    • Cache embedding computations (semantic cache)               ║
║    • Cache frequent query results with TTL                       ║
║    • Use LangChain CacheBackedEmbeddings for re-use              ║
║                                                                  ║
║  Cost Optimization                                               ║
║    • Use text-embedding-3-small (6x cheaper than large)          ║
║    • Use gpt-4o-mini for grading, gpt-4o for final generation    ║
║    • Reduce chunk overlap on non-critical corpora                ║
║                                                                  ║
║  Monitoring & Debugging                                          ║
║    • Enable LangSmith tracing (LANGCHAIN_TRACING_V2=true)        ║
║    • Log retrieval latency and hit rates per query               ║
║    • Run RAGAS on a rolling 5% of prod queries                   ║
║                                                                  ║
║  Common Pitfalls & Solutions                                     ║
║    • Chunk too small → poor context → use parent doc retriever   ║
║    • Chunk too large → low precision → reduce chunk_size         ║
║    • Embedding drift → re-embed when model version changes       ║
║    • Stale index → implement incremental ingestion pipeline      ║
║                                                                  ║
║  Security & Compliance                                           ║
║    • Strip PII before embedding (regex or NER)                   ║
║    • Encrypt vector store at rest                                ║
║    • Implement access control per collection                     ║
║    • Audit log all retrievals for regulated industries           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument("--output",   type=str, default="eval_report.json")
    parser.add_argument("--checklist", action="store_true",
                        help="Print the production best-practices checklist")
    args = parser.parse_args()

    if args.checklist:
        print(PRODUCTION_CHECKLIST)
    else:
        run_evaluation(output_file=args.output)
