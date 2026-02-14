"""Retrieval evaluation framework: compute MRR and Recall@K against test queries."""
import json
import logging
import sys
from pathlib import Path

from logging_config import setup_logging

logger = logging.getLogger(__name__)

EVAL_QUERIES_PATH = Path(__file__).parent / "queries.json"


def load_eval_queries(path: Path = EVAL_QUERIES_PATH) -> list[dict]:
    """Load evaluation queries from JSON file.

    Expected format:
    [
        {
            "question": "What optimizer was used?",
            "expected_sources": ["paper1.pdf"],
            "expected_pages": [3, 4]  // optional
        },
        ...
    ]
    """
    if not path.exists():
        logger.error("No eval queries found at %s", path)
        logger.info("Create queries.json with test questions and expected sources.")
        sys.exit(1)
    return json.loads(path.read_text())


def reciprocal_rank(expected_sources: list[str], retrieved_docs: list) -> float:
    """Compute reciprocal rank: 1/rank of first relevant document."""
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "")
        if source in expected_sources:
            return 1.0 / i
    return 0.0


def recall_at_k(expected_sources: list[str], retrieved_docs: list, k: int) -> float:
    """Compute Recall@K: fraction of expected sources found in top-K results."""
    retrieved_sources = {doc.metadata.get("source", "") for doc in retrieved_docs[:k]}
    if not expected_sources:
        return 1.0
    found = sum(1 for s in expected_sources if s in retrieved_sources)
    return found / len(expected_sources)


def run_eval():
    """Run evaluation and print metrics."""
    from query import load_vectorstore, build_retriever

    setup_logging()

    queries = load_eval_queries()
    logger.info("Loaded %d evaluation queries", len(queries))

    vs = load_vectorstore()
    retriever = build_retriever(vs)

    mrr_sum = 0.0
    recall_sum = 0.0
    k = retriever.k

    for i, q in enumerate(queries, start=1):
        question = q["question"]
        expected = q["expected_sources"]

        docs = retriever.invoke(question)
        mrr = reciprocal_rank(expected, docs)
        recall = recall_at_k(expected, docs, k)

        mrr_sum += mrr
        recall_sum += recall

        retrieved_sources = [d.metadata.get("source", "?") for d in docs]
        status = "HIT" if mrr > 0 else "MISS"
        logger.info(
            "[%s] Q%d: %s | MRR=%.2f Recall@%d=%.2f | Retrieved: %s",
            status, i, question[:60], mrr, k, recall, retrieved_sources,
        )

    n = len(queries)
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n} queries)")
    print(f"{'='*60}")
    print(f"  Mean Reciprocal Rank (MRR):  {mrr_sum / n:.3f}")
    print(f"  Mean Recall@{k}:              {recall_sum / n:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_eval()
