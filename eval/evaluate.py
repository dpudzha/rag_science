"""Retrieval evaluation framework: source-level and chunk-level retrieval metrics."""
import argparse
import json
import hashlib
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logging_config import setup_logging

logger = logging.getLogger(__name__)

EVAL_QUERIES_PATH = Path(__file__).parent / "golden_dataset.json"


def load_eval_queries(path: Path = EVAL_QUERIES_PATH) -> list[dict]:
    """Load evaluation queries from JSON file.

    Expected format:
    [
        {
            "question": "What optimizer was used?",
            "expected_sources": ["paper1.pdf"],
            "expected_pages": [3, 4],
            "expected_chunk_ids": [
              "paper1.pdf|p3|0a1b2c3d4e5f"
            ]
        },
        ...
    ]
    """
    if not path.exists():
        logger.error("No eval queries found at %s", path)
        logger.info("Create golden_dataset.json with test questions and expected sources.")
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


def _normalize_text(text: str) -> str:
    """Normalize whitespace for stable content hashing."""
    return " ".join(text.split())


def doc_chunk_id(doc) -> str:
    """Create a stable chunk identifier from metadata + normalized chunk text.

    If the document already has metadata["chunk_id"], use it directly.
    """
    metadata = getattr(doc, "metadata", {}) or {}
    existing = metadata.get("chunk_id")
    if existing:
        return str(existing)

    source = metadata.get("source", "")
    page = metadata.get("page", "")
    text = _normalize_text(getattr(doc, "page_content", ""))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}|p{page}|{digest}"


def extract_retrieved_chunk_ids(retrieved_docs: list) -> list[str]:
    """Return deterministic chunk IDs for retrieved documents."""
    return [doc_chunk_id(doc) for doc in retrieved_docs]


def reciprocal_rank_chunks(expected_chunk_ids: list[str], retrieved_docs: list) -> float:
    """Compute reciprocal rank against expected chunk IDs."""
    expected = set(expected_chunk_ids)
    for i, chunk_id in enumerate(extract_retrieved_chunk_ids(retrieved_docs), start=1):
        if chunk_id in expected:
            return 1.0 / i
    return 0.0


def precision_at_k_chunks(expected_chunk_ids: list[str], retrieved_docs: list, k: int) -> float:
    """Compute Precision@K for chunk-level retrieval."""
    retrieved = extract_retrieved_chunk_ids(retrieved_docs)[:k]
    if not retrieved:
        return 0.0
    expected = set(expected_chunk_ids)
    true_positives = sum(1 for chunk_id in retrieved if chunk_id in expected)
    return true_positives / len(retrieved)


def recall_at_k_chunks(expected_chunk_ids: list[str], retrieved_docs: list, k: int) -> float:
    """Compute Recall@K for chunk-level retrieval."""
    expected = set(expected_chunk_ids)
    if not expected:
        return 1.0
    retrieved = set(extract_retrieved_chunk_ids(retrieved_docs)[:k])
    true_positives = sum(1 for chunk_id in expected if chunk_id in retrieved)
    return true_positives / len(expected)


def run_eval(dataset_path: Path | None = None):
    """Run evaluation and print metrics."""
    from retriever import load_vectorstore, build_retriever

    setup_logging()

    queries = load_eval_queries(dataset_path or EVAL_QUERIES_PATH)
    logger.info("Loaded %d evaluation queries", len(queries))

    vs = load_vectorstore()
    retriever = build_retriever(vs)

    mrr_sum = 0.0
    recall_sum = 0.0
    chunk_mrr_sum = 0.0
    chunk_recall_sum = 0.0
    chunk_precision_sum = 0.0
    chunk_labeled_n = 0
    k = retriever.k

    for i, q in enumerate(queries, start=1):
        question = q["question"]
        expected = q.get("expected_sources", [])
        expected_chunk_ids = q.get("expected_chunk_ids", [])

        docs = retriever.invoke(question)
        mrr = reciprocal_rank(expected, docs)
        recall = recall_at_k(expected, docs, k)

        mrr_sum += mrr
        recall_sum += recall

        retrieved_sources = [d.metadata.get("source", "?") for d in docs]
        retrieved_chunk_ids = extract_retrieved_chunk_ids(docs)
        status = "HIT" if mrr > 0 else "MISS"
        log_line = (
            f"[{status}] Q{i}: {question[:60]} | "
            f"Source MRR={mrr:.2f} Source Recall@{k}={recall:.2f} | "
            f"Retrieved sources: {retrieved_sources}"
        )

        if expected_chunk_ids:
            chunk_labeled_n += 1
            chunk_mrr = reciprocal_rank_chunks(expected_chunk_ids, docs)
            chunk_precision = precision_at_k_chunks(expected_chunk_ids, docs, k)
            chunk_recall = recall_at_k_chunks(expected_chunk_ids, docs, k)

            chunk_mrr_sum += chunk_mrr
            chunk_precision_sum += chunk_precision
            chunk_recall_sum += chunk_recall

            log_line += (
                f" | Chunk MRR={chunk_mrr:.2f} "
                f"Chunk Precision@{k}={chunk_precision:.2f} "
                f"Chunk Recall@{k}={chunk_recall:.2f}"
            )

        log_line += f" | Retrieved chunk IDs (top 3): {retrieved_chunk_ids[:3]}"
        logger.info(log_line)

    n = len(queries)
    source_mrr = mrr_sum / n if n else 0.0
    source_recall = recall_sum / n if n else 0.0
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n} queries)")
    print(f"{'='*60}")
    print(f"  Mean Reciprocal Rank (MRR):  {source_mrr:.3f}")
    print(f"  Mean Recall@{k}:              {source_recall:.3f}")
    if chunk_labeled_n:
        print("")
        print(f"  Chunk-labeled queries:        {chunk_labeled_n}/{n}")
        print(f"  Chunk Mean Reciprocal Rank:   {chunk_mrr_sum / chunk_labeled_n:.3f}")
        print(f"  Chunk Mean Precision@{k}:      {chunk_precision_sum / chunk_labeled_n:.3f}")
        print(f"  Chunk Mean Recall@{k}:         {chunk_recall_sum / chunk_labeled_n:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVAL_QUERIES_PATH,
        help="Path to golden evaluation dataset JSON file",
    )
    args = parser.parse_args()
    run_eval(args.dataset)
