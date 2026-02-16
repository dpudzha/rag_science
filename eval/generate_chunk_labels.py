"""Generate chunk-level labels for golden datasets from indexed chunks.

This tool uses the persisted BM25 docs (`vectorstore/bm25_index.pkl`) and
existing source/page labels in a golden dataset to infer `expected_chunk_ids`.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.evaluate import doc_chunk_id

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def normalize_source_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower()


def score_doc(question: str, ideal_answer: str, page_content: str) -> float:
    """Simple lexical overlap score for candidate ranking."""
    q_tokens = tokenize(question)
    a_tokens = tokenize(ideal_answer)
    d_tokens = tokenize(page_content)

    if not d_tokens:
        return 0.0

    q_overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
    a_overlap = len(a_tokens & d_tokens) / max(1, len(a_tokens))
    return 0.35 * q_overlap + 0.65 * a_overlap


def load_bm25_docs(path: Path) -> list:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["docs"]


def build_source_page_index(docs: list) -> tuple[dict[tuple[str, int], list], dict[str, list], dict[str, list]]:
    by_source_page: dict[tuple[str, int], list] = {}
    by_source: dict[str, list] = {}
    by_source_norm: dict[str, list] = {}
    for doc in docs:
        source = doc.metadata.get("source")
        if not source:
            continue
        page = int(doc.metadata.get("page", 0) or 0)
        by_source_page.setdefault((source, page), []).append(doc)
        by_source.setdefault(source, []).append(doc)
        by_source_norm.setdefault(normalize_source_name(source), []).append(doc)
    return by_source_page, by_source, by_source_norm


def select_expected_chunks(
    item: dict,
    by_source_page: dict[tuple[str, int], list],
    by_source: dict[str, list],
    by_source_norm: dict[str, list],
    max_chunks: int,
    min_score: float,
) -> list[str]:
    question = item.get("question", "")
    ideal_answer = item.get("ideal_answer", "")
    expected_sources = item.get("expected_sources", []) or []
    expected_pages = item.get("expected_pages", []) or []

    candidates = []
    if expected_sources and expected_pages and len(expected_sources) == len(expected_pages):
        for source, page in zip(expected_sources, expected_pages):
            candidates.extend(by_source_page.get((source, int(page)), []))

        # If exact source-page matching yields nothing, fallback to source-level matches.
        if not candidates:
            for source in expected_sources:
                candidates.extend(by_source.get(source, []))
    else:
        if expected_sources and expected_pages and len(expected_sources) != len(expected_pages):
            logger.warning(
                "expected_sources (%d) and expected_pages (%d) length mismatch for question: %s — ignoring pages",
                len(expected_sources), len(expected_pages), question[:60],
            )
        for source in expected_sources:
            candidates.extend(by_source.get(source, []))

    # Accent-insensitive source fallback (e.g., Möbius vs Mobius) if still empty.
    if not candidates:
        for source in expected_sources:
            candidates.extend(by_source_norm.get(normalize_source_name(source), []))

    # Dedupe by chunk id before ranking.
    unique_docs = {}
    for doc in candidates:
        unique_docs[doc_chunk_id(doc)] = doc

    scored = []
    for cid, doc in unique_docs.items():
        s = score_doc(question, ideal_answer, doc.page_content)
        scored.append((s, cid, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [cid for s, cid, _ in scored if s >= min_score][:max_chunks]

    # If strict threshold filters everything, still keep the strongest candidate.
    if not selected and scored:
        selected = [scored[0][1]]

    return selected


def generate_labels(
    input_path: Path,
    output_path: Path,
    bm25_path: Path,
    max_chunks: int,
    min_score: float,
) -> tuple[int, int]:
    dataset = json.loads(input_path.read_text())
    docs = load_bm25_docs(bm25_path)
    by_source_page, by_source, by_source_norm = build_source_page_index(docs)

    labeled = 0
    for item in dataset:
        chunk_ids = select_expected_chunks(
            item,
            by_source_page=by_source_page,
            by_source=by_source,
            by_source_norm=by_source_norm,
            max_chunks=max_chunks,
            min_score=min_score,
        )
        item["expected_chunk_ids"] = chunk_ids
        if chunk_ids:
            labeled += 1

    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    return len(dataset), labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expected_chunk_ids for a golden dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Input golden dataset JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Output dataset JSON path",
    )
    parser.add_argument(
        "--bm25",
        type=Path,
        default=Path("vectorstore/bm25_index.pkl"),
        help="Path to persisted BM25 index file",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
        help="Maximum expected chunks to keep per question",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.12,
        help="Minimum lexical overlap score for selecting a chunk",
    )
    args = parser.parse_args()

    total, labeled = generate_labels(
        input_path=args.input,
        output_path=args.output,
        bm25_path=args.bm25,
        max_chunks=args.max_chunks,
        min_score=args.min_score,
    )
    print(f"Labeled {labeled}/{total} questions with expected_chunk_ids")


if __name__ == "__main__":
    main()
