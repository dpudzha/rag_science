"""Minimal RAG experimentation utilities.

This module supports a compact end-to-end workflow:
1. Build a chunked dataset from documents.
2. Build a golden dataset with relevant chunk IDs.
3. Embed chunks and index them with FAISS.
4. Retrieve and evaluate with Recall@k and Precision@k.
5. Run a simple RAG-style generation pass grounded on retrieved chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text.strip()) if s.strip()]
    return sentences


def chunk_documents(documents: list[dict], sentences_per_chunk: int = 5) -> tuple[list[dict], list[dict]]:
    """Chunk documents into fixed-size sentence windows.

    Returns:
        chunked_documents: input documents with embedded "chunks" field.
        chunk_records: flat list of chunk rows for indexing/evaluation.
    """
    chunked_documents = []
    chunk_records = []

    for doc in documents:
        doc_id = doc["doc_id"]
        source = doc["source"]
        sentences = split_sentences(doc["text"])

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            window = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            chunk_id = f"{doc_id}_chunk_{i // sentences_per_chunk}"

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": source,
                "start_sentence": i,
                "end_sentence": i + len(window) - 1,
                "text": chunk_text,
            }
            chunks.append(chunk)
            chunk_records.append(chunk)

        doc_with_chunks = {**doc, "chunks": chunks}
        chunked_documents.append(doc_with_chunks)

    return chunked_documents, chunk_records


def build_golden_dataset(golden_questions: list[dict], chunks: list[dict]) -> list[dict]:
    """Attach relevant chunk IDs by matching evidence snippets in chunk text."""
    golden = []

    for row in golden_questions:
        snippets = row.get("evidence_snippets", [])
        relevant = set()

        for snippet in snippets:
            snippet_norm = snippet.lower().strip()
            for chunk in chunks:
                if snippet_norm in chunk["text"].lower():
                    relevant.add(chunk["chunk_id"])

        golden.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "relevant_chunks": sorted(relevant),
            }
        )

    return golden


@dataclass
class ChunkIndex:
    embedding_model_name: str
    embedding_model: Any
    index: Any
    chunks: list[dict]
    embeddings: np.ndarray



def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_chunk_index(chunks: list[dict], embedding_model_name: str) -> ChunkIndex:
    import faiss
    from sentence_transformers import SentenceTransformer

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    embeddings = _l2_normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return ChunkIndex(
        embedding_model_name=embedding_model_name,
        embedding_model=model,
        index=index,
        chunks=chunks,
        embeddings=embeddings,
    )


def retrieve_top_k(question: str, chunk_index: ChunkIndex, k: int) -> list[dict]:
    query_vec = chunk_index.embedding_model.encode([question], convert_to_numpy=True, show_progress_bar=False)
    query_vec = query_vec.astype(np.float32)
    query_vec = _l2_normalize(query_vec)

    scores, ids = chunk_index.index.search(query_vec, k)
    out = []

    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        chunk = chunk_index.chunks[idx]
        out.append({"chunk_id": chunk["chunk_id"], "score": float(score), "chunk": chunk})

    return out


def precision_recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> tuple[float, float]:
    retrieved = retrieved_chunk_ids[:k]
    relevant = set(relevant_chunk_ids)

    if not retrieved:
        return 0.0, 0.0

    true_positives = sum(1 for c in retrieved if c in relevant)
    precision = true_positives / len(retrieved)
    recall = (true_positives / len(relevant)) if relevant else 1.0
    return precision, recall


def evaluate_retrieval(golden_dataset: list[dict], chunk_index: ChunkIndex, k: int = 3) -> dict:
    rows = []
    precision_scores = []
    recall_scores = []

    for row in golden_dataset:
        retrieved = retrieve_top_k(row["question"], chunk_index, k=k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        precision, recall = precision_recall_at_k(retrieved_ids, row["relevant_chunks"], k)

        rows.append(
            {
                "question": row["question"],
                "relevant_chunks": row["relevant_chunks"],
                "retrieved_chunks": retrieved_ids,
                f"precision@{k}": precision,
                f"recall@{k}": recall,
            }
        )
        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        "mean_precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "mean_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "rows": rows,
    }


class FaissRetriever:
    """Retriever wrapper connected to a custom FAISS chunk index."""

    def __init__(self, chunk_index: ChunkIndex, k: int = 3):
        self.chunk_index = chunk_index
        self.k = k

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        return retrieve_top_k(question, self.chunk_index, k or self.k)


class MinimalRAGSequence:
    """Simple RAG architecture with tokenizer, retriever, and sequence generator."""

    def __init__(
        self,
        retriever: FaissRetriever,
        generator_model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 80,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.rag_sequence_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, k: int | None = None) -> dict:
        hits = self.retriever.retrieve(question, k=k)
        contexts = [h["chunk"]["text"] for h in hits]

        prompt = (
            "Answer the question using only the provided context. "
            "If context is insufficient, say so clearly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n- " + "\n- ".join(contexts)
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = self.rag_sequence_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "question": question,
            "answer": answer,
            "retrieved": hits,
        }


def run_experiment_grid(
    documents: list[dict],
    golden_questions: list[dict],
    chunk_sizes: list[int],
    embedding_models: list[str],
    k: int = 3,
) -> list[dict]:
    """Run retrieval experiments across chunk sizes and embedding models."""
    results = []

    for chunk_size in chunk_sizes:
        chunked_docs, chunks = chunk_documents(documents, sentences_per_chunk=chunk_size)
        golden = build_golden_dataset(golden_questions, chunks)

        for embed_model in embedding_models:
            chunk_index = build_chunk_index(chunks, embed_model)
            metrics = evaluate_retrieval(golden, chunk_index, k=k)
            results.append(
                {
                    "chunk_size_sentences": chunk_size,
                    "embedding_model": embed_model,
                    f"precision@{k}": metrics["mean_precision"],
                    f"recall@{k}": metrics["mean_recall"],
                    "num_chunks": len(chunks),
                    "chunked_documents": chunked_docs,
                    "golden_dataset": golden,
                }
            )

    return results
