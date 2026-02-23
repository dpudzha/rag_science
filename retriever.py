"""Retrieval components: hybrid search, cross-encoder reranking, vectorstore loading."""
import logging
import math
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from utils import get_default_llm, get_default_embeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import (
    VECTORSTORE_DIR,
    TOP_K,
    TOP_K_CANDIDATES,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RERANK_MODEL,
    ENABLE_PARENT_RETRIEVAL,
)
from utils import tokenize

logger = logging.getLogger(__name__)


def _prob_to_logit(p: float) -> float:
    """Convert a 0-1 probability to a logit (inverse sigmoid)."""
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


class CohereReranker:
    """Wraps the Cohere rerank API, exposing the same .predict() interface as CrossEncoder."""

    def __init__(self, model: str, api_key: str):
        import cohere
        self.model = model
        self.client = cohere.ClientV2(api_key=api_key)

    def predict(self, pairs: list[list[str]], **kwargs) -> list[float]:
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        # Build score array in original document order
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = _prob_to_logit(result.relevance_score)
        return scores


class JinaReranker:
    """Wraps the Jina rerank API, exposing the same .predict() interface as CrossEncoder."""

    JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def predict(self, pairs: list[list[str]], **kwargs) -> list[float]:
        import httpx
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        response = httpx.post(
            self.JINA_RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        scores = [0.0] * len(documents)
        for result in data["results"]:
            scores[result["index"]] = _prob_to_logit(result["relevance_score"])
        return scores


# Module-level reranker singleton (lazy-loaded)
_reranker: Any = None
_reranker_key: tuple[str, str] | None = None


class VectorstoreNotFoundError(FileNotFoundError):
    """Raised when the FAISS vectorstore is missing from disk."""


def _get_reranker() -> Any:
    global _reranker, _reranker_key
    import config as _cfg
    backend = _cfg.RERANK_BACKEND
    if backend == "cohere":
        current_key = (backend, _cfg.COHERE_RERANK_MODEL)
    elif backend == "jina":
        current_key = (backend, _cfg.JINA_RERANK_MODEL)
    else:
        current_key = (backend, _cfg.RERANK_MODEL)

    if _reranker is None or _reranker_key != current_key:
        if backend == "cohere":
            logger.info("Loading Cohere reranker: %s", _cfg.COHERE_RERANK_MODEL)
            _reranker = CohereReranker(
                model=_cfg.COHERE_RERANK_MODEL,
                api_key=_cfg.COHERE_API_KEY,
            )
        elif backend == "jina":
            logger.info("Loading Jina reranker: %s", _cfg.JINA_RERANK_MODEL)
            _reranker = JinaReranker(
                model=_cfg.JINA_RERANK_MODEL,
                api_key=_cfg.JINA_API_KEY,
            )
        else:
            logger.info("Loading cross-encoder model: %s", _cfg.RERANK_MODEL)
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            _reranker = CrossEncoder(_cfg.RERANK_MODEL, max_length=512)
        _reranker_key = current_key
    return _reranker


def reset_reranker() -> None:
    """Reset the reranker singleton, forcing a reload on next access."""
    global _reranker, _reranker_key
    _reranker = None
    _reranker_key = None


# Backward compatibility alias
reset_cross_encoder = reset_reranker


QA_PROMPT = PromptTemplate.from_template(
    """You are a scientific research assistant answering questions based on \
provided paper excerpts. Use ONLY the context below to answer. Be precise and \
use technical terminology appropriate to the field.

Rules:
- Always respond in English.
- If the context does not contain enough information to answer, say "I don't \
have enough information in the provided papers to answer this."
- Cite specific findings, numbers, and methodologies from the context.
- When multiple papers are relevant, synthesize across them and note agreements \
or disagreements.
- Do not speculate beyond what the context supports.

Context:
{context}

Question: {question}

Answer:"""
)


class HybridRetriever(BaseRetriever):
    """Combines FAISS dense + BM25 keyword search, then reranks with a cross-encoder."""

    vectorstore: FAISS
    bm25: BM25Okapi
    bm25_docs: list[Document]
    cross_encoder: Any
    parent_chunks: list[Document] | None = None
    metadata_filters: dict | None = None
    last_top_rerank_score: float = 0.0
    k: int = TOP_K
    k_candidates: int = TOP_K_CANDIDATES
    bm25_weight: float = BM25_WEIGHT
    dense_weight: float = DENSE_WEIGHT

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Content-based key so the same chunk from FAISS and BM25 merges scores."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        return f"{source}:{page}:{doc.page_content[:200]}"

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        doc_map: dict[str, Document] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        rrf_k = 60  # Standard RRF constant

        # Dense retrieval via FAISS
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.k_candidates)
        for rank, (doc, _score) in enumerate(dense_results):
            key = self._doc_key(doc)
            doc_map[key] = doc
            dense_ranks[key] = rank

        # BM25 keyword retrieval
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:self.k_candidates]
        for rank, idx in enumerate(bm25_top):
            doc = self.bm25_docs[idx]
            key = self._doc_key(doc)
            doc_map[key] = doc
            bm25_ranks[key] = rank

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for key in doc_map:
            score = 0.0
            if key in dense_ranks:
                score += self.dense_weight / (rrf_k + dense_ranks[key])
            if key in bm25_ranks:
                score += self.bm25_weight / (rrf_k + bm25_ranks[key])
            rrf_scores[key] = score

        # Sort by RRF score, take top candidates for reranking
        ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        candidates = [doc_map[key] for key in ranked[:self.k_candidates]]

        if not candidates:
            self.last_top_rerank_score = 0.0
            return []

        # Cross-encoder reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        self.last_top_rerank_score = float(reranked[0][1])

        # Apply metadata filters BEFORE final k-slice
        filtered = [doc for doc, _ in reranked]
        if self.metadata_filters:
            from metadata_extractor import MetadataFilterApplier
            filtered = MetadataFilterApplier.apply(filtered, self.metadata_filters)

        top_docs = filtered[:self.k]

        # If parent-document retrieval is enabled, expand child chunks to parent chunks
        if self.parent_chunks is not None:
            expanded = []
            seen_parents = set()
            for doc in top_docs:
                parent_idx = doc.metadata.get("parent_idx")
                if parent_idx is not None and parent_idx < len(self.parent_chunks):
                    if parent_idx not in seen_parents:
                        seen_parents.add(parent_idx)
                        expanded.append(self.parent_chunks[parent_idx])
                else:
                    expanded.append(doc)
            return expanded

        return top_docs

    def get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Public retrieval interface for direct callers and legacy integrations."""
        return self._get_relevant_documents(query, **kwargs)


class PreloadedRetriever(BaseRetriever):
    """Wrapper that returns pre-fetched docs instead of re-retrieving."""
    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.docs


def load_vectorstore() -> FAISS:
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        message = (
            f"No vectorstore found at {VECTORSTORE_DIR}/. "
            "Run 'python ingest.py' first to ingest PDFs."
        )
        logger.error(message)
        raise VectorstoreNotFoundError(message)
    embeddings = get_default_embeddings()
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


def load_bm25() -> tuple[BM25Okapi, list[Document]] | None:
    """Load persisted BM25 index from disk. Returns None if not found."""
    bm25_path = Path(VECTORSTORE_DIR) / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        logger.info("BM25 index loaded from %s", bm25_path)
        return data["bm25"], data["docs"]
    return None


def load_parent_chunks() -> list[Document] | None:
    """Load parent chunks for parent-document retrieval."""
    parents_path = Path(VECTORSTORE_DIR) / "parent_chunks.pkl"
    if parents_path.exists():
        with open(parents_path, "rb") as f:
            return pickle.load(f)
    return None


def build_retriever(vectorstore: FAISS) -> HybridRetriever:
    # Try to load persisted BM25 index first
    bm25_data = load_bm25()
    if bm25_data:
        bm25, bm25_docs = bm25_data
    else:
        # Fallback: rebuild from docstore (backward compatibility)
        logger.warning("No persisted BM25 index found, rebuilding from docstore")
        docstore = vectorstore.docstore
        all_docs = [docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]
        tokenized = [tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized)
        bm25_docs = all_docs

    cross_encoder = _get_reranker()

    parent_chunks = None
    if ENABLE_PARENT_RETRIEVAL:
        parent_chunks = load_parent_chunks()
        if parent_chunks:
            logger.info("Parent-document retrieval enabled with %d parent chunks", len(parent_chunks))

    return HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        bm25_docs=bm25_docs,
        cross_encoder=cross_encoder,
        parent_chunks=parent_chunks,
    )


def build_qa_chain(retriever: HybridRetriever, streaming: bool = False) -> ConversationalRetrievalChain:
    llm = get_default_llm(temperature=0, streaming=streaming)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
