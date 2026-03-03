"""LlamaIndex-based hybrid retrieval backend (drop-in for HybridRetriever)."""
import logging
import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import (
    LLAMAINDEX_VECTORSTORE_DIR,
    TOP_K,
    TOP_K_CANDIDATES,
    BM25_WEIGHT,
    DENSE_WEIGHT,
)

logger = logging.getLogger(__name__)

# Module-level imports for testability (all patchable via patch("retriever_llamaindex.X")).
# These are set to None when llama-index is not installed; the code paths that use
# them will raise naturally at call time if the library is missing.
try:
    from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.llms.mock import MockLLM
except ImportError:
    VectorIndexRetriever = None  # type: ignore[assignment,misc]
    QueryFusionRetriever = None  # type: ignore[assignment,misc]
    BM25Retriever = None  # type: ignore[assignment,misc]
    FaissVectorStore = None  # type: ignore[assignment,misc]
    StorageContext = None  # type: ignore[assignment,misc]
    load_index_from_storage = None  # type: ignore[assignment,misc]
    MockLLM = None  # type: ignore[assignment,misc]


class LlamaIndexVectorstoreNotFoundError(FileNotFoundError):
    """Raised when the LlamaIndex vectorstore is missing from disk."""


def _node_to_document(node) -> Document:
    """Convert a LlamaIndex NodeWithScore or TextNode to a LangChain Document."""
    if hasattr(node, "node"):
        # NodeWithScore wrapper
        actual_node = node.node
    else:
        actual_node = node
    return Document(
        page_content=actual_node.get_content(),
        metadata=dict(actual_node.metadata),
    )


def _node_text(node) -> str:
    """Extract text content from a LlamaIndex node or NodeWithScore."""
    if hasattr(node, "node"):
        return node.node.get_content()
    return node.get_content()


def _get_llamaindex_reranker():
    """Return the shared cross-encoder reranker singleton from retriever.py.

    Reuses the existing singleton so the model is loaded only once regardless
    of which backend is active, and avoids the Python-3.13-incompatible
    llama-index-postprocessor-sentence-transformer-rerank package.
    """
    from retriever import _get_reranker
    return _get_reranker()


def reset_llamaindex_reranker() -> None:
    """Reset the shared cross-encoder reranker singleton."""
    from retriever import reset_reranker
    reset_reranker()


class LlamaIndexHybridRetriever:
    """Drop-in replacement for HybridRetriever using LlamaIndex components.

    Exposes the same instance attributes (bm25_weight, dense_weight,
    metadata_filters, k, k_candidates, last_top_rerank_score) so that
    query.py's preprocess_query() can mutate them without changes.

    Reranking delegates to the shared cross-encoder singleton in retriever.py
    via predict([[query, text], ...]) — the same logit-compatible interface used
    by HybridRetriever, so relevance_checker.check_from_score() (sigmoid) works
    identically for both backends.
    """

    def __init__(
        self,
        index,
        bm25_nodes: list,
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
        k: int = TOP_K,
        k_candidates: int = TOP_K_CANDIDATES,
        metadata_filters: dict | None = None,
        last_top_rerank_score: float = 0.0,
    ):
        self._index = index
        self._bm25_nodes = bm25_nodes
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.k = k
        self.k_candidates = k_candidates
        self.metadata_filters = metadata_filters
        self.last_top_rerank_score = last_top_rerank_score

    def for_request(self) -> "LlamaIndexHybridRetriever":
        """Create a lightweight per-request copy sharing heavy internals."""
        return LlamaIndexHybridRetriever(
            index=self._index,
            bm25_nodes=self._bm25_nodes,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            k=self.k,
            k_candidates=self.k_candidates,
            metadata_filters=None,
            last_top_rerank_score=0.0,
        )

    def invoke(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)

    def get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Dense retrieval via FAISS VectorIndex
        dense_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self.k_candidates,
        )

        # BM25 keyword retrieval
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self._bm25_nodes,
            similarity_top_k=self.k_candidates,
        )

        # Reciprocal Rank Fusion via QueryFusionRetriever.
        # num_queries=1 skips LLM query generation so the LLM is never invoked.
        # We still pass MockLLM explicitly to prevent QueryFusionRetriever.__init__
        # from resolving Settings.llm (which would attempt to import llama-index-llms-openai).
        fusion_retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            retriever_weights=[self.dense_weight, self.bm25_weight],
            mode="reciprocal_rerank",
            num_queries=1,
            use_async=False,
            llm=MockLLM(),
        )

        fused_nodes = fusion_retriever.retrieve(query)

        if not fused_nodes:
            self.last_top_rerank_score = 0.0
            return []

        # Cross-encoder reranking — reuse the shared singleton from retriever.py.
        # predict() returns raw logit scores compatible with relevance_checker
        # (which applies sigmoid internally via check_from_score).
        reranker = _get_llamaindex_reranker()
        pairs = [[query, _node_text(n)] for n in fused_nodes]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
        reranked = sorted(zip(fused_nodes, rerank_scores), key=lambda x: x[1], reverse=True)

        self.last_top_rerank_score = float(reranked[0][1]) if reranked else 0.0

        # Convert to LangChain Documents
        docs = [_node_to_document(node) for node, _ in reranked]

        # Apply metadata filters BEFORE final k-slice
        if self.metadata_filters:
            from metadata_extractor import MetadataFilterApplier
            docs = MetadataFilterApplier.apply(docs, self.metadata_filters)

        return docs[:self.k]


def load_llamaindex_vectorstore():
    """Load the LlamaIndex VectorStoreIndex from disk.

    Raises LlamaIndexVectorstoreNotFoundError if the index files are missing.
    """
    index_path = Path(LLAMAINDEX_VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        message = (
            f"No LlamaIndex vectorstore found at {LLAMAINDEX_VECTORSTORE_DIR}/. "
            "Run 'python ingest_llamaindex.py' first to ingest documents."
        )
        logger.error(message)
        raise LlamaIndexVectorstoreNotFoundError(message)

    from utils import get_llamaindex_embeddings

    embed_model = get_llamaindex_embeddings()
    vector_store = FaissVectorStore.from_persist_dir(LLAMAINDEX_VECTORSTORE_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=LLAMAINDEX_VECTORSTORE_DIR,
    )
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    logger.info("LlamaIndex vectorstore loaded from %s", LLAMAINDEX_VECTORSTORE_DIR)
    return index


def load_llamaindex_bm25_nodes() -> list:
    """Load persisted BM25 nodes from disk. Returns [] if not found."""
    nodes_path = Path(LLAMAINDEX_VECTORSTORE_DIR) / "bm25_nodes.pkl"
    if nodes_path.exists():
        try:
            with open(nodes_path, "rb") as f:
                nodes = pickle.load(f)
            logger.info("LlamaIndex BM25 nodes loaded from %s (%d nodes)", nodes_path, len(nodes))
            return nodes
        except Exception as e:
            logger.warning("Failed to load LlamaIndex BM25 nodes: %s", e)
    return []


def build_llamaindex_retriever() -> LlamaIndexHybridRetriever:
    """Top-level factory: load vectorstore + BM25 nodes and build retriever."""
    index = load_llamaindex_vectorstore()
    bm25_nodes = load_llamaindex_bm25_nodes()
    return LlamaIndexHybridRetriever(
        index=index,
        bm25_nodes=bm25_nodes,
    )
