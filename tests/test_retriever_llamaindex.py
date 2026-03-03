"""Tests for retriever_llamaindex.py and the auto-dispatch factory."""
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(text: str, metadata: dict | None = None):
    """Create a minimal LlamaIndex-like TextNode stub (no 'node' attribute)."""
    node = MagicMock(spec=["get_content", "metadata"])
    node.get_content.return_value = text
    node.metadata = metadata or {}
    return node


def _make_node_with_score(text: str, score: float, metadata: dict | None = None):
    """Create a NodeWithScore-like stub (has .node and .score)."""
    stub = MagicMock()
    stub.score = score
    stub.node = _make_node(text, metadata)
    stub.get_content.return_value = text
    stub.metadata = metadata or {}
    return stub


# ---------------------------------------------------------------------------
# TestNodeToDocument
# ---------------------------------------------------------------------------

class TestNodeToDocument:
    def test_plain_node(self):
        from retriever_llamaindex import _node_to_document
        node = _make_node("hello world", {"source": "paper.pdf", "page": 1})
        doc = _node_to_document(node)
        assert isinstance(doc, Document)
        assert doc.page_content == "hello world"
        assert doc.metadata["source"] == "paper.pdf"
        assert doc.metadata["page"] == 1

    def test_node_with_score_wrapper(self):
        from retriever_llamaindex import _node_to_document
        node_ws = _make_node_with_score("wrapped text", 0.9, {"source": "paper.pdf"})
        doc = _node_to_document(node_ws)
        assert doc.page_content == "wrapped text"
        assert doc.metadata["source"] == "paper.pdf"

    def test_empty_metadata(self):
        from retriever_llamaindex import _node_to_document
        node = _make_node("text only")
        doc = _node_to_document(node)
        assert doc.metadata == {}


# ---------------------------------------------------------------------------
# TestLastTopRerankScoreStored
# ---------------------------------------------------------------------------

class TestLastTopRerankScoreStored:
    """Verify last_top_rerank_score stores the raw cross-encoder logit score."""

    def test_raw_reranker_score_stored(self):
        """Cross-encoder predict() scores are stored as-is (logit-compatible)."""
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        retriever = LlamaIndexHybridRetriever(index=mock_index, bm25_nodes=[])

        reranked_node = _make_node_with_score("text", score=2.5)

        with (
            patch("retriever_llamaindex.VectorIndexRetriever"),
            patch("retriever_llamaindex.BM25Retriever"),
            patch("retriever_llamaindex.QueryFusionRetriever") as mock_fusion,
            patch("retriever_llamaindex._get_llamaindex_reranker") as mock_reranker_fn,
        ):
            mock_fusion_instance = MagicMock()
            mock_fusion_instance.retrieve.return_value = [reranked_node]
            mock_fusion.return_value = mock_fusion_instance

            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [2.5]
            mock_reranker_fn.return_value = mock_reranker

            retriever._get_relevant_documents("test query")

        assert retriever.last_top_rerank_score == pytest.approx(2.5)

    def test_empty_results_sets_zero(self):
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        retriever = LlamaIndexHybridRetriever(index=mock_index, bm25_nodes=[])

        with (
            patch("retriever_llamaindex.VectorIndexRetriever"),
            patch("retriever_llamaindex.BM25Retriever"),
            patch("retriever_llamaindex.QueryFusionRetriever") as mock_fusion,
            patch("retriever_llamaindex._get_llamaindex_reranker"),
        ):
            mock_fusion_instance = MagicMock()
            mock_fusion_instance.retrieve.return_value = []
            mock_fusion.return_value = mock_fusion_instance

            retriever._get_relevant_documents("test query")

        assert retriever.last_top_rerank_score == 0.0

    def test_reranker_called_with_query_text_pairs(self):
        """predict() should receive [[query, node_text], ...] pairs."""
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        retriever = LlamaIndexHybridRetriever(index=mock_index, bm25_nodes=[])
        node = _make_node("chunk content")

        with (
            patch("retriever_llamaindex.VectorIndexRetriever"),
            patch("retriever_llamaindex.BM25Retriever"),
            patch("retriever_llamaindex.QueryFusionRetriever") as mock_fusion,
            patch("retriever_llamaindex._get_llamaindex_reranker") as mock_reranker_fn,
        ):
            mock_fusion_instance = MagicMock()
            mock_fusion_instance.retrieve.return_value = [node]
            mock_fusion.return_value = mock_fusion_instance

            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [1.0]
            mock_reranker_fn.return_value = mock_reranker

            retriever._get_relevant_documents("my query")

        pairs = mock_reranker.predict.call_args[0][0]
        assert pairs == [["my query", "chunk content"]]


# ---------------------------------------------------------------------------
# TestLlamaIndexHybridRetrieverForRequest
# ---------------------------------------------------------------------------

class TestLlamaIndexHybridRetrieverForRequest:
    def test_for_request_shares_heavy_internals(self):
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        mock_nodes = [MagicMock()]
        original = LlamaIndexHybridRetriever(
            index=mock_index,
            bm25_nodes=mock_nodes,
            bm25_weight=0.4,
            dense_weight=0.6,
        )
        original.metadata_filters = {"author": "Smith"}
        original.last_top_rerank_score = 2.5

        copy = original.for_request()

        assert copy._index is mock_index
        assert copy._bm25_nodes is mock_nodes

    def test_for_request_resets_mutable_state(self):
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        original = LlamaIndexHybridRetriever(index=mock_index, bm25_nodes=[])
        original.metadata_filters = {"author": "Smith"}
        original.last_top_rerank_score = 2.5

        copy = original.for_request()

        assert copy.metadata_filters is None
        assert copy.last_top_rerank_score == 0.0

    def test_for_request_preserves_weights(self):
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        original = LlamaIndexHybridRetriever(
            index=mock_index, bm25_nodes=[], bm25_weight=0.4, dense_weight=0.6
        )
        copy = original.for_request()

        assert copy.bm25_weight == 0.4
        assert copy.dense_weight == 0.6

    def test_mutations_dont_affect_original(self):
        from retriever_llamaindex import LlamaIndexHybridRetriever

        mock_index = MagicMock()
        original = LlamaIndexHybridRetriever(index=mock_index, bm25_nodes=[])
        copy = original.for_request()

        copy.bm25_weight = 0.9
        copy.metadata_filters = {"new": "filter"}

        assert original.bm25_weight != 0.9
        assert original.metadata_filters is None


# ---------------------------------------------------------------------------
# TestLlamaIndexVectorstoreNotFound
# ---------------------------------------------------------------------------

class TestLlamaIndexVectorstoreNotFound:
    def test_is_subclass_of_file_not_found_error(self):
        from retriever_llamaindex import LlamaIndexVectorstoreNotFoundError
        assert issubclass(LlamaIndexVectorstoreNotFoundError, FileNotFoundError)

    def test_raised_when_index_missing(self, tmp_path):
        with patch("retriever_llamaindex.LLAMAINDEX_VECTORSTORE_DIR", str(tmp_path)):
            from retriever_llamaindex import load_llamaindex_vectorstore, LlamaIndexVectorstoreNotFoundError
            with pytest.raises(LlamaIndexVectorstoreNotFoundError):
                load_llamaindex_vectorstore()

    def test_error_message_contains_directory(self, tmp_path):
        with patch("retriever_llamaindex.LLAMAINDEX_VECTORSTORE_DIR", str(tmp_path)):
            from retriever_llamaindex import load_llamaindex_vectorstore, LlamaIndexVectorstoreNotFoundError
            with pytest.raises(LlamaIndexVectorstoreNotFoundError, match=str(tmp_path)):
                load_llamaindex_vectorstore()


# ---------------------------------------------------------------------------
# TestLoadLlamaIndexBm25Nodes
# ---------------------------------------------------------------------------

class TestLoadLlamaIndexBm25Nodes:
    def test_returns_empty_when_missing(self, tmp_path):
        with patch("retriever_llamaindex.LLAMAINDEX_VECTORSTORE_DIR", str(tmp_path)):
            from retriever_llamaindex import load_llamaindex_bm25_nodes
            result = load_llamaindex_bm25_nodes()
        assert result == []

    def test_loads_nodes_from_pickle(self, tmp_path):
        # Use simple picklable objects rather than MagicMock
        nodes = ["node1", "node2"]
        pkl_path = tmp_path / "bm25_nodes.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(nodes, f)

        with patch("retriever_llamaindex.LLAMAINDEX_VECTORSTORE_DIR", str(tmp_path)):
            from retriever_llamaindex import load_llamaindex_bm25_nodes
            result = load_llamaindex_bm25_nodes()

        assert len(result) == 2

    def test_returns_empty_on_corrupt_file(self, tmp_path):
        pkl_path = tmp_path / "bm25_nodes.pkl"
        pkl_path.write_bytes(b"not a pickle")

        with patch("retriever_llamaindex.LLAMAINDEX_VECTORSTORE_DIR", str(tmp_path)):
            from retriever_llamaindex import load_llamaindex_bm25_nodes
            result = load_llamaindex_bm25_nodes()

        assert result == []


# ---------------------------------------------------------------------------
# TestRerankerDelegation
# ---------------------------------------------------------------------------

class TestRerankerDelegation:
    """Verify _get_llamaindex_reranker delegates to retriever._get_reranker."""

    def test_delegates_to_retriever_get_reranker(self):
        from retriever_llamaindex import _get_llamaindex_reranker
        mock_reranker = MagicMock()
        with patch("retriever._get_reranker", return_value=mock_reranker):
            result = _get_llamaindex_reranker()
        assert result is mock_reranker

    def test_reset_calls_retriever_reset_reranker(self):
        from retriever_llamaindex import reset_llamaindex_reranker
        with patch("retriever.reset_reranker") as mock_reset:
            reset_llamaindex_reranker()
        mock_reset.assert_called_once()


# ---------------------------------------------------------------------------
# TestBuildRetrieverAutoDispatch
# ---------------------------------------------------------------------------

class TestBuildRetrieverAutoDispatch:
    def test_langchain_backend_uses_build_retriever(self):
        with patch("config.RETRIEVER_BACKEND", "langchain"):
            with (
                patch("retriever.load_vectorstore") as mock_load_vs,
                patch("retriever.build_retriever") as mock_build,
            ):
                mock_vs = MagicMock()
                mock_load_vs.return_value = mock_vs
                mock_build.return_value = MagicMock()

                from retriever import build_retriever_auto
                result = build_retriever_auto()

                mock_load_vs.assert_called_once()
                mock_build.assert_called_once_with(mock_vs)

    def test_llamaindex_backend_uses_llamaindex_builder(self):
        with patch("config.RETRIEVER_BACKEND", "llamaindex"):
            mock_retriever = MagicMock()
            with patch("retriever_llamaindex.build_llamaindex_retriever", return_value=mock_retriever):
                from retriever import build_retriever_auto
                result = build_retriever_auto()

        assert result is mock_retriever


# ---------------------------------------------------------------------------
# TestLlamaIndexRetrieverAdapterInBuildQaChain
# ---------------------------------------------------------------------------

class TestLlamaIndexRetrieverAdapterInBuildQaChain:
    def test_adapter_wraps_non_base_retriever(self):
        from retriever import _LlamaIndexRetrieverAdapter, build_qa_chain

        mock_li_retriever = MagicMock()
        mock_li_retriever.invoke.return_value = []

        with patch("retriever.ConversationalRetrievalChain") as mock_chain_cls:
            mock_chain_cls.from_llm.return_value = MagicMock()
            with patch("retriever.get_default_llm", return_value=MagicMock()):
                build_qa_chain(mock_li_retriever)

        call_kwargs = mock_chain_cls.from_llm.call_args
        passed_retriever = call_kwargs[1].get("retriever") or call_kwargs[0][1]
        assert isinstance(passed_retriever, _LlamaIndexRetrieverAdapter)

    def test_base_retriever_passed_through_unchanged(self):
        from retriever import build_qa_chain
        from langchain_core.retrievers import BaseRetriever

        mock_retriever = MagicMock(spec=BaseRetriever)

        with patch("retriever.ConversationalRetrievalChain") as mock_chain_cls:
            mock_chain_cls.from_llm.return_value = MagicMock()
            with patch("retriever.get_default_llm", return_value=MagicMock()):
                build_qa_chain(mock_retriever)

        call_kwargs = mock_chain_cls.from_llm.call_args
        passed_retriever = call_kwargs[1].get("retriever") or call_kwargs[0][1]
        assert passed_retriever is mock_retriever

    def test_adapter_delegates_to_inner_invoke(self):
        from retriever import _LlamaIndexRetrieverAdapter

        docs = [Document(page_content="result", metadata={})]
        mock_inner = MagicMock()
        mock_inner.invoke.return_value = docs

        adapter = _LlamaIndexRetrieverAdapter(inner=mock_inner)
        result = adapter._get_relevant_documents("my query")

        mock_inner.invoke.assert_called_once_with("my query")
        assert result == docs
