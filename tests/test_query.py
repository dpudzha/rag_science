"""Tests for query.py and retriever.py: hybrid scoring, reranking, BM25 loading."""
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class TestTokenize:
    def test_basic(self):
        from utils import tokenize
        assert tokenize("Hello World!") == ["hello", "world"]

    def test_alphanumeric(self):
        from utils import tokenize
        assert "42" in tokenize("The answer is 42.")


class TestLoadBm25:
    def test_load_from_pickle(self, tmp_vectorstore, sample_documents):
        from ingest import tokenize
        tokenized = [tokenize(d.page_content) for d in sample_documents]
        bm25 = BM25Okapi(tokenized)

        pkl_path = tmp_vectorstore / "bm25_index.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": sample_documents}, f)

        with patch("retriever.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from retriever import load_bm25
            result = load_bm25()

        assert result is not None
        loaded_bm25, loaded_docs = result
        assert len(loaded_docs) == len(sample_documents)

    def test_returns_none_when_missing(self, tmp_vectorstore):
        with patch("retriever.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from retriever import load_bm25
            assert load_bm25() is None


class TestLoadVectorstore:
    def test_raises_when_missing(self, tmp_vectorstore):
        with patch("retriever.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from retriever import load_vectorstore, VectorstoreNotFoundError
            with pytest.raises(VectorstoreNotFoundError):
                load_vectorstore()


class TestLoadParentChunks:
    def test_load_when_exists(self, tmp_vectorstore, sample_documents):
        pkl_path = tmp_vectorstore / "parent_chunks.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(sample_documents, f)

        with patch("retriever.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from retriever import load_parent_chunks
            result = load_parent_chunks()
        assert result is not None
        assert len(result) == len(sample_documents)

    def test_returns_none_when_missing(self, tmp_vectorstore):
        with patch("retriever.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from retriever import load_parent_chunks
            assert load_parent_chunks() is None


class TestCrossEncoderSingleton:
    def test_lazy_loaded_once(self):
        import retriever
        retriever._cross_encoder = None  # Reset
        with patch("retriever.CrossEncoder") as MockCE:
            mock_instance = MagicMock()
            MockCE.return_value = mock_instance

            result1 = retriever._get_cross_encoder()
            result2 = retriever._get_cross_encoder()

            assert result1 is result2
            MockCE.assert_called_once()

        retriever._cross_encoder = None  # Clean up


class TestHybridRetrieverDocKey:
    def test_doc_key_format(self):
        from retriever import HybridRetriever
        doc = Document(page_content="Hello world " * 50, metadata={"source": "test.pdf", "page": 1})
        key = HybridRetriever._doc_key(doc)
        assert key.startswith("test.pdf:1:")
        assert len(key.split(":", 2)[2]) <= 200

    def test_public_get_relevant_documents_delegates(self):
        from retriever import HybridRetriever
        retriever = MagicMock(spec=HybridRetriever)
        expected = [Document(page_content="test", metadata={})]
        retriever._get_relevant_documents.return_value = expected

        result = HybridRetriever.get_relevant_documents(retriever, "query")
        assert result == expected
        retriever._get_relevant_documents.assert_called_once_with("query")


class TestHybridRetrieverRerankerScore:
    def test_last_top_rerank_score_set_after_retrieval(self):
        from retriever import HybridRetriever
        from langchain_community.vectorstores import FAISS
        from sentence_transformers import CrossEncoder
        import numpy as np

        mock_vs = MagicMock(spec=FAISS)
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="doc1", metadata={"source": "a.pdf", "page": 1}), 0.9),
            (Document(page_content="doc2", metadata={"source": "b.pdf", "page": 1}), 0.8),
        ]

        mock_bm25 = MagicMock(spec=BM25Okapi)
        mock_bm25.get_scores.return_value = np.array([0.5, 0.3])

        bm25_docs = [
            Document(page_content="doc1", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="doc2", metadata={"source": "b.pdf", "page": 1}),
        ]

        mock_ce = MagicMock(spec=CrossEncoder)
        mock_ce.predict.return_value = np.array([3.5, -1.0])

        retriever = HybridRetriever(
            vectorstore=mock_vs, bm25=mock_bm25, bm25_docs=bm25_docs,
            cross_encoder=mock_ce, k=2, k_candidates=10,
        )
        retriever._get_relevant_documents("test query")
        assert retriever.last_top_rerank_score == pytest.approx(3.5)

    def test_last_top_rerank_score_zero_when_no_candidates(self):
        from retriever import HybridRetriever
        from langchain_community.vectorstores import FAISS
        from sentence_transformers import CrossEncoder

        mock_vs = MagicMock(spec=FAISS)
        mock_vs.similarity_search_with_score.return_value = []

        mock_bm25 = MagicMock(spec=BM25Okapi)
        mock_bm25.get_scores.return_value = []

        mock_ce = MagicMock(spec=CrossEncoder)

        retriever = HybridRetriever(
            vectorstore=mock_vs, bm25=mock_bm25, bm25_docs=[],
            cross_encoder=mock_ce, k=2, k_candidates=10,
        )
        retriever._get_relevant_documents("test query")
        assert retriever.last_top_rerank_score == 0.0


class TestPreloadedRetriever:
    def test_returns_cached_docs(self):
        from retriever import PreloadedRetriever
        docs = [
            Document(page_content="doc1", metadata={"source": "a.pdf"}),
            Document(page_content="doc2", metadata={"source": "b.pdf"}),
        ]
        retriever = PreloadedRetriever(docs=docs)
        result = retriever.invoke("any query")
        assert result == docs

    def test_ignores_query_text(self):
        from retriever import PreloadedRetriever
        docs = [Document(page_content="fixed", metadata={})]
        retriever = PreloadedRetriever(docs=docs)
        assert retriever.invoke("query A") == retriever.invoke("query B")


class TestRelevanceCheckerIntegration:
    def test_get_relevance_checker_when_enabled(self):
        with patch("query.RELEVANCE_CHECK_ENABLED", True), \
             patch("query.RELEVANCE_THRESHOLD", 0.5), \
             patch("utils.get_default_llm"):
            from query import _get_relevance_checker
            checker = _get_relevance_checker()
            assert checker is not None
            assert checker.threshold == 0.5

    def test_get_relevance_checker_when_disabled(self):
        with patch("query.RELEVANCE_CHECK_ENABLED", False):
            from query import _get_relevance_checker
            checker = _get_relevance_checker()
            assert checker is None
