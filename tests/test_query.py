"""Tests for query.py: hybrid scoring, reranking, BM25 loading."""
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class TestTokenize:
    def test_basic(self):
        from query import tokenize
        assert tokenize("Hello World!") == ["hello", "world"]

    def test_alphanumeric(self):
        from query import tokenize
        assert "42" in tokenize("The answer is 42.")


class TestLoadBm25:
    def test_load_from_pickle(self, tmp_vectorstore, sample_documents):
        from ingest import tokenize
        tokenized = [tokenize(d.page_content) for d in sample_documents]
        bm25 = BM25Okapi(tokenized)

        pkl_path = tmp_vectorstore / "bm25_index.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": sample_documents}, f)

        with patch("query.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from query import load_bm25
            result = load_bm25()

        assert result is not None
        loaded_bm25, loaded_docs = result
        assert len(loaded_docs) == len(sample_documents)

    def test_returns_none_when_missing(self, tmp_vectorstore):
        with patch("query.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from query import load_bm25
            assert load_bm25() is None


class TestLoadParentChunks:
    def test_load_when_exists(self, tmp_vectorstore, sample_documents):
        pkl_path = tmp_vectorstore / "parent_chunks.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(sample_documents, f)

        with patch("query.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from query import load_parent_chunks
            result = load_parent_chunks()
        assert result is not None
        assert len(result) == len(sample_documents)

    def test_returns_none_when_missing(self, tmp_vectorstore):
        with patch("query.VECTORSTORE_DIR", str(tmp_vectorstore)):
            from query import load_parent_chunks
            assert load_parent_chunks() is None


class TestCrossEncoderSingleton:
    def test_lazy_loaded_once(self):
        import query
        query._cross_encoder = None  # Reset
        with patch("query.CrossEncoder") as MockCE:
            mock_instance = MagicMock()
            MockCE.return_value = mock_instance

            result1 = query._get_cross_encoder()
            result2 = query._get_cross_encoder()

            assert result1 is result2
            MockCE.assert_called_once()

        query._cross_encoder = None  # Clean up


class TestHybridRetrieverDocKey:
    def test_doc_key_format(self):
        from query import HybridRetriever
        doc = Document(page_content="Hello world " * 50, metadata={"source": "test.pdf", "page": 1})
        key = HybridRetriever._doc_key(doc)
        assert key.startswith("test.pdf:1:")
        assert len(key.split(":", 2)[2]) <= 200


class TestRelevanceCheckerIntegration:
    def test_get_relevance_checker_when_enabled(self):
        with patch("query.RELEVANCE_CHECK_ENABLED", True), \
             patch("query.RELEVANCE_THRESHOLD", 0.5), \
             patch("relevance_checker.ChatOllama"):
            from query import _get_relevance_checker
            checker = _get_relevance_checker()
            assert checker is not None
            assert checker.threshold == 0.5

    def test_get_relevance_checker_when_disabled(self):
        with patch("query.RELEVANCE_CHECK_ENABLED", False):
            from query import _get_relevance_checker
            checker = _get_relevance_checker()
            assert checker is None
