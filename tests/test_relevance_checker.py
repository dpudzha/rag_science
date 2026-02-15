"""Tests for relevance_checker.py."""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class TestRelevanceChecker:
    def _make_checker(self, response_text, threshold=0.6):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response_text)
        from relevance_checker import RelevanceChecker
        return RelevanceChecker(llm=mock_llm, threshold=threshold)

    def _sample_docs(self):
        return [
            Document(page_content="Neural networks are powerful models.",
                     metadata={"source": "paper1.pdf"}),
        ]

    def test_relevant_score(self):
        checker = self._make_checker("SCORE: 0.8\nSUGGESTION: none")
        result = checker.check("What are neural networks?", self._sample_docs())
        assert result["score"] == 0.8
        assert result["is_relevant"] is True
        assert result["suggestion"] is None

    def test_irrelevant_score_with_suggestion(self):
        checker = self._make_checker("SCORE: 0.3\nSUGGESTION: neural network architectures and types")
        result = checker.check("What is quantum computing?", self._sample_docs())
        assert result["score"] == 0.3
        assert result["is_relevant"] is False
        assert result["suggestion"] is not None

    def test_empty_docs_returns_zero(self):
        checker = self._make_checker("SCORE: 0.0\nSUGGESTION: none")
        result = checker.check("test", [])
        assert result["score"] == 0.0
        assert result["is_relevant"] is False

    def test_score_clamped_to_range(self):
        checker = self._make_checker("SCORE: 1.5\nSUGGESTION: none")
        result = checker.check("test", self._sample_docs())
        assert result["score"] <= 1.0

    def test_fallback_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        from relevance_checker import RelevanceChecker
        checker = RelevanceChecker(llm=mock_llm)
        result = checker.check("test", self._sample_docs())
        assert result["is_relevant"] is True  # Assumes relevant on failure

    def test_custom_threshold(self):
        checker = self._make_checker("SCORE: 0.5\nSUGGESTION: none", threshold=0.4)
        result = checker.check("test", self._sample_docs())
        assert result["is_relevant"] is True

        checker2 = self._make_checker("SCORE: 0.5\nSUGGESTION: none", threshold=0.7)
        result2 = checker2.check("test", self._sample_docs())
        assert result2["is_relevant"] is False


class TestRetrieveWithRelevanceCheck:
    def test_relevant_no_retry(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="relevant content", metadata={"source": "test.pdf"})
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SCORE: 0.9\nSUGGESTION: none")
        checker = RelevanceChecker(llm=mock_llm)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker)
        assert len(docs) == 1
        assert info["retry_count"] == 0
        assert info["is_relevant"] is True

    def test_irrelevant_triggers_retry(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="some content", metadata={"source": "test.pdf"})
        ]

        # First check: low score with suggestion; second: high score
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="SCORE: 0.3\nSUGGESTION: better query"),
            AIMessage(content="SCORE: 0.8\nSUGGESTION: none"),
        ]
        checker = RelevanceChecker(llm=mock_llm)

        docs, info = retrieve_with_relevance_check(mock_retriever, "bad query", checker, max_retries=1)
        assert info["retry_count"] == 1
        assert info["final_query"] == "better query"

    def test_max_retries_respected(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="irrelevant", metadata={"source": "test.pdf"})
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SCORE: 0.2\nSUGGESTION: try again")
        checker = RelevanceChecker(llm=mock_llm)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker, max_retries=1)
        assert info["retry_count"] == 1
        # Should have called retriever twice (original + 1 retry)
        assert mock_retriever.invoke.call_count == 2

    def test_no_suggestion_stops_retry(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="irrelevant", metadata={"source": "test.pdf"})
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SCORE: 0.2\nSUGGESTION: none")
        checker = RelevanceChecker(llm=mock_llm)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker, max_retries=1)
        assert info["retry_count"] == 0  # No retry because no suggestion
        assert mock_retriever.invoke.call_count == 1
