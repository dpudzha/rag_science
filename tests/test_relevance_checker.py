"""Tests for relevance_checker.py."""
import math
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


class TestCheckFromScore:
    """Tests for cross-encoder score as relevance proxy."""

    def test_high_logit_is_relevant(self):
        from relevance_checker import RelevanceChecker
        checker = RelevanceChecker(llm=MagicMock(), threshold=0.6)
        result = checker.check_from_score(5.0)  # sigmoid(5) ≈ 0.993
        assert result["is_relevant"] is True
        assert result["score"] == pytest.approx(1.0 / (1.0 + math.exp(-5.0)))
        assert result["suggestion"] is None

    def test_low_logit_is_irrelevant(self):
        from relevance_checker import RelevanceChecker
        checker = RelevanceChecker(llm=MagicMock(), threshold=0.6)
        result = checker.check_from_score(-5.0)  # sigmoid(-5) ≈ 0.007
        assert result["is_relevant"] is False
        assert result["score"] < 0.01

    def test_zero_logit_gives_half(self):
        from relevance_checker import RelevanceChecker
        checker = RelevanceChecker(llm=MagicMock(), threshold=0.6)
        result = checker.check_from_score(0.0)  # sigmoid(0) = 0.5
        assert result["score"] == pytest.approx(0.5)
        assert result["is_relevant"] is False  # 0.5 < 0.6 threshold

    def test_threshold_boundary(self):
        from relevance_checker import RelevanceChecker
        # sigmoid(0.405) ≈ 0.5999, sigmoid(0.41) ≈ 0.601
        checker = RelevanceChecker(llm=MagicMock(), threshold=0.6)
        result_below = checker.check_from_score(0.4)
        result_above = checker.check_from_score(0.5)
        assert result_below["is_relevant"] is False
        assert result_above["is_relevant"] is True


class TestRetrieveWithRelevanceCheck:
    def test_relevant_no_retry(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="relevant content", metadata={"source": "test.pdf"})
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SCORE: 0.9\nSUGGESTION: none")
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=False)

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
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=False)

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
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=False)

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
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=False)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker, max_retries=1)
        assert info["retry_count"] == 0  # No retry because no suggestion
        assert mock_retriever.invoke.call_count == 1

    def test_uses_cross_encoder_score_when_available(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        mock_retriever.last_top_rerank_score = 5.0  # sigmoid(5) ≈ 0.993
        mock_retriever.invoke.return_value = [
            Document(page_content="relevant content", metadata={"source": "test.pdf"})
        ]

        mock_llm = MagicMock()
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=True)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker)
        assert info["is_relevant"] is True
        assert info["score"] == pytest.approx(1.0 / (1.0 + math.exp(-5.0)))
        # LLM should NOT have been called
        mock_llm.invoke.assert_not_called()

    def test_falls_back_to_llm_when_no_score_attr(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock(spec=[])  # no last_top_rerank_score attr
        mock_retriever.invoke = MagicMock(return_value=[
            Document(page_content="content", metadata={"source": "test.pdf"})
        ])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SCORE: 0.9\nSUGGESTION: none")
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=True)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker)
        assert info["is_relevant"] is True
        mock_llm.invoke.assert_called_once()

    def test_cross_encoder_low_score_triggers_retry(self):
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check

        mock_retriever = MagicMock()
        # First call: low score, second call: high score
        mock_retriever.invoke.return_value = [
            Document(page_content="content", metadata={"source": "test.pdf"})
        ]
        type(mock_retriever).last_top_rerank_score = MagicMock(side_effect=[-5.0, 5.0])

        mock_llm = MagicMock()
        checker = RelevanceChecker(llm=mock_llm, use_cross_encoder_score=True)

        docs, info = retrieve_with_relevance_check(mock_retriever, "test", checker, max_retries=1)
        assert info["retry_count"] == 1
        assert mock_retriever.invoke.call_count == 2
        mock_llm.invoke.assert_not_called()
