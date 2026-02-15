"""Tests for query_resolver.py: follow-up query resolution with chat history."""
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


class TestQueryResolver:
    def _make_resolver(self, llm_response: str):
        from query_resolver import QueryResolver
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=llm_response)
        return QueryResolver(llm=mock_llm), mock_llm

    def test_returns_original_when_no_history(self):
        resolver, mock_llm = self._make_resolver("")
        result = resolver.resolve("What is the current?", [])
        assert result == "What is the current?"
        mock_llm.invoke.assert_not_called()

    def test_resolves_follow_up_with_history(self):
        resolved = "What is the total current drawn by the whole chip?"
        resolver, mock_llm = self._make_resolver(resolved)

        history = [
            ("What is the current drawn by the chip?", "The current per pixel is ~4.5 uA.")
        ]
        result = resolver.resolve("But I am asking by whole chip", history)
        assert result == resolved
        mock_llm.invoke.assert_called_once()

    def test_prompt_includes_history_and_question(self):
        resolver, mock_llm = self._make_resolver("resolved question")

        history = [("First question", "First answer")]
        resolver.resolve("Follow up", history)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "First question" in prompt_text
        assert "First answer" in prompt_text
        assert "Follow up" in prompt_text

    def test_truncates_history_to_last_3(self):
        resolver, mock_llm = self._make_resolver("resolved")

        history = [(f"Q{i}", f"A{i}") for i in range(5)]
        resolver.resolve("Follow up", history)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        # Only last 3 turns should be in the prompt
        assert "Q0" not in prompt_text
        assert "Q1" not in prompt_text
        assert "Q2" in prompt_text
        assert "Q3" in prompt_text
        assert "Q4" in prompt_text

    def test_falls_back_on_empty_response(self):
        resolver, _ = self._make_resolver("")
        history = [("Q", "A")]
        result = resolver.resolve("Follow up", history)
        assert result == "Follow up"

    def test_falls_back_on_llm_error(self):
        from query_resolver import QueryResolver
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        resolver = QueryResolver(llm=mock_llm)

        result = resolver.resolve("Follow up", [("Q", "A")])
        assert result == "Follow up"


class TestQueryResolutionConfig:
    def test_disabled_returns_none(self):
        with patch("query.QUERY_RESOLUTION_ENABLED", False):
            from query import _get_query_resolver
            assert _get_query_resolver() is None

    @patch("query.QUERY_RESOLUTION_ENABLED", True)
    @patch("query_resolver.ChatOllama")
    def test_enabled_returns_resolver(self, mock_ollama):
        from query import _get_query_resolver
        resolver = _get_query_resolver()
        assert resolver is not None
