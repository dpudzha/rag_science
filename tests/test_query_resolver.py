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

    def test_resolves_follow_up_with_pronoun(self):
        resolved = "What is the total current drawn by the whole chip?"
        resolver, mock_llm = self._make_resolver(resolved)

        history = [
            ("What is the current drawn by the chip?", "The current per pixel is ~4.5 uA.")
        ]
        result = resolver.resolve("What is the total current for it?", history)
        assert result == resolved
        mock_llm.invoke.assert_called_once()

    def test_standalone_question_skips_llm(self):
        resolver, mock_llm = self._make_resolver("resolved question")

        history = [("What is optoboard?", "Optoboard hosts ASICs.")]
        result = resolver.resolve("What is lpgbt?", history)
        assert result == "What is lpgbt?"
        mock_llm.invoke.assert_not_called()

    def test_standalone_question_with_domain_context_skips_llm(self):
        resolver, mock_llm = self._make_resolver("resolved question")

        history = [
            ("What is optoboard?", "Optoboard hosts ASICs."),
            ("what is lpgbt", "LpGBT multiplexes uplink data."),
        ]
        result = resolver.resolve("How 1 MHz requirement is related to per pixel rate", history)
        assert result == "How 1 MHz requirement is related to per pixel rate"
        mock_llm.invoke.assert_not_called()

    def test_prompt_includes_history_and_question(self):
        resolver, mock_llm = self._make_resolver("resolved question")

        history = [("First question", "First answer")]
        resolver.resolve("What else does it do?", history)

        call_args = mock_llm.invoke.call_args[0][0]
        system_text = call_args[0].content
        user_text = call_args[1].content
        assert "First question" in system_text
        assert "First answer" in system_text
        assert "What else does it do?" == user_text

    def test_truncates_history_to_last_3(self):
        resolver, mock_llm = self._make_resolver("resolved")

        history = [(f"Q{i}", f"A{i}") for i in range(5)]
        resolver.resolve("What else does it do?", history)

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
        result = resolver.resolve("How does this work?", history)
        assert result == "How does this work?"

    def test_falls_back_on_llm_error(self):
        from query_resolver import QueryResolver
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        resolver = QueryResolver(llm=mock_llm)

        result = resolver.resolve("How does it work?", [("Q", "A")])
        assert result == "How does it work?"


class TestNeedsResolution:
    def test_pronoun_it(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("What does it do?") is True

    def test_pronoun_this(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("How does this work?") is True

    def test_demonstrative_that(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("Can you explain that?") is True

    def test_plural_they(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("How do they communicate?") is True

    def test_standalone_technical_question(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("How 1 MHz requirement is related to per pixel rate") is False

    def test_standalone_definition_question(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("What is lpgbt?") is False

    def test_standalone_new_topic(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("What is the main purpose of optoboard?") is False

    def test_the_same_triggers_resolution(self):
        from query_resolver import _needs_resolution
        assert _needs_resolution("Does the same apply to GBCR?") is True


class TestQueryResolutionConfig:
    def test_disabled_returns_none(self):
        with patch("query.QUERY_RESOLUTION_ENABLED", False):
            from query import _get_query_resolver
            assert _get_query_resolver() is None

    @patch("query.QUERY_RESOLUTION_ENABLED", True)
    @patch("utils.get_default_llm")
    def test_enabled_returns_resolver(self, mock_ollama):
        from query import _get_query_resolver
        resolver = _get_query_resolver()
        assert resolver is not None
