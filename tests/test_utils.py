"""Tests for utils.py: tokenize and get_default_llm."""
from unittest.mock import patch, MagicMock

from utils import tokenize


class TestTokenize:
    def test_basic_words(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_numbers_included(self):
        assert tokenize("GPT-4 has 175B params") == ["gpt", "4", "has", "175b", "params"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_punctuation_stripped(self):
        assert tokenize("what's up? nothing!") == ["what", "s", "up", "nothing"]

    def test_mixed_case(self):
        assert tokenize("UPPER lower MiXeD") == ["upper", "lower", "mixed"]


class TestGetDefaultLLM:
    @patch("utils.ChatOllama")
    def test_returns_chat_ollama_instance(self, mock_cls):
        from utils import get_default_llm
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        result = get_default_llm()
        assert result is mock_instance
        mock_cls.assert_called_once()

    @patch("utils.ChatOllama")
    def test_default_temperature_zero(self, mock_cls):
        from utils import get_default_llm
        get_default_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0

    @patch("utils.ChatOllama")
    def test_custom_temperature(self, mock_cls):
        from utils import get_default_llm
        get_default_llm(temperature=0.7)
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.7
