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
    @patch("utils.config")
    def test_ollama_backend(self, mock_config):
        mock_config.LLM_BACKEND = "ollama"
        mock_config.LLM_MODEL = "gemma3:12b"
        mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            from utils import get_default_llm
            result = get_default_llm()
            assert result is mock_instance
            mock_cls.assert_called_once()

    @patch("utils.config")
    def test_default_temperature_zero(self, mock_config):
        mock_config.LLM_BACKEND = "ollama"
        mock_config.LLM_MODEL = "gemma3:12b"
        mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            from utils import get_default_llm
            get_default_llm()
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0

    @patch("utils.config")
    def test_custom_temperature(self, mock_config):
        mock_config.LLM_BACKEND = "ollama"
        mock_config.LLM_MODEL = "gemma3:12b"
        mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            from utils import get_default_llm
            get_default_llm(temperature=0.7)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.7

    @patch("utils.config")
    def test_streaming_parameter(self, mock_config):
        mock_config.LLM_BACKEND = "ollama"
        mock_config.LLM_MODEL = "gemma3:12b"
        mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            from utils import get_default_llm
            get_default_llm(streaming=True)
            _, kwargs = mock_cls.call_args
            assert kwargs["streaming"] is True


class TestGetDefaultEmbeddings:
    @patch("utils.config")
    def test_ollama_backend(self, mock_config):
        mock_config.LLM_BACKEND = "ollama"
        mock_config.EMBEDDING_MODEL = "nomic-embed-text"
        mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        with patch("langchain_ollama.OllamaEmbeddings") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            from utils import get_default_embeddings
            result = get_default_embeddings()
            assert result is mock_instance

    @patch("utils.config")
    def test_anthropic_falls_back_to_openai(self, mock_config):
        mock_config.LLM_BACKEND = "anthropic"
        mock_config.OPENAI_API_KEY = "sk-test"
        mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        with patch("langchain_openai.OpenAIEmbeddings") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            from utils import get_default_embeddings
            result = get_default_embeddings()
            assert result is mock_instance

    @patch("utils.config")
    def test_anthropic_no_openai_key_raises(self, mock_config):
        mock_config.LLM_BACKEND = "anthropic"
        mock_config.OPENAI_API_KEY = ""
        import pytest
        from utils import get_default_embeddings
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_default_embeddings()
