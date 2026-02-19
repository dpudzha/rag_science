"""Shared utilities for RAG Science."""

import re

from langchain_core.language_models import BaseChatModel

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def get_default_llm(temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    """Create a chat LLM instance based on the configured backend."""
    backend = config.LLM_BACKEND

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
            streaming=streaming,
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
            streaming=streaming,
        )

    # Default: ollama
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )


def get_default_embeddings():
    """Create an embeddings instance based on the configured backend.

    Anthropic doesn't provide an embeddings API, so it falls back to
    OpenAI embeddings (requires OPENAI_API_KEY to be set).
    """
    backend = config.LLM_BACKEND

    if backend in ("openai", "anthropic"):
        from langchain_openai import OpenAIEmbeddings
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                f"LLM_BACKEND={backend} requires OPENAI_API_KEY for embeddings "
                "(Anthropic does not provide an embeddings API)"
            )
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )

    # Default: ollama
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
