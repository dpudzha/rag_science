"""Shared utilities for RAG Science."""

import re
from langchain_ollama import ChatOllama
from config import LLM_MODEL, OLLAMA_BASE_URL

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def get_default_llm(temperature: float = 0) -> ChatOllama:
    """Create a ChatOllama instance with standard settings."""
    return ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=temperature
    )
