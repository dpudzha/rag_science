"""Shared fixtures for RAG Science tests."""
import json
import pickle
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


@pytest.fixture
def tmp_vectorstore(tmp_path):
    """Create a temporary vectorstore directory."""
    vs_dir = tmp_path / "vectorstore"
    vs_dir.mkdir()
    return vs_dir


@pytest.fixture
def sample_pdf_text():
    """Sample multi-page PDF text structure."""
    return {
        "pages": [
            {"text": "This is the title of the paper\n\nAbstract\nThis paper studies neural networks.", "page": 1},
            {"text": "1. INTRODUCTION\nDeep learning has revolutionized many fields.", "page": 2},
            {"text": "2. METHODS\nWe used a transformer architecture with attention.", "page": 3},
        ],
        "source": "test_paper.pdf",
        "title": "This is the title of the paper",
        "hash": "abc123",
    }


@pytest.fixture
def sample_documents():
    """Sample LangChain Document list."""
    return [
        Document(page_content="Neural networks are powerful models.", metadata={"source": "paper1.pdf", "page": 1}),
        Document(page_content="Transformers use attention mechanisms.", metadata={"source": "paper1.pdf", "page": 2}),
        Document(page_content="BERT is a bidirectional model.", metadata={"source": "paper2.pdf", "page": 1}),
        Document(page_content="GPT uses autoregressive generation.", metadata={"source": "paper2.pdf", "page": 3}),
    ]


@pytest.fixture
def mock_ollama():
    """Mock Ollama health check to always succeed."""
    with patch("health.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        yield mock_get


@pytest.fixture
def ingest_record_path(tmp_vectorstore):
    """Path to ingested.json in temp vectorstore."""
    return tmp_vectorstore / "ingested.json"


@pytest.fixture
def bm25_index_path(tmp_vectorstore):
    """Path to BM25 pickle in temp vectorstore."""
    return tmp_vectorstore / "bm25_index.pkl"
