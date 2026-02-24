"""Tests for the POST /query/stream SSE endpoint."""
import json
import types
from dataclasses import dataclass

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture
def client():
    """Create a test client with mocked Ollama health check."""
    with patch("api.check_backend"):
        from api import app
        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset API state between tests.

    Disables intent classification and query resolution by default so tests
    don't depend on a running LLM.  Tests that specifically need these
    features should provide their own mocks.
    """
    import api
    api._qa_chain = None
    api._retriever = None
    api._agent = None
    api._intent_classifier = None
    api._query_resolver = None
    api._sessions.clear()
    with patch("api.INTENT_CLASSIFICATION_ENABLED", False), \
         patch("api.QUERY_RESOLUTION_ENABLED", False):
        yield
    api._qa_chain = None
    api._retriever = None
    api._agent = None
    api._intent_classifier = None
    api._query_resolver = None
    api._sessions.clear()


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    for line in text.strip().split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: "):])
            events.append({"event": current_event, "data": data})
            current_event = None
    return events


@dataclass
class FakeChunk:
    """Mimics a streaming LLM chunk with a .content attribute."""
    content: str


def _make_mock_llm(tokens: list[str], fail_after: int | None = None):
    """Create a mock LLM that yields tokens via astream.

    If fail_after is set, raises RuntimeError after that many tokens.
    """
    mock_llm = MagicMock()

    async def _fake_astream(*args, **kwargs):
        for i, token in enumerate(tokens):
            if fail_after is not None and i >= fail_after:
                raise RuntimeError("LLM crashed")
            yield FakeChunk(content=token)

    mock_llm.astream = _fake_astream
    return mock_llm


class TestStreamEventOrdering:
    def test_stream_metadata_tokens_done(self, client):
        """SSE events should follow: metadata → token(s) → done."""
        mock_docs = [
            Document(
                metadata={"source": "paper.pdf", "page": 1},
                page_content="content",
            ),
        ]
        mock_retriever = MagicMock()
        mock_retriever.for_request.return_value = mock_retriever
        mock_retriever.invoke.return_value = mock_docs

        mock_llm = _make_mock_llm(["The", " answer"])

        with (
            patch("api._get_qa", return_value=MagicMock()),
            patch("api._retriever", mock_retriever),
            patch("api.RELEVANCE_CHECK_ENABLED", False),
            patch("api.get_default_llm", return_value=mock_llm),
        ):
            resp = client.post(
                "/query/stream",
                json={"question": "What is AI?"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert len(events) >= 3

        # First event is metadata with sources
        assert events[0]["event"] == "metadata"
        assert "sources" in events[0]["data"]
        assert events[0]["data"]["sources"] == [{"file": "paper.pdf", "page": 1}]

        # Middle events are tokens
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["data"]["token"] == "The"
        assert token_events[1]["data"]["token"] == " answer"

        # Last event is done
        assert events[-1]["event"] == "done"
        assert events[-1]["data"] == {}

    def test_stream_includes_relevance_metadata(self, client):
        """Metadata event includes relevance_score and retry_count when available."""
        mock_retriever = MagicMock()
        mock_retriever.for_request.return_value = mock_retriever

        fake_rel_module = types.SimpleNamespace(
            RelevanceChecker=MagicMock(return_value=object()),
            retrieve_with_relevance_check=MagicMock(return_value=(
                [Document(metadata={"source": "a.pdf", "page": 1}, page_content="x")],
                {"score": 0.83, "is_relevant": True, "retry_count": 0, "final_query": "q"},
            )),
        )

        mock_llm = _make_mock_llm(["ok"])

        with (
            patch.dict("sys.modules", {"relevance_checker": fake_rel_module}),
            patch("api._get_qa", return_value=MagicMock()),
            patch("api._retriever", mock_retriever),
            patch("api.RELEVANCE_CHECK_ENABLED", True),
            patch("api.get_default_llm", return_value=mock_llm),
        ):
            resp = client.post(
                "/query/stream",
                json={"question": "test?"},
            )

        events = _parse_sse(resp.text)
        metadata = events[0]["data"]
        assert metadata["relevance_score"] == 0.83
        assert metadata["retry_count"] == 0


class TestStreamChitchat:
    def test_chitchat_streams_canned_response(self, client):
        """Chitchat should emit metadata → single token (canned response) → done."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = "GREETING"
        mock_classifier.get_chitchat_response.return_value = "Hello! How can I help?"

        with patch("api._get_intent_classifier", return_value=mock_classifier):
            resp = client.post(
                "/query/stream",
                json={"question": "Hi there"},
            )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert events[0]["event"] == "metadata"
        assert events[0]["data"]["sources"] == []
        assert events[1]["event"] == "token"
        assert events[1]["data"]["token"] == "Hello! How can I help?"
        assert events[2]["event"] == "done"


class TestStreamErrors:
    def test_vectorstore_missing_returns_error_event(self, client):
        with patch("api._get_qa", side_effect=FileNotFoundError("missing")):
            resp = client.post(
                "/query/stream",
                json={"question": "test?"},
            )

        assert resp.status_code == 200  # SSE always returns 200
        events = _parse_sse(resp.text)
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "Vectorstore not found" in events[0]["data"]["detail"]

    def test_ollama_down_returns_error_event(self, client):
        with patch("api._get_qa", side_effect=ConnectionError("refused")):
            resp = client.post(
                "/query/stream",
                json={"question": "test?"},
            )

        events = _parse_sse(resp.text)
        assert events[0]["event"] == "error"
        assert "LLM backend unavailable" in events[0]["data"]["detail"]

    def test_agent_path_returns_error_event(self, client):
        with patch("api.ENABLE_SQL_AGENT", True):
            resp = client.post(
                "/query/stream",
                json={"question": "test?"},
            )

        events = _parse_sse(resp.text)
        assert events[0]["event"] == "error"
        assert "not supported" in events[0]["data"]["detail"].lower()

    def test_generation_error_emits_error_event(self, client):
        """If the LLM fails mid-stream, an error event is emitted."""
        mock_retriever = MagicMock()
        mock_retriever.for_request.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        mock_llm = _make_mock_llm(["partial", "BOOM"], fail_after=1)

        with (
            patch("api._get_qa", return_value=MagicMock()),
            patch("api._retriever", mock_retriever),
            patch("api.RELEVANCE_CHECK_ENABLED", False),
            patch("api.get_default_llm", return_value=mock_llm),
        ):
            resp = client.post(
                "/query/stream",
                json={"question": "test?"},
            )

        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "metadata" in event_types
        assert "token" in event_types
        assert "error" in event_types
        # No "done" event after error
        assert event_types[-1] == "error"


class TestStreamSessionHistory:
    def test_stream_updates_session_history(self, client):
        """After streaming completes, session history should be updated."""
        import api

        mock_retriever = MagicMock()
        mock_retriever.for_request.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        mock_llm = _make_mock_llm(["Hello", " world"])

        with (
            patch("api._get_qa", return_value=MagicMock()),
            patch("api._retriever", mock_retriever),
            patch("api.RELEVANCE_CHECK_ENABLED", False),
            patch("api.get_default_llm", return_value=mock_llm),
        ):
            resp = client.post(
                "/query/stream",
                json={"question": "test?", "session_id": "stream-sess"},
            )

        # Consume full response
        assert resp.status_code == 200
        assert "stream-sess" in api._sessions
        assert api._sessions["stream-sess"]["history"][0] == ("test?", "Hello world")


class TestStreamValidation:
    def test_empty_question_rejected(self, client):
        """Validation should still work for the stream endpoint."""
        resp = client.post("/query/stream", json={"question": "   "})
        assert resp.status_code == 422

    def test_too_long_question_rejected(self, client):
        resp = client.post("/query/stream", json={"question": "x" * 5001})
        assert resp.status_code == 422
