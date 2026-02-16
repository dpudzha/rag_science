"""Tests for api.py: endpoints, error responses, session management, validation."""
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Ollama health check."""
    with patch("api.check_ollama"):
        from api import app
        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset API state between tests."""
    import api
    api._qa_chain = None
    api._retriever = None
    api._sessions.clear()
    yield
    api._qa_chain = None
    api._retriever = None
    api._sessions.clear()


class TestHealthEndpoint:
    def test_health_ok(self, client):
        with patch("api.check_ollama"):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ollama"] == "ok"

    def test_health_degraded(self, client):
        with patch("api.check_ollama", side_effect=ConnectionError("down")):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ollama"] == "unreachable"


class TestQueryValidation:
    def test_empty_question_rejected(self, client):
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_too_long_question_rejected(self, client):
        resp = client.post("/query", json={"question": "x" * 5001})
        assert resp.status_code == 422

    def test_invalid_session_id_rejected(self, client):
        resp = client.post("/query", json={"question": "test?", "session_id": "x" * 129})
        assert resp.status_code == 422


class TestQueryEndpoint:
    def test_query_success(self, client):
        mock_result = {
            "answer": "Test answer",
            "source_documents": [
                MagicMock(metadata={"source": "paper.pdf", "page": 1}, page_content="content"),
            ],
        }
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("api._get_qa", return_value=mock_chain):
            resp = client.post("/query", json={"question": "What is AI?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert len(data["sources"]) == 1

    def test_query_ollama_down(self, client):
        with patch("api._get_qa", side_effect=ConnectionError("no connection")):
            resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 503

    def test_query_no_vectorstore(self, client):
        with patch("api._get_qa", side_effect=FileNotFoundError("missing")):
            resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 503


class TestSessionManagement:
    def test_query_with_session_id(self, client):
        mock_result = {
            "answer": "Answer 1",
            "source_documents": [],
        }
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("api._get_qa", return_value=mock_chain):
            resp = client.post("/query", json={"question": "Q1?", "session_id": "sess-1"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "sess-1"

        # Verify session was stored
        import api
        assert "sess-1" in api._sessions
        assert len(api._sessions["sess-1"]["history"]) == 1

    def test_delete_session(self, client):
        import api
        api._sessions["sess-1"] = {"history": [("q", "a")], "last_access": 0}

        resp = client.delete("/sessions/sess-1")
        assert resp.status_code == 200
        assert "sess-1" not in api._sessions

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/sessions/nope")
        assert resp.status_code == 404


class TestErrorResponses:
    """Verify error responses don't leak internal details."""

    def test_unexpected_error_hides_details(self, client):
        mock_qa = MagicMock()
        mock_qa.invoke.side_effect = RuntimeError("secret db password leaked")
        with patch("api._get_qa", return_value=mock_qa):
            resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "secret" not in detail
        assert "password" not in detail
        assert "internal error" in detail.lower()

    def test_ingest_error_hides_details(self, client):
        with patch("ingest.ingest", side_effect=RuntimeError("disk path /secret/data")):
            resp = client.post("/ingest")
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "/secret" not in detail
        assert "failed" in detail.lower()


class TestIngestEndpoint:
    def test_ingest_success(self, client):
        with patch("ingest.ingest") as mock_run:
            resp = client.post("/ingest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_run.assert_called_once()

    def test_ingest_ollama_down(self, client):
        with patch("ingest.ingest", side_effect=ConnectionError("offline")):
            resp = client.post("/ingest")
        assert resp.status_code == 503
