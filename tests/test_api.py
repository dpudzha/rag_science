"""Tests for api.py: endpoints, error responses, session management, validation."""
import concurrent.futures
import time
import types
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

    def test_query_retry_reprocesses_final_query(self, client):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Retry answer", "source_documents": []}
        mock_retriever = MagicMock()
        fake_rel_module = types.SimpleNamespace(
            RelevanceChecker=MagicMock(return_value=object()),
            retrieve_with_relevance_check=MagicMock(return_value=([], {
                "score": 0.4,
                "is_relevant": False,
                "retry_count": 1,
                "final_query": "rewritten question",
            })),
        )

        with (
            patch.dict("sys.modules", {"relevance_checker": fake_rel_module}),
            patch("api._get_qa", return_value=mock_chain),
            patch("api.RELEVANCE_CHECK_ENABLED", True),
            patch("api._retriever", mock_retriever),
            patch("api._apply_query_preprocessing",
                  side_effect=lambda q, r=None: f"processed::{q}") as preprocess_mock,
        ):
            resp = client.post("/query", json={"question": "original question"})

        assert resp.status_code == 200
        # First arg is the question; second is the per-request retriever copy
        assert preprocess_mock.call_args_list[0].args[0] == "original question"
        assert preprocess_mock.call_args_list[1].args[0] == "rewritten question"
        invoke_payload = mock_chain.invoke.call_args.args[0]
        assert invoke_payload["question"] == "processed::rewritten question"


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

    def test_delete_session_concurrent_calls_are_safe(self):
        import api

        api._sessions["sess-1"] = {"history": [("q", "a")], "last_access": time.time()}

        def _delete():
            try:
                return api.delete_session("sess-1")
            except Exception as exc:
                return exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: _delete(), range(8)))

        successes = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
        not_found = [r for r in results if getattr(r, "status_code", None) == 404]
        unexpected = [r for r in results if r not in successes and r not in not_found]
        assert len(successes) == 1
        assert len(not_found) == 7
        assert not unexpected


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

    def test_ingest_conflict_when_already_running(self, client):
        import api

        acquired = api._ingest_lock.acquire(blocking=False)
        assert acquired
        try:
            resp = client.post("/ingest")
        finally:
            api._ingest_lock.release()

        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"].lower()


class TestConcurrencySafety:
    def test_get_qa_lazy_initialization_thread_safe(self):
        import api

        sentinel_chain = object()
        calls = {"load": 0, "retriever": 0, "chain": 0}

        def _slow_vectorstore():
            calls["load"] += 1
            time.sleep(0.03)
            return "vs"

        def _build_retriever(_):
            calls["retriever"] += 1
            return "retriever"

        def _build_qa_chain(_):
            calls["chain"] += 1
            return sentinel_chain

        fake_retriever_module = types.SimpleNamespace(
            load_vectorstore=_slow_vectorstore,
            build_retriever=_build_retriever,
            build_qa_chain=_build_qa_chain,
        )

        with (
            patch("api.ENABLE_SQL_AGENT", False),
            patch.dict("sys.modules", {"retriever": fake_retriever_module}),
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(lambda _: api._get_qa(), range(12)))

        assert all(r is sentinel_chain for r in results)
        assert calls["load"] == 1
        assert calls["retriever"] == 1
        assert calls["chain"] == 1


class TestAgentSources:
    def test_sql_agent_response_includes_sources_when_available(self, client):
        import api

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "answer": "Agent answer",
            "tool_used": "sql_tool",
            "sources": [{"file": "table.csv", "page": "row:1"}],
        }

        with (
            patch("api.ENABLE_SQL_AGENT", True),
            patch("api._agent", mock_agent),
            patch("api._get_qa", return_value=mock_agent),
        ):
            resp = client.post("/query", json={"question": "sql question"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Agent answer"
        assert body["tool_used"] == "sql_tool"
        assert body["sources"] == [{"file": "table.csv", "page": "row:1"}]
