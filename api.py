"""FastAPI layer for the RAG Science pipeline."""
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from config import (
    LLM_BACKEND, SESSION_TTL_SECONDS, CORS_ORIGINS,
    INTENT_CLASSIFICATION_ENABLED, ENABLE_SQL_AGENT,
    RELEVANCE_CHECK_ENABLED, RELEVANCE_THRESHOLD, MAX_RETRIEVAL_RETRIES,
    QUERY_RESOLUTION_ENABLED, USE_CROSS_ENCODER_RELEVANCE,
    get_tunable_config, apply_config, save_config, load_config,
)
from health import check_backend
from utils import get_default_llm
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# --- Lazy-loaded singleton state ---
_qa_chain = None
_retriever = None
_agent = None
_intent_classifier = None
_query_resolver = None

# --- Session storage: session_id -> {"history": [...], "last_access": timestamp} ---
_sessions: dict[str, dict] = {}
_qa_init_lock = threading.Lock()
_sessions_lock = threading.RLock()
_ingest_lock = threading.Lock()
MAX_SESSIONS = 100
MAX_HISTORY_LENGTH = 20


def _get_qa():
    """Lazy-load vectorstore, retriever, and QA chain (or agent) on first query."""
    global _qa_chain, _retriever, _agent
    if _qa_chain is None and _agent is None:
        with _qa_init_lock:
            if _qa_chain is None and _agent is None:
                from retriever import load_vectorstore, build_retriever, build_qa_chain
                vs = load_vectorstore()
                _retriever = build_retriever(vs)
                if ENABLE_SQL_AGENT:
                    from query import build_agent
                    _agent = build_agent(_retriever)
                else:
                    _qa_chain = build_qa_chain(_retriever)
    return _agent or _qa_chain


def _get_intent_classifier():
    """Lazy-load intent classifier if enabled."""
    global _intent_classifier
    if not INTENT_CLASSIFICATION_ENABLED:
        return None
    if _intent_classifier is None:
        from intent_classifier import IntentClassifier
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def _get_query_resolver():
    """Lazy-load query resolver if enabled."""
    global _query_resolver
    if not QUERY_RESOLUTION_ENABLED:
        return None
    if _query_resolver is None:
        from query_resolver import QueryResolver
        _query_resolver = QueryResolver()
    return _query_resolver


def _get_session_history(session_id: str | None) -> list:
    """Get or create chat history for a session. Evicts expired sessions."""
    if session_id is None:
        return []

    now = time.time()
    with _sessions_lock:
        # Evict expired sessions
        expired = [
            sid for sid, data in _sessions.items()
            if now - data["last_access"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            _sessions.pop(sid, None)

        if session_id not in _sessions:
            if len(_sessions) >= MAX_SESSIONS:
                oldest = min(_sessions, key=lambda s: _sessions[s]["last_access"])
                _sessions.pop(oldest, None)
            _sessions[session_id] = {"history": [], "last_access": now}

        _sessions[session_id]["last_access"] = now
        # Return a snapshot for read/use outside the session lock.
        return list(_sessions[session_id]["history"])


def _append_session_history(session_id: str | None, question: str, answer: str) -> None:
    """Append an exchange to session history in a threadsafe way."""
    if session_id is None:
        return
    now = time.time()
    with _sessions_lock:
        session = _sessions.setdefault(session_id, {"history": [], "last_access": now})
        session["history"].append((question, answer))
        if len(session["history"]) > MAX_HISTORY_LENGTH:
            session["history"] = session["history"][-MAX_HISTORY_LENGTH:]
        session["last_access"] = now


def _extract_sources_from_result(result: dict | None) -> list["Source"]:
    """Normalize source payloads from chain/agent output to API source models."""
    if not result:
        return []

    seen: set[tuple[str, int | str]] = set()
    sources: list[Source] = []

    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=src, page=page))

    for item in result.get("sources", []):
        src = "unknown"
        page: int | str = "?"
        if isinstance(item, dict):
            src = item.get("file") or item.get("source") or "unknown"
            page = item.get("page", "?")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            src = str(item[0])
            page = item[1]
        elif isinstance(item, str):
            src = item

        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=src, page=page))

    return sources


def _apply_query_preprocessing(question: str, retriever=None) -> str:
    """Run query preprocessing with a per-request retriever copy."""
    r = retriever or _retriever
    if r is None:
        return question
    from query import (
        preprocess_query,
        _get_archetype_detector,
        _get_metadata_extractor,
        _get_domain_terminology,
    )
    return preprocess_query(
        question,
        r,
        detector=_get_archetype_detector(),
        metadata_extractor=_get_metadata_extractor(),
        domain_terms=_get_domain_terminology(),
    )


def _resolve_followup_question(question: str, chat_history: list) -> str:
    """Resolve follow-up questions into standalone questions when enabled."""
    resolver = _get_query_resolver()
    if resolver and chat_history:
        return resolver.resolve(question, chat_history)
    return question


def _get_chitchat_response(question: str) -> str | None:
    """Return canned chitchat response when intent classifier indicates one."""
    classifier = _get_intent_classifier()
    if not classifier:
        return None
    intent = classifier.classify(question)
    return classifier.get_chitchat_response(intent)


def _prepare_query_execution(
    resolved_question: str,
    request_retriever,
    include_docs: bool = False,
) -> tuple[str, float | None, int | None, list]:
    """Prepare preprocessed query and optional prefetched docs for execution."""
    processed_question = _apply_query_preprocessing(resolved_question, request_retriever)

    relevance_score = None
    retry_count = None
    docs = []

    if RELEVANCE_CHECK_ENABLED and request_retriever is not None:
        from relevance_checker import RelevanceChecker, retrieve_with_relevance_check
        checker = RelevanceChecker(
            threshold=RELEVANCE_THRESHOLD,
            use_cross_encoder_score=USE_CROSS_ENCODER_RELEVANCE,
        )
        docs, rel_info = retrieve_with_relevance_check(
            request_retriever, processed_question, checker,
            max_retries=MAX_RETRIEVAL_RETRIES,
        )
        relevance_score = rel_info["score"]
        retry_count = rel_info["retry_count"]
        if rel_info["retry_count"] > 0:
            processed_question = _apply_query_preprocessing(
                rel_info["final_query"], request_retriever,
            )
    elif include_docs and request_retriever is not None:
        docs = request_retriever.invoke(processed_question)

    return processed_question, relevance_score, retry_count, docs


# --- Lifespan: health check on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        check_backend()
    except ConnectionError:
        logger.warning("LLM backend not available at startup — queries will fail until it's reachable")
    yield


app = FastAPI(title="RAG Science API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)


# --- Request / response models ---
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_be_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        if len(v) > 5000:
            raise ValueError("Question must be 5000 characters or fewer")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_format(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v or len(v) > 128:
                raise ValueError("session_id must be 1-128 characters")
        return v


class Source(BaseModel):
    file: str
    page: int | str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str | None = None
    tool_used: str | None = None
    relevance_score: float | None = None
    retry_count: int | None = None


class ConfigResponse(BaseModel):
    config: dict


class IngestResponse(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    ollama: str  # kept for backward compat; reports backend status


# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chat_history = _get_session_history(req.session_id)

    # Resolve follow-up questions before intent classification so that
    # bare follow-ups like "Why?" are resolved into full questions first.
    resolved_question = _resolve_followup_question(req.question, chat_history)

    # Intent classification: skip RAG for greetings/chitchat
    chitchat_response = _get_chitchat_response(resolved_question)
    if chitchat_response:
        return QueryResponse(answer=chitchat_response, sources=[], session_id=req.session_id)

    try:
        qa = _get_qa()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Vectorstore not found. Run ingestion first.")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {e}")

    try:
        # Request-scoped retriever avoids mutating shared state without cloning heavy internals.
        request_retriever = _retriever.for_request() if _retriever is not None else None

        processed_question, relevance_score, retry_count, _ = _prepare_query_execution(
            resolved_question, request_retriever, include_docs=False
        )

        # Check if using agent or chain
        if ENABLE_SQL_AGENT and _agent is not None:
            result = _agent.invoke(processed_question, chat_history=chat_history)
            answer = result.get("answer", "")
            tool_used = result.get("tool_used")
            sources = _extract_sources_from_result(result)

            _append_session_history(req.session_id, req.question, answer)

            return QueryResponse(
                answer=answer, sources=sources, session_id=req.session_id,
                tool_used=tool_used, relevance_score=relevance_score, retry_count=retry_count,
            )

        result = qa.invoke({"question": processed_question, "chat_history": chat_history})
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {e}")
    except Exception as e:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs.")

    # Update session history
    _append_session_history(req.session_id, req.question, result["answer"])

    sources = _extract_sources_from_result(result)
    return QueryResponse(
        answer=result["answer"], sources=sources, session_id=req.session_id,
        relevance_score=relevance_score, retry_count=retry_count,
    )


def _extract_sources_from_docs(docs) -> list["Source"]:
    """Extract deduplicated sources from a list of LangChain Documents."""
    seen: set[tuple[str, int | str]] = set()
    sources: list[Source] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=src, page=page))
    return sources


def _format_sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Stream the answer via Server-Sent Events.

    Runs all pre-generation pipeline stages (intent, archetype, retrieval,
    relevance check), emits a metadata event with sources, then streams
    answer tokens incrementally.
    """
    # Agent path: not supported for streaming
    if ENABLE_SQL_AGENT:
        async def _agent_error():
            yield _format_sse("error", {
                "detail": "Streaming is not supported with the SQL agent. Use /query instead.",
            })
        return StreamingResponse(
            _agent_error(), media_type="text/event-stream",
        )

    chat_history = _get_session_history(req.session_id)

    # Resolve follow-up questions before intent classification so that
    # bare follow-ups like "Why?" are resolved into full questions first.
    resolved_question = _resolve_followup_question(req.question, chat_history)

    # Intent classification: short-circuit for greetings/chitchat
    chitchat_response = _get_chitchat_response(resolved_question)
    if chitchat_response:
        async def _chitchat_stream():
            yield _format_sse("metadata", {
                "sources": [], "session_id": req.session_id,
            })
            yield _format_sse("token", {"token": chitchat_response})
            yield _format_sse("done", {})
        return StreamingResponse(
            _chitchat_stream(), media_type="text/event-stream",
        )

    # Load QA chain / retriever
    try:
        _get_qa()  # ensure retriever is loaded
    except FileNotFoundError:
        async def _missing():
            yield _format_sse("error", {
                "detail": "Vectorstore not found. Run ingestion first.",
            })
        return StreamingResponse(_missing(), media_type="text/event-stream")
    except ConnectionError as e:
        detail = f"LLM backend unavailable: {e}"

        async def _conn_err():
            yield _format_sse("error", {"detail": detail})

        return StreamingResponse(_conn_err(), media_type="text/event-stream")

    # Request-scoped retriever avoids mutating shared state without cloning heavy internals.
    request_retriever = _retriever.for_request() if _retriever is not None else None

    processed_question, relevance_score, retry_count, docs = _prepare_query_execution(
        resolved_question, request_retriever, include_docs=True
    )

    # Stream the LLM directly — ConversationalRetrievalChain doesn't yield
    # incremental tokens, so we bypass the chain and call the LLM directly.
    from retriever import QA_PROMPT

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = QA_PROMPT.format(context=context, question=processed_question)
    llm = get_default_llm(temperature=0, streaming=True)

    sources = _extract_sources_from_docs(docs)

    async def _event_stream():
        # Emit metadata (sources known before generation)
        metadata = {
            "sources": [s.model_dump() for s in sources],
            "session_id": req.session_id,
        }
        if relevance_score is not None:
            metadata["relevance_score"] = relevance_score
        if retry_count is not None:
            metadata["retry_count"] = retry_count
        yield _format_sse("metadata", metadata)

        # Stream answer tokens
        full_answer = []
        try:
            async for chunk in llm.astream(prompt):
                token = chunk.content
                if token:
                    full_answer.append(token)
                    yield _format_sse("token", {"token": token})
        except Exception:
            logger.exception("Error during streaming generation")
            yield _format_sse("error", {"detail": "Generation failed. Check server logs."})
            return

        yield _format_sse("done", {})

        # Update session history with the complete answer
        _append_session_history(req.session_id, req.question, "".join(full_answer))

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    global _qa_chain, _retriever, _agent
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Ingestion already in progress")
    try:
        from ingest import ingest as run_ingest
        run_ingest()
        # Reset cached chain so next query picks up new documents
        with _qa_init_lock:
            _qa_chain = None
            _retriever = None
            _agent = None
        return IngestResponse(status="ok", detail="Ingestion complete")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")
    finally:
        _ingest_lock.release()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    with _sessions_lock:
        if _sessions.pop(session_id, None) is not None:
            return {"status": "ok", "detail": f"Session '{session_id}' cleared"}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        check_backend(retries=1, delay=0)
        ollama_status = "ok"
    except ConnectionError:
        ollama_status = "unreachable"
    return HealthResponse(
        status="ok" if ollama_status == "ok" else "degraded",
        ollama=ollama_status,
    )


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(config=get_tunable_config())


@app.put("/config", response_model=ConfigResponse)
def update_config(updates: dict) -> ConfigResponse:
    try:
        apply_config(updates)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ConfigResponse(config=get_tunable_config())


@app.post("/config/save")
def save_config_endpoint() -> dict:
    try:
        save_config()
    except OSError as e:
        logger.exception("Failed to save config")
        raise HTTPException(status_code=500, detail="Failed to save config. Check server logs.")
    return {"status": "ok"}


@app.post("/config/load", response_model=ConfigResponse)
def load_config_endpoint() -> ConfigResponse:
    try:
        data = load_config()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No saved config file found")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logger.exception("Failed to load config")
        raise HTTPException(status_code=500, detail="Failed to load config. Check server logs.")
    return ConfigResponse(config=get_tunable_config())
