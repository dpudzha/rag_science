"""FastAPI layer for the RAG Science pipeline."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import OLLAMA_BASE_URL, SESSION_TTL_SECONDS, CORS_ORIGINS
from health import check_ollama
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# --- Lazy-loaded singleton state ---
_qa_chain = None
_retriever = None

# --- Session storage: session_id -> {"history": [...], "last_access": timestamp} ---
_sessions: dict[str, dict] = {}


def _get_qa():
    """Lazy-load vectorstore, retriever, and QA chain on first query."""
    global _qa_chain, _retriever
    if _qa_chain is None:
        from query import load_vectorstore, build_retriever, build_qa_chain
        vs = load_vectorstore()
        _retriever = build_retriever(vs)
        _qa_chain = build_qa_chain(_retriever)
    return _qa_chain


def _get_session_history(session_id: str | None) -> list:
    """Get or create chat history for a session. Evicts expired sessions."""
    if session_id is None:
        return []

    now = time.time()

    # Evict expired sessions
    expired = [sid for sid, data in _sessions.items()
               if now - data["last_access"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]

    if session_id not in _sessions:
        _sessions[session_id] = {"history": [], "last_access": now}

    _sessions[session_id]["last_access"] = now
    return _sessions[session_id]["history"]


# --- Lifespan: health check on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        check_ollama()
    except ConnectionError:
        logger.warning("Ollama not available at startup — queries will fail until it's reachable")
    yield


app = FastAPI(title="RAG Science API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class IngestResponse(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    ollama: str


# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        qa = _get_qa()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Vectorstore not found. Run ingestion first.")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")

    chat_history = _get_session_history(req.session_id)

    try:
        result = qa.invoke({"question": req.question, "chat_history": chat_history})
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    except Exception as e:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {e}")

    # Update session history
    if req.session_id is not None:
        chat_history.append((req.question, result["answer"]))

    seen = set()
    sources = []
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=src, page=page))
    return QueryResponse(answer=result["answer"], sources=sources, session_id=req.session_id)


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global _qa_chain, _retriever
    try:
        from ingest import ingest as run_ingest
        run_ingest()
        # Reset cached chain so next query picks up new documents
        _qa_chain = None
        _retriever = None
        return IngestResponse(status="ok", detail="Ingestion complete")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "ok", "detail": f"Session '{session_id}' cleared"}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        check_ollama(retries=1, delay=0)
        ollama_status = "ok"
    except ConnectionError:
        ollama_status = "unreachable"
    return HealthResponse(
        status="ok" if ollama_status == "ok" else "degraded",
        ollama=ollama_status,
    )
