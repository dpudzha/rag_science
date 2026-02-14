"""FastAPI layer for the RAG Science pipeline."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import OLLAMA_BASE_URL
from health import check_ollama

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Lazy-loaded singleton state ---
_qa_chain = None
_retriever = None


def _get_qa():
    """Lazy-load vectorstore, retriever, and QA chain on first query."""
    global _qa_chain, _retriever
    if _qa_chain is None:
        from query import load_vectorstore, build_retriever, build_qa_chain
        vs = load_vectorstore()
        _retriever = build_retriever(vs)
        _qa_chain = build_qa_chain(_retriever)
    return _qa_chain


# --- Lifespan: health check on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        check_ollama()
    except ConnectionError:
        logger.warning("Ollama not available at startup — queries will fail until it's reachable")
    yield


app = FastAPI(title="RAG Science API", lifespan=lifespan)


# --- Request / response models ---
class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: int | str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    ollama: str


# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    qa = _get_qa()
    result = qa.invoke({"question": req.question, "chat_history": []})
    seen = set()
    sources = []
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=src, page=page))
    return QueryResponse(answer=result["answer"], sources=sources)


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
