# RAG Science — Detailed Implementation Plan (All 5 Phases)

## Context

4-agent review (Code Quality, RAG Expert, Security, Portfolio) identified 2 critical bugs, 2 critical security issues, 5 high-impact code quality problems, and significant portfolio underselling. This plan provides exact, line-by-line implementation steps to address all findings across 5 phases. The project is already top 5-10% of RAG portfolio projects — these changes will make it exceptional.

---

## Phase 1: Critical Fixes & Quick Wins

### 1.1 Fix Double-Retrieval Bug

**Problem**: `retrieve_with_relevance_check()` calls `retriever.invoke()` (relevance_checker.py:92), returns docs — but callers discard them. Then `qa.invoke()` calls retriever again internally via `ConversationalRetrievalChain`.

**Files**: `relevance_checker.py`, `query.py`, `api.py`

**Step 1** — `query.py`: Modify `ask()` (lines 419-431) to use relevance-checked docs directly:

```python
# BEFORE (lines 419-431):
if relevance_checker:
    from relevance_checker import retrieve_with_relevance_check
    docs, rel_info = retrieve_with_relevance_check(...)
    if rel_info["retry_count"] > 0:
        processed_question = rel_info["final_query"]
result = qa.invoke({"question": processed_question, "chat_history": []})

# AFTER:
if relevance_checker:
    from relevance_checker import retrieve_with_relevance_check
    docs, rel_info = retrieve_with_relevance_check(
        retriever, processed_question, relevance_checker,
        max_retries=MAX_RETRIEVAL_RETRIES,
    )
    if rel_info["retry_count"] > 0:
        processed_question = rel_info["final_query"]
    # Use pre-retrieved docs directly — skip chain's internal retrieval
    from langchain.chains.combine_documents import create_stuff_documents_chain
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    result = {
        "answer": create_stuff_documents_chain(llm, QA_PROMPT).invoke(
            {"context": docs, "question": processed_question}
        ),
        "source_documents": docs,
    }
else:
    result = qa.invoke({"question": processed_question, "chat_history": []})
```

Wait — this introduces a new dependency and changes the chain structure. Simpler approach: create a helper that wraps `StuffDocumentsChain` to match the existing result format.

**Better approach** — Add a `answer_with_docs()` function in `query.py`:

```python
def answer_with_docs(docs: list[Document], question: str, chat_history: list) -> dict:
    """Generate answer from pre-retrieved docs without re-retrieval."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
    )
    answer = stuff_chain.invoke({"input_documents": docs, "question": question})
    return {"answer": answer["output_text"], "source_documents": docs}
```

Then update `ask()` (line 430), `interactive()` (line 389), and `api.py` (line 216) to call `answer_with_docs(docs, processed_question, chat_history)` when relevance check returned docs.

**Actually, simplest correct approach**: Instead of restructuring chains, make the retriever return cached docs on second call. Add a `CachedRetriever` wrapper:

**Final approach (chosen for minimal disruption)**:

**Step 1a** — Add to `query.py` after the `HybridRetriever` class (~line 168):

```python
class PreloadedRetriever(BaseRetriever):
    """Wrapper that returns pre-fetched docs instead of re-retrieving."""
    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.docs
```

**Step 1a-ii** — Update `build_qa_chain` type hint (line 230):

```python
# BEFORE:
def build_qa_chain(retriever: HybridRetriever):
# AFTER:
def build_qa_chain(retriever: BaseRetriever):
```

**Step 1b** — Update `ask()` (lines 419-431):

```python
if relevance_checker:
    from relevance_checker import retrieve_with_relevance_check
    docs, rel_info = retrieve_with_relevance_check(
        retriever, processed_question, relevance_checker,
        max_retries=MAX_RETRIEVAL_RETRIES,
    )
    if rel_info["retry_count"] > 0:
        processed_question = rel_info["final_query"]
    # Swap retriever so QA chain uses pre-retrieved docs
    qa = build_qa_chain(PreloadedRetriever(docs=docs))

result = qa.invoke({"question": processed_question, "chat_history": []})
```

**Step 1c** — Same change in `interactive()` (lines 380-389):

```python
if relevance_checker:
    from relevance_checker import retrieve_with_relevance_check
    docs, rel_info = retrieve_with_relevance_check(
        retriever, processed_question, relevance_checker,
        max_retries=MAX_RETRIEVAL_RETRIES,
    )
    if rel_info["retry_count"] > 0:
        processed_question = rel_info["final_query"]
    result = build_qa_chain(PreloadedRetriever(docs=docs)).invoke(
        {"question": processed_question, "chat_history": chat_history}
    )
else:
    result = qa.invoke({"question": processed_question, "chat_history": chat_history})
```

**Step 1d** — Same change in `api.py` (lines 201-216):

```python
if RELEVANCE_CHECK_ENABLED and _retriever is not None:
    from relevance_checker import RelevanceChecker, retrieve_with_relevance_check
    checker = RelevanceChecker(threshold=RELEVANCE_THRESHOLD)
    docs, rel_info = retrieve_with_relevance_check(
        _retriever, resolved_question, checker,
        max_retries=MAX_RETRIEVAL_RETRIES,
    )
    relevance_score = rel_info["score"]
    retry_count = rel_info["retry_count"]
    if rel_info["retry_count"] > 0:
        resolved_question = rel_info["final_query"]
    from query import PreloadedRetriever, build_qa_chain
    qa_for_query = build_qa_chain(PreloadedRetriever(docs=docs))
    result = qa_for_query.invoke({"question": resolved_question, "chat_history": chat_history})
else:
    result = qa.invoke({"question": resolved_question, "chat_history": chat_history})
```

**Tests to update**: `tests/test_relevance_checker.py`, `tests/test_query.py`, `tests/test_api.py` — add test that `PreloadedRetriever` returns cached docs.

---

### 1.2 SQL Security Hardening

**Files**: `sql_database.py`, `config.py`

**Step 1** — `sql_database.py:13-16`: Expand `_UNSAFE_PATTERN`:

```python
# BEFORE:
_UNSAFE_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

# AFTER:
_UNSAFE_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|EXEC|EXECUTE"
    r"|ATTACH|DETACH|PRAGMA|LOAD_EXTENSION|SAVEPOINT|RELEASE|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)
```

**Step 2** — `sql_database.py:28-29`: Use read-only connection:

```python
# BEFORE:
def _connect(self) -> sqlite3.Connection:
    return sqlite3.connect(self._db_path)

# AFTER:
def _connect(self, readonly: bool = True) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{self._db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(self._db_path)
    return conn
```

**Step 3** — `sql_database.py:31-37` (`create_table`): Pass `readonly=False`:

```python
def create_table(self, name: str, df) -> None:
    conn = self._connect(readonly=False)
    try:
        df.to_sql(name, conn, if_exists="replace", index=False)
    finally:
        conn.close()
```

**Step 4** — `sql_database.py:97-99`: Fix table name quoting in `get_sample_rows()`:

```python
# BEFORE:
def get_sample_rows(self, table_name: str, limit: int = 3) -> list[dict]:
    return self.execute_query(f"SELECT * FROM '{table_name}' LIMIT {limit}")

# AFTER:
def get_sample_rows(self, table_name: str, limit: int = 3) -> list[dict]:
    safe_name = table_name.replace('"', '""')
    return self.execute_query(f'SELECT * FROM "{safe_name}" LIMIT {limit}')
```

**Step 5** — Same fix for `get_schema()` (lines 70, 75) and `get_table_names()` (line 62) — verify they use safe quoting. `get_table_names()` queries `sqlite_master` directly which is fine. `get_schema()` uses `PRAGMA table_info('{table_name}')` — but PRAGMA is now blocked by our regex. Change `get_schema()` to use `sqlite_master` instead:

```python
# BEFORE (line 70-78):
def get_schema(self) -> str:
    tables = self.get_table_names()
    if not tables:
        return "No tables found."
    schema_parts = []
    for table_name in tables:
        conn = self._connect()
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        conn.close()
        col_defs = [f"  {col[1]} {col[2]}" for col in columns]
        schema_parts.append(f"Table: {table_name}\n" + "\n".join(col_defs))
    return "\n\n".join(schema_parts)

# AFTER:
def get_schema(self) -> str:
    tables = self.get_table_names()
    if not tables:
        return "No tables found."
    schema_parts = []
    conn = self._connect()
    try:
        for table_name in tables:
            safe_name = table_name.replace('"', '""')
            # Get column info without PRAGMA
            cursor = conn.execute(f'SELECT * FROM "{safe_name}" LIMIT 0')
            col_names = [desc[0] for desc in cursor.description]
            col_defs = [f"  {name}" for name in col_names]
            schema_parts.append(f"Table: {table_name}\n" + "\n".join(col_defs))
    finally:
        conn.close()
    return "\n\n".join(schema_parts)
```

Note: This loses type info from PRAGMA, but column names are sufficient for SQL generation. Alternatively, keep PRAGMA but use internal `_connect(readonly=False)` only for schema — but that weakens our read-only guarantee. Better: use `SELECT sql FROM sqlite_master WHERE name = ?` which returns the CREATE TABLE statement:

```python
def get_schema(self) -> str:
    tables = self.get_table_names()
    if not tables:
        return "No tables found."
    conn = self._connect()
    try:
        schema_parts = []
        for table_name in tables:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                schema_parts.append(row[0])
        return "\n\n".join(schema_parts)
    finally:
        conn.close()
```

This is the cleanest approach — returns the actual CREATE TABLE DDL, uses parameterized query, no PRAGMA needed.

**Step 6** — Also fix `sql_database.py:45`: Remove unnecessary f-string:

```python
# BEFORE:
raise ValueError(f"Unsafe SQL operation detected. Only SELECT queries are allowed.")
# AFTER:
raise ValueError("Unsafe SQL operation detected. Only SELECT queries are allowed.")
```

**Tests to update**: `tests/test_sql_database.py` — add tests for ATTACH, PRAGMA, LOAD_EXTENSION being blocked; test read-only connection; test table name with special chars.

---

### 1.3 Switch to Reciprocal Rank Fusion (RRF)

**File**: `query.py` — `HybridRetriever._get_relevant_documents()` (lines 107-146)

**Replace** the entire score normalization and combination block (lines 107-146) with RRF:

```python
def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
    doc_map: dict[str, Document] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    rrf_k = 60  # Standard RRF constant

    # Dense retrieval via FAISS
    dense_results = self.vectorstore.similarity_search_with_score(
        query, k=self.k_candidates
    )
    for rank, (doc, _score) in enumerate(dense_results):
        key = self._doc_key(doc)
        doc_map[key] = doc
        dense_ranks[key] = rank

    # BM25 keyword retrieval
    tokenized_query = tokenize(query)
    bm25_scores = self.bm25.get_scores(tokenized_query)
    bm25_top = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:self.k_candidates]
    for rank, idx in enumerate(bm25_top):
        doc = self.bm25_docs[idx]
        key = self._doc_key(doc)
        doc_map[key] = doc
        bm25_ranks[key] = rank

    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    for key in doc_map:
        score = 0.0
        if key in dense_ranks:
            score += self.dense_weight / (rrf_k + dense_ranks[key])
        if key in bm25_ranks:
            score += self.bm25_weight / (rrf_k + bm25_ranks[key])
        rrf_scores[key] = score

    # Sort by RRF score, take top candidates for reranking
    ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    candidates = [doc_map[key] for key in ranked[:self.k_candidates]]

    if not candidates:
        return []

    # Cross-encoder reranking
    pairs = [[query, doc.page_content] for doc in candidates]
    rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
    reranked = sorted(
        zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
    )

    # Apply metadata filters BEFORE final selection (moved from after)
    filtered = [doc for doc, _ in reranked]
    if self.metadata_filters:
        from metadata_extractor import MetadataFilterApplier
        filtered = MetadataFilterApplier.apply(filtered, self.metadata_filters)

    top_docs = filtered[:self.k]

    # Parent-document expansion (unchanged)
    if self.parent_chunks is not None:
        expanded = []
        seen_parents = set()
        for doc in top_docs:
            parent_idx = doc.metadata.get("parent_idx")
            if parent_idx is not None and parent_idx < len(self.parent_chunks):
                if parent_idx not in seen_parents:
                    seen_parents.add(parent_idx)
                    expanded.append(self.parent_chunks[parent_idx])
            else:
                expanded.append(doc)
        return expanded

    return top_docs
```

Key changes:
1. Rank-based fusion instead of score-based (eliminates normalization issues)
2. Weights still influence via `self.dense_weight` and `self.bm25_weight` in RRF formula
3. Metadata filtering moved BEFORE `[:self.k]` slice (was after — could reduce below k)

**Tests to update**: `tests/test_query.py` — update any tests that check score values.

---

### 1.4 CORS Fix

**File**: `config.py:29`

```python
# BEFORE:
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# AFTER:
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
```

**File**: `api.py:102-108`

```python
# BEFORE:
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AFTER:
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
```

---

### 1.5 Docker & API Hardening

**File**: `Dockerfile` — Add non-root user (after `COPY . .`, before `EXPOSE`):

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**New file**: `.dockerignore`:

```
.venv/
__pycache__/
*.pyc
.DS_Store
vectorstore/
.env
.git/
.github/
*.md
!requirements*.txt
experiments/
tests/
CLAUDE.md
```

**File**: `api.py` — Add session limits. Modify `_get_session_history()` (lines 70-87):

```python
MAX_SESSIONS = 100
MAX_HISTORY_LENGTH = 20

def _get_session_history(session_id: str | None) -> list:
    if session_id is None:
        return []

    now = time.time()

    # Evict expired sessions
    expired = [sid for sid, data in _sessions.items()
               if now - data["last_access"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]

    if session_id not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            # Evict oldest session
            oldest = min(_sessions, key=lambda s: _sessions[s]["last_access"])
            del _sessions[oldest]
        _sessions[session_id] = {"history": [], "last_access": now}

    _sessions[session_id]["last_access"] = now
    history = _sessions[session_id]["history"]

    # Trim history to max length
    if len(history) > MAX_HISTORY_LENGTH:
        _sessions[session_id]["history"] = history[-MAX_HISTORY_LENGTH:]
        history = _sessions[session_id]["history"]

    return history
```

**File**: `api.py` — Sanitize error responses. Lines 219-221:

```python
# BEFORE:
except Exception as e:
    logger.exception("Unexpected error during query")
    raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {e}")

# AFTER:
except Exception as e:
    logger.exception("Unexpected error during query")
    raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs.")
```

Same for ingest endpoint (line 255-257):

```python
# BEFORE:
except Exception as e:
    logger.exception("Ingestion failed")
    raise HTTPException(status_code=500, detail=str(e))

# AFTER:
except Exception as e:
    logger.exception("Ingestion failed")
    raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")
```

---

### 1.6 Retrieval Parameter Tuning

**File**: `config.py` — Update defaults:

```python
# BEFORE:
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))  # Line 17
TOP_K = int(os.getenv("TOP_K", "4"))                          # Line 18

# AFTER:
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "30"))
TOP_K = int(os.getenv("TOP_K", "6"))
```

Metadata filtering is already moved before final selection in Step 1.3 above.

---

## Phase 2: Code Quality & Architecture

### 2.1 Extract Shared Utilities — DRY Fixes

**New file**: `utils.py`

```python
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
```

**Update imports** — Replace `tokenize()` in:
- `ingest.py:38-45`: Remove `_TOKEN_RE` and `tokenize()`, add `from utils import tokenize`
- `query.py:41-59`: Remove `_TOKEN_RE` and `tokenize()`, add `from utils import tokenize`

**Update LLM construction** — Replace `ChatOllama(model=LLM_MODEL, ...)` in:
- `intent_classifier.py:57-60`: `self._llm = llm or get_default_llm()`
- `archetype_detector.py:56-59` (ArchetypeDetector): same
- `archetype_detector.py:84-87` (QueryReformulator): same
- `metadata_extractor.py:21-24`: same
- `relevance_checker.py:24-27`: same
- `query_resolver.py:19-22`: same
- `agent.py:23-26`: same
- `query.py:231` (build_qa_chain): `llm = get_default_llm()`

Each file: add `from utils import get_default_llm`, remove direct `ChatOllama` import (if no longer needed), replace the construction.

`tools/sql_tool.py:48-51` (`_get_llm`): Change to:
```python
def _get_llm(self):
    if self.llm is None:
        from utils import get_default_llm
        self.llm = get_default_llm()
    return self.llm
```

---

### 2.2 Deduplicate ask() / interactive()

**File**: `query.py`

**Step 1** — Extract shared pipeline into `_run_pipeline()` (~insert after `preprocess_query`, before `interactive`):

```python
def _run_pipeline(
    question: str,
    retriever: HybridRetriever,
    qa,
    chat_history: list,
    classifier=None,
    detector=None,
    reformulator=None,
    meta_extractor=None,
    relevance_checker=None,
    query_resolver=None,
) -> dict | None:
    """Core query pipeline shared by ask() and interactive().

    Returns result dict or None if handled (e.g., greeting).
    """
    # Intent classification
    if classifier:
        intent = classifier.classify(question)
        response = classifier.get_chitchat_response(intent)
        if response:
            print(f"\n{response}\n")
            return None

    # Resolve follow-ups
    resolved = question
    if query_resolver and chat_history:
        resolved = query_resolver.resolve(question, chat_history)

    # Archetype + reformulation + metadata
    processed = preprocess_query(
        resolved, retriever, detector, reformulator, meta_extractor
    )

    # Relevance checking with retry
    if relevance_checker:
        from relevance_checker import retrieve_with_relevance_check
        docs, rel_info = retrieve_with_relevance_check(
            retriever, processed, relevance_checker,
            max_retries=MAX_RETRIEVAL_RETRIES,
        )
        if rel_info["retry_count"] > 0:
            processed = rel_info["final_query"]
        result = build_qa_chain(PreloadedRetriever(docs=docs)).invoke(
            {"question": processed, "chat_history": chat_history}
        )
    else:
        result = qa.invoke({"question": processed, "chat_history": chat_history})

    return result
```

**Step 2** — Simplify `ask()`:

```python
def ask(question: str):
    from health import check_ollama
    check_ollama()

    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = build_qa_chain(retriever)

    result = _run_pipeline(
        question, retriever, qa,
        chat_history=[],
        classifier=_get_intent_classifier(),
        detector=_get_archetype_detector(),
        reformulator=_get_query_reformulator(),
        meta_extractor=_get_metadata_extractor(),
        relevance_checker=_get_relevance_checker(),
    )
    if result:
        print_result(result)
```

**Step 3** — Simplify `interactive()`:

```python
def interactive():
    from health import check_ollama
    check_ollama()

    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = build_qa_chain(retriever)
    chat_history = []
    classifier = _get_intent_classifier()
    detector = _get_archetype_detector()
    reformulator = _get_query_reformulator()
    meta_extractor = _get_metadata_extractor()
    rel_checker = _get_relevance_checker()
    resolver = _get_query_resolver()

    print("RAG Science — ask questions about your papers (type 'quit' to exit)\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        result = _run_pipeline(
            question, retriever, qa,
            chat_history=chat_history,
            classifier=classifier,
            detector=detector,
            reformulator=reformulator,
            meta_extractor=meta_extractor,
            relevance_checker=rel_checker,
            query_resolver=resolver,
        )
        if result:
            print_result(result)
            chat_history.append((question, result["answer"]))
```

---

### 2.3 Fix Thread-Safety — Immutable Retriever State

**File**: `query.py`

**Step 1** — Change `preprocess_query()` (lines 309-333) to return weights/filters instead of mutating:

```python
def preprocess_query(
    question: str,
    retriever: HybridRetriever,
    detector=None,
    reformulator=None,
    metadata_extractor=None,
) -> str:
    """Apply archetype detection, query reformulation, and metadata extraction."""
    archetype = None
    if detector:
        archetype = detector.detect(question)
        bm25_w, dense_w = detector.get_weights(archetype)
        retriever.bm25_weight = bm25_w
        retriever.dense_weight = dense_w
        logger.info(
            "Adjusted weights: bm25=%.2f, dense=%.2f for archetype %s",
            bm25_w, dense_w, archetype,
        )

    if reformulator and archetype:
        question = reformulator.reformulate(question, archetype)

    if metadata_extractor:
        metadata = metadata_extractor.extract(question)
        if metadata_extractor.has_filters(metadata):
            retriever.metadata_filters = metadata
        else:
            retriever.metadata_filters = None

    return question
```

The mutation is acceptable for `query.py` CLI (single-threaded), but for `api.py` we need per-request isolation.

**Step 2** — In `api.py`, create a per-request retriever copy. Update the query endpoint (around line 190):

```python
# In the query endpoint, after getting _retriever:
import copy
request_retriever = copy.copy(_retriever)  # Shallow copy — shares vectorstore/bm25 (read-only) but gets own weight/filter fields
```

Then use `request_retriever` in `preprocess_query()` and `retrieve_with_relevance_check()` calls.

`HybridRetriever` extends Pydantic's `BaseRetriever`, so we need to verify `copy.copy()` works. Since it uses Pydantic model fields, we can use `.model_copy()`:

```python
request_retriever = _retriever.model_copy()
```

This creates a shallow copy — FAISS vectorstore, BM25, cross_encoder are shared (good, they're read-only), but `bm25_weight`, `dense_weight`, `metadata_filters` are per-request.

---

### 2.4 Cache Lazy-Loaders in query.py

**File**: `query.py` (lines 261-307)

Add module-level caching like `api.py` does:

```python
_intent_classifier = None
_archetype_detector = None
_query_reformulator = None
_metadata_extractor = None
_relevance_checker = None
_query_resolver = None


def _get_intent_classifier():
    global _intent_classifier
    if not INTENT_CLASSIFICATION_ENABLED:
        return None
    if _intent_classifier is None:
        from intent_classifier import IntentClassifier
        _intent_classifier = IntentClassifier()
    return _intent_classifier
```

Repeat pattern for all 6 lazy-loaders. Each should:
1. Declare `global _variable`
2. Check feature flag
3. Check if already created
4. Create only if None
5. Return cached instance

---

### 2.5 Type Contracts

**File**: `parsers/__init__.py` — Add Protocol:

```python
from typing import Protocol


class DocumentDict:
    """Type alias for parser return format (documented, not enforced at runtime)."""
    # pages: list[dict with "text" and "page" keys]
    # source: str
    # title: str
    # creation_date: str
    # authors: str
    # tables: list[dict] (optional)
    pass


class Parser(Protocol):
    def parse(self, path: str) -> dict: ...


def get_parser(path: str) -> Parser:
    ...  # existing implementation
```

**File**: `tools/rag_tool.py:20`:

```python
# BEFORE:
retriever: object
# AFTER:
retriever: Any  # HybridRetriever (can't import due to circular dep)
```

Add `from typing import Any` at top.

**File**: `tools/sql_tool.py:44`:

```python
# BEFORE:
llm: object = None
# AFTER:
llm: Any = None  # ChatOllama | None
```

---

### 2.6 Dead Code & Cleanup

**File**: `ingest.py:109-129` — Delete `load_new_pdfs()` function entirely.

**File**: `config.py:38` — Delete `DOMAIN_TERMINOLOGY_PATH` line.

**File**: `metadata_extractor.py:46`:

```python
# BEFORE:
except (json.JSONDecodeError, Exception) as e:
# AFTER:
except Exception as e:
```

**File**: `parsers/pdf_parser.py` — Move `extract_text_from_pdf` here from `ingest.py`:

Step 1: Move the function `extract_text_from_pdf()` (ingest.py:63-107) into `parsers/pdf_parser.py`.
Step 2: Update `PDFParser.parse()` to call the local function.
Step 3: In `ingest.py`, add `from parsers.pdf_parser import extract_text_from_pdf` (only if still needed elsewhere — check if anything else calls it directly). If only `PDFParser.parse()` calls it, no import needed in `ingest.py`.

Actually, checking the codebase: `ingest.py` calls `get_parser(path).parse(path)` for all formats (line 176). The old `extract_text_from_pdf()` in `ingest.py` is only called by `load_new_pdfs()` (dead code) and by `parsers/pdf_parser.py`. So after deleting `load_new_pdfs()`, only `parsers/pdf_parser.py` uses it. Move it there cleanly.

---

### 2.7 Prompt Injection Mitigation

**All prompt files** use `.replace("{query}", query)` which is injectable.

For each LLM-based module, switch from `HumanMessage` with string interpolation to system/user message separation:

**File**: `intent_classifier.py:79-83`:

```python
# BEFORE:
prompt = _PROMPT_TEMPLATE.replace("{query}", query)
response = self._llm.invoke([HumanMessage(content=prompt)])

# AFTER:
from langchain_core.messages import SystemMessage
response = self._llm.invoke([
    SystemMessage(content=_PROMPT_TEMPLATE),
    HumanMessage(content=query),
])
```

Update `prompts/intent_classification.txt` to end with instruction like "Classify the following user query:" instead of containing `{query}` placeholder.

Apply same pattern to:
- `archetype_detector.py:99-103` (ArchetypeDetector.detect)
- `archetype_detector.py:108-113` (QueryReformulator.reformulate) — this one has `{query}`, `{archetype}`, `{domain_terms}`. Keep archetype/terms in system message, query in user message.
- `metadata_extractor.py:27-29`
- `relevance_checker.py:46-50` — has `{query}` and `{documents}`. Put template + documents in system, query in user.
- `query_resolver.py:29-33` — has `{chat_history}` and `{question}`. Put template + history in system, question in user.
- `tools/sql_tool.py:65-68` — has `{schema}`, `{samples}`, `{question}`. Put template + schema + samples in system, question in user.

Each prompt file needs minor adjustment: remove `{query}` (or `{question}`) placeholder, add instruction like "The user's query follows." The system message contains the template + any non-user data; the user message contains only the user's input.

---

## Phase 3: RAG Quality Improvements

### 3.1 Chunking Optimization

**File**: `config.py:7-8`:

```python
# BEFORE:
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# AFTER:
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
```

Note: Changing chunk size requires re-ingestion. Document this in the commit message.

### 3.2 Merge Archetype Detection + Reformulation

**File**: `archetype_detector.py`

Combine `ArchetypeDetector.detect()` and `QueryReformulator.reformulate()` into a single LLM call:

**New file**: `prompts/archetype_and_reformulation.txt`:

```
You are classifying a scientific query and optionally reformulating it.

Step 1: Classify the query into one of these archetypes:
- WHAT_INFORMATION: Seeks specific facts or data
- HOW_METHODOLOGY: Asks about methods or procedures
- COMPARISON: Compares two or more things
- DEFINITION: Asks for a definition or explanation
- WHY_REASONING: Asks about causes or reasoning
- SUMMARY: Requests an overview or summary

Step 2: If the query contains abbreviations or informal terms, reformulate it using precise scientific terminology. Otherwise, keep the original query.

Domain terms for reference:
{domain_terms}

Respond in this exact format:
ARCHETYPE: <one of the six types>
QUERY: <reformulated query or original if no changes needed>
```

**Modify** `archetype_detector.py` — Add combined method to `ArchetypeDetector`:

```python
def detect_and_reformulate(self, query: str, domain_terms: dict) -> tuple[str, str]:
    """Detect archetype and reformulate in a single LLM call.

    Returns (archetype, reformulated_query).
    """
    # Load combined prompt, invoke once, parse both ARCHETYPE and QUERY lines
    ...
```

**Update** `query.py:preprocess_query()` to use single combined call instead of two separate ones.

**Update** `_run_pipeline()` — Remove `reformulator` parameter, use combined method.

---

### 3.3 Cross-Encoder Upgrade

**File**: `config.py:23`:

```python
# BEFORE:
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# AFTER:
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
```

The `-L-12` variant is a drop-in replacement (same API, more accurate, slightly slower). No code changes needed — the model name is already configurable via `RERANK_MODEL`.

---

### 3.4 Better Citation Tracking

**File**: `prompts/qa_prompt.txt` (or wherever the QA prompt is defined) — Add citation instruction:

Add to the QA prompt:
```
When citing findings, reference the source document as [Source: filename, p.X].
```

**File**: `query.py` — `_doc_key()` (line ~100):

```python
# BEFORE:
def _doc_key(self, doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content[:200]}"

# AFTER:
import hashlib as _hashlib

def _doc_key(self, doc: Document) -> str:
    content_hash = _hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
    return f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{content_hash}"
```

---

### 3.5 Relevance Check Optimization

**File**: `relevance_checker.py:42` — Increase doc preview:

```python
# BEFORE:
doc_texts.append(f"[{i}] ({source}): {doc.page_content[:300]}")

# AFTER:
doc_texts.append(f"[{i}] ({source}): {doc.page_content[:500]}")
```

---

## Phase 4: Portfolio Presentation

### 4.1 README Rewrite

**File**: `README.md` — Major rewrite. Key additions:

1. **Architecture diagram** (Mermaid) at the top:
```markdown
## Architecture

```mermaid
graph TD
    A[User Query] --> B[Intent Classification]
    B -->|Greeting/Chitchat| C[Canned Response]
    B -->|Substantive| D[Archetype Detection]
    D --> E[Query Reformulation]
    E --> F[Metadata Extraction]
    F --> G{Agent Tool Selection}
    G -->|RAG| H[Hybrid Retrieval<br/>FAISS + BM25]
    G -->|SQL| I[Text-to-SQL]
    H --> J[Cross-Encoder Reranking]
    J --> K[Relevance Check]
    K -->|Low Score| E
    K -->|Relevant| L[Generate Answer with Sources]
    I --> L
```​
```

2. **Key Features section** highlighting the 13 "wow factors":
   - Multi-stage agentic pipeline with tool selection
   - Hybrid retrieval (FAISS + BM25 + cross-encoder reranking)
   - 6 query archetypes with dynamic retrieval weight tuning
   - Relevance checking with automatic retry
   - Multi-format ingestion (PDF, DOCX, XLSX) with table extraction
   - Text-to-SQL for large tabular data
   - Incremental ingestion with atomic saves
   - 9 independent feature flags
   - 157+ tests
   - Evaluation framework with golden dataset + RAGAS
   - Conversational follow-up resolution
   - Intent classification (skip RAG for non-substantive queries)
   - Experiment runner with parameter sweeps

3. **Link to HOW_IT_WORKS.md**

4. **Benchmark results**: "MRR: 0.860 on 25-query golden evaluation set"

5. **Updated project structure** listing ALL modules

6. **Mention `/docs` endpoint** for interactive API exploration

7. **Badges** at top: Python 3.13, Tests 157+, License MIT

### 4.2 Add LICENSE File

**New file**: `LICENSE` — MIT License with user's name and 2026 year.

### 4.3 Add GitHub Actions CI

**New file**: `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, dev/*]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/ -v
```

### 4.4 Minor Polish

**File**: `query.py:173` — Replace `sys.exit(1)` with exception:

```python
# BEFORE:
if not Path(VECTORSTORE_DIR).exists():
    logger.error("Vectorstore not found at %s. Run ingest.py first.", VECTORSTORE_DIR)
    sys.exit(1)

# AFTER:
if not Path(VECTORSTORE_DIR).exists():
    raise FileNotFoundError(
        f"Vectorstore not found at {VECTORSTORE_DIR}. Run ingest.py first."
    )
```

Remove `import sys` if no longer used.

**File**: `ingest.py:49` — Switch MD5 to SHA-256:

```python
# BEFORE:
def file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

# AFTER:
def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

Note: This will cause all documents to be re-ingested on next run (different hashes). Mention in commit message.

**File**: `requirements.txt` — Pin upper bounds:

```
langchain>=1.2,<2.0
langchain-classic>=1.0,<2.0
langchain-community>=0.4,<1.0
langchain-ollama>=1.0,<2.0
langchain-text-splitters>=1.1,<2.0
PyMuPDF>=1.27,<2.0
faiss-cpu>=1.13,<2.0
rank-bm25>=0.2.2,<1.0
sentence-transformers>=2.2.0,<4.0
fastapi>=0.109,<1.0
uvicorn[standard]>=0.27,<1.0
httpx>=0.27,<1.0
python-docx>=1.0,<2.0
openpyxl>=3.1,<4.0
pandas>=2.0,<3.0
```

---

## Phase 5: Advanced RAG (Future Enhancements)

### 5.1 Contextual Retrieval

**Concept**: For each chunk, generate a 1-2 sentence context summary explaining what the chunk covers within the document. Prepend this to the chunk before embedding.

**Implementation**:

**File**: `ingest.py` — Add `_generate_chunk_context()` after chunking, before embedding:

```python
def _generate_chunk_context(chunk_text: str, doc_title: str, section: str) -> str:
    """Generate a brief context prefix for a chunk (contextual retrieval)."""
    from utils import get_default_llm
    llm = get_default_llm()
    prompt = f"""Describe what this chunk covers in 1-2 sentences, given it comes from "{doc_title}", section "{section}":

{chunk_text[:500]}

Context summary:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
```

Add a feature flag in `config.py`:
```python
ENABLE_CONTEXTUAL_RETRIEVAL = os.getenv("ENABLE_CONTEXTUAL_RETRIEVAL", "false").lower() == "true"
```

In `chunk_documents()`, after creating chunks, optionally prepend context:
```python
if ENABLE_CONTEXTUAL_RETRIEVAL:
    for chunk in chunks:
        context = _generate_chunk_context(
            chunk.page_content,
            chunk.metadata.get("paper_title", ""),
            chunk.metadata.get("section_header", ""),
        )
        chunk.page_content = f"[Context: {context}]\n{chunk.page_content}"
```

Note: This is expensive (1 LLM call per chunk). Only enable for small corpora or batch processing. Consider caching context summaries.

### 5.2 Query Decomposition for Comparison Queries

**File**: `archetype_detector.py` — When archetype is COMPARISON, decompose:

```python
def decompose_comparison(self, query: str) -> list[str]:
    """Decompose a comparison query into sub-queries."""
    prompt = f"""Break this comparison question into 2-3 simpler sub-questions that can be answered independently:

Question: {query}

Sub-questions (one per line):"""
    response = self._llm.invoke([HumanMessage(content=prompt)])
    lines = [l.strip().lstrip("0123456789.-) ") for l in response.content.strip().split("\n") if l.strip()]
    return lines[:3]
```

**File**: `query.py` — In `_run_pipeline()`, when archetype is COMPARISON:

```python
if archetype == "COMPARISON":
    sub_queries = detector.decompose_comparison(question)
    all_docs = []
    for sq in sub_queries:
        docs = retriever.invoke(sq)
        all_docs.extend(docs)
    # Deduplicate and rerank combined docs
    ...
```

### 5.3 Evaluation Expansion

**File**: `eval/golden_dataset.json` — Create or expand:

```json
[
    {
        "query": "What methods were used in the Smith 2023 paper?",
        "expected_sources": ["smith_2023.pdf"],
        "expected_pages": [3, 4],
        "ideal_answer": "The Smith 2023 paper used...",
        "archetype": "HOW_METHODOLOGY"
    }
]
```

**File**: `eval/evaluate.py` — Add metrics:
- NDCG@K
- Per-stage latency tracking (wrap each pipeline stage with `time.perf_counter()`)
- Ablation mode: accept `--disable` flag to skip specific pipeline stages

### 5.4 RAPTOR (Hierarchical Summarization)

This is a larger feature. High-level approach:

1. After chunking, cluster chunks by semantic similarity (k-means on embeddings)
2. For each cluster, generate a summary using the LLM
3. Recursively cluster and summarize until a single root summary
4. Store summaries as additional documents in the vectorstore with `level` metadata
5. At retrieval time, search across all levels

Add feature flag: `ENABLE_RAPTOR = os.getenv("ENABLE_RAPTOR", "false").lower() == "true"`

This is best implemented as a separate module `raptor.py` with:
- `build_raptor_tree(chunks, embeddings) -> list[Document]`
- Integrate into `ingest.py` after chunking

---

## Verification Plan

After each phase, run:

```bash
# Run all tests
pytest tests/ -v

# Check imports work
python -c "import query; import ingest; import api; import utils"

# If Ollama is available, test end-to-end:
python query.py "What methods were used?"

# Test API starts:
timeout 5 uvicorn api:app --port 8765 || true
```

### Phase-specific checks:

**Phase 1**:
- Test that relevance check docs are passed through (add unit test)
- Test SQL blocklist rejects ATTACH, PRAGMA, LOAD_EXTENSION
- Test read-only connection rejects writes
- Verify CORS headers with: `curl -H "Origin: http://evil.com" http://localhost:8000/health -v`
- Verify error responses don't leak internals

**Phase 2**:
- Verify `from utils import tokenize, get_default_llm` works from all modules
- Verify `ask()` and `interactive()` produce same results as before
- Verify `model_copy()` creates independent retriever for API requests
- Run `python -c "import query; import ingest; import api"` to check no circular imports

**Phase 3**:
- Re-ingest with new chunk size and verify retrieval quality
- Run eval if golden dataset exists: `python eval/evaluate.py`
- Compare retrieval results with old vs new cross-encoder model

**Phase 4**:
- Verify README renders correctly on GitHub
- Verify Docker build works: `docker-compose build`
- Verify CI workflow syntax: `act -n` (if available) or push to branch

**Phase 5**:
- Each feature behind a flag — test with flag on and off
- Measure latency impact of contextual retrieval
- Run eval after each change to measure retrieval quality delta

---

## Implementation Order

Execute phases sequentially (1 → 2 → 3 → 4 → 5), with tests run after each phase.

Within each phase, steps should be done in order listed (dependencies exist between steps).

**Estimated scope**: ~30 files modified/created across all 5 phases.
