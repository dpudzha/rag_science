# RAG Science — Portfolio Deep Dive

A ground-up guide to this Retrieval-Augmented Generation system for scientific papers. Starts from the foundational concepts, walks through how the system is built, and ends with the advanced techniques that make it production-grade.

---

## What This Project Is

A question-answering system for scientific papers. You drop PDF/DOCX/XLSX files into a folder, the system processes them, and then you can ask natural language questions — it finds relevant passages and generates grounded answers with source citations.

Everything runs locally via Ollama (no cloud APIs, no API keys, no data leaving your machine). It's served through a FastAPI backend with real-time streaming and a React frontend.

---

## Part 1: The Fundamentals

### The Core Problem RAG Solves

Large Language Models know a lot, but they have two critical limitations:

1. **Knowledge cutoff**: They don't know about documents they weren't trained on (your private papers, recent publications).
2. **Hallucination**: When asked about something they don't know, they may confidently make things up rather than saying "I don't know."

**Retrieval-Augmented Generation (RAG)** solves both problems by adding a retrieval step before generation. Instead of relying on what the LLM memorized during training, you:

1. **Search** your own document collection for relevant passages
2. **Inject** those passages into the LLM's prompt as context
3. **Generate** an answer grounded in the retrieved evidence

The LLM doesn't need to "know" the answer — it just needs to read the retrieved passages and synthesize a response. This is the same thing a human does: find relevant sources, read them, then answer.

### Step 1: Document Ingestion — From Files to Searchable Data

Raw documents (PDFs, Word files, spreadsheets) aren't searchable by an LLM. They need to be transformed into a format that enables fast, accurate retrieval. This is the ingestion pipeline.

#### 1a. Parsing — Extracting Text from Documents

Different file formats require different extraction methods:

- **PDFs**: Use PyMuPDF (fitz) to extract text page by page. PDFs are notoriously messy — they store visual layout, not logical structure. The parser also extracts metadata (creation date, authors) and attempts to identify the paper title (heuristic: first non-empty line longer than 10 characters).
- **DOCX**: Use `python-docx` to extract paragraphs and tables.
- **XLSX**: Use `openpyxl`/`pandas` to extract spreadsheet data as DataFrames.

A parser factory (`parsers/__init__.py`) selects the right parser based on file extension — the rest of the system doesn't need to know what format the document was.

**Tables get special treatment**: Small tables (≤100 rows) are chunked row-by-row into the vector store. Large tables (>100 rows) go into a SQLite database where they can be queried with SQL — embedding hundreds of rows individually would be wasteful, and semantic search isn't great for tabular lookups anyway.

#### 1b. Chunking — Breaking Documents into Pieces

An LLM has a limited context window, and embedding models work best on focused passages. You can't embed an entire 30-page paper as one unit — the embedding would be a vague average of everything in the paper, matching lots of queries poorly rather than a few queries well.

**Chunking** splits documents into smaller, focused passages. This project uses:

- **Chunk size: 500 tokens** — large enough to contain a complete paragraph or argument, small enough to stay on-topic
- **Overlap: 50 tokens** — adjacent chunks share 50 tokens at their boundary, so sentences that span a chunk boundary aren't lost
- **Smart splitting**: The splitter tries to break at natural boundaries in this order of preference: double newlines (`\n\n`), single newlines (`\n`), sentence endings (`. `), spaces (` `). It avoids splitting mid-sentence whenever possible.

Each chunk gets enriched with context it would lose by being separated from the full document:

```
[Paper: Attention Is All You Need]
[Section: 3.2 Multi-Head Attention]
Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions...
```

These prefixes help the embedding model and LLM understand what the chunk is about even in isolation.

#### 1c. Metadata Enrichment

Every chunk carries structured metadata:

| Field | Example | Purpose |
|-------|---------|---------|
| source | "attention_paper.pdf" | Cite which document the answer came from |
| page | 5 | Point the user to the exact page |
| title | "Attention Is All You Need" | Display in citations |
| section | "3.2 Multi-Head Attention" | Show which section was relevant |
| creation_date | "2023-06-15" | Enable date-based filtering |
| authors | "Vaswani et al." | Enable author-based filtering |
| chunk_id | "attention_paper.pdf\|p5\|a1b2c3d4e5f6" | Stable unique identifier |

This metadata enables the filtering and citation features discussed later.

#### 1d. Embedding — Turning Text into Vectors

This is the key step that makes semantic search possible.

An **embedding model** converts text into a high-dimensional numerical vector (a list of numbers). The critical property: texts with similar meanings get vectors that are close together in this high-dimensional space.

```
"What optimizer was used to train the model?"  →  [0.12, -0.34, 0.56, ...]  (768 numbers)
"The model was trained using Adam optimizer"   →  [0.11, -0.33, 0.55, ...]  (similar vector)
"The weather is nice today"                    →  [0.87, 0.22, -0.91, ...]  (very different vector)
```

This project uses **nomic-embed-text** running locally via Ollama. Every chunk gets embedded, and the resulting vectors are stored in a FAISS index.

#### 1e. FAISS Index — The Vector Store

**FAISS** (Facebook AI Similarity Search) is a library for fast nearest-neighbor search over vectors. When you have thousands of chunk embeddings and a query embedding, FAISS finds the most similar chunks in milliseconds.

FAISS is an in-process library (not a separate server). This means:
- Zero infrastructure — no database to run, no network calls
- Fast cold starts — the index loads from disk in milliseconds
- Simple deployment — just a file on disk

The trade-off vs. a managed vector database (Pinecone, Weaviate): no built-in filtering, no replication, no multi-user access. For a local single-user tool, FAISS is the right choice.

#### 1f. BM25 Index — Keyword Search

Alongside the vector embeddings, the system builds a **BM25** index. BM25 is a classical keyword-matching algorithm (the same family of algorithms behind traditional search engines like Elasticsearch).

BM25 scores documents based on exact term frequency — how often the query words appear in the document, adjusted for document length and term rarity. No neural network, no embeddings — pure word matching.

**Why have BM25 when you already have embeddings?** Because they catch different things:

- Embeddings find **semantically similar** text: "cardiac arrest" matches "heart attack"
- BM25 finds **exact keyword matches**: "BERT" matches "BERT" (an embedding model might also match "transformer" or "language model," diluting the result)

For scientific text with precise terminology, you need both. This is called **hybrid retrieval** — combining them is one of this project's key techniques (covered in Part 3).

The BM25 index is built during ingestion and saved as a pickle file. It loads from disk at query time, just like the FAISS index.

#### 1g. Incremental Ingestion

Re-embedding an entire corpus every time you add one document would be slow and wasteful. This system tracks which files have been processed using SHA-256 hashes stored in `ingested.json`. When you re-run ingestion:

- New files get processed and added
- Changed files (different hash) get reprocessed
- Unchanged files are skipped entirely

#### 1h. Atomic Saves — Preventing Corruption

The vector store save uses an atomic swap pattern:

1. Build all indices in a temporary directory (`vectorstore.tmp.{random}`)
2. Rename the existing directory to a backup
3. Rename the temp directory to the final location
4. Delete the backup

If the process crashes at any point, either the old or new data is intact — never a corrupted half-written state. This is the same pattern databases use.

### Step 2: Retrieval — Finding Relevant Passages

When a user asks a question, it needs to be turned into the same vector representation as the stored chunks, then matched against them.

#### The Basic Retrieval Flow

```
User question: "What optimizer was used?"
  │
  ├─ 1. Embed the question using the same model (nomic-embed-text)
  │     → [0.12, -0.34, 0.56, ...] (768-dim vector)
  │
  ├─ 2. Search FAISS for nearest neighbors
  │     → Top 20 most semantically similar chunks
  │
  ├─ 3. Search BM25 for keyword matches
  │     → Top 20 chunks with highest term-frequency scores
  │
  ├─ 4. Merge and deduplicate results
  │     → Combined ranked list (details in Part 3)
  │
  └─ 5. Return top 4 chunks as context for the LLM
```

The number of initial candidates (20) is intentionally much larger than the final result count (4). This gives the reranking and filtering stages enough material to work with.

### Step 3: Answer Generation — The LLM Step

The retrieved chunks are injected into a prompt alongside the user's question:

```
System: You are a helpful assistant that answers questions based on the
provided context. If the context doesn't contain the answer, say so.

Context:
[Paper: Attention Is All You Need, Section: 5.1 Training]
We used the Adam optimizer with β1 = 0.9, β2 = 0.98, and ε = 10−9...

[Paper: Attention Is All You Need, Section: 5.2 Hardware]
We trained our models on 8 NVIDIA P100 GPUs...

Question: What optimizer was used?
```

The LLM reads the context and generates an answer grounded in the evidence. Because the answer is based on specific retrieved passages, the system can cite exactly which document, page, and section the answer came from.

This project uses models running locally via **Ollama** — an open-source tool that runs LLMs on your own hardware. The default model is `gemma3:12b` (12 billion parameters). The architecture is model-agnostic: swapping to a different Ollama model or even a cloud API requires changing one config value.

### Summary: The Naive RAG Pipeline

At its simplest, the entire system is:

```
Documents → Parse → Chunk → Embed → Store in FAISS
                                          │
User Question → Embed → Search FAISS → Top chunks → LLM → Answer
```

This is "naive RAG" — it works, but it has many failure modes. The rest of this guide covers the techniques that make it production-grade.

---

## Part 2: The Ingestion Pipeline in Detail

The full ingestion pipeline goes beyond basic chunking:

```
Documents (PDF/DOCX/XLSX)
  │
  ├─ 1. Format-specific parsing (PyMuPDF, python-docx, openpyxl)
  ├─ 2. Title and metadata extraction
  ├─ 3. Table extraction and routing
  │     ├─ Small tables (≤100 rows) → chunked as rows into vectorstore
  │     └─ Large tables (>100 rows) → SQLite database
  ├─ 4. Text chunking (500 tokens, 50 token overlap)
  │     ├─ Section header detection via regex
  │     └─ Page number tracking via binary search (bisect)
  ├─ 5. Metadata enrichment per chunk
  ├─ 6. Embedding (Ollama nomic-embed-text)
  ├─ 7. FAISS index construction
  ├─ 8. BM25 index construction + pickle serialization
  └─ 9. Atomic save (temp dir → rename swap)
```

### Section Header Detection

When chunking, the system looks backward from each chunk's position to find the most recent section header using regex patterns (matching common formats like "3.2 Methods", "## Results", "INTRODUCTION"). This header becomes part of the chunk's metadata and prefix, giving the embedding model and LLM valuable structural context.

### Page Number Tracking

After chunking, each chunk needs to know which page it came from. The system tracks cumulative character offsets per page during parsing, then uses binary search (`bisect`) to map each chunk's character position back to a page number. This is more efficient than checking every page boundary linearly.

### Parent-Document Retrieval (Optional)

An advanced strategy where embedding and retrieval use different chunk sizes:

- **Embed small "child" chunks** (400 tokens) — smaller passages match queries more precisely
- **Return larger "parent" chunks** (500 tokens) — the LLM gets more context for answer generation

The child chunks are stored in FAISS for searching, but when a child is found, its parent chunk is what gets sent to the LLM. This separates the "finding" unit from the "reading" unit.

---

## Part 3: Advanced Retrieval Techniques

This is where the system goes beyond naive RAG. Each technique addresses a specific failure mode.

### Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

**The problem**: Dense (embedding-based) retrieval catches semantic similarity but misses exact keyword matches. BM25 catches keywords but misses paraphrased concepts. Neither alone is sufficient for scientific text with precise terminology.

**The solution**: Run both in parallel and merge their rankings.

But you can't just average scores — FAISS cosine similarity and BM25 scores are on completely different scales. A FAISS score of 0.85 and a BM25 score of 12.3 aren't comparable.

**Reciprocal Rank Fusion (RRF)** solves this by working with ranks instead of scores:

```
For each document d:
  score(d) = bm25_weight / (rrf_k + bm25_rank) + dense_weight / (rrf_k + dense_rank)
```

Where `rrf_k = 60` is a smoothing constant. A document ranked #1 by both systems gets the highest fused score. A document ranked #1 by one and #50 by the other gets a moderate score. RRF doesn't care about the absolute score values — only relative ordering — making it robust across different scoring systems.

Documents are deduplicated before fusion using a content-based key (`source:page:content[:200]`) so that the same passage found by both FAISS and BM25 isn't counted twice.

### Archetype-Aware Weight Adjustment

Different types of questions benefit from different BM25/dense balances:

| Archetype | BM25 Weight | Dense Weight | Rationale |
|-----------|-------------|--------------|-----------|
| DEFINITION | 0.5 | 0.5 | Definitions need exact terms AND semantic context |
| WHAT_INFORMATION | 0.4 | 0.6 | Slightly favor semantic for factual queries |
| HOW_METHODOLOGY | 0.3 | 0.7 | Methods described with varied vocabulary |
| COMPARISON | 0.3 | 0.7 | Comparing concepts requires semantic understanding |
| WHY_REASONING | 0.2 | 0.8 | Reasoning queries rarely match keywords directly |
| SUMMARY | 0.2 | 0.8 | Summaries need broad semantic coverage |

For example, "What is BERT?" benefits from BM25 (the exact acronym matters), while "Why does the model generalize well?" benefits from dense retrieval (the answer might use words like "regularization" or "dropout" that don't appear in the query).

### Cross-Encoder Reranking

**The problem**: The embedding model (a "bi-encoder") encodes the query and each document independently. It can't model fine-grained word-by-word interactions between them. It's fast (you embed once, search millions) but imprecise.

**The solution**: After hybrid retrieval returns ~20 candidates, a **cross-encoder** (`BAAI/bge-reranker-v2-m3`) re-scores them. Unlike a bi-encoder, a cross-encoder takes the query AND the document as a single combined input, enabling full token-level attention between them.

```
Bi-encoder:   embed(query) · embed(document)  →  fast, rough score
Cross-encoder: score(query + document)          →  slow, precise score
```

This is a classic two-stage pattern in information retrieval:
1. **Recall stage** (FAISS + BM25): Cast a wide net, get many candidates cheaply
2. **Precision stage** (cross-encoder): Re-score the candidates accurately

The cross-encoder is loaded once as a module-level singleton (lazy-loaded on first use) to avoid the ~2 second model loading time on every query. Its top score is cached for use as a relevance proxy (see below).

### Metadata Extraction and Filtering

**The problem**: "What did Smith et al. find in their 2023 paper?" — the answer requires filtering by author and date, not just semantic similarity.

**The solution**: An LLM extracts structured metadata (dates, authors, paper names, source filenames) from the query. These become filters applied to the retrieval results.

**Key design choice**: Filters are applied *after* reranking but *before* the final top-k slice. Why this order?

- If you filter first, a slightly wrong filter (user says "2023" but the paper's metadata says "2022-12-15") could eliminate the correct document before it's ever scored
- By reranking all candidates first, you get the best possible relevance ordering, then narrow by metadata as a refinement
- If filters eliminate everything, the system falls back to unfiltered results rather than returning nothing

### Relevance Checking with Retry

**The problem**: Sometimes the corpus simply doesn't contain the answer, or the query is too vague to retrieve anything useful. Generating an answer from irrelevant passages produces confident-sounding nonsense.

**The solution**: A two-tier relevance check before answer generation:

1. **Fast path (cross-encoder proxy)**: The reranker already scored the top documents. Apply sigmoid normalization to its logit score (`1/(1+exp(-score))`) and compare to a threshold (default 0.6). This costs zero additional computation — you're reusing a score you already have.

2. **Slow path (LLM-based)**: If the cross-encoder score isn't available, ask the LLM to score relevance 0.0-1.0 and optionally suggest a better query formulation.

If relevance is below threshold, the system retries once with either the LLM's suggestion or a reformulated query. A `PreloadedRetriever` wrapper avoids re-running the entire retrieval pipeline on retry — it returns pre-fetched documents directly.

**Why this is clever**: The cross-encoder proxy reuses a model you've already paid for (reranking) as a free relevance signal, eliminating an LLM round-trip in the common case (happy path).

---

## Part 4: Query Preprocessing Pipeline

Before retrieval even happens, the query goes through several processing stages. Each one addresses a different source of retrieval failure.

### Follow-up Resolution

**The problem**: In a conversation, users say things like "What about its performance?" — the pronoun "its" refers to something from earlier in the conversation.

**The solution**: Regex patterns detect follow-up indicators (pronouns, demonstratives like "this", "that", "those"). If found, the LLM rewrites the question to be self-contained using conversation history. If no indicators are found, the query passes through unchanged (avoiding an unnecessary LLM call).

### Intent Classification

**The problem**: Running the full RAG pipeline for "Hello" or "Thanks!" wastes compute and returns nonsensical source citations.

**The solution**: A two-tier classifier:

1. **Fast regex patterns**: Obvious greetings ("hi", "hello", "hey") and farewells ("bye", "goodbye") are caught instantly with no LLM call.
2. **LLM fallback**: Ambiguous cases go to the LLM, which classifies into GREETING, FAREWELL, CHITCHAT, or SUBSTANTIVE.

Non-substantive queries get canned responses and skip the entire retrieval pipeline. Falls back to SUBSTANTIVE on any error — better to run RAG unnecessarily than to skip it for a real question.

### Archetype Detection + Query Reformulation

**The problem**: User queries use casual language that doesn't match the formal terminology in scientific papers. "How does the ML model work?" won't match passages about "machine learning architecture."

**The solution**: A single LLM call does two things:

1. **Archetype detection**: Classifies the query into one of six types (WHAT_INFORMATION, HOW_METHODOLOGY, COMPARISON, DEFINITION, WHY_REASONING, SUMMARY) to adjust retrieval weights.
2. **Query reformulation**: Expands abbreviations using a terminology dictionary (ML → machine learning, BERT → Bidirectional Encoder Representations from Transformers, etc.) and rewrites with domain-specific terms.

**Drift protection**: The reformulator includes a safety check that rejects rewrites where:
- Content token overlap drops below 50% (the rewrite wandered too far from the original)
- Named entities from the original are missing
- For methodology queries, new method terms were injected that weren't in the original

This prevents the LLM from "hallucinating" a different question during reformulation.

**Why combine in one LLM call**: Two separate calls would double latency (~2-4 extra seconds). Since both tasks operate on the same input, combining them in a single prompt halves the LLM-attributed cost.

### The Complete Query Pipeline

Putting it all together, every question passes through:

```
User Query
  │
  ├─ 1. Follow-up Resolution ── resolve pronouns using conversation history
  │
  ├─ 2. Intent Classification ── greeting/chitchat? → canned response (skip RAG)
  │
  ├─ 3. Archetype Detection + Query Reformulation ── single LLM call
  │     ├─ classify query type (6 archetypes)
  │     ├─ adjust retrieval weights per archetype
  │     └─ rewrite query with domain terminology
  │
  ├─ 4. Metadata Extraction ── extract dates, authors, sources for filtering
  │
  ├─ 5. Hybrid Retrieval + Reranking
  │     ├─ FAISS dense search (semantic similarity)
  │     ├─ BM25 keyword search (lexical matching)
  │     ├─ Reciprocal Rank Fusion (merge rankings)
  │     ├─ Cross-encoder reranking (fine-grained relevance)
  │     └─ Metadata filtering (post-rerank, pre-slice)
  │
  ├─ 6. Relevance Check ── below threshold? → reformulate + retry once
  │
  └─ 7. Answer Generation ── LLM generates answer grounded in retrieved docs
```

Every stage is independently toggleable via feature flags and falls back gracefully on error:

| Stage | Fallback on Error | Rationale |
|-------|-------------------|-----------|
| Intent classification | SUBSTANTIVE | Better to run RAG unnecessarily than miss a real question |
| Archetype detection | WHAT_INFORMATION | Most common type, balanced weights |
| Query reformulation | Original query | The original still works for retrieval |
| Metadata extraction | Empty dict | No filtering, all docs considered |
| Relevance check | Assume relevant (1.0) | Better to return possibly-irrelevant results than nothing |

---

## Part 5: The Agentic Tool System

### Why an Agent?

**The problem**: Not all data fits neatly into a vector store. Large tabular data (>100 rows) can't be meaningfully chunked and embedded — "What's the average value in column X?" requires computation, not passage retrieval.

**The solution**: An optional LangChain agent that chooses between two tools:

1. **`search_papers`** (RAG tool): Wraps the hybrid retriever for passage-based questions.
2. **`query_tables`** (SQL tool): Generates SQL from natural language, executes against SQLite, and returns formatted results.

The agent reads the question and decides which tool fits. For "What does the paper say about transformers?" it uses RAG. For "What's the average accuracy across all experiments?" it uses SQL.

On agent failure, the system falls back to direct RAG retrieval — the agent is an enhancement, not a hard dependency.

### SQL Safety

The SQL tool has three layers of defense against injection:

1. **Regex blocklist**: Blocks DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, REPLACE, EXEC, ATTACH, DETACH, PRAGMA, and more — only SELECT is allowed
2. **Read-only connections**: SQLite URI mode `?mode=ro` — even if the blocklist is bypassed, writes physically can't happen
3. **Parameterized queries**: Schema introspection uses `WHERE name=?` instead of string interpolation

---

## Part 6: The API Layer

### FastAPI with SSE Streaming

The backend provides two query modes:

- **`POST /query`**: Returns the complete answer in a single JSON response. Simple but the user waits for everything.
- **`POST /query/stream`**: Server-Sent Events (SSE) stream with event types:
  - `metadata` — sources, session_id, relevance_score, tool_used, retry_count (emitted immediately after retrieval, before answer generation starts)
  - `token` — individual answer tokens as they're generated
  - `done` — stream complete
  - `error` — something went wrong

Streaming lets the frontend show retrieval metadata (sources, relevance) immediately while the answer is still being generated token by token.

### Session Management

In-memory sessions with bounded resource usage:

- **TTL**: 1 hour per session (configurable)
- **Max sessions**: 100 (oldest evicted when full)
- **Max history**: 20 exchanges per session (sliding window)
- **Thread safety**: `RLock` guards all session access

For a local-first tool, in-memory is simpler and faster than Redis or a database. The TTL and max limits prevent unbounded memory growth. If scaling to many concurrent users, swapping to Redis would be straightforward — the session interface is already dict-based.

### Thread Safety

The `HybridRetriever` holds mutable state (weights, metadata filters) that varies per request. Without protection, concurrent requests would corrupt each other:

```python
# UNSAFE: Request A sets weights for SUMMARY while Request B is mid-retrieval
retriever.dense_weight = 0.8  # ← Request A
retriever.bm25_weight = 0.2   # ← Request A overwrites B's weights
```

Solution: `model_copy()` creates a per-request copy with independent state:

```python
request_retriever = _retriever.model_copy()  # each request gets its own copy
```

### Error Sanitization

The API returns generic error messages to clients ("An error occurred while processing your query") while logging full stack traces server-side. This prevents leaking internal state, file paths, or model details — a basic but important security practice.

### Runtime Configuration

The config system supports hot-reloading without restart:

- `GET /config` — returns current tunable parameters
- `PUT /config` — applies changes in memory (validates types)
- `POST /config/save` — persists to `config.json`
- `POST /config/load` — restores from `config.json`

This enables experimenting with retrieval parameters (chunk size, top-k, weights) without restarting the server.

---

## Part 7: Security

### Prompt Injection Mitigation

All LLM prompts use **SystemMessage/HumanMessage separation**:

```python
response = llm.invoke([
    SystemMessage(content=system_prompt),  # instructions (from file)
    HumanMessage(content=user_input),      # user data (isolated)
])
```

User input never enters the system prompt — it's always in a separate HumanMessage. This makes it harder for adversarial inputs to override system instructions. Prompt templates are loaded from files, not constructed from user input.

### Request Validation

- Question length: 1-5000 characters (prevents payload abuse)
- Session ID: 1-128 alphanumeric characters (prevents injection)
- Pydantic validators enforce types at the API boundary

### Non-root Docker

The container runs as `appuser` instead of root. If the application is compromised, the attacker has limited filesystem access — a standard container hardening practice.

---

## Part 8: Frontend

### React + Vite + TypeScript

A minimal single-page app with three panels:

- **Chat**: Message list with SSE streaming, source citations, session management
- **Ingest**: Trigger document ingestion, view status
- **Config**: Grouped parameter editor with Apply/Save/Load

No component library — plain CSS with a dark theme. Keeps the bundle small and avoids dependency on a UI framework.

### SSE Streaming Implementation

The frontend consumes streaming events via the `ReadableStream` API:

```typescript
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  // Parse SSE lines: "event: token\ndata: {...}\n\n"
  // Dispatch to callbacks: onToken, onMetadata, onDone, onError
}
```

This provides real-time token-by-token display while showing retrieval metadata (sources, relevance score) before the answer even starts generating.

### Dev/Prod Proxy

- **Development**: Vite proxies `/api/*` to `http://localhost:8000/*` (strips `/api` prefix)
- **Production**: Nginx proxies `/api/*` to the backend container, serves static files for everything else (SPA fallback)

---

## Part 9: Testing & Evaluation

### Test Suite

194 pytest tests across 17 test files covering every module:

- **Mocking external dependencies**: All Ollama calls, FAISS operations, and file I/O are mocked. Tests never require a running Ollama instance or real vectorstore.
- **Config patching**: Feature flags toggled per-test using `patch()` context managers, ensuring test isolation.
- **State reset**: API singletons (cached QA chain, retriever) are reset between tests to prevent cross-test contamination.
- **Boundary testing**: Edge cases — empty queries, oversized inputs, malformed LLM responses, concurrent access patterns.

### Retrieval Evaluation Framework

A golden dataset framework (`eval/evaluate.py`) measures retrieval quality:

- **Mean Reciprocal Rank (MRR)**: Average of `1/rank` of the first relevant document. MRR = 1.0 means the correct document is always ranked first.
- **Recall@K**: Fraction of expected source documents found in the top K results. Measures whether relevant documents are being retrieved at all.
- **Chunk-level precision/recall**: If the golden dataset includes specific chunk IDs, measures whether the right passages (not just the right documents) are retrieved.

Golden dataset format:
```json
[
  {
    "question": "What optimizer was used?",
    "expected_sources": ["paper1.pdf"],
    "expected_pages": [3, 4],
    "expected_chunk_ids": ["paper1.pdf|p3|0a1b2c3d4e5f"]
  }
]
```

This enables quantitative measurement — you can verify that a new chunking strategy or weight adjustment actually improves results rather than relying on gut feeling.

---

## Part 10: Docker & Deployment

### Multi-Container Setup

```yaml
services:
  rag:        # Python API on port 8000
  frontend:   # React app on port 3000 (Nginx on port 80 in container)
  ollama:     # LLM inference on port 11434 (with GPU passthrough)
```

The backend connects to Ollama via Docker internal DNS (`http://ollama:11434`). Volumes persist `papers/` and `vectorstore/` across container restarts.

### Backend Dockerfile

- Non-root user (`appuser`) for security
- Healthcheck: periodic HTTP GET to `/health` with 5s timeout, 3 retries
- 10s start period to allow model loading before first health check

### Frontend Dockerfile

Multi-stage build:
1. **Build stage** (node:20-alpine): Installs dependencies, runs `npm run build`
2. **Serve stage** (nginx:alpine): Copies built static files, serves via Nginx with API proxy

---

## Part 11: Design Patterns & Code Quality

### Feature Flags

Every pipeline stage can be independently disabled via environment variables:

```bash
INTENT_CLASSIFICATION_ENABLED=true
ARCHETYPE_DETECTION_ENABLED=true
QUERY_REFORMULATION_ENABLED=true
METADATA_EXTRACTION_ENABLED=true
RELEVANCE_CHECK_ENABLED=true
USE_CROSS_ENCODER_RELEVANCE=true
ENABLE_SQL_AGENT=false
ENABLE_TABLE_EXTRACTION=true
ENABLE_PARENT_RETRIEVAL=false
```

This serves multiple purposes:
1. **Debugging**: Disable stages to isolate issues
2. **Performance tuning**: Skip expensive stages when latency matters more than quality
3. **A/B testing**: Compare pipeline variants quantitatively using the eval framework
4. **Graceful degradation**: If a model fails, disable its stage without crashing the system

### Singleton Lazy Loading

Expensive resources (cross-encoder model, FAISS index, LLM) are loaded once on first use and cached at module level:

```python
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder
```

This means cold starts are fast (nothing loads until needed) and subsequent requests reuse already-loaded resources.

### Graceful Degradation

The system is designed to get worse, not break. Every LLM-based stage catches exceptions and falls back to a safe default. A transient Ollama timeout doesn't show an error — it shows a slightly less optimized but still functional result.

---

## Part 12: What Could Be Improved

### Retrieval Quality
- **Cross-encoder upgrade**: Replace `bge-reranker-v2-m3` with a newer, more capable reranker model
- **Learned sparse retrieval**: Replace BM25 with SPLADE (a neural sparse model) for better keyword-semantic hybrid performance
- **Query expansion**: Use pseudo-relevance feedback — extract key terms from top results and expand the query for a second retrieval pass
- **Embedding model upgrade**: `nomic-embed-text` is functional but not state-of-the-art; a stronger embedding model would improve dense retrieval directly

### Architecture
- **Persistent sessions**: Replace in-memory with Redis/SQLite for crash resilience and multi-instance deployments
- **Async ingestion**: Currently blocks on `POST /ingest`. A background task queue would let the API respond immediately
- **Streaming with agent**: SSE endpoint doesn't support the agent path — would require streaming tool call results

### Evaluation
- **CI integration**: Run the eval framework in CI to catch retrieval quality regressions
- **End-to-end evaluation**: Current eval only measures retrieval (MRR, Recall@K). Adding answer quality evaluation (LLM-as-judge) would measure the full pipeline
- **Larger golden dataset**: More diverse test questions across archetypes and document types

---

## What You Should Take Away From This Project

### RAG Engineering

Production RAG is far more than "embed documents, retrieve top-k, generate answer." A robust system needs:

1. **Multi-modal retrieval**: Combining dense and sparse retrieval covers both semantic and lexical matching gaps
2. **Reranking**: Bi-encoder retrieval is a fast first pass; cross-encoder reranking provides the precision needed for quality answers
3. **Relevance gating**: Not every retrieval is good enough to generate from — checking and retrying prevents confident-sounding wrong answers
4. **Query understanding**: Classifying intent, detecting query type, and reformulating queries all improve retrieval before it even happens
5. **Graceful degradation**: Every component should have a fallback. The system should get worse, not break.

### System Design

- **Feature flags** enable incremental development and A/B testing
- **Atomic operations** prevent data corruption
- **Thread safety** through immutable copies rather than locks on shared mutable state
- **Lazy loading** of expensive resources for fast startup
- **Error sanitization** at the API boundary (generic messages to clients, details in logs)

### Security in ML Systems

- SQL injection prevention through defense-in-depth (blocklist + read-only + parameterized queries)
- Prompt injection mitigation through message role separation
- Request validation at system boundaries
- Non-root containers in production

### Local-First AI

This project shows that a sophisticated AI application can run entirely locally. No cloud APIs, no API keys, no data leaving the machine. The trade-off is model quality (local models vs. frontier models), but the architecture makes swapping models trivial.

---

## Project Structure Reference

```
rag_science/
├── config.py                 # All configuration with env var overrides
├── utils.py                  # Shared utilities (tokenizer, LLM factory)
├── ingest.py                 # Document ingestion pipeline
├── query.py                  # Query pipeline orchestrator
├── retriever.py              # Hybrid retrieval (FAISS + BM25 + RRF + reranking)
├── api.py                    # FastAPI REST layer with SSE streaming
├── agent.py                  # LangChain agent with RAG + SQL tools
├── intent_classifier.py      # Greeting/chitchat routing
├── archetype_detector.py     # Query type classification + reformulation
├── metadata_extractor.py     # Structured metadata extraction from queries
├── relevance_checker.py      # Relevance scoring with retry
├── query_resolver.py         # Follow-up question resolution
├── health.py                 # Ollama health checks with backoff
├── logging_config.py         # Centralized logging setup
├── domain_terminology.json   # Abbreviations and synonyms for reformulation
├── parsers/
│   ├── __init__.py           # Parser factory (selects by file extension)
│   ├── pdf_parser.py         # PDF text/table extraction (PyMuPDF)
│   ├── docx_parser.py        # DOCX parsing (python-docx)
│   ├── xlsx_parser.py        # XLSX parsing (openpyxl/pandas)
│   └── table_extractor.py    # PDF table extraction
├── tools/
│   ├── rag_tool.py           # LangChain tool wrapping HybridRetriever
│   ├── sql_tool.py           # LangChain tool for text-to-SQL
│   └── sql_database.py       # SQLite wrapper with safety checks
├── prompts/                  # All LLM prompt templates
├── eval/
│   ├── evaluate.py           # MRR/Recall@K evaluation framework
│   └── golden_dataset.json   # Test questions with expected results
├── tests/                    # 194 pytest tests across 17 files
├── frontend/                 # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── App.tsx           # Layout and tab routing
│   │   ├── api.ts            # API client with SSE streaming
│   │   ├── types.ts          # Shared TypeScript types
│   │   └── components/
│   │       ├── ChatPanel.tsx  # Chat UI with streaming
│   │       ├── Sidebar.tsx    # Navigation
│   │       ├── IngestPanel.tsx# Document ingestion UI
│   │       └── ConfigPanel.tsx# Runtime config editor
│   ├── nginx.conf            # Production proxy config
│   └── Dockerfile            # Multi-stage build (node → nginx)
├── Dockerfile                # Backend container (non-root)
├── docker-compose.yml        # Full stack: backend + frontend + Ollama
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Dev dependencies (pytest, etc.)
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| LangChain | LLM orchestration, chains, agents, tool framework |
| PyMuPDF (fitz) | PDF text and table extraction |
| FAISS (faiss-cpu) | In-process vector similarity search |
| rank-bm25 | BM25 keyword retrieval algorithm |
| sentence-transformers | Cross-encoder reranking model |
| Ollama | Local LLM inference (embedding + generation) |
| FastAPI + Uvicorn | Async REST API with SSE support |
| React + Vite | Frontend with hot-reload development |
