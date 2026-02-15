# How RAG Science Works

A high-level guide to understanding this Retrieval-Augmented Generation (RAG) system for scientific papers.

---

## What Is RAG?

RAG is a pattern where you **don't** fine-tune an LLM on your data. Instead, you:

1. Store your documents in a searchable index
2. When a question comes in, **retrieve** the most relevant chunks from that index
3. Feed those chunks as context to the LLM and ask it to **generate** an answer

The LLM never "memorizes" your papers — it reads relevant excerpts on the fly.

---

## The Two Main Stages

### Stage 1: Ingestion (`ingest.py`) — PDF to Searchable Index

This is the offline/preparation step. You run it once (and again when you add new papers).

```
PDF files  →  Extract text  →  Split into chunks  →  Compute embeddings  →  Save to FAISS index
```

#### Step-by-step:

**1. Extract text from PDFs** — `fitz` (PyMuPDF library)

```python
fitz.open(pdf_path)         # opens a PDF
page.get_text("text")       # extracts plain text from each page
```

PyMuPDF reads each page of a PDF and returns raw text. We keep track of which page each piece of text came from so we can cite it later.

**2. Skip already-ingested files** — `hashlib.md5`

```python
hashlib.md5(file_bytes).hexdigest()   # fingerprint each PDF
```

Before processing, we compute an MD5 hash of each file and check it against `ingested.json`. If the hash is already recorded, the file is skipped. This makes re-running `ingest.py` fast — it only processes new or changed PDFs.

**3. Split text into chunks** — `RecursiveCharacterTextSplitter` (LangChain)

```python
RecursiveCharacterTextSplitter(
    chunk_size=1500,                              # max characters per chunk
    chunk_overlap=200,                            # overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""]      # split hierarchy
)
```

An LLM has a limited context window and embeddings work better on focused passages. This splitter tries to break text at paragraph boundaries first (`\n\n`), then newlines, then sentences (`. `), then words. The 200-char overlap ensures that a sentence split across two chunks still appears fully in at least one of them.

We concatenate all pages of a PDF into one long string first, then split. A `bisect`-based lookup maps each chunk back to its original page number for citations.

**4. Compute embeddings** — `OllamaEmbeddings` with `nomic-embed-text`

```python
OllamaEmbeddings(model="nomic-embed-text")
```

An embedding model converts text into a dense numerical vector (a list of ~768 floats). Texts with similar meaning get vectors that are close together in this vector space. `nomic-embed-text` runs locally via Ollama — no API keys needed.

**5. Store in FAISS** — `FAISS.from_documents()`

```python
FAISS.from_documents(chunks, embeddings)   # build the index
store.save_local("./vectorstore/")         # persist to disk
```

FAISS (Facebook AI Similarity Search) is a library optimized for fast nearest-neighbor search over vectors. It stores all chunk embeddings in an index file on disk. When querying, it can find the most similar vectors in milliseconds, even with thousands of chunks. If a vectorstore already exists, new chunks are merged into it.

---

### Stage 2: Querying (`query.py`) — Question to Answer

This is the online/runtime step. It runs each time you ask a question.

```
Question  →  Hybrid retrieval (FAISS + BM25)  →  Cross-encoder reranking  →  LLM generates answer
```

#### Step-by-step:

**1. Load the index and build retrievers**

```python
vectorstore = FAISS.load_local(...)                  # load the saved FAISS index
bm25 = BM25Okapi(tokenized_corpus)                   # build BM25 index from all chunks
cross_encoder = CrossEncoder("ms-marco-MiniLM-L-6-v2")  # load reranker model
```

Three components are initialized:
- The FAISS vector index (for semantic/meaning-based search)
- A BM25 index (for keyword/term-frequency-based search)
- A cross-encoder model (for reranking)

**2. Hybrid retrieval** — combining two search strategies

**Dense retrieval (FAISS):** Embeds the question into a vector and finds chunks whose vectors are closest. Good at understanding meaning — "cardiac arrest" matches "heart attack" even though the words differ.

```python
vectorstore.similarity_search_with_score(query, k=20)
```

**Sparse retrieval (BM25):** Classic keyword search based on term frequency. Good at exact term matching — if the question mentions "CRISPR", BM25 will prioritize chunks containing that exact word.

```python
bm25.get_scores(tokenized_query)    # score every chunk by keyword overlap
```

`BM25Okapi` (from `rank_bm25`) is the Okapi BM25 algorithm — it scores documents by how many query terms they contain, weighted by how rare those terms are across the corpus (TF-IDF style).

**Combining scores:** Each retriever returns its top 20 candidates. Scores are normalized to 0–1 and combined with weights: 70% dense + 30% BM25. Documents found by both retrievers get a boosted combined score. A content-based key (source + page + first 200 chars) ensures the same chunk from both sources is correctly merged.

**3. Cross-encoder reranking** — `CrossEncoder` (sentence-transformers)

```python
pairs = [[query, doc.page_content] for doc in candidates]
rerank_scores = cross_encoder.predict(pairs)
```

The hybrid step is fast but approximate. The cross-encoder is a more powerful (but slower) model that reads the question and each candidate chunk together as a pair and produces a relevance score. It uses the `ms-marco-MiniLM-L-6-v2` model, trained specifically for passage ranking. The top 4 chunks after reranking are kept as final context.

Why not use the cross-encoder for everything? It's too slow — it must process every chunk paired with the query. The hybrid step narrows thousands of chunks down to ~20 candidates, making cross-encoder reranking feasible.

**4. Generate the answer** — `ChatOllama` with `llama3.1:8b`

```python
ChatOllama(model="llama3.1:8b", temperature=0)
```

The top 4 chunks are inserted into a prompt template as context. The LLM (`llama3.1:8b` running locally via Ollama) reads the context and the question, then generates an answer. `temperature=0` means deterministic output — no randomness, same question gives same answer.

**5. Conversational chain** — `ConversationalRetrievalChain` (LangChain)

```python
ConversationalRetrievalChain.from_llm(llm, retriever, ...)
```

This LangChain component wraps everything into a chain that also accepts `chat_history`. In interactive mode, previous Q&A pairs are passed along so the LLM can handle follow-up questions like "Can you explain that further?" or "What about the second study?"

---

## Data Flow Diagram

```
./papers/*.pdf
      │
      ▼
┌─────────────┐     fitz (PyMuPDF)
│  Extract     │──── reads each PDF page → raw text
│  Text        │
└──────┬──────┘
       │
       ▼
┌─────────────┐     RecursiveCharacterTextSplitter
│  Chunk       │──── splits text into 1500-char pieces with 200-char overlap
│  Text        │     tries paragraph → line → sentence → word boundaries
└──────┬──────┘
       │
       ▼
┌─────────────┐     OllamaEmbeddings (nomic-embed-text)
│  Embed       │──── converts each chunk into a ~768-dimension vector
│  Chunks      │
└──────┬──────┘
       │
       ▼
┌─────────────┐     FAISS
│  Vector      │──── stores all vectors for fast similarity search
│  Store       │     saved to ./vectorstore/
└──────┬──────┘
       │
       │  ← query time ─────────────────────────────────
       │
       ▼
┌─────────────┐     FAISS (dense) + BM25Okapi (sparse)
│  Hybrid      │──── 70% semantic similarity + 30% keyword match
│  Retrieval   │     returns top 20 candidates
└──────┬──────┘
       │
       ▼
┌─────────────┐     CrossEncoder (ms-marco-MiniLM-L-6-v2)
│  Rerank      │──── reads each (question, chunk) pair
│              │     picks the 4 most relevant chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐     ChatOllama (llama3.1:8b)
│  Generate    │──── LLM reads the 4 chunks + question
│  Answer      │     produces a grounded answer with citations
└─────────────┘
```

---

## Key Libraries Summary

| Library | What it does in this project |
|---|---|
| **PyMuPDF (fitz)** | Extracts raw text from PDF files, page by page |
| **LangChain Text Splitters** | Splits long text into overlapping chunks at natural boundaries |
| **Ollama + nomic-embed-text** | Converts text chunks into numerical vectors (embeddings) locally |
| **FAISS (faiss-cpu)** | Stores vectors and performs fast nearest-neighbor similarity search |
| **rank-bm25 (BM25Okapi)** | Classic keyword search — scores chunks by term frequency and rarity |
| **sentence-transformers (CrossEncoder)** | Reranks candidate chunks by reading question+chunk pairs together |
| **Ollama + llama3.1:8b** | Local LLM that generates the final answer from retrieved context |
| **LangChain (ConversationalRetrievalChain)** | Orchestrates the retrieval→prompt→LLM pipeline with chat memory |

---

## Config at a Glance (`config.py`)

| Setting | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 1500 chars | Large enough for a full paragraph, small enough for focused retrieval |
| `CHUNK_OVERLAP` | 200 chars | Prevents losing context at chunk boundaries |
| `TOP_K_CANDIDATES` | 20 | Number of chunks passed to the reranker |
| `TOP_K` | 4 | Final chunks given to the LLM as context |
| `DENSE_WEIGHT` | 0.7 | Semantic search contributes 70% of hybrid score |
| `BM25_WEIGHT` | 0.3 | Keyword search contributes 30% of hybrid score |
| `temperature` | 0 | Deterministic LLM output (no randomness) |
