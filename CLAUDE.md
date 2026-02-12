# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for question-answering over scientific papers using local LLMs via Ollama. Everything runs locally — no cloud APIs or keys required.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ingest PDFs into FAISS vectorstore (requires Ollama running)
python ingest.py

# Single question query
python query.py "Your question here"

# Interactive conversational mode (supports follow-up questions)
python query.py
```

Ollama must be running at `http://localhost:11434` with models `nomic-embed-text` and `llama3.1:8b` pulled.

## Architecture

Three-file pipeline:

- **`config.py`** — All constants: directories, chunk sizes, Ollama URLs, model names, retrieval/reranking settings
- **`ingest.py`** — PDF extraction (PyMuPDF/fitz) → text chunking (RecursiveCharacterTextSplitter, 1500 chars / 200 overlap, sentence-aware separators) → embedding (Ollama `nomic-embed-text`) → FAISS vectorstore saved to `./vectorstore/`. Supports incremental ingestion via MD5 hash tracking in `ingested.json`.
- **`query.py`** — Loads FAISS index → hybrid retrieval (FAISS dense + BM25 keyword search) → cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) → answers via LangChain ConversationalRetrievalChain with `ChatOllama` (llama3.1:8b, temperature=0). Supports both single-shot and interactive conversational mode with memory.

Data flow: PDFs in `./papers/` → `ingest.py` → `./vectorstore/` → `query.py` → answer with source citations (file + page number).

## Key Dependencies

- **LangChain ecosystem**: `langchain`, `langchain_community` (FAISS), `langchain_ollama`, `langchain_classic` (ConversationalRetrievalChain), `langchain_text_splitters`
- **PyMuPDF (fitz)**: PDF text extraction
- **FAISS (faiss-cpu)**: Vector similarity search
- **rank-bm25**: BM25 keyword retrieval
- **sentence-transformers**: Cross-encoder re-ranking
- **Ollama**: Local LLM inference

## Notes

- No build system, tests, or CI/CD currently exist
- Chunk metadata tracks source PDF filename and page number for citation display
- Ingestion is incremental — re-running `ingest.py` only processes new/changed PDFs
- The cross-encoder model (`ms-marco-MiniLM-L-6-v2`) is downloaded on first run
