# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for question-answering over scientific papers using local LLMs via Ollama. Everything runs locally — no cloud APIs or keys required.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Ingest PDFs into FAISS vectorstore (requires Ollama running)
python ingest.py

# Query the vectorstore
python query.py "Your question here"
```

Ollama must be running at `http://localhost:11434` with models `nomic-embed-text` and `llama3.1:8b` pulled.

## Architecture

Three-file pipeline:

- **`config.py`** — All constants: directories, chunk sizes, Ollama URLs, model names
- **`ingest.py`** — PDF extraction (PyMuPDF/fitz) → text chunking (RecursiveCharacterTextSplitter, 1000 chars / 200 overlap) → embedding (Ollama `nomic-embed-text`) → FAISS vectorstore saved to `./vectorstore/`
- **`query.py`** — Loads FAISS index → retrieves top-4 chunks → answers via LangChain RetrievalQA with `ChatOllama` (llama3.1:8b, temperature=0)

Data flow: PDFs in `./papers/` → `ingest.py` → `./vectorstore/` → `query.py` → answer with source citations.

## Key Dependencies

- **LangChain ecosystem**: `langchain`, `langchain_community` (FAISS), `langchain_ollama`, `langchain_classic` (RetrievalQA), `langchain_text_splitters`
- **PyMuPDF (fitz)**: PDF text extraction
- **FAISS (faiss-cpu)**: Vector similarity search
- **Ollama**: Local LLM inference

## Notes

- No build system, tests, or CI/CD currently exist
- `config.py` uses `from config import *` pattern — all constants are imported as globals
- Chunk metadata tracks source PDF filename for citation display
