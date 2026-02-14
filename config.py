import os

PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
INGEST_RECORD = os.getenv("INGEST_RECORD", "./vectorstore/ingested.json")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# Retrieval settings
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
TOP_K = int(os.getenv("TOP_K", "4"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))

# Reranking
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Session management
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Parent document retrieval (Phase 3)
ENABLE_PARENT_RETRIEVAL = os.getenv("ENABLE_PARENT_RETRIEVAL", "false").lower() == "true"
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
