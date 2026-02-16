import os

PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
INGEST_RECORD = os.getenv("INGEST_RECORD", "./vectorstore/ingested.json")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# Retrieval settings
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "30"))
TOP_K = int(os.getenv("TOP_K", "6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))

# Reranking
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")

# Session management
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")]

# Intent classification
INTENT_CLASSIFICATION_ENABLED = os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true"

# Archetype detection and query reformulation
ARCHETYPE_DETECTION_ENABLED = os.getenv("ARCHETYPE_DETECTION_ENABLED", "true").lower() == "true"
QUERY_REFORMULATION_ENABLED = os.getenv("QUERY_REFORMULATION_ENABLED", "true").lower() == "true"
QUERY_RESOLUTION_ENABLED = os.getenv("QUERY_RESOLUTION_ENABLED", "true").lower() == "true"

# Metadata extraction
METADATA_EXTRACTION_ENABLED = os.getenv("METADATA_EXTRACTION_ENABLED", "true").lower() == "true"

# Relevance checking
RELEVANCE_CHECK_ENABLED = os.getenv("RELEVANCE_CHECK_ENABLED", "true").lower() == "true"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))

# Agent / SQL
ENABLE_SQL_AGENT = os.getenv("ENABLE_SQL_AGENT", "false").lower() == "true"
SQL_DATABASE_PATH = os.getenv("SQL_DATABASE_PATH", "")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

# Multi-format ingestion
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "pdf,docx,xlsx").split(",")
ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
LARGE_TABLE_THRESHOLD = int(os.getenv("LARGE_TABLE_THRESHOLD", "100"))

# Parent document retrieval (Phase 3)
ENABLE_PARENT_RETRIEVAL = os.getenv("ENABLE_PARENT_RETRIEVAL", "false").lower() == "true"
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
