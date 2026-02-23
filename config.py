import os
from dotenv import load_dotenv

# Load environment variables from .env file (explicit path so notebooks find it)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(_PROJECT_ROOT, "papers"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(_PROJECT_ROOT, "vectorstore"))
INGEST_RECORD = os.getenv("INGEST_RECORD", os.path.join(_PROJECT_ROOT, "vectorstore", "ingested.json"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# LLM backend: "ollama" (default), "anthropic", "openai"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b") # llama3.1:8b
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b") # quite good quality/speed ratio

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval settings
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
TOP_K = int(os.getenv("TOP_K", "4"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))

# Reranking
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "local").lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")

# Session management
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

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
USE_CROSS_ENCODER_RELEVANCE = os.getenv("USE_CROSS_ENCODER_RELEVANCE", "true").lower() == "true"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))

# Agent / SQL
ENABLE_SQL_AGENT = os.getenv("ENABLE_SQL_AGENT", "false").lower() == "true"
# Optional explicit SQLite path. If empty, defaults to VECTORSTORE_DIR/tables.db.
SQL_DATABASE_PATH = os.getenv("SQL_DATABASE_PATH", "")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

# Multi-format ingestion
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "pdf,docx,xlsx").split(",")
ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
LARGE_TABLE_THRESHOLD = int(os.getenv("LARGE_TABLE_THRESHOLD", "100"))

# S3 ingestion (optional)
ENABLE_S3_INGEST = os.getenv("ENABLE_S3_INGEST", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_REGION = os.getenv("S3_REGION", "")
S3_LOOKBACK_HOURS = int(os.getenv("S3_LOOKBACK_HOURS", "3"))

# Parent document retrieval (Phase 3)
ENABLE_PARENT_RETRIEVAL = os.getenv("ENABLE_PARENT_RETRIEVAL", "false").lower() == "true"
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# --- Config persistence ---
CONFIG_FILE = os.path.join(_PROJECT_ROOT, "config.json")

_TUNABLE_KEYS = [
    "LLM_BACKEND", "LLM_MODEL", "EMBEDDING_MODEL",
    "RERANK_BACKEND", "RERANK_MODEL",
    "COHERE_RERANK_MODEL", "JINA_RERANK_MODEL",
    "ANTHROPIC_MODEL", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL",
    "TOP_K", "TOP_K_CANDIDATES", "BM25_WEIGHT", "DENSE_WEIGHT", "RELEVANCE_THRESHOLD",
    "CHUNK_SIZE", "CHUNK_OVERLAP",
    "INTENT_CLASSIFICATION_ENABLED", "ARCHETYPE_DETECTION_ENABLED",
    "QUERY_REFORMULATION_ENABLED", "QUERY_RESOLUTION_ENABLED",
    "METADATA_EXTRACTION_ENABLED", "RELEVANCE_CHECK_ENABLED",
    "USE_CROSS_ENCODER_RELEVANCE", "ENABLE_SQL_AGENT",
    "ENABLE_TABLE_EXTRACTION", "ENABLE_PARENT_RETRIEVAL",
]

import json as _json


def get_tunable_config() -> dict:
    this = __import__(__name__)
    return {key: getattr(this, key) for key in _TUNABLE_KEYS}


def apply_config(updates: dict) -> None:
    this = __import__(__name__)
    tunable = get_tunable_config()
    for key, value in updates.items():
        if key not in tunable:
            raise ValueError(f"Unknown config key: {key}")
        expected_type = type(tunable[key])
        if not isinstance(value, expected_type):
            # Allow int where float is expected
            if expected_type is float and isinstance(value, int):
                value = float(value)
            else:
                raise TypeError(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        setattr(this, key, value)


def save_config(path: str = CONFIG_FILE) -> None:
    with open(path, "w") as f:
        _json.dump(get_tunable_config(), f, indent=2)


def load_config(path: str = CONFIG_FILE) -> dict:
    with open(path) as f:
        data = _json.load(f)
    apply_config(data)
    return data
