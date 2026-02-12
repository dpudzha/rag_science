PAPERS_DIR = "./papers"
VECTORSTORE_DIR = "./vectorstore"
INGEST_RECORD = "./vectorstore/ingested.json"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

# Retrieval settings
TOP_K_CANDIDATES = 20  # candidates from hybrid retrieval before reranking
TOP_K = 4              # final results after reranking
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# Reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
