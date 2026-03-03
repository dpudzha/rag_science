"""LlamaIndex ingestion pipeline — runs in parallel with ingest.py."""
import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from llama_index.core.schema import TextNode

from config import (
    PAPERS_DIR,
    LLAMAINDEX_VECTORSTORE_DIR,
    LLAMAINDEX_INGEST_RECORD,
)
from ingest import load_new_documents, chunk_documents, file_hash

logger = logging.getLogger(__name__)


def _load_ingest_record() -> dict:
    """Load the LlamaIndex-specific ingest record (separate from ingest.py's record)."""
    record_path = Path(LLAMAINDEX_INGEST_RECORD)
    if record_path.exists():
        try:
            return json.loads(record_path.read_text())
        except Exception as e:
            logger.warning("Failed to load LlamaIndex ingest record: %s", e)
    return {}


def _lc_doc_to_node(doc: Document) -> TextNode:
    """Convert a LangChain Document to a LlamaIndex TextNode."""
    return TextNode(
        text=doc.page_content,
        metadata=dict(doc.metadata),
    )


def _detect_embedding_dim(embed_model) -> int:
    """Embed a test string to detect the embedding dimension."""
    result = embed_model.get_text_embedding("test")
    return len(result)


def _save_atomically(
    index,
    new_nodes: list,
    existing_nodes: list,
    ingest_record: dict,
) -> None:
    """Atomically save FAISS vectorstore, BM25 nodes, and ingest record to disk."""
    dest = Path(LLAMAINDEX_VECTORSTORE_DIR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=dest.parent, prefix="vectorstore_llamaindex.tmp.")

    try:
        # Save LlamaIndex storage context (FAISS + docstore)
        index.storage_context.persist(persist_dir=tmp_dir)

        # Save FAISS index file (LlamaIndex saves as default_faiss.index or similar)
        # We also need to copy the faiss vector store separately
        from llama_index.vector_stores.faiss import FaissVectorStore
        vs = index.storage_context.vector_store
        if hasattr(vs, '_faiss_index') or hasattr(vs, 'faiss_index'):
            faiss_idx = getattr(vs, '_faiss_index', None) or getattr(vs, 'faiss_index', None)
            if faiss_idx is not None:
                import faiss
                faiss.write_index(faiss_idx, os.path.join(tmp_dir, "index.faiss"))

        # Save BM25 nodes (existing + new)
        all_nodes = existing_nodes + new_nodes
        nodes_path = Path(tmp_dir) / "bm25_nodes.pkl"
        with open(nodes_path, "wb") as f:
            pickle.dump(all_nodes, f)
        logger.info("Saved %d BM25 nodes", len(all_nodes))

        # Save ingest record
        record_path = Path(tmp_dir) / "ingested.json"
        with open(record_path, "w") as f:
            json.dump(ingest_record, f, indent=2)

        # Atomic swap
        backup = None
        if dest.exists():
            backup = str(dest) + ".bak"
            os.rename(str(dest), backup)
        os.rename(tmp_dir, str(dest))
        if backup and Path(backup).exists():
            shutil.rmtree(backup)

        logger.info("LlamaIndex vectorstore atomically saved to %s", dest)
    except Exception:
        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)
        raise


def ingest_llamaindex() -> None:
    """Main LlamaIndex ingestion entry point."""
    from health import check_backend
    check_backend()

    from utils import get_llamaindex_embeddings

    # Load LlamaIndex-specific ingest record (separate from ingest.py's record)
    li_record = _load_ingest_record()

    # Load new documents — pass the LlamaIndex-specific record so that files
    # already ingested by the LangChain backend are NOT skipped here.
    new_docs, _large_tables = load_new_documents(PAPERS_DIR, record=li_record)

    if not new_docs:
        logger.info("Nothing new to ingest for LlamaIndex backend.")
        return

    logger.info("LlamaIndex: ingesting %d new documents", len(new_docs))

    # Chunk documents via shared chunker
    lc_chunks: list[Document] = chunk_documents(new_docs)
    if not lc_chunks:
        logger.info("No chunks generated; skipping LlamaIndex vectorstore update.")
        return

    # Convert to LlamaIndex TextNodes
    new_nodes = [_lc_doc_to_node(doc) for doc in lc_chunks]
    logger.info("Converted %d chunks to LlamaIndex TextNodes", len(new_nodes))

    # Get embedding model and detect dimension
    embed_model = get_llamaindex_embeddings()
    embed_dim = _detect_embedding_dim(embed_model)
    logger.info("Embedding dimension: %d", embed_dim)

    # Build or update VectorStoreIndex
    dest = Path(LLAMAINDEX_VECTORSTORE_DIR)
    index_path = dest / "index.faiss"

    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

    if index_path.exists():
        # Incremental: load existing index and insert new nodes
        logger.info("Loading existing LlamaIndex vectorstore for incremental update")
        vector_store = FaissVectorStore.from_persist_dir(str(dest))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(dest),
        )
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        index.insert_nodes(new_nodes)
        logger.info("Inserted %d new nodes into existing index", len(new_nodes))
        # Load existing BM25 nodes
        existing_nodes = _load_existing_bm25_nodes()
    else:
        # Fresh ingest: create new FAISS index
        logger.info("Creating new LlamaIndex FAISS vectorstore")
        faiss_index = faiss.IndexFlatL2(embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=new_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        existing_nodes = []

    # Update ingest record
    for doc in new_docs:
        li_record[doc["hash"]] = doc["source"]

    # Save atomically
    _save_atomically(
        index=index,
        new_nodes=new_nodes,
        existing_nodes=existing_nodes,
        ingest_record=li_record,
    )


def _load_existing_bm25_nodes() -> list:
    """Load existing BM25 nodes from disk if available."""
    nodes_path = Path(LLAMAINDEX_VECTORSTORE_DIR) / "bm25_nodes.pkl"
    if nodes_path.exists():
        try:
            with open(nodes_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load existing BM25 nodes: %s", e)
    return []


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging()
    ingest_llamaindex()
