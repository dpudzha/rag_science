"""Read PDFs from folder, chunk text, store in FAISS with BM25 index."""
import bisect
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import tempfile

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from config import (
    PAPERS_DIR,
    VECTORSTORE_DIR,
    INGEST_RECORD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    ENABLE_PARENT_RETRIEVAL,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    SUPPORTED_FORMATS,
    ENABLE_TABLE_EXTRACTION,
    LARGE_TABLE_THRESHOLD,
)

from parsers.pdf_parser import extract_text_from_pdf, _extract_title  # noqa: F401

try:
    from utils import tokenize
except ImportError:
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

logger = logging.getLogger(__name__)

_SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.[\d.]*\s+[A-Z][^\n]*|[A-Z][A-Z\s]{3,})$", re.MULTILINE
)


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ingest_record() -> dict:
    record_path = Path(INGEST_RECORD)
    if record_path.exists():
        return json.loads(record_path.read_text())
    return {}


def save_ingest_record(record: dict):
    Path(INGEST_RECORD).write_text(json.dumps(record, indent=2))


def _detect_section_header(text: str) -> str | None:
    """Return the last section header found before this chunk, if any."""
    match = None
    for m in _SECTION_HEADER_RE.finditer(text):
        match = m
    if match:
        return match.group(0).strip()
    return None


def load_new_documents(folder: str) -> tuple[list[dict], list[dict]]:
    """Load new documents of all supported formats.

    Returns (docs, large_tables) where large_tables are tables with >LARGE_TABLE_THRESHOLD rows
    to be routed to SQLite.
    """
    record = load_ingest_record()
    folder_path = Path(folder)

    # Collect all supported files
    format_map = {
        "pdf": "*.pdf",
        "docx": "*.docx",
        "xlsx": "*.xlsx",
    }

    all_files = []
    for fmt in SUPPORTED_FORMATS:
        fmt = fmt.strip().lower()
        if fmt in format_map:
            all_files.extend(folder_path.rglob(format_map[fmt]))

    logger.info("Found %d documents total across formats: %s", len(all_files), SUPPORTED_FORMATS)

    docs = []
    large_tables = []

    for file_path in all_files:
        h = file_hash(str(file_path))
        if h in record:
            logger.info("  - %s (already ingested, skipping)", file_path.name)
            continue

        try:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                doc = extract_text_from_pdf(str(file_path))
                # Extract tables from PDFs if enabled
                if ENABLE_TABLE_EXTRACTION:
                    from parsers.table_extractor import TableExtractor
                    extractor = TableExtractor()
                    tables = extractor.extract_tables(str(file_path))
                    doc["tables"] = tables
            else:
                from parsers import get_parser
                parser = get_parser(str(file_path))
                doc = parser.parse(str(file_path))

            doc["hash"] = h

            # Route large tables to SQL
            for table in doc.get("tables", []):
                num_rows = table.get("num_rows", len(table.get("data", [])) - 1)
                if num_rows > LARGE_TABLE_THRESHOLD:
                    large_tables.append({
                        "source": doc["source"],
                        "table": table,
                    })

            docs.append(doc)
            logger.info("  + %s (new)", doc["source"])
        except Exception as e:
            logger.warning("  ! %s: %s", file_path.name, e)

    logger.info("%d new documents to ingest, %d large tables for SQL", len(docs), len(large_tables))
    return docs, large_tables


def _page_at_offset(page_offsets: list[int], page_numbers: list[int], offset: int) -> int:
    """Return the page number for a given character offset in the concatenated text."""
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return page_numbers[max(0, idx)]


def chunk_documents(docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    all_chunks = []
    for doc in docs:
        full_text = ""
        page_offsets = []
        page_numbers = []
        for page_info in doc["pages"]:
            page_offsets.append(len(full_text))
            page_numbers.append(page_info["page"])
            full_text += page_info["text"] + "\n"

        if not full_text.strip():
            continue

        title = doc.get("title", "")

        # Use split_text_with_offsets-like approach: track offset by scanning
        chunks = splitter.split_text(full_text)
        search_start = 0
        for chunk_text in chunks:
            offset = full_text.find(chunk_text, search_start)
            if offset == -1:
                offset = full_text.find(chunk_text)
            if offset != -1:
                search_start = offset + 1

            page = _page_at_offset(page_offsets, page_numbers, max(0, offset))

            # Detect section header from text preceding this chunk
            preceding_text = full_text[max(0, offset - 500):max(0, offset)]
            section = _detect_section_header(preceding_text)

            # Build enriched content with title/section prefix
            prefix_parts = []
            if title:
                prefix_parts.append(f"[Paper: {title}]")
            if section:
                prefix_parts.append(f"[Section: {section}]")
            prefix = " ".join(prefix_parts)
            enriched_content = f"{prefix}\n{chunk_text}" if prefix else chunk_text

            all_chunks.append(Document(
                page_content=enriched_content,
                metadata={
                    "source": doc["source"],
                    "page": page,
                    "title": title,
                    **({"section": section} if section else {}),
                    **({"creation_date": doc["creation_date"]} if doc.get("creation_date") else {}),
                    **({"authors": doc["authors"]} if doc.get("authors") else {}),
                },
            ))

    # Add small tables as chunks (row-per-chunk with header context)
    for doc in docs:
        for table in doc.get("tables", []):
            num_rows = table.get("num_rows", len(table.get("data", [])) - 1)
            if num_rows <= LARGE_TABLE_THRESHOLD and table.get("data"):
                data = table["data"]
                header = data[0] if data else []
                header_str = " | ".join(header)
                for row in data[1:]:
                    row_str = " | ".join(row)
                    content = f"[Table from {doc['source']}]\nHeader: {header_str}\nRow: {row_str}"
                    all_chunks.append(Document(
                        page_content=content,
                        metadata={
                            "source": doc["source"],
                            "page": table.get("page", 1),
                            "title": doc.get("title", ""),
                            "content_type": "table_row",
                        },
                    ))

    logger.info("Created %d chunks", len(all_chunks))
    return all_chunks


def chunk_documents_child(docs: list[dict]) -> tuple[list[Document], list[Document]]:
    """Create both parent (large) and child (small) chunks for parent-document retrieval."""
    parent_chunks = chunk_documents(docs)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    child_chunks = []
    for i, parent in enumerate(parent_chunks):
        children = child_splitter.split_text(parent.page_content)
        for child_text in children:
            child_chunks.append(Document(
                page_content=child_text,
                metadata={**parent.metadata, "parent_idx": i},
            ))

    logger.info("Created %d child chunks from %d parent chunks", len(child_chunks), len(parent_chunks))
    return parent_chunks, child_chunks


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_bm25(docs: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Build BM25 index from documents."""
    tokenized = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


def save_bm25(bm25: BM25Okapi, docs: list[Document], directory: str):
    """Save BM25 index and document list to disk."""
    bm25_path = Path(directory) / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    logger.info("BM25 index saved to %s", bm25_path)


def build_vectorstore(chunks, existing_store=None) -> FAISS:
    embeddings = get_embeddings()
    new_store = FAISS.from_documents(chunks, embeddings)
    if existing_store:
        existing_store.merge_from(new_store)
        return existing_store
    return new_store


def _save_vectorstore_atomic(store: FAISS, bm25: BM25Okapi, bm25_docs: list[Document],
                             parent_chunks: list[Document] | None = None):
    """Save vectorstore, BM25 index, and optionally parent chunks atomically."""
    dest = Path(VECTORSTORE_DIR)
    tmp_dir = tempfile.mkdtemp(dir=dest.parent, prefix="vectorstore.tmp.")

    try:
        store.save_local(tmp_dir)
        save_bm25(bm25, bm25_docs, tmp_dir)

        if parent_chunks is not None:
            parents_path = Path(tmp_dir) / "parent_chunks.pkl"
            with open(parents_path, "wb") as f:
                pickle.dump(parent_chunks, f)

        # Preserve ingested.json if it exists in current vectorstore
        ingest_record = Path(INGEST_RECORD)
        if ingest_record.exists():
            shutil.copy2(str(ingest_record), os.path.join(tmp_dir, "ingested.json"))

        # Atomic swap: rename old dir, rename tmp to target, remove old
        backup = None
        if dest.exists():
            backup = str(dest) + ".bak"
            os.rename(str(dest), backup)
        os.rename(tmp_dir, str(dest))
        if backup and Path(backup).exists():
            shutil.rmtree(backup)

        logger.info("Vectorstore atomically saved to %s", dest)
    except Exception:
        # Clean up temp dir on failure
        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)
        raise


def load_existing_vectorstore():
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            VECTORSTORE_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    return None


def _get_all_docs_from_store(store: FAISS) -> list[Document]:
    """Extract all documents from a FAISS vectorstore's docstore."""
    return [store.docstore.search(doc_id) for doc_id in store.index_to_docstore_id.values()]


def _save_large_tables_to_sql(large_tables: list[dict]) -> list[Document]:
    """Save large tables to SQLite and return description chunks for the vectorstore.

    Returns a list of Document objects describing each table so the agent can
    discover SQL tables through vector retrieval.
    """
    description_chunks = []
    if not large_tables:
        return description_chunks
    try:
        from sql_database import SQLDatabase
        db = SQLDatabase()
        for item in large_tables:
            source = item["source"]
            table = item["table"]
            data = table["data"]
            if len(data) < 2:
                continue
            header = data[0]
            rows = data[1:]
            table_name = Path(source).stem.replace("-", "_").replace(" ", "_")
            sheet_name = table.get("sheet_name")
            if sheet_name:
                table_name = f"{table_name}_{sheet_name.replace(' ', '_')}"
            # Sanitize table name
            table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)

            import pandas as pd
            if "dataframe" in table:
                df = table["dataframe"]
            else:
                df = pd.DataFrame(rows, columns=header)
            db.create_table_from_dataframe(table_name, df)
            logger.info("Saved large table '%s' (%d rows) to SQL", table_name, len(rows))

            # Create a description chunk for vectorstore discovery
            columns_str = ", ".join(header)
            sample_rows = rows[:3]
            sample_str = "\n".join(" | ".join(row) for row in sample_rows)
            description = (
                f"[SQL Table: {table_name}]\n"
                f"Source: {source}\n"
                f"Columns: {columns_str}\n"
                f"Total rows: {len(rows)}\n"
                f"Sample data:\n{sample_str}\n\n"
                f"This table is stored in a SQL database. Use the query_tables tool "
                f"to run SQL queries against table '{table_name}'."
            )
            description_chunks.append(Document(
                page_content=description,
                metadata={
                    "source": source,
                    "page": 1,
                    "title": f"SQL Table: {table_name}",
                    "content_type": "sql_table_description",
                },
            ))
    except ImportError:
        logger.warning("sql_database module not available, skipping SQL table storage")
    except Exception as e:
        logger.warning("Failed to save large tables to SQL: %s", e)
    return description_chunks


def ingest():
    from health import check_ollama
    check_ollama()

    # Use multi-format document loading
    docs, large_tables = load_new_documents(PAPERS_DIR)
    if not docs:
        logger.info("Nothing new to ingest.")
        return

    # Save large tables to SQLite and get description chunks for vectorstore
    table_description_chunks = _save_large_tables_to_sql(large_tables)

    existing = load_existing_vectorstore()

    if ENABLE_PARENT_RETRIEVAL:
        parent_chunks, child_chunks = chunk_documents_child(docs)
        child_chunks.extend(table_description_chunks)
        store = build_vectorstore(child_chunks, existing)
        # Combine all docs for BM25 (use parent chunks for better keyword matching)
        all_parents = parent_chunks
        if existing:
            parents_path = Path(VECTORSTORE_DIR) / "parent_chunks.pkl"
            if parents_path.exists():
                with open(parents_path, "rb") as f:
                    existing_parents = pickle.load(f)
                all_parents = existing_parents + parent_chunks
        all_store_docs = _get_all_docs_from_store(store)
        bm25, bm25_docs = build_bm25(all_store_docs)
        _save_vectorstore_atomic(store, bm25, bm25_docs, all_parents)
    else:
        chunks = chunk_documents(docs)
        chunks.extend(table_description_chunks)
        store = build_vectorstore(chunks, existing)
        all_docs = _get_all_docs_from_store(store)
        bm25, bm25_docs = build_bm25(all_docs)
        _save_vectorstore_atomic(store, bm25, bm25_docs)

    record = load_ingest_record()
    for doc in docs:
        record[doc["hash"]] = doc["source"]
    save_ingest_record(record)


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging()
    ingest()
