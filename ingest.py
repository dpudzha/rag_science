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
from datetime import datetime, timedelta, timezone

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import get_default_embeddings
from rank_bm25 import BM25Okapi
from config import (
    PAPERS_DIR,
    VECTORSTORE_DIR,
    INGEST_RECORD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    ENABLE_PARENT_RETRIEVAL,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    SUPPORTED_FORMATS,
    ENABLE_TABLE_EXTRACTION,
    LARGE_TABLE_THRESHOLD,
    ENABLE_S3_INGEST,
    S3_BUCKET,
    S3_PREFIX,
    S3_REGION,
    S3_LOOKBACK_HOURS,
    LLM_BACKEND,
    EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.[\d.]*\s+[A-Z][^\n]*|[A-Z][A-Z\s]{3,})$", re.MULTILINE
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _embedding_model_id() -> str:
    """Return a string identifying the current embedding backend + model."""
    if LLM_BACKEND in ("openai", "anthropic"):
        return f"openai/{OPENAI_EMBEDDING_MODEL}"
    return f"ollama/{EMBEDDING_MODEL}"


def load_ingest_record() -> dict:
    record_path = Path(INGEST_RECORD)
    if record_path.exists():
        record = json.loads(record_path.read_text())
        stored_model = record.get("_embedding_model")
        current_model = _embedding_model_id()
        if stored_model and stored_model != current_model:
            logger.info(
                "Embedding model changed (%s → %s), re-ingesting all documents",
                stored_model,
                current_model,
            )
            return {}
        return record
    return {}


def save_ingest_record(record: dict) -> None:
    _write_json_atomic(Path(INGEST_RECORD), record)


def _write_json_atomic(path: Path, payload: dict):
    """Atomically write a JSON file in the same directory as its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(path.parent),
        prefix=f".{path.name}.tmp.",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _to_markdown_lines(text: str) -> str:
    """Normalize plain text into lightweight markdown paragraphs."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


def _markdownify_pages(pages: list[dict], title: str) -> list[dict]:
    """Wrap extracted page text in markdown headings."""
    md_pages = []
    for page_info in pages:
        page_num = page_info.get("page", 1)
        body = _to_markdown_lines(page_info.get("text", ""))
        if not body:
            continue
        md_text = f"# {title}\n\n## Page {page_num}\n\n{body}"
        md_pages.append({"text": md_text, "page": page_num})
    return md_pages


def _split_text_by_token_limit(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks bounded by token count using token spans."""
    chunks_with_offsets = _split_text_by_token_limit_with_offsets(text, chunk_size, chunk_overlap)
    return [chunk for chunk, _offset in chunks_with_offsets]


def _split_text_by_token_limit_with_offsets(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, int]]:
    """Split text into token-bounded chunks and return each chunk's start offset."""
    lowered = text.lower()
    matches = list(_TOKEN_RE.finditer(lowered))
    if not matches:
        if text.strip():
            stripped = text.strip()
            offset = text.find(stripped)
            return [(stripped, max(0, offset))]
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    stride = chunk_size - overlap

    chunks: list[tuple[str, int]] = []
    start = 0
    total_tokens = len(matches)
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        char_start = matches[start].start()
        char_end = matches[end - 1].end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append((chunk, char_start))
        if end >= total_tokens:
            break
        start += stride
    return chunks


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
                from parsers.pdf_parser import extract_text_from_pdf
                doc = extract_text_from_pdf(str(file_path))
                doc["pages"] = _markdownify_pages(doc["pages"], doc["title"])
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


def sync_recent_s3_documents(target_dir: str) -> int:
    """Download documents uploaded within the configured lookback window."""
    if not ENABLE_S3_INGEST:
        return 0
    if not S3_BUCKET:
        logger.warning("ENABLE_S3_INGEST is true but S3_BUCKET is empty; skipping S3 sync")
        return 0

    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed; skipping S3 sync")
        return 0

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, S3_LOOKBACK_HOURS))
    downloaded = 0

    client_kwargs = {"region_name": S3_REGION} if S3_REGION else {}
    s3 = boto3.client("s3", **client_kwargs)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key:
                continue
            if obj.get("LastModified") and obj["LastModified"] < cutoff:
                continue
            suffix = Path(key).suffix.lower().lstrip(".")
            if suffix not in {fmt.strip().lower() for fmt in SUPPORTED_FORMATS}:
                continue
            destination = target / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(destination))
            downloaded += 1

    logger.info("Downloaded %d recent documents from s3://%s/%s", downloaded, S3_BUCKET, S3_PREFIX)
    return downloaded


def _page_at_offset(page_offsets: list[int], page_numbers: list[int], offset: int) -> int:
    """Return the page number for a given character offset in the concatenated text."""
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return page_numbers[max(0, idx)]


def chunk_documents(docs: list[dict]) -> list[Document]:
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

        chunks = _split_text_by_token_limit_with_offsets(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_text, offset in chunks:
            page = _page_at_offset(page_offsets, page_numbers, offset)

            # Detect section header from text preceding this chunk
            preceding_text = full_text[max(0, offset - 500):max(0, offset)] if offset >= 0 else ""
            section = _detect_section_header(preceding_text)

            # Build enriched content with title/section prefix
            prefix_parts = []
            if title:
                prefix_parts.append(f"[Paper: {title}]")
            if section:
                prefix_parts.append(f"[Section: {section}]")
            prefix = " ".join(prefix_parts)
            enriched_content = f"{prefix}\n{chunk_text}" if prefix else chunk_text

            # Compute stable chunk ID: source|page|sha256(normalized_text)[:12]
            normalized = " ".join(enriched_content.split())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{doc['source']}|p{page}|{digest}"

            all_chunks.append(Document(
                page_content=enriched_content,
                metadata={
                    "source": doc["source"],
                    "page": page,
                    "title": title,
                    "chunk_id": chunk_id,
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
                    table_page = table.get("page", 1)
                    normalized = " ".join(content.split())
                    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
                    chunk_id = f"{doc['source']}|p{table_page}|{digest}"
                    all_chunks.append(Document(
                        page_content=content,
                        metadata={
                            "source": doc["source"],
                            "page": table_page,
                            "title": doc.get("title", ""),
                            "content_type": "table_row",
                            "chunk_id": chunk_id,
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


def get_embeddings():
    return get_default_embeddings()


def build_bm25(docs: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Build BM25 index from documents."""
    bm25, _tokenized = _build_bm25_with_tokens(docs)
    return bm25, docs


def _build_bm25_with_tokens(docs: list[Document]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized = [tokenize(doc.page_content) for doc in docs]
    return BM25Okapi(tokenized), tokenized


def _load_bm25_data(directory: str) -> dict | None:
    """Load BM25 payload from disk, returning None when unavailable or invalid."""
    bm25_path = Path(directory) / "bm25_index.pkl"
    if not bm25_path.exists():
        return None
    try:
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            logger.warning("Invalid BM25 payload type at %s; ignoring", bm25_path)
            return None
        if "bm25" not in data or "docs" not in data:
            logger.warning("BM25 payload missing required keys at %s; ignoring", bm25_path)
            return None
        return data
    except Exception as e:
        logger.warning("Failed to load BM25 payload from %s: %s", bm25_path, e)
        return None


def save_bm25(
    bm25: BM25Okapi,
    docs: list[Document],
    directory: str,
    tokenized_docs: list[list[str]] | None = None,
) -> None:
    """Save BM25 index and document list to disk."""
    bm25_path = Path(directory) / "bm25_index.pkl"
    if tokenized_docs is None:
        tokenized_docs = [tokenize(doc.page_content) for doc in docs]
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs, "tokenized_docs": tokenized_docs}, f)
    logger.info("BM25 index saved to %s", bm25_path)


# nomic-embed-text context is 2048 BPE tokens.  CHUNK_SIZE=500 keeps the
# vast majority of chunks within this limit.  As a safety net for rare
# outliers (e.g. table-heavy content), we truncate at the embedding stage.
# Full chunk text is still stored in FAISS for retrieval.
_EMBED_CHAR_LIMIT = 3500


def build_vectorstore(chunks, existing_store=None) -> FAISS | None:
    if not chunks:
        return existing_store
    embeddings = get_embeddings()

    # Truncate oversized text for embedding while keeping full content
    # in the stored Document.  Use batch embed_documents() to avoid
    # per-chunk HTTP overhead against Ollama.
    embed_texts = [chunk.page_content[:_EMBED_CHAR_LIMIT] for chunk in chunks]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    logger.info("Embedding %d chunks in batch…", len(chunks))
    vectors = embeddings.embed_documents(embed_texts)

    text_embedding_pairs = list(zip(texts, vectors))
    new_store = FAISS.from_embeddings(
        text_embedding_pairs, embeddings, metadatas=metadatas
    )
    if existing_store:
        existing_store.merge_from(new_store)
        return existing_store
    return new_store


def _save_vectorstore_atomic(
    store: FAISS,
    bm25: BM25Okapi,
    bm25_docs: list[Document],
    bm25_tokenized_docs: list[list[str]] | None = None,
    parent_chunks: list[Document] | None = None,
    ingest_record: dict | None = None,
):
    """Save vectorstore, BM25 index, and optionally parent chunks atomically."""
    dest = Path(VECTORSTORE_DIR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=dest.parent, prefix="vectorstore.tmp.")

    try:
        store.save_local(tmp_dir)
        save_bm25(bm25, bm25_docs, tmp_dir, tokenized_docs=bm25_tokenized_docs)

        if parent_chunks is not None:
            parents_path = Path(tmp_dir) / "parent_chunks.pkl"
            with open(parents_path, "wb") as f:
                pickle.dump(parent_chunks, f)

        ingest_record_path = Path(INGEST_RECORD)
        resolved_dest = dest.resolve(strict=False)
        resolved_ingest = ingest_record_path.resolve(strict=False)
        ingest_under_vectorstore = (
            resolved_ingest == resolved_dest
            or resolved_dest in resolved_ingest.parents
        )

        if ingest_record is not None and ingest_under_vectorstore:
            relative_ingest = resolved_ingest.relative_to(resolved_dest)
            _write_json_atomic(Path(tmp_dir) / relative_ingest, ingest_record)
        elif ingest_record is None and ingest_record_path.exists() and ingest_under_vectorstore:
            relative_ingest = resolved_ingest.relative_to(resolved_dest)
            target = Path(tmp_dir) / relative_ingest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(ingest_record_path), str(target))

        # Atomic swap: rename old dir, rename tmp to target, remove old
        backup = None
        if dest.exists():
            backup = str(dest) + ".bak"
            os.rename(str(dest), backup)
        os.rename(tmp_dir, str(dest))
        if backup and Path(backup).exists():
            shutil.rmtree(backup)

        # If ingest record is configured outside VECTORSTORE_DIR, swap + record cannot
        # be done in one rename operation. Use atomic file replacement as best effort.
        if ingest_record is not None and not ingest_under_vectorstore:
            logger.warning(
                "INGEST_RECORD is outside VECTORSTORE_DIR; updating separately at %s",
                ingest_record_path,
            )
            _write_json_atomic(ingest_record_path, ingest_record)

        logger.info("Vectorstore atomically saved to %s", dest)
    except Exception:
        # Clean up temp dir on failure
        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)
        raise


def load_existing_vectorstore() -> FAISS | None:
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            VECTORSTORE_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    return None


def _get_all_docs_from_store(store: FAISS) -> list[Document]:
    """Extract all documents from a FAISS vectorstore's docstore."""
    return [store.docstore.search(doc_id) for doc_id in store.index_to_docstore_id.values()]


def _build_or_update_bm25(
    store: FAISS,
    new_docs: list[Document],
    allow_incremental: bool,
) -> tuple[BM25Okapi, list[Document], list[list[str]]]:
    """Build BM25 from scratch or update incrementally from persisted tokenized docs."""
    if allow_incremental:
        existing = _load_bm25_data(VECTORSTORE_DIR)
        if existing:
            existing_docs = existing.get("docs") or []
            existing_tokenized = existing.get("tokenized_docs")
            if (
                isinstance(existing_docs, list)
                and isinstance(existing_tokenized, list)
                and len(existing_docs) == len(existing_tokenized)
            ):
                tokenized_new_docs = [tokenize(doc.page_content) for doc in new_docs]
                combined_docs = existing_docs + new_docs
                combined_tokenized = existing_tokenized + tokenized_new_docs
                bm25 = BM25Okapi(combined_tokenized)
                logger.info(
                    "BM25 incremental update: %d existing + %d new documents",
                    len(existing_docs),
                    len(new_docs),
                )
                return bm25, combined_docs, combined_tokenized
            logger.info("BM25 incremental update unavailable (missing/invalid tokenized docs); rebuilding")

    all_docs = _get_all_docs_from_store(store)
    bm25, tokenized_docs = _build_bm25_with_tokens(all_docs)
    return bm25, all_docs, tokenized_docs


def _save_large_tables_to_sql(large_tables: list[dict]) -> list[Document]:
    """Save large tables to SQLite and return description chunks for the vectorstore.

    Returns a list of Document objects describing each table so the agent can
    discover SQL tables through vector retrieval.
    """
    description_chunks = []
    if not large_tables:
        return description_chunks
    try:
        from tools.sql_database import SQLDatabase
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


def ingest() -> None:
    from health import check_backend
    check_backend()

    sync_recent_s3_documents(PAPERS_DIR)

    # Use multi-format document loading
    docs, large_tables = load_new_documents(PAPERS_DIR)
    if not docs:
        logger.info("Nothing new to ingest.")
        return
    updated_record = load_ingest_record()
    # Detect if this is a full re-ingest due to embedding model change
    embedding_changed = "_embedding_model" not in updated_record
    for doc in docs:
        updated_record[doc["hash"]] = doc["source"]
    updated_record["_embedding_model"] = _embedding_model_id()

    # Save large tables to SQLite and get description chunks for vectorstore
    table_description_chunks = _save_large_tables_to_sql(large_tables)

    # Don't merge with existing vectorstore if embedding model changed
    existing = None if embedding_changed else load_existing_vectorstore()

    if ENABLE_PARENT_RETRIEVAL:
        parent_chunks, child_chunks = chunk_documents_child(docs)
        child_chunks.extend(table_description_chunks)
        store = build_vectorstore(child_chunks, existing)
        if store is None:
            logger.info("No chunks generated; skipping vectorstore update.")
            save_ingest_record(updated_record)
            return
        # Combine all docs for BM25
        all_parents = parent_chunks
        if existing:
            parents_path = Path(VECTORSTORE_DIR) / "parent_chunks.pkl"
            if parents_path.exists():
                with open(parents_path, "rb") as f:
                    existing_parents = pickle.load(f)
                all_parents = existing_parents + parent_chunks
        bm25, bm25_docs, bm25_tokenized_docs = _build_or_update_bm25(
            store=store,
            new_docs=child_chunks,
            allow_incremental=existing is not None,
        )
        _save_vectorstore_atomic(
            store,
            bm25,
            bm25_docs,
            bm25_tokenized_docs=bm25_tokenized_docs,
            parent_chunks=all_parents,
            ingest_record=updated_record,
        )
    else:
        chunks = chunk_documents(docs)
        chunks.extend(table_description_chunks)
        if not chunks:
            logger.info("No chunks generated; skipping vectorstore update.")
            save_ingest_record(updated_record)
            return
        store = build_vectorstore(chunks, existing)
        if store is None:
            logger.info("No vectorstore available; skipping vectorstore update.")
            save_ingest_record(updated_record)
            return
        bm25, bm25_docs, bm25_tokenized_docs = _build_or_update_bm25(
            store=store,
            new_docs=chunks,
            allow_incremental=existing is not None,
        )
        _save_vectorstore_atomic(
            store,
            bm25,
            bm25_docs,
            bm25_tokenized_docs=bm25_tokenized_docs,
            ingest_record=updated_record,
        )


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging()
    ingest()
