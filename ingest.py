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

import fitz
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
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.[\d.]*\s+[A-Z][^\n]*|[A-Z][A-Z\s]{3,})$", re.MULTILINE
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_ingest_record() -> dict:
    record_path = Path(INGEST_RECORD)
    if record_path.exists():
        return json.loads(record_path.read_text())
    return {}


def save_ingest_record(record: dict):
    Path(INGEST_RECORD).write_text(json.dumps(record, indent=2))


def extract_text_from_pdf(pdf_path: str) -> dict:
    pages = []
    title = None
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"text": text, "page": page_num})
                if title is None:
                    title = _extract_title(text)
    return {
        "pages": pages,
        "source": Path(pdf_path).name,
        "title": title or Path(pdf_path).stem,
    }


def _extract_title(first_page_text: str) -> str | None:
    """Extract paper title from the first page (first non-empty line, heuristic)."""
    for line in first_page_text.split("\n"):
        line = line.strip()
        if len(line) > 10 and not line.startswith("http"):
            return line
    return None


def _detect_section_header(text: str) -> str | None:
    """Return the last section header found before this chunk, if any."""
    match = None
    for m in _SECTION_HEADER_RE.finditer(text):
        match = m
    if match:
        return match.group(0).strip()
    return None


def load_new_pdfs(folder: str) -> list[dict]:
    pdfs = list(Path(folder).glob("*.pdf"))
    record = load_ingest_record()
    logger.info("Found %d PDFs total", len(pdfs))

    docs = []
    for pdf_path in pdfs:
        h = file_hash(str(pdf_path))
        if h in record:
            logger.info("  - %s (already ingested, skipping)", pdf_path.name)
            continue
        try:
            doc = extract_text_from_pdf(str(pdf_path))
            doc["hash"] = h
            docs.append(doc)
            logger.info("  + %s (new)", doc["source"])
        except Exception as e:
            logger.warning("  ! %s: %s", pdf_path.name, e)

    logger.info("%d new PDFs to ingest", len(docs))
    return docs


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


def ingest():
    from health import check_ollama
    check_ollama()

    docs = load_new_pdfs(PAPERS_DIR)
    if not docs:
        logger.info("Nothing new to ingest.")
        return

    existing = load_existing_vectorstore()

    if ENABLE_PARENT_RETRIEVAL:
        parent_chunks, child_chunks = chunk_documents_child(docs)
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
