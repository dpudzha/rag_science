"""Read PDFs from folder, chunk text, store in FAISS."""
import bisect
import hashlib
import json
import fitz
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import (
    PAPERS_DIR,
    VECTORSTORE_DIR,
    INGEST_RECORD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)


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
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"text": text, "page": page_num})
    return {
        "pages": pages,
        "source": Path(pdf_path).name,
    }


def load_new_pdfs(folder: str) -> list[dict]:
    pdfs = list(Path(folder).glob("*.pdf"))
    record = load_ingest_record()
    print(f"Found {len(pdfs)} PDFs total")

    docs = []
    for pdf_path in pdfs:
        h = file_hash(str(pdf_path))
        if h in record:
            print(f"  - {pdf_path.name} (already ingested, skipping)")
            continue
        try:
            doc = extract_text_from_pdf(str(pdf_path))
            doc["hash"] = h
            docs.append(doc)
            print(f"  + {doc['source']} (new)")
        except Exception as e:
            print(f"  ! {pdf_path.name}: {e}")

    print(f"{len(docs)} new PDFs to ingest")
    return docs


def _page_at_offset(page_offsets: list[int], page_numbers: list[int], offset: int) -> int:
    """Return the page number for a given character offset in the concatenated text."""
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return page_numbers[max(0, idx)]


def chunk_documents(docs: list[dict]) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    all_chunks = []
    for doc in docs:
        # Concatenate all pages into one text, tracking page boundary offsets
        full_text = ""
        page_offsets = []   # character offset where each page starts
        page_numbers = []   # corresponding page number
        for page_info in doc["pages"]:
            page_offsets.append(len(full_text))
            page_numbers.append(page_info["page"])
            full_text += page_info["text"] + "\n"

        if not full_text.strip():
            continue

        # Chunk the full document text (paragraphs spanning pages stay together)
        chunks = splitter.split_text(full_text)
        for chunk_text in chunks:
            offset = full_text.find(chunk_text)
            page = _page_at_offset(page_offsets, page_numbers, offset)
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={"source": doc["source"], "page": page},
            ))
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vectorstore(chunks, existing_store=None) -> FAISS:
    embeddings = get_embeddings()
    new_store = FAISS.from_documents(chunks, embeddings)
    if existing_store:
        existing_store.merge_from(new_store)
        existing_store.save_local(VECTORSTORE_DIR)
        print(f"Merged into existing vectorstore at {VECTORSTORE_DIR}")
        return existing_store
    new_store.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved to {VECTORSTORE_DIR}")
    return new_store


def load_existing_vectorstore():
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            VECTORSTORE_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    return None


def ingest():
    docs = load_new_pdfs(PAPERS_DIR)
    if not docs:
        print("Nothing new to ingest.")
        return

    chunks = chunk_documents(docs)
    existing = load_existing_vectorstore()
    build_vectorstore(chunks, existing)

    record = load_ingest_record()
    for doc in docs:
        record[doc["hash"]] = doc["source"]
    save_ingest_record(record)


if __name__ == "__main__":
    ingest()
