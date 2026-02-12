"""Read PDFs from folder, chunk text, store in FAISS."""
import fitz
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import *


def extract_text_from_pdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return {
        "text": text,
        "source": Path(pdf_path).name,
    }


def load_all_pdfs(folder: str) -> list[dict]:
    pdfs = list(Path(folder).glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")
    docs = []
    for pdf_path in pdfs:
        try:
            doc = extract_text_from_pdf(str(pdf_path))
            docs.append(doc)
            print(f"  ✓ {doc['source']}")
        except Exception as e:
            print(f"  ✗ {pdf_path.name}: {e}")
    return docs


def chunk_documents(docs: list[dict]) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.create_documents(
            texts=[doc["text"]],
            metadatas=[{"source": doc["source"]}],
        )
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def build_vectorstore(chunks) -> FAISS:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved to {VECTORSTORE_DIR}")
    return vectorstore


def ingest():
    docs = load_all_pdfs(PAPERS_DIR)
    if not docs:
        print("No PDFs found. Drop some into ./papers/")
        return
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)


if __name__ == "__main__":
    ingest()