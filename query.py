"""Retrieve relevant chunks and answer questions with hybrid search."""
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from config import (
    VECTORSTORE_DIR,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    BM25_WEIGHT,
    DENSE_WEIGHT,
)


class HybridRetriever(BaseRetriever):
    """Combines FAISS dense retrieval with BM25 keyword search."""

    vectorstore: FAISS
    bm25: BM25Okapi
    bm25_docs: list[Document]
    k: int = TOP_K
    bm25_weight: float = BM25_WEIGHT
    dense_weight: float = DENSE_WEIGHT

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # Dense retrieval via FAISS
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.k)

        # BM25 keyword retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]

        # Combine scores from both retrievers
        doc_scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        # Dense scores (lower distance = better, so invert)
        if dense_results:
            max_dist = max(s for _, s in dense_results) or 1.0
            for doc, dist in dense_results:
                doc_id = id(doc)
                doc_map[doc_id] = doc
                doc_scores[doc_id] = self.dense_weight * (1 - dist / (max_dist + 1e-6))

        # BM25 scores
        max_bm25 = max((bm25_scores[i] for i in bm25_top), default=1.0) or 1.0
        for idx in bm25_top:
            doc = self.bm25_docs[idx]
            doc_id = id(doc)
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.bm25_weight * (bm25_scores[idx] / max_bm25)

        # Return top-k by combined score
        ranked = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
        return [doc_map[d] for d in ranked[:self.k]]


def load_vectorstore() -> FAISS:
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        print(f"No vectorstore found at {VECTORSTORE_DIR}/")
        print("Run 'python ingest.py' first to ingest PDFs.")
        sys.exit(1)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


def build_retriever(vectorstore: FAISS) -> HybridRetriever:
    docstore = vectorstore.docstore
    all_docs = [docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]
    tokenized = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)
    return HybridRetriever(vectorstore=vectorstore, bm25=bm25, bm25_docs=all_docs)


def build_qa_chain(retriever: HybridRetriever):
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


def print_result(result: dict):
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    seen = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  - {source} (p.{page}): {snippet}...")


def interactive():
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = build_qa_chain(retriever)
    chat_history = []

    print("RAG Science — ask questions about your papers (type 'quit' to exit)\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        result = qa.invoke({"question": question, "chat_history": chat_history})
        print_result(result)
        chat_history.append((question, result["answer"]))


def ask(question: str):
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = build_qa_chain(retriever)
    result = qa.invoke({"question": question, "chat_history": []})
    print_result(result)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        ask(" ".join(args))
    else:
        interactive()
