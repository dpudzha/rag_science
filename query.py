"""Retrieve relevant chunks and answer questions with hybrid search + reranking."""
import re
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from config import (
    VECTORSTORE_DIR,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    TOP_K_CANDIDATES,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RERANK_MODEL,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens, stripping all punctuation."""
    return _TOKEN_RE.findall(text.lower())


QA_PROMPT = PromptTemplate.from_template(
    """You are a scientific research assistant answering questions based on \
provided paper excerpts. Use ONLY the context below to answer. Be precise and \
use technical terminology appropriate to the field.

Rules:
- If the context does not contain enough information to answer, say "I don't \
have enough information in the provided papers to answer this."
- Cite specific findings, numbers, and methodologies from the context.
- When multiple papers are relevant, synthesize across them and note agreements \
or disagreements.
- Do not speculate beyond what the context supports.

Context:
{context}

Question: {question}

Answer:"""
)


class HybridRetriever(BaseRetriever):
    """Combines FAISS dense + BM25 keyword search, then reranks with a cross-encoder."""

    vectorstore: FAISS
    bm25: BM25Okapi
    bm25_docs: list[Document]
    cross_encoder: CrossEncoder
    k: int = TOP_K
    k_candidates: int = TOP_K_CANDIDATES
    bm25_weight: float = BM25_WEIGHT
    dense_weight: float = DENSE_WEIGHT

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Content-based key so the same chunk from FAISS and BM25 merges scores."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        return f"{source}:{page}:{doc.page_content[:200]}"

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # Dense retrieval via FAISS — fetch more candidates for reranking
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.k_candidates)

        # BM25 keyword retrieval
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k_candidates]

        # Combine scores from both retrievers using content-based keys
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # Dense scores (lower distance = better, so invert)
        if dense_results:
            max_dist = max(s for _, s in dense_results) or 1.0
            for doc, dist in dense_results:
                key = self._doc_key(doc)
                doc_map[key] = doc
                doc_scores[key] = self.dense_weight * (1 - dist / (max_dist + 1e-6))

        # BM25 scores
        max_bm25 = max((bm25_scores[i] for i in bm25_top), default=1.0) or 1.0
        for idx in bm25_top:
            doc = self.bm25_docs[idx]
            key = self._doc_key(doc)
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0) + self.bm25_weight * (bm25_scores[idx] / max_bm25)

        # Take top candidates by hybrid score, then rerank with cross-encoder
        ranked = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
        candidates = [doc_map[d] for d in ranked[:self.k_candidates]]

        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:self.k]]


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
    tokenized = [tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized)
    cross_encoder = CrossEncoder(RERANK_MODEL)
    return HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        bm25_docs=all_docs,
        cross_encoder=cross_encoder,
    )


def build_qa_chain(retriever: HybridRetriever):
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
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
