"""Retrieve relevant chunks and answer questions with hybrid search + reranking."""
import logging
import pickle
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
    ENABLE_PARENT_RETRIEVAL,
    INTENT_CLASSIFICATION_ENABLED,
    ARCHETYPE_DETECTION_ENABLED,
    METADATA_EXTRACTION_ENABLED,
    QUERY_RESOLUTION_ENABLED,
    RELEVANCE_CHECK_ENABLED,
    RELEVANCE_THRESHOLD,
    MAX_RETRIEVAL_RETRIES,
    ENABLE_SQL_AGENT,
    AGENT_MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Module-level cross-encoder singleton (lazy-loaded)
_cross_encoder: CrossEncoder | None = None


class VectorstoreNotFoundError(FileNotFoundError):
    """Raised when the FAISS vectorstore is missing from disk."""


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model: %s", RERANK_MODEL)
        # Suppress noisy model load logs from sentence-transformers
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


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
    parent_chunks: list[Document] | None = None
    metadata_filters: dict | None = None
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
        doc_map: dict[str, Document] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        rrf_k = 60  # Standard RRF constant

        # Dense retrieval via FAISS
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.k_candidates)
        for rank, (doc, _score) in enumerate(dense_results):
            key = self._doc_key(doc)
            doc_map[key] = doc
            dense_ranks[key] = rank

        # BM25 keyword retrieval
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:self.k_candidates]
        for rank, idx in enumerate(bm25_top):
            doc = self.bm25_docs[idx]
            key = self._doc_key(doc)
            doc_map[key] = doc
            bm25_ranks[key] = rank

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for key in doc_map:
            score = 0.0
            if key in dense_ranks:
                score += self.dense_weight / (rrf_k + dense_ranks[key])
            if key in bm25_ranks:
                score += self.bm25_weight / (rrf_k + bm25_ranks[key])
            rrf_scores[key] = score

        # Sort by RRF score, take top candidates for reranking
        ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        candidates = [doc_map[key] for key in ranked[:self.k_candidates]]

        if not candidates:
            return []

        # Cross-encoder reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        # Apply metadata filters BEFORE final k-slice
        filtered = [doc for doc, _ in reranked]
        if self.metadata_filters:
            from metadata_extractor import MetadataFilterApplier
            filtered = MetadataFilterApplier.apply(filtered, self.metadata_filters)

        top_docs = filtered[:self.k]

        # If parent-document retrieval is enabled, expand child chunks to parent chunks
        if self.parent_chunks is not None:
            expanded = []
            seen_parents = set()
            for doc in top_docs:
                parent_idx = doc.metadata.get("parent_idx")
                if parent_idx is not None and parent_idx < len(self.parent_chunks):
                    if parent_idx not in seen_parents:
                        seen_parents.add(parent_idx)
                        expanded.append(self.parent_chunks[parent_idx])
                else:
                    expanded.append(doc)
            return expanded

        return top_docs

    def get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Public retrieval interface for direct callers and legacy integrations."""
        return self._get_relevant_documents(query, **kwargs)


class PreloadedRetriever(BaseRetriever):
    """Wrapper that returns pre-fetched docs instead of re-retrieving."""
    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.docs


def load_vectorstore() -> FAISS:
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        message = (
            f"No vectorstore found at {VECTORSTORE_DIR}/. "
            "Run 'python ingest.py' first to ingest PDFs."
        )
        logger.error(message)
        raise VectorstoreNotFoundError(message)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


def load_bm25() -> tuple[BM25Okapi, list[Document]] | None:
    """Load persisted BM25 index from disk. Returns None if not found."""
    bm25_path = Path(VECTORSTORE_DIR) / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        logger.info("BM25 index loaded from %s", bm25_path)
        return data["bm25"], data["docs"]
    return None


def load_parent_chunks() -> list[Document] | None:
    """Load parent chunks for parent-document retrieval."""
    parents_path = Path(VECTORSTORE_DIR) / "parent_chunks.pkl"
    if parents_path.exists():
        with open(parents_path, "rb") as f:
            return pickle.load(f)
    return None


def build_retriever(vectorstore: FAISS) -> HybridRetriever:
    # Try to load persisted BM25 index first
    bm25_data = load_bm25()
    if bm25_data:
        bm25, bm25_docs = bm25_data
    else:
        # Fallback: rebuild from docstore (backward compatibility)
        logger.warning("No persisted BM25 index found, rebuilding from docstore")
        docstore = vectorstore.docstore
        all_docs = [docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]
        tokenized = [tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized)
        bm25_docs = all_docs

    cross_encoder = _get_cross_encoder()

    parent_chunks = None
    if ENABLE_PARENT_RETRIEVAL:
        parent_chunks = load_parent_chunks()
        if parent_chunks:
            logger.info("Parent-document retrieval enabled with %d parent chunks", len(parent_chunks))

    return HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        bm25_docs=bm25_docs,
        cross_encoder=cross_encoder,
        parent_chunks=parent_chunks,
    )


def build_qa_chain(retriever: HybridRetriever):
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )


def build_agent(retriever: HybridRetriever):
    """Build a RAG agent with tool selection (RAG + optional SQL)."""
    from agent import RAGAgent
    return RAGAgent(retriever=retriever, max_iterations=AGENT_MAX_ITERATIONS)


def print_result(result: dict):
    print(f"\nAnswer:\n{result['answer']}\n")
    seen = set()
    pages = []
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            pages.append(f"{source} p.{page}")
    if pages:
        print(f"Sources: {', '.join(pages)}")


def _get_intent_classifier():
    """Lazy-load intent classifier if enabled."""
    if not INTENT_CLASSIFICATION_ENABLED:
        return None
    from intent_classifier import IntentClassifier
    return IntentClassifier()


def _get_archetype_detector():
    """Lazy-load archetype detector if enabled."""
    if not ARCHETYPE_DETECTION_ENABLED:
        return None
    from archetype_detector import ArchetypeDetector
    return ArchetypeDetector()


def _get_domain_terminology() -> dict:
    """Load domain terminology for query reformulation."""
    from archetype_detector import _load_domain_terminology
    return _load_domain_terminology()


def _get_query_resolver():
    """Lazy-load query resolver if enabled."""
    if not QUERY_RESOLUTION_ENABLED:
        return None
    from query_resolver import QueryResolver
    return QueryResolver()


def _get_metadata_extractor():
    """Lazy-load metadata extractor if enabled."""
    if not METADATA_EXTRACTION_ENABLED:
        return None
    from metadata_extractor import MetadataExtractor
    return MetadataExtractor()


def _get_relevance_checker():
    """Lazy-load relevance checker if enabled."""
    if not RELEVANCE_CHECK_ENABLED:
        return None
    from relevance_checker import RelevanceChecker
    return RelevanceChecker(threshold=RELEVANCE_THRESHOLD)


def preprocess_query(question: str, retriever: HybridRetriever,
                     detector=None,
                     metadata_extractor=None,
                     domain_terms: dict | None = None) -> str:
    """Apply archetype detection + reformulation (single LLM call) and metadata extraction."""
    if detector:
        archetype, question = detector.detect_and_reformulate(
            question, domain_terms or {},
        )
        bm25_w, dense_w = detector.get_weights(archetype)
        retriever.bm25_weight = bm25_w
        retriever.dense_weight = dense_w
        logger.info("Adjusted weights: bm25=%.2f, dense=%.2f for archetype %s",
                     bm25_w, dense_w, archetype)

    # Extract metadata filters and set on retriever
    if metadata_extractor:
        metadata = metadata_extractor.extract(question)
        if metadata_extractor.has_filters(metadata):
            retriever.metadata_filters = metadata
        else:
            retriever.metadata_filters = None

    return question


def _run_pipeline(
    question: str,
    retriever: HybridRetriever,
    qa,
    agent,
    chat_history: list,
    classifier=None,
    detector=None,
    meta_extractor=None,
    relevance_checker=None,
    query_resolver=None,
    domain_terms: dict | None = None,
) -> dict | None:
    """Core query pipeline shared by ask() and interactive().

    Returns result dict, or None if the query was handled (e.g. greeting).
    """
    # Intent classification: skip RAG for greetings/chitchat
    if classifier:
        intent = classifier.classify(question)
        response = classifier.get_chitchat_response(intent)
        if response:
            print(f"\n{response}\n")
            return None

    # Resolve follow-up questions into standalone queries
    resolved = question
    if query_resolver and chat_history:
        resolved = query_resolver.resolve(question, chat_history)

    # Archetype detection + reformulation + metadata extraction
    processed = preprocess_query(
        resolved, retriever, detector, meta_extractor, domain_terms,
    )

    # Relevance checking with retry
    if relevance_checker:
        from relevance_checker import retrieve_with_relevance_check
        docs, rel_info = retrieve_with_relevance_check(
            retriever, processed, relevance_checker,
            max_retries=MAX_RETRIEVAL_RETRIES,
        )
        if rel_info["retry_count"] > 0:
            processed = rel_info["final_query"]
        # Use PreloadedRetriever to avoid double-retrieval in QA chain
        if qa is not None:
            qa.retriever = PreloadedRetriever(docs=docs)

    if agent is not None:
        result = agent.invoke(processed, chat_history=chat_history)
    else:
        result = qa.invoke({"question": processed, "chat_history": chat_history})

    # Restore original retriever for subsequent queries
    if relevance_checker and qa is not None:
        qa.retriever = retriever

    return result


def interactive():
    from health import check_ollama
    check_ollama()

    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = None if ENABLE_SQL_AGENT else build_qa_chain(retriever)
    agent = build_agent(retriever) if ENABLE_SQL_AGENT else None
    chat_history = []
    classifier = _get_intent_classifier()
    detector = _get_archetype_detector()
    meta_extractor = _get_metadata_extractor()
    relevance_checker = _get_relevance_checker()
    query_resolver = _get_query_resolver()
    domain_terms = _get_domain_terminology()

    print("RAG Science — ask questions about your papers (type 'quit' to exit)\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        result = _run_pipeline(
            question, retriever, qa, agent,
            chat_history=chat_history,
            classifier=classifier, detector=detector,
            meta_extractor=meta_extractor,
            relevance_checker=relevance_checker,
            query_resolver=query_resolver,
            domain_terms=domain_terms,
        )
        if result:
            print_result(result)
            chat_history.append((question, result["answer"]))


def ask(question: str):
    from health import check_ollama
    check_ollama()

    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    qa = None if ENABLE_SQL_AGENT else build_qa_chain(retriever)
    agent = build_agent(retriever) if ENABLE_SQL_AGENT else None

    result = _run_pipeline(
        question, retriever, qa, agent,
        chat_history=[],
        classifier=_get_intent_classifier(),
        detector=_get_archetype_detector(),
        meta_extractor=_get_metadata_extractor(),
        relevance_checker=_get_relevance_checker(),
        domain_terms=_get_domain_terminology(),
    )
    if result:
        print_result(result)


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging()
    args = sys.argv[1:]
    if args:
        ask(" ".join(args))
    else:
        interactive()
