"""Retrieve relevant chunks and answer questions with hybrid search + reranking."""
import logging
import sys

from config import (
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
from retriever import (
    HybridRetriever,
    PreloadedRetriever,
    load_vectorstore,
    build_retriever,
    build_qa_chain,
)

logger = logging.getLogger(__name__)


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
