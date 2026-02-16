"""RAG tool wrapping the existing HybridRetriever for agent use."""
import logging
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    """Search the scientific paper knowledge base using hybrid retrieval."""

    name: str = "search_papers"
    description: str = (
        "Search the scientific paper knowledge base. Use this tool to find information "
        "about research papers, methods, results, and findings. Input should be a "
        "research question or search query."
    )
    retriever: object  # HybridRetriever

    model_config = {"arbitrary_types_allowed": True}

    def _retrieve_docs(self, query: str) -> list:
        """Retrieve docs across retriever interface variants."""
        invoke = getattr(self.retriever, "invoke", None)
        if callable(invoke):
            return invoke(query)

        get_relevant_documents = getattr(self.retriever, "get_relevant_documents", None)
        if callable(get_relevant_documents):
            return get_relevant_documents(query)

        legacy_get_relevant_documents = getattr(self.retriever, "_get_relevant_documents", None)
        if callable(legacy_get_relevant_documents):
            return legacy_get_relevant_documents(query)

        logger.error("Retriever does not expose a supported retrieval interface")
        return []

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        docs = self._retrieve_docs(query)
        if not docs:
            return "No relevant documents found for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            results.append(f"[{i}] ({source}, p.{page}):\n{doc.page_content[:500]}")

        return "\n\n".join(results)
