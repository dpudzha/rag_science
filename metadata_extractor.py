"""Metadata extraction from queries and filtering of retrieval results."""
import json
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config import OLLAMA_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "metadata_extraction.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()


class MetadataExtractor:
    """Extracts structured metadata (dates, authors, papers, sources) from queries."""

    def __init__(self, llm: ChatOllama | None = None):
        self._llm = llm or ChatOllama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
        )

    def extract(self, query: str) -> dict:
        """Extract metadata from a query. Returns dict with dates/authors/papers/sources."""
        prompt = _PROMPT_TEMPLATE.replace("{query}", query)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                metadata = json.loads(text[start:end])
                # Normalize: ensure all expected keys exist
                result = {
                    "dates": metadata.get("dates"),
                    "authors": metadata.get("authors"),
                    "papers": metadata.get("papers"),
                    "sources": metadata.get("sources"),
                }
                logger.info("Extracted metadata: %s", result)
                return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Metadata extraction failed (%s), returning empty", e)
        return {"dates": None, "authors": None, "papers": None, "sources": None}

    def has_filters(self, metadata: dict) -> bool:
        """Check if any metadata filters were extracted."""
        return any(v for v in metadata.values() if v)


class MetadataFilterApplier:
    """Filters retrieved documents based on extracted metadata."""

    @staticmethod
    def apply(docs: list[Document], metadata: dict) -> list[Document]:
        """Filter documents by metadata. Returns all docs if no filters match."""
        if not any(v for v in metadata.values() if v):
            return docs

        filtered = []
        for doc in docs:
            if MetadataFilterApplier._matches(doc, metadata):
                filtered.append(doc)

        # If filters are too strict and nothing matches, return original
        if not filtered:
            logger.info("Metadata filters too strict, returning unfiltered results")
            return docs

        logger.info("Metadata filtering: %d -> %d documents", len(docs), len(filtered))
        return filtered

    @staticmethod
    def _matches(doc: Document, metadata: dict) -> bool:
        """Check if a document matches ALL active metadata filters (AND logic)."""
        doc_meta = doc.metadata
        content = doc.page_content.lower()

        # Source filter: match by filename
        sources = metadata.get("sources")
        if sources:
            doc_source = doc_meta.get("source", "").lower()
            if not any(s.lower() in doc_source for s in sources):
                return False

        # Author filter: check content or metadata
        authors = metadata.get("authors")
        if authors:
            author_meta = doc_meta.get("authors", "").lower()
            if not any(
                author.lower() in content or author.lower() in author_meta
                for author in authors
            ):
                return False

        # Date filter: check content or metadata
        dates = metadata.get("dates")
        if dates:
            doc_date = str(doc_meta.get("creation_date", ""))
            if not any(date in content or date in doc_date for date in dates):
                return False

        # Paper title filter: check title metadata or content
        papers = metadata.get("papers")
        if papers:
            doc_title = doc_meta.get("title", "").lower()
            if not any(
                paper.lower() in doc_title or paper.lower() in content
                for paper in papers
            ):
                return False

        return True
