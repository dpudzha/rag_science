"""Relevance checking and query retry for retrieval results."""
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_default_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "relevance_check.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_SCORE_RE = re.compile(r"SCORE:\s*([\d.]+)")
_SUGGESTION_RE = re.compile(r"SUGGESTION:\s*(.+)", re.IGNORECASE)


class RelevanceChecker:
    """Scores the relevance of retrieved documents against a query."""

    def __init__(self, llm=None, threshold: float = 0.6):
        self._llm = llm or get_default_llm()
        self.threshold = threshold

    def check(self, query: str, docs: list[Document]) -> dict:
        """Score relevance and optionally suggest reformulation.

        Returns dict with 'score' (float), 'is_relevant' (bool),
        and 'suggestion' (str or None).
        """
        if not docs:
            return {"score": 0.0, "is_relevant": False, "suggestion": None}

        # Format documents for the prompt
        doc_texts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            doc_texts.append(f"[{i}] ({source}): {doc.page_content[:500]}")
        documents_str = "\n\n".join(doc_texts)

        system_prompt = _PROMPT_TEMPLATE.replace("{documents}", documents_str)

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ])
            text = response.content.strip()

            # Parse score
            score_match = _SCORE_RE.search(text)
            score = float(score_match.group(1)) if score_match else 0.5
            score = max(0.0, min(1.0, score))

            # Parse suggestion
            suggestion = None
            sugg_match = _SUGGESTION_RE.search(text)
            if sugg_match:
                sugg = sugg_match.group(1).strip()
                if sugg.lower() != "none":
                    suggestion = sugg

            is_relevant = score >= self.threshold
            logger.info("Relevance score: %.2f (threshold: %.2f, relevant: %s)",
                        score, self.threshold, is_relevant)

            return {
                "score": score,
                "is_relevant": is_relevant,
                "suggestion": suggestion,
            }
        except Exception as e:
            logger.warning("Relevance check failed (%s), assuming relevant", e)
            return {"score": 1.0, "is_relevant": True, "suggestion": None}


def retrieve_with_relevance_check(retriever, query: str, checker: RelevanceChecker,
                                   max_retries: int = 1) -> tuple[list[Document], dict]:
    """Retrieve documents with relevance checking and optional retry.

    Returns (docs, relevance_info) where relevance_info contains score,
    is_relevant, retry_count, and final_query.
    """
    current_query = query
    retry_count = 0

    for attempt in range(1 + max_retries):
        docs = retriever.invoke(current_query)
        result = checker.check(current_query, docs)

        if result["is_relevant"] or attempt >= max_retries:
            return docs, {
                "score": result["score"],
                "is_relevant": result["is_relevant"],
                "retry_count": retry_count,
                "final_query": current_query,
            }

        # Retry with suggested reformulation
        if result["suggestion"]:
            logger.info("Relevance low (%.2f), retrying with: %s",
                        result["score"], result["suggestion"])
            current_query = result["suggestion"]
            retry_count += 1
        else:
            # No suggestion, return current results
            return docs, {
                "score": result["score"],
                "is_relevant": result["is_relevant"],
                "retry_count": retry_count,
                "final_query": current_query,
            }

    return docs, {
        "score": result["score"],
        "is_relevant": result["is_relevant"],
        "retry_count": retry_count,
        "final_query": current_query,
    }
