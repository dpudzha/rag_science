"""Resolve follow-up questions into standalone queries using conversation history."""
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_default_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_resolution.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

# Pronouns and demonstratives that signal a genuine follow-up reference
_FOLLOW_UP_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|theirs"
    r"|such|the\s+same|the\s+other|the\s+mentioned|the\s+above|the\s+previous)\b",
    re.IGNORECASE,
)


def _needs_resolution(question: str) -> bool:
    """Return True only if the question contains references that require history."""
    return bool(_FOLLOW_UP_RE.search(question))


class QueryResolver:
    """Rewrites follow-up questions as standalone queries using chat history."""

    def __init__(self, llm=None):
        self._llm = llm or get_default_llm()

    def resolve(self, question: str, chat_history: list[tuple[str, str]]) -> str:
        """Resolve a follow-up into a standalone question.

        Args:
            question: The current user question.
            chat_history: List of (user_question, assistant_answer) tuples.

        Returns:
            A standalone question incorporating context from history.
            Falls back to the original question on error or empty history.
        """
        if not chat_history:
            return question

        # Skip LLM call for questions that are clearly standalone
        if not _needs_resolution(question):
            logger.debug("Query resolution skipped — no follow-up indicators: '%s'", question[:60])
            return question

        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]
        )

        system_prompt = _PROMPT_TEMPLATE.replace("{chat_history}", history_text)
        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ])
            resolved = response.content.strip()
            if resolved:
                logger.info("Resolved: '%s' -> '%s'", question[:60], resolved[:60])
                return resolved
            return question
        except Exception as e:
            logger.warning("Query resolution failed (%s), using original", e)
            return question
