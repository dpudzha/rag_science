"""Resolve follow-up questions into standalone queries using conversation history."""
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_default_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_resolution.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()


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
