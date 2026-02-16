"""Intent classification to route greetings/chitchat away from RAG pipeline."""
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_default_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "intent_classification.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

INTENTS = {"GREETING", "FAREWELL", "CHITCHAT", "SUBSTANTIVE"}

_CHITCHAT_RESPONSES = {
    "GREETING": "Hello! I'm a scientific research assistant. Ask me questions about the papers in the knowledge base.",
    "FAREWELL": "Goodbye! Feel free to come back anytime.",
    "CHITCHAT": (
        "I'm a research assistant that answers questions about the scientific papers "
        "ingested into my knowledge base. You can ask things like:\n"
        '  - "What methods were used in [paper]?"\n'
        '  - "What were the main results?"\n'
        '  - "Compare the findings across papers"\n'
        '  - "Summarize the conclusions"\n'
        "Just ask a specific research question and I'll search the papers for you."
    ),
}

# Fast regex patterns for obvious intents (checked before LLM call)
_FAREWELL_PATTERNS = re.compile(
    r"^\s*(bye|goodbye|see\s+you|later|farewell|good\s*night|take\s+care)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|cheers)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(what\s+(can|do)\s+you\s+do|what\s+are\s+(you|your\s+capabilities)|"
    r"who\s+are\s+you|how\s+are\s+you|tell\s+me\s+(a\s+joke|about\s+yourself)|"
    r"what\s+can\s+(i|we)\s+ask\b.*|what\s+kind\s+of\s+questions\b.*|"
    r"what\s+can\s+you\s+(help|assist)\b.*|help\s+me|what\s+do\s+you\s+know|"
    r"what\s+are\s+you\s+(good\s+at|for)|are\s+you\s+a\s+(bot|ai|robot)|"
    r"questions?\s+about\s+what|about\s+what|what\s+papers|"
    r"what\s+topics|what\s+documents|like\s+what|such\s+as)\s*[?!.]*\s*$",
    re.IGNORECASE,
)


class IntentClassifier:
    """Classifies user queries as GREETING, CHITCHAT, or SUBSTANTIVE using an LLM."""

    def __init__(self, llm=None):
        self._llm = llm or get_default_llm()

    @staticmethod
    def _fast_classify(query: str) -> str | None:
        """Fast regex check for obvious greetings/chitchat/farewell. Returns None if uncertain."""
        if _FAREWELL_PATTERNS.match(query):
            return "FAREWELL"
        if _GREETING_PATTERNS.match(query):
            return "GREETING"
        if _CHITCHAT_PATTERNS.search(query):
            return "CHITCHAT"
        return None

    def classify(self, query: str) -> str:
        """Return the intent label for a query. Falls back to SUBSTANTIVE on error."""
        # Try fast regex first — avoids LLM call for obvious cases
        fast_result = self._fast_classify(query)
        if fast_result:
            logger.info("Intent classified as %s (fast) for query: %s", fast_result, query[:80])
            return fast_result

        try:
            response = self._llm.invoke([
                SystemMessage(content=_PROMPT_TEMPLATE),
                HumanMessage(content=query),
            ])
            intent = response.content.strip().split("\n")[0].strip().upper()
            if intent in INTENTS:
                logger.info("Intent classified as %s for query: %s", intent, query[:80])
                return intent
            logger.warning("Unrecognized intent '%s', falling back to SUBSTANTIVE", intent)
            return "SUBSTANTIVE"
        except Exception as e:
            logger.warning("Intent classification failed (%s), falling back to SUBSTANTIVE", e)
            return "SUBSTANTIVE"

    def get_chitchat_response(self, intent: str) -> str | None:
        """Return a canned response for non-substantive intents, or None for SUBSTANTIVE."""
        return _CHITCHAT_RESPONSES.get(intent)
