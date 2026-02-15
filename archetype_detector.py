"""Archetype detection and query reformulation for improved retrieval."""
import json
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import OLLAMA_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

_ARCHETYPE_PROMPT_PATH = Path(__file__).parent / "prompts" / "archetype_detection.txt"
_REFORMULATION_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_reformulation.txt"
_DOMAIN_TERMINOLOGY_PATH = Path(__file__).parent / "domain_terminology.json"

_ARCHETYPE_PROMPT = _ARCHETYPE_PROMPT_PATH.read_text()
_REFORMULATION_PROMPT = _REFORMULATION_PROMPT_PATH.read_text()

ARCHETYPES = {
    "WHAT_INFORMATION",
    "HOW_METHODOLOGY",
    "COMPARISON",
    "DEFINITION",
    "WHY_REASONING",
    "SUMMARY",
}

# Default archetype-to-weight mapping: (bm25_weight, dense_weight)
ARCHETYPE_WEIGHTS = {
    "WHAT_INFORMATION": (0.4, 0.6),
    "HOW_METHODOLOGY": (0.3, 0.7),
    "COMPARISON": (0.3, 0.7),
    "DEFINITION": (0.5, 0.5),
    "WHY_REASONING": (0.2, 0.8),
    "SUMMARY": (0.2, 0.8),
}

DEFAULT_WEIGHTS = (0.3, 0.7)


def _load_domain_terminology(path: Path | None = None) -> dict:
    """Load domain terminology from JSON file."""
    p = path or _DOMAIN_TERMINOLOGY_PATH
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load domain terminology: %s", e)
    return {}


class ArchetypeDetector:
    """Detects the archetype of a research query using an LLM."""

    def __init__(self, llm: ChatOllama | None = None):
        self._llm = llm or ChatOllama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
        )

    def detect(self, query: str) -> str:
        """Return the archetype label for a query. Falls back to WHAT_INFORMATION."""
        prompt = _ARCHETYPE_PROMPT.replace("{query}", query)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            archetype = response.content.strip().split("\n")[0].strip().upper()
            if archetype in ARCHETYPES:
                logger.info("Archetype: %s for query: %s", archetype, query[:80])
                return archetype
            logger.warning("Unrecognized archetype '%s', falling back to WHAT_INFORMATION", archetype)
            return "WHAT_INFORMATION"
        except Exception as e:
            logger.warning("Archetype detection failed (%s), falling back to WHAT_INFORMATION", e)
            return "WHAT_INFORMATION"

    def get_weights(self, archetype: str) -> tuple[float, float]:
        """Return (bm25_weight, dense_weight) for an archetype."""
        return ARCHETYPE_WEIGHTS.get(archetype, DEFAULT_WEIGHTS)


class QueryReformulator:
    """Rewrites queries using archetype context and domain terminology."""

    def __init__(self, llm: ChatOllama | None = None, terminology_path: Path | None = None):
        self._llm = llm or ChatOllama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
        )
        self._terminology = _load_domain_terminology(terminology_path)

    def reformulate(self, query: str, archetype: str) -> str:
        """Rewrite the query with domain awareness. Falls back to original on error."""
        domain_terms = ""
        if self._terminology:
            # Find relevant abbreviations
            abbrevs = self._terminology.get("abbreviations", {})
            relevant = {k: v for k, v in abbrevs.items() if k.lower() in query.lower()}
            if relevant:
                domain_terms = ", ".join(f"{k} = {v}" for k, v in relevant.items())

        prompt = (_REFORMULATION_PROMPT
                  .replace("{query}", query)
                  .replace("{archetype}", archetype)
                  .replace("{domain_terms}", domain_terms or "none"))
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            reformulated = response.content.strip()
            if reformulated:
                logger.info("Reformulated: '%s' -> '%s'", query[:60], reformulated[:60])
                return reformulated
            return query
        except Exception as e:
            logger.warning("Query reformulation failed (%s), using original", e)
            return query
