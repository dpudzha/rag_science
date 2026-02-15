"""Archetype detection and query reformulation for improved retrieval."""
import json
import logging
import re
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

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
            "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "they",
            "this", "to", "was", "what", "when", "where", "which", "who", "why", "will",
            "with",
        }
        tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", text.lower()))
        return {tok for tok in tokens if tok not in stop_words}

    @staticmethod
    def _entity_tokens(text: str) -> set[str]:
        return {
            tok for tok in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9\-]{1,}\b", text)
            if any(ch.isupper() for ch in tok) or any(ch.isdigit() for ch in tok)
        }

    @staticmethod
    def _asks_methodology(text: str) -> bool:
        lowered = text.lower()
        method_cues = ("method", "methodology", "procedure", "protocol", "technique", "trained", "training", "how ")
        return any(cue in lowered for cue in method_cues)

    def _is_safe_rewrite(self, original: str, rewritten: str, archetype: str) -> bool:
        """Reject rewrites that drift away from the original query intent."""
        if original.strip().lower() == rewritten.strip().lower():
            return True

        original_tokens = self._content_tokens(original)
        rewritten_tokens = self._content_tokens(rewritten)
        if original_tokens:
            overlap = len(original_tokens & rewritten_tokens) / len(original_tokens)
            if overlap < 0.5:
                return False

        # Preserve high-signal entities like ITk, BERT, Stage-2, years, versions.
        missing_entities = {
            ent.lower() for ent in self._entity_tokens(original)
            if ent.lower() not in rewritten.lower()
        }
        if missing_entities:
            return False

        # Guard against misclassified archetype injecting methodology wording.
        if archetype == "HOW_METHODOLOGY":
            has_method_terms = any(
                term in rewritten.lower() for term in ("methodology", "experimental procedure", "procedure")
            )
            if has_method_terms and not self._asks_methodology(original):
                return False

        return True

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
                if self._is_safe_rewrite(query, reformulated, archetype):
                    logger.info("Reformulated: '%s' -> '%s'", query[:60], reformulated[:60])
                    return reformulated
                logger.info("Rejected reformulation drift, keeping original query: '%s'", query[:60])
            return query
        except Exception as e:
            logger.warning("Query reformulation failed (%s), using original", e)
            return query
