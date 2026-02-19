"""Backend connectivity health check with retries."""
import logging
import time

import httpx

from config import OLLAMA_BASE_URL, LLM_BACKEND, ANTHROPIC_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def check_ollama(retries: int = 5, delay: float = 2) -> bool:
    """Verify Ollama is reachable at OLLAMA_BASE_URL.

    Uses exponential backoff: delay, delay*2, delay*4, ...
    Returns True on success, raises ConnectionError after all retries fail.
    """
    url = f"{OLLAMA_BASE_URL}/api/tags"
    wait = delay

    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(url, timeout=5)
            resp.raise_for_status()
            logger.info("Ollama is reachable at %s", OLLAMA_BASE_URL)
            return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.warning("Ollama not ready (attempt %d/%d): %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(wait)
                wait *= 2

    raise ConnectionError(
        f"Could not reach Ollama at {OLLAMA_BASE_URL} after {retries} attempts. "
        "Is Ollama running?"
    )


def check_backend(retries: int = 5, delay: float = 2) -> bool:
    """Verify the configured LLM backend is available.

    For ollama: hits /api/tags with retries.
    For anthropic/openai: verifies the API key is set.
    """
    if LLM_BACKEND == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ConnectionError(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set."
            )
        logger.info("Anthropic backend configured (API key set)")
        return True

    if LLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise ConnectionError(
                "LLM_BACKEND=openai but OPENAI_API_KEY is not set."
            )
        logger.info("OpenAI backend configured (API key set)")
        return True

    return check_ollama(retries=retries, delay=delay)
