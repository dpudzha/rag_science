"""Ollama connectivity health check with retries."""
import logging
import time

import httpx

from config import OLLAMA_BASE_URL

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
