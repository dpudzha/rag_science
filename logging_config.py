"""Centralized logging configuration."""
import logging
import os

LOG_FORMAT = "%(levelname)s: %(message)s"

# Libraries that are excessively noisy at INFO level
QUIET_LOGGERS = [
    "httpx",
    "httpcore",
    "faiss",
    "faiss.loader",
    "sentence_transformers",
    "urllib3",
    "transformers",
    "huggingface_hub",
]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Suppress sentence-transformers tqdm progress bars
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
