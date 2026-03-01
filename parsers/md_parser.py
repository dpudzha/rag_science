"""Markdown parser — extracts text from .md files."""
import re
from pathlib import Path

from parsers import DocumentDict


class MDParser:
    """Parse Markdown files into the standard document dict format."""

    def parse(self, path: str) -> DocumentDict:
        text = Path(path).read_text(encoding="utf-8")

        title = self._extract_title(text) or Path(path).stem

        return {
            "pages": [{"text": text, "page": 1}],
            "source": Path(path).name,
            "title": title,
            "creation_date": "",
            "authors": "",
            "tables": [],
        }

    @staticmethod
    def _extract_title(text: str) -> str:
        """Return the first H1 heading, or empty string if none."""
        match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        return match.group(1).strip() if match else ""
