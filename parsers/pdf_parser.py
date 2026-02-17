"""PDF parser — extract text, metadata, and title from PDF files."""
from pathlib import Path

import fitz

from parsers import DocumentDict


def _extract_title(first_page_text: str) -> str | None:
    """Extract paper title from the first page (first non-empty line, heuristic)."""
    for line in first_page_text.split("\n"):
        line = line.strip()
        if len(line) > 10 and not line.startswith("http"):
            return line
    return None


def extract_text_from_pdf(pdf_path: str) -> DocumentDict:
    pages = []
    title = None
    creation_date = None
    authors = None
    with fitz.open(pdf_path) as doc:
        # Extract PDF metadata
        pdf_meta = doc.metadata
        if pdf_meta:
            creation_date = pdf_meta.get("creationDate", "")
            authors = pdf_meta.get("author", "")

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"text": text, "page": page_num})
                if title is None:
                    title = _extract_title(text)
    return {
        "pages": pages,
        "source": Path(pdf_path).name,
        "title": title or Path(pdf_path).stem,
        "creation_date": creation_date or "",
        "authors": authors or "",
    }


class PDFParser:
    """Parse PDF files into the standard document dict format."""

    def parse(self, path: str) -> DocumentDict:
        return extract_text_from_pdf(path)
