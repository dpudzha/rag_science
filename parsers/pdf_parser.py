"""PDF parser — wraps existing extract_text_from_pdf for unified interface."""
from ingest import extract_text_from_pdf


class PDFParser:
    """Parse PDF files into the standard document dict format."""

    def parse(self, path: str) -> dict:
        return extract_text_from_pdf(path)
