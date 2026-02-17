"""Document parsers for multiple file formats."""
from pathlib import Path
from typing import Protocol, TypedDict


class DocumentDict(TypedDict, total=False):
    """Shape of the dict returned by all parsers and used throughout the ingest pipeline."""
    pages: list[dict]          # [{"text": str, "page": int}, ...]
    source: str                # filename (e.g. "paper.pdf")
    title: str                 # extracted or filename-derived title
    creation_date: str         # PDF metadata or ""
    authors: str               # PDF metadata or ""
    tables: list[dict]         # extracted tables
    hash: str                  # SHA-256 file hash (added during ingestion)


class Parser(Protocol):
    def parse(self, path: str) -> DocumentDict: ...


def get_parser(path: str) -> Parser:
    """Return the appropriate parser for a file based on its extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        from parsers.pdf_parser import PDFParser
        return PDFParser()
    elif ext == ".docx":
        from parsers.docx_parser import DOCXParser
        return DOCXParser()
    elif ext in (".xlsx", ".xls"):
        from parsers.xlsx_parser import XLSXParser
        return XLSXParser()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
