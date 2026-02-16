"""Document parsers for multiple file formats."""
from pathlib import Path
from typing import Protocol


class Parser(Protocol):
    def parse(self, path: str) -> dict: ...


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
