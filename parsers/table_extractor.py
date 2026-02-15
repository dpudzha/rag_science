"""PDF table extraction using PyMuPDF's find_tables()."""
import logging

import fitz

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables from PDF files using PyMuPDF."""

    def extract_tables(self, pdf_path: str) -> list[dict]:
        """Extract all tables from a PDF. Returns list of table dicts."""
        tables = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    page_tables = page.find_tables()
                    for i, table in enumerate(page_tables.tables):
                        data = table.extract()
                        if data and len(data) > 1:  # At least header + 1 row
                            # Clean None values
                            cleaned = []
                            for row in data:
                                cleaned.append([str(cell) if cell else "" for cell in row])
                            tables.append({
                                "data": cleaned,
                                "index": len(tables),
                                "page": page_num,
                                "num_rows": len(cleaned) - 1,  # Exclude header
                            })
        except Exception as e:
            logger.warning("Table extraction failed for %s: %s", pdf_path, e)
        return tables
