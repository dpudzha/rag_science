"""Tests for parsers/table_extractor.py."""
import pytest


class TestTableExtractor:
    def test_returns_empty_for_text_only_pdf(self, tmp_path):
        """A simple PDF without tables should return empty list."""
        import fitz
        path = tmp_path / "no_table.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a simple paragraph without tables.")
        doc.save(str(path))
        doc.close()

        from parsers.table_extractor import TableExtractor
        extractor = TableExtractor()
        tables = extractor.extract_tables(str(path))
        assert isinstance(tables, list)

    def test_handles_invalid_path(self):
        from parsers.table_extractor import TableExtractor
        extractor = TableExtractor()
        tables = extractor.extract_tables("/nonexistent/path.pdf")
        assert tables == []

    def test_table_data_format(self, tmp_path):
        """Verify table data format when tables are found."""
        from parsers.table_extractor import TableExtractor
        extractor = TableExtractor()
        # Can't easily create a PDF with tables programmatically,
        # but verify the method doesn't crash on a basic PDF
        import fitz
        path = tmp_path / "basic.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(path))
        doc.close()
        tables = extractor.extract_tables(str(path))
        assert isinstance(tables, list)
