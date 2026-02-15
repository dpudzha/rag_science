"""Tests for the parser factory."""
import pytest


class TestGetParser:
    def test_pdf_parser(self):
        from parsers import get_parser
        from parsers.pdf_parser import PDFParser
        parser = get_parser("test.pdf")
        assert isinstance(parser, PDFParser)

    def test_docx_parser(self):
        from parsers import get_parser
        from parsers.docx_parser import DOCXParser
        parser = get_parser("test.docx")
        assert isinstance(parser, DOCXParser)

    def test_xlsx_parser(self):
        from parsers import get_parser
        from parsers.xlsx_parser import XLSXParser
        parser = get_parser("test.xlsx")
        assert isinstance(parser, XLSXParser)

    def test_unsupported_format(self):
        from parsers import get_parser
        with pytest.raises(ValueError, match="Unsupported"):
            get_parser("test.txt")
