"""Tests for parsers/xlsx_parser.py."""
import pytest
import pandas as pd


@pytest.fixture
def sample_xlsx(tmp_path):
    """Create a sample XLSX file."""
    path = tmp_path / "test.xlsx"
    df = pd.DataFrame({
        "Method": ["CNN", "RNN", "Transformer"],
        "Accuracy": [95.2, 92.1, 97.3],
        "F1 Score": [0.94, 0.91, 0.96],
    })
    df.to_excel(str(path), index=False)
    return str(path)


@pytest.fixture
def multi_sheet_xlsx(tmp_path):
    """Create a multi-sheet XLSX file."""
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(str(path)) as writer:
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"X": [5, 6], "Y": [7, 8]}).to_excel(writer, sheet_name="Sheet2", index=False)
    return str(path)


class TestXLSXParser:
    def test_extracts_tables(self, sample_xlsx):
        from parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        result = parser.parse(sample_xlsx)
        assert result["tables"]
        table = result["tables"][0]
        assert "Method" in table["data"][0]
        assert table["num_rows"] == 3

    def test_extracts_text_representation(self, sample_xlsx):
        from parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        result = parser.parse(sample_xlsx)
        assert result["pages"]
        text = result["pages"][0]["text"]
        assert "Method" in text
        assert "Accuracy" in text

    def test_source_is_filename(self, sample_xlsx):
        from parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        result = parser.parse(sample_xlsx)
        assert result["source"] == "test.xlsx"

    def test_multi_sheet(self, multi_sheet_xlsx):
        from parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        result = parser.parse(multi_sheet_xlsx)
        assert len(result["tables"]) == 2
        assert len(result["pages"]) == 2

    def test_includes_dataframe(self, sample_xlsx):
        from parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        result = parser.parse(sample_xlsx)
        assert result["tables"][0]["dataframe"] is not None
        assert isinstance(result["tables"][0]["dataframe"], pd.DataFrame)
