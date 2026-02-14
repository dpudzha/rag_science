"""Tests for ingest.py: chunking, MD5 dedup, page attribution, metadata enrichment."""
import json
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


class TestChunkDocuments:
    def test_creates_chunks_from_documents(self, sample_pdf_text):
        from ingest import chunk_documents
        chunks = chunk_documents([sample_pdf_text])
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_metadata_has_source_and_page(self, sample_pdf_text):
        from ingest import chunk_documents
        chunks = chunk_documents([sample_pdf_text])
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert chunk.metadata["source"] == "test_paper.pdf"

    def test_chunk_metadata_has_title(self, sample_pdf_text):
        from ingest import chunk_documents
        chunks = chunk_documents([sample_pdf_text])
        for chunk in chunks:
            assert "title" in chunk.metadata
            assert chunk.metadata["title"] == "This is the title of the paper"

    def test_enriched_content_has_paper_prefix(self, sample_pdf_text):
        from ingest import chunk_documents
        chunks = chunk_documents([sample_pdf_text])
        assert any("[Paper:" in c.page_content for c in chunks)

    def test_empty_document_produces_no_chunks(self):
        from ingest import chunk_documents
        doc = {"pages": [{"text": "   ", "page": 1}], "source": "empty.pdf", "title": ""}
        chunks = chunk_documents([doc])
        assert len(chunks) == 0

    def test_page_attribution_sequential(self):
        """Chunks should attribute to correct pages even with repeated text."""
        from ingest import chunk_documents
        doc = {
            "pages": [
                {"text": "A " * 500, "page": 1},
                {"text": "B " * 500, "page": 2},
            ],
            "source": "test.pdf",
            "title": "Test",
        }
        chunks = chunk_documents([doc])
        # First chunk should be page 1, last chunk(s) should be page 2
        assert chunks[0].metadata["page"] == 1
        assert chunks[-1].metadata["page"] == 2


class TestPageAtOffset:
    def test_first_page(self):
        from ingest import _page_at_offset
        assert _page_at_offset([0, 100, 200], [1, 2, 3], 50) == 1

    def test_second_page(self):
        from ingest import _page_at_offset
        assert _page_at_offset([0, 100, 200], [1, 2, 3], 150) == 2

    def test_exact_boundary(self):
        from ingest import _page_at_offset
        assert _page_at_offset([0, 100, 200], [1, 2, 3], 100) == 2

    def test_zero_offset(self):
        from ingest import _page_at_offset
        assert _page_at_offset([0, 100], [1, 2], 0) == 1


class TestIngestRecord:
    def test_load_empty_record(self, tmp_path):
        from ingest import load_ingest_record
        with patch("ingest.INGEST_RECORD", str(tmp_path / "missing.json")):
            record = load_ingest_record()
        assert record == {}

    def test_save_and_load_record(self, tmp_path):
        from ingest import load_ingest_record, save_ingest_record
        record_path = str(tmp_path / "ingested.json")
        with patch("ingest.INGEST_RECORD", record_path):
            save_ingest_record({"hash1": "file1.pdf"})
            record = load_ingest_record()
        assert record == {"hash1": "file1.pdf"}


class TestFileHash:
    def test_produces_hex_string(self, tmp_path):
        from ingest import file_hash
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = file_hash(str(f))
        assert len(h) == 32  # MD5 hex digest length
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self, tmp_path):
        from ingest import file_hash
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert file_hash(str(f1)) != file_hash(str(f2))


class TestBuildBm25:
    def test_builds_index(self, sample_documents):
        from ingest import build_bm25
        bm25, docs = build_bm25(sample_documents)
        assert docs == sample_documents
        scores = bm25.get_scores(["neural"])
        assert len(scores) == len(sample_documents)

    def test_save_and_load_bm25(self, tmp_vectorstore, sample_documents):
        from ingest import build_bm25, save_bm25
        bm25, docs = build_bm25(sample_documents)
        save_bm25(bm25, docs, str(tmp_vectorstore))

        pkl_path = tmp_vectorstore / "bm25_index.pkl"
        assert pkl_path.exists()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        assert "bm25" in data
        assert "docs" in data
        assert len(data["docs"]) == len(sample_documents)


class TestTokenize:
    def test_basic_tokenization(self):
        from ingest import tokenize
        assert tokenize("Hello World!") == ["hello", "world"]

    def test_numbers_included(self):
        from ingest import tokenize
        assert tokenize("GPT-3 has 175B params") == ["gpt", "3", "has", "175b", "params"]

    def test_empty_string(self):
        from ingest import tokenize
        assert tokenize("") == []


class TestExtractTitle:
    def test_extracts_first_long_line(self):
        from ingest import _extract_title
        text = "Short\nThis is a long enough title for extraction\nOther text"
        assert _extract_title(text) == "This is a long enough title for extraction"

    def test_skips_urls(self):
        from ingest import _extract_title
        text = "http://example.com/paper\nActual Paper Title Here"
        assert _extract_title(text) == "Actual Paper Title Here"

    def test_returns_none_for_short_text(self):
        from ingest import _extract_title
        assert _extract_title("Hi\nBye") is None


class TestDetectSectionHeader:
    def test_detects_numbered_header(self):
        from ingest import _detect_section_header
        text = "some text\n1. Introduction\nmore text"
        result = _detect_section_header(text)
        assert result is not None
        assert "Introduction" in result

    def test_detects_caps_header(self):
        from ingest import _detect_section_header
        text = "some text\nMETHODS\nmore text"
        result = _detect_section_header(text)
        assert result is not None

    def test_returns_none_for_no_header(self):
        from ingest import _detect_section_header
        assert _detect_section_header("just regular text here") is None
