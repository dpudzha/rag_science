"""Tests for ingest.py: chunking, MD5 dedup, page attribution, metadata enrichment."""
import json
import pickle
import types
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
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
        with patch("ingest.CHUNK_SIZE", 200), patch("ingest.CHUNK_OVERLAP", 20):
            chunks = chunk_documents([doc])
        # First chunk should be page 1, last chunk(s) should be page 2
        assert chunks[0].metadata["page"] == 1
        assert chunks[-1].metadata["page"] == 2

    def test_respects_token_limit(self):
        from ingest import chunk_documents
        doc = {
            "pages": [{"text": "word " * 1300, "page": 1}],
            "source": "tokens.pdf",
            "title": "Token Limit",
        }
        with patch("ingest.CHUNK_SIZE", 1200), patch("ingest.CHUNK_OVERLAP", 120):
            chunks = chunk_documents([doc])
        assert len(chunks) >= 2
        for chunk in chunks:
            token_count = len([t for t in chunk.page_content.split() if t.isalnum()])
            assert token_count <= 1220  # includes title/section prefix tokens


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


class TestSaveLargeTablesToSql:
    def test_returns_description_chunks(self, tmp_path):
        from ingest import _save_large_tables_to_sql
        large_tables = [{
            "source": "data.xlsx",
            "table": {
                "data": [
                    ["Name", "Value", "Category"],
                    *([["item", "100", "A"]] * 105),
                ],
                "num_rows": 105,
                "sheet_name": "Sheet1",
            },
        }]
        with patch("sql_database.SQLDatabase") as MockDB:
            MockDB.return_value = MagicMock()
            chunks = _save_large_tables_to_sql(large_tables)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert "SQL Table" in chunk.page_content
        assert "query_tables" in chunk.page_content
        assert chunk.metadata["content_type"] == "sql_table_description"
        assert chunk.metadata["source"] == "data.xlsx"

    def test_empty_tables_returns_empty(self):
        from ingest import _save_large_tables_to_sql
        assert _save_large_tables_to_sql([]) == []

    def test_description_includes_columns_and_sample(self, tmp_path):
        from ingest import _save_large_tables_to_sql
        large_tables = [{
            "source": "report.xlsx",
            "table": {
                "data": [
                    ["Year", "Revenue"],
                    ["2020", "1M"],
                    ["2021", "2M"],
                    *([["2022", "3M"]] * 102),
                ],
                "num_rows": 104,
            },
        }]
        with patch("sql_database.SQLDatabase") as MockDB:
            MockDB.return_value = MagicMock()
            chunks = _save_large_tables_to_sql(large_tables)

        content = chunks[0].page_content
        assert "Year" in content
        assert "Revenue" in content
        assert "2020" in content  # sample row


class TestIngestFlow:
    def test_ingest_skips_vectorstore_build_when_no_chunks(self):
        import ingest

        docs = [{
            "hash": "newhash",
            "source": "empty.pdf",
            "pages": [{"text": " ", "page": 1}],
            "title": "Empty",
        }]

        with patch("health.check_ollama"), \
             patch("ingest.load_new_documents", return_value=(docs, [])), \
             patch("ingest.load_ingest_record", return_value={}), \
             patch("ingest._save_large_tables_to_sql", return_value=[]), \
             patch("ingest.load_existing_vectorstore", return_value=None), \
             patch("ingest.chunk_documents", return_value=[]), \
             patch("ingest.build_vectorstore") as mock_build_vectorstore, \
             patch("ingest.save_ingest_record") as mock_save_ingest_record:
            ingest.ingest()

        mock_build_vectorstore.assert_not_called()
        mock_save_ingest_record.assert_called_once_with({"newhash": "empty.pdf"})

    def test_ingest_passes_updated_record_into_atomic_swap(self, sample_documents):
        import ingest

        docs = [{
            "hash": "newhash",
            "source": "paper.pdf",
            "pages": [{"text": "content", "page": 1}],
            "title": "Paper",
        }]
        fake_store = object()

        with patch("health.check_ollama"), \
             patch("ingest.load_new_documents", return_value=(docs, [])), \
             patch("ingest.load_ingest_record", return_value={"oldhash": "old.pdf"}), \
             patch("ingest._save_large_tables_to_sql", return_value=[]), \
             patch("ingest.load_existing_vectorstore", return_value=None), \
             patch("ingest.chunk_documents", return_value=sample_documents), \
             patch("ingest.build_vectorstore", return_value=fake_store), \
             patch("ingest._get_all_docs_from_store", return_value=sample_documents), \
             patch("ingest.build_bm25", return_value=(MagicMock(), sample_documents)), \
             patch("ingest._save_vectorstore_atomic") as mock_save_atomic, \
             patch("ingest.save_ingest_record") as mock_save_ingest_record:
            ingest.ingest()

        assert mock_save_atomic.call_count == 1
        assert mock_save_atomic.call_args.kwargs["ingest_record"] == {
            "oldhash": "old.pdf",
            "newhash": "paper.pdf",
        }
        mock_save_ingest_record.assert_not_called()

    def test_ingest_syncs_s3_before_loading_documents(self):
        import ingest

        with patch("health.check_ollama"), \
             patch("ingest.sync_recent_s3_documents") as mock_sync, \
             patch("ingest.load_new_documents", return_value=([], [])):
            ingest.ingest()

        mock_sync.assert_called_once()


class TestS3Sync:
    def test_returns_zero_when_disabled(self):
        from ingest import sync_recent_s3_documents
        with patch("ingest.ENABLE_S3_INGEST", False):
            assert sync_recent_s3_documents("/tmp") == 0

    def test_downloads_only_recent_supported_files(self, tmp_path):
        from ingest import sync_recent_s3_documents

        class FakePaginator:
            def paginate(self, Bucket, Prefix):
                now = datetime.now(timezone.utc)
                return [{
                    "Contents": [
                        {"Key": "new/report.pdf", "LastModified": now},
                        {"Key": "new/table.xlsx", "LastModified": now},
                        {"Key": "old/doc.pdf", "LastModified": now - timedelta(hours=10)},
                        {"Key": "new/image.png", "LastModified": now},
                    ],
                }]

        fake_s3 = MagicMock()
        fake_s3.get_paginator.return_value = FakePaginator()

        fake_boto3 = types.SimpleNamespace(client=MagicMock(return_value=fake_s3))

        with patch("ingest.ENABLE_S3_INGEST", True), \
             patch("ingest.S3_BUCKET", "bucket"), \
             patch("ingest.S3_PREFIX", "new/"), \
             patch("ingest.S3_LOOKBACK_HOURS", 3), \
             patch("ingest.SUPPORTED_FORMATS", ["pdf", "xlsx"]), \
             patch.dict("sys.modules", {"boto3": fake_boto3}):
            downloaded = sync_recent_s3_documents(str(tmp_path))

        assert downloaded == 2
        assert fake_s3.download_file.call_count == 2
