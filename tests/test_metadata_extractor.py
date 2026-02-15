"""Tests for metadata_extractor.py: extraction and filtering."""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class TestMetadataExtractor:
    def _make_extractor(self, response_text):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response_text)
        from metadata_extractor import MetadataExtractor
        return MetadataExtractor(llm=mock_llm)

    def test_extracts_dates(self):
        response = json.dumps({"dates": ["2023"], "authors": None, "papers": None, "sources": None})
        extractor = self._make_extractor(response)
        result = extractor.extract("What was found in 2023?")
        assert result["dates"] == ["2023"]

    def test_extracts_authors(self):
        response = json.dumps({"dates": None, "authors": ["Smith"], "papers": None, "sources": None})
        extractor = self._make_extractor(response)
        result = extractor.extract("What did Smith find?")
        assert result["authors"] == ["Smith"]

    def test_extracts_sources(self):
        response = json.dumps({"dates": None, "authors": None, "papers": None, "sources": ["paper1.pdf"]})
        extractor = self._make_extractor(response)
        result = extractor.extract("Results from paper1.pdf")
        assert result["sources"] == ["paper1.pdf"]

    def test_no_metadata_found(self):
        response = json.dumps({"dates": None, "authors": None, "papers": None, "sources": None})
        extractor = self._make_extractor(response)
        result = extractor.extract("What is attention?")
        assert not extractor.has_filters(result)

    def test_fallback_on_invalid_json(self):
        extractor = self._make_extractor("not valid json")
        result = extractor.extract("test")
        assert result == {"dates": None, "authors": None, "papers": None, "sources": None}

    def test_fallback_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        from metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor(llm=mock_llm)
        result = extractor.extract("test")
        assert result == {"dates": None, "authors": None, "papers": None, "sources": None}

    def test_has_filters_true(self):
        from metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor(llm=MagicMock())
        assert extractor.has_filters({"dates": ["2023"], "authors": None, "papers": None, "sources": None})

    def test_has_filters_false(self):
        from metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor(llm=MagicMock())
        assert not extractor.has_filters({"dates": None, "authors": None, "papers": None, "sources": None})


class TestMetadataFilterApplier:
    @pytest.fixture
    def docs(self):
        return [
            Document(page_content="Neural nets in 2023", metadata={"source": "paper1.pdf", "page": 1, "title": "Neural Networks Study", "authors": "Smith"}),
            Document(page_content="Transformers in 2022", metadata={"source": "paper2.pdf", "page": 1, "title": "Transformer Architecture", "authors": "Zhang"}),
            Document(page_content="BERT models overview", metadata={"source": "paper3.pdf", "page": 1, "title": "BERT Survey", "creation_date": "2021"}),
        ]

    def test_filter_by_source(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": None, "authors": None, "papers": None, "sources": ["paper1.pdf"]}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == 1
        assert result[0].metadata["source"] == "paper1.pdf"

    def test_filter_by_author(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": None, "authors": ["Smith"], "papers": None, "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == 1
        assert result[0].metadata["authors"] == "Smith"

    def test_filter_by_date_in_content(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": ["2023"], "authors": None, "papers": None, "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == 1
        assert "2023" in result[0].page_content

    def test_filter_by_paper_title(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": None, "authors": None, "papers": ["BERT"], "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == 1
        assert "BERT" in result[0].metadata["title"]

    def test_no_filters_returns_all(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": None, "authors": None, "papers": None, "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == len(docs)

    def test_strict_filter_falls_back_to_all(self, docs):
        from metadata_extractor import MetadataFilterApplier
        metadata = {"dates": None, "authors": None, "papers": None, "sources": ["nonexistent.pdf"]}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == len(docs)

    def test_and_logic_multiple_filters(self, docs):
        """Multiple active filters use AND logic: doc must match ALL filters."""
        from metadata_extractor import MetadataFilterApplier
        # paper1 has author Smith AND date 2023 in content
        metadata = {"dates": ["2023"], "authors": ["Smith"], "papers": None, "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        assert len(result) == 1
        assert result[0].metadata["source"] == "paper1.pdf"

    def test_and_logic_no_doc_matches_all_filters(self, docs):
        """When no doc matches ALL filters, falls back to unfiltered."""
        from metadata_extractor import MetadataFilterApplier
        # Smith is in paper1 but 2022 is in paper2 — no doc has both
        metadata = {"dates": ["2022"], "authors": ["Smith"], "papers": None, "sources": None}
        result = MetadataFilterApplier.apply(docs, metadata)
        # Falls back to all docs since AND yields zero matches
        assert len(result) == len(docs)
