"""Tests for archetype_detector.py: archetype detection, weight mapping, reformulation."""
from unittest.mock import MagicMock
from pathlib import Path
import json
import pytest

from langchain_core.messages import AIMessage


class TestArchetypeDetector:
    def _make_detector(self, response_text):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response_text)
        from archetype_detector import ArchetypeDetector
        return ArchetypeDetector(llm=mock_llm)

    def test_what_information(self):
        detector = self._make_detector("WHAT_INFORMATION")
        assert detector.detect("What was the accuracy?") == "WHAT_INFORMATION"

    def test_how_methodology(self):
        detector = self._make_detector("HOW_METHODOLOGY")
        assert detector.detect("How was the experiment conducted?") == "HOW_METHODOLOGY"

    def test_comparison(self):
        detector = self._make_detector("COMPARISON")
        assert detector.detect("Compare method A and B") == "COMPARISON"

    def test_definition(self):
        detector = self._make_detector("DEFINITION")
        assert detector.detect("What is attention?") == "DEFINITION"

    def test_why_reasoning(self):
        detector = self._make_detector("WHY_REASONING")
        assert detector.detect("Why did they choose this?") == "WHY_REASONING"

    def test_summary(self):
        detector = self._make_detector("SUMMARY")
        assert detector.detect("Summarize the findings") == "SUMMARY"

    def test_fallback_on_invalid(self):
        detector = self._make_detector("INVALID_TYPE")
        assert detector.detect("test") == "WHAT_INFORMATION"

    def test_fallback_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        from archetype_detector import ArchetypeDetector
        detector = ArchetypeDetector(llm=mock_llm)
        assert detector.detect("test") == "WHAT_INFORMATION"


class TestArchetypeWeights:
    def test_known_archetype_returns_weights(self):
        from archetype_detector import ArchetypeDetector
        detector = ArchetypeDetector(llm=MagicMock())
        bm25_w, dense_w = detector.get_weights("DEFINITION")
        assert bm25_w == 0.5
        assert dense_w == 0.5

    def test_unknown_archetype_returns_default(self):
        from archetype_detector import ArchetypeDetector
        detector = ArchetypeDetector(llm=MagicMock())
        bm25_w, dense_w = detector.get_weights("NONEXISTENT")
        assert bm25_w == 0.3
        assert dense_w == 0.7

    def test_all_archetypes_have_weights(self):
        from archetype_detector import ARCHETYPES, ARCHETYPE_WEIGHTS
        for archetype in ARCHETYPES:
            assert archetype in ARCHETYPE_WEIGHTS


class TestQueryReformulator:
    def _make_reformulator(self, response_text, terminology_path=None):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response_text)
        from archetype_detector import QueryReformulator
        return QueryReformulator(llm=mock_llm, terminology_path=terminology_path)

    def test_reformulates_query(self):
        reformulator = self._make_reformulator("What methodology and experimental procedure was used?")
        result = reformulator.reformulate("What methods were used?", "HOW_METHODOLOGY")
        assert "methodology" in result.lower()

    def test_fallback_on_empty_response(self):
        reformulator = self._make_reformulator("")
        result = reformulator.reformulate("original query", "WHAT_INFORMATION")
        assert result == "original query"

    def test_fallback_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        from archetype_detector import QueryReformulator
        reformulator = QueryReformulator(llm=mock_llm)
        assert reformulator.reformulate("test", "WHAT_INFORMATION") == "test"

    def test_loads_domain_terminology(self, tmp_path):
        terminology = {
            "abbreviations": {"ML": "machine learning"},
            "synonyms": {"model": ["architecture"]}
        }
        term_path = tmp_path / "terms.json"
        term_path.write_text(json.dumps(terminology))

        reformulator = self._make_reformulator(
            "What machine learning model was used?",
            terminology_path=term_path,
        )
        result = reformulator.reformulate("What ML model was used?", "WHAT_INFORMATION")
        assert result  # Just verify it runs without error

    def test_missing_terminology_file(self, tmp_path):
        reformulator = self._make_reformulator(
            "reformulated query",
            terminology_path=tmp_path / "nonexistent.json",
        )
        result = reformulator.reformulate("test query", "WHAT_INFORMATION")
        assert result == "reformulated query"

    def test_rejects_methodology_drift_for_entity_lookup(self):
        reformulator = self._make_reformulator(
            "What experimental procedure and methodology will the experiment using ITk follow?"
        )
        original = "Which experiment will be using ITk?"
        result = reformulator.reformulate(original, "HOW_METHODOLOGY")
        assert result == original

    def test_rejects_rewrite_that_drops_key_entity(self):
        reformulator = self._make_reformulator("What dataset was used in the experiment?")
        original = "What accuracy did BERT get?"
        result = reformulator.reformulate(original, "WHAT_INFORMATION")
        assert result == original
