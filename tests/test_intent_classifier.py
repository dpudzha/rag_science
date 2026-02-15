"""Tests for intent_classifier.py."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


class TestIntentClassifier:
    def _make_classifier(self, response_text):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response_text)
        from intent_classifier import IntentClassifier
        return IntentClassifier(llm=mock_llm)

    def test_greeting(self):
        classifier = self._make_classifier("GREETING")
        assert classifier.classify("Hello!") == "GREETING"

    def test_chitchat(self):
        classifier = self._make_classifier("CHITCHAT")
        assert classifier.classify("What can you do?") == "CHITCHAT"

    def test_substantive(self):
        classifier = self._make_classifier("SUBSTANTIVE")
        assert classifier.classify("What methods were used?") == "SUBSTANTIVE"

    def test_fallback_on_invalid_response(self):
        classifier = self._make_classifier("UNKNOWN_INTENT")
        assert classifier.classify("test") == "SUBSTANTIVE"

    def test_fallback_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("connection failed")
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=mock_llm)
        assert classifier.classify("test") == "SUBSTANTIVE"

    def test_multiline_response_takes_first_line(self):
        classifier = self._make_classifier("GREETING\nSome explanation here")
        assert classifier.classify("Hi") == "GREETING"

    def test_case_insensitive(self):
        classifier = self._make_classifier("greeting")
        assert classifier.classify("Hello") == "GREETING"


class TestFastClassify:
    """Test regex-based fast classification (no LLM call)."""

    def test_greeting_patterns(self):
        from intent_classifier import IntentClassifier
        assert IntentClassifier._fast_classify("Hi") == "GREETING"
        assert IntentClassifier._fast_classify("hello!") == "GREETING"
        assert IntentClassifier._fast_classify("Hey") == "GREETING"
        assert IntentClassifier._fast_classify("Good morning") == "GREETING"
        assert IntentClassifier._fast_classify("Thanks!") == "GREETING"

    def test_farewell_patterns(self):
        from intent_classifier import IntentClassifier
        assert IntentClassifier._fast_classify("bye") == "FAREWELL"
        assert IntentClassifier._fast_classify("Goodbye!") == "FAREWELL"
        assert IntentClassifier._fast_classify("See you") == "FAREWELL"
        assert IntentClassifier._fast_classify("take care") == "FAREWELL"
        assert IntentClassifier._fast_classify("later") == "FAREWELL"

    def test_chitchat_patterns(self):
        from intent_classifier import IntentClassifier
        assert IntentClassifier._fast_classify("What can you do") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What can you do?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("Who are you?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("How are you") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What can I ask") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What kind of questions can I ask?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What do you know") == "CHITCHAT"
        assert IntentClassifier._fast_classify("Are you a bot?") == "CHITCHAT"

    def test_followup_meta_questions(self):
        from intent_classifier import IntentClassifier
        assert IntentClassifier._fast_classify("Questions about what") == "CHITCHAT"
        assert IntentClassifier._fast_classify("About what?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What papers?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("Like what?") == "CHITCHAT"
        assert IntentClassifier._fast_classify("What topics?") == "CHITCHAT"

    def test_substantive_not_matched(self):
        from intent_classifier import IntentClassifier
        assert IntentClassifier._fast_classify("What methods were used?") is None
        assert IntentClassifier._fast_classify("Compare results from paper A and B") is None
        assert IntentClassifier._fast_classify("What is the accuracy of the model?") is None

    def test_fast_path_skips_llm(self):
        """Fast-classified queries should not call the LLM."""
        mock_llm = MagicMock()
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=mock_llm)
        assert classifier.classify("Hello!") == "GREETING"
        mock_llm.invoke.assert_not_called()

        assert classifier.classify("What can you do?") == "CHITCHAT"
        mock_llm.invoke.assert_not_called()


class TestChitchatResponse:
    def test_greeting_response(self):
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=MagicMock())
        response = classifier.get_chitchat_response("GREETING")
        assert response is not None
        assert "research assistant" in response.lower() or "scientific" in response.lower()

    def test_chitchat_response(self):
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=MagicMock())
        response = classifier.get_chitchat_response("CHITCHAT")
        assert response is not None

    def test_farewell_response(self):
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=MagicMock())
        response = classifier.get_chitchat_response("FAREWELL")
        assert response is not None
        assert "goodbye" in response.lower() or "bye" in response.lower()

    def test_substantive_returns_none(self):
        from intent_classifier import IntentClassifier
        classifier = IntentClassifier(llm=MagicMock())
        assert classifier.get_chitchat_response("SUBSTANTIVE") is None
