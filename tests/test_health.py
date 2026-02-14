"""Tests for health.py: retry logic, timeout handling."""
import pytest
from unittest.mock import patch, MagicMock, call

import httpx


class TestCheckOllama:
    def test_success_on_first_try(self):
        from health import check_ollama
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("health.httpx.get", return_value=mock_resp) as mock_get:
            result = check_ollama(retries=1, delay=0)

        assert result is True
        mock_get.assert_called_once()

    def test_retries_on_connect_error(self):
        from health import check_ollama
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("health.httpx.get") as mock_get:
            mock_get.side_effect = [
                httpx.ConnectError("refused"),
                mock_resp,
            ]
            with patch("health.time.sleep"):
                result = check_ollama(retries=2, delay=0.01)

        assert result is True
        assert mock_get.call_count == 2

    def test_raises_after_all_retries(self):
        from health import check_ollama

        with patch("health.httpx.get", side_effect=httpx.ConnectError("down")):
            with patch("health.time.sleep"):
                with pytest.raises(ConnectionError, match="Could not reach Ollama"):
                    check_ollama(retries=3, delay=0.01)

    def test_handles_timeout(self):
        from health import check_ollama
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("health.httpx.get") as mock_get:
            mock_get.side_effect = [
                httpx.TimeoutException("timeout"),
                mock_resp,
            ]
            with patch("health.time.sleep"):
                result = check_ollama(retries=2, delay=0.01)

        assert result is True

    def test_exponential_backoff(self):
        from health import check_ollama

        with patch("health.httpx.get", side_effect=httpx.ConnectError("down")):
            with patch("health.time.sleep") as mock_sleep:
                with pytest.raises(ConnectionError):
                    check_ollama(retries=3, delay=1)

        # delay=1, then 2, but not after the last attempt
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)
