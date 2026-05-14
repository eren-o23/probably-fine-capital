"""Tests for SentimentAnalyst.

Runs without hitting any external APIs — call_llm is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.sentiment_analyst import SentimentAnalyst, _build_prompt
from core.models import AnalystReport


_TICKER = "AAPLx/USD"
_HEADLINES = [
    "Apple beats earnings expectations by wide margin",
    "iPhone sales surge in emerging markets",
    "Apple announces $110B share buyback programme",
]


def _make_llm_result(signal: str = "buy", confidence: float = 0.75, reasoning: str = "positive news"):
    """Return a mock _SentimentLLMResponse-like object."""
    from agents.sentiment_analyst import _SentimentLLMResponse
    return _SentimentLLMResponse(signal=signal, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_ticker():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert _TICKER in prompt


def test_build_prompt_contains_all_headlines():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    for headline in _HEADLINES:
        assert headline in prompt


def test_build_prompt_numbers_headlines():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert "1." in prompt
    assert "2." in prompt
    assert "3." in prompt


# ---------------------------------------------------------------------------
# empty headlines → hold at 0.0 (no LLM call)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_headlines_returns_hold():
    with patch("agents.sentiment_analyst.call_llm") as mock_llm:
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, [])

    assert isinstance(report, AnalystReport)
    assert report.signal == "hold"
    assert report.confidence == 0.0
    assert report.analyst_type == "sentiment"
    assert "no headlines" in report.reasoning.lower()
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# happy path — 3 headlines, confident buy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_returns_analyst_report():
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=_make_llm_result())):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert isinstance(report, AnalystReport)
    assert report.ticker == _TICKER
    assert report.signal == "buy"
    assert report.confidence == 0.75
    assert report.analyst_type == "sentiment"


# ---------------------------------------------------------------------------
# LLM failure → safe hold at 0.0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_returns_safe_hold():
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=None)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert isinstance(report, AnalystReport)
    assert report.signal == "hold"
    assert report.confidence == 0.0
    assert "unavailable" in report.reasoning.lower()


# ---------------------------------------------------------------------------
# below MIN_CONFIDENCE → returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_min_confidence_returns_none():
    low_confidence_result = _make_llm_result(signal="sell", confidence=0.30)
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=low_confidence_result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report is None


@pytest.mark.asyncio
async def test_exactly_at_min_confidence_is_returned():
    """Confidence equal to MIN_CONFIDENCE should pass (threshold is exclusive)."""
    import config
    at_threshold = _make_llm_result(signal="buy", confidence=config.MIN_CONFIDENCE)
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=at_threshold)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert isinstance(report, AnalystReport)
    assert report.confidence == config.MIN_CONFIDENCE
