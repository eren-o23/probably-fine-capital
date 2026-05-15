"""Tests for SentimentAnalyst.

Runs without hitting any external APIs — call_llm is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.sentiment_analyst import (
    SentimentAnalyst,
    _SentimentLLMResponse,
    _build_prompt,
    _label_headlines,
)
from core.models import AnalystReport


_TICKER = "AAPLx/USD"
_HEADLINES = [
    "Apple beats earnings expectations by wide margin",
    "iPhone sales surge in emerging markets",
    "Apple announces $110B share buyback programme",
]


def _make_llm_result(
    signal: str = "buy",
    confidence: float = 0.75,
    reasoning: str = "Strong direct news about AAPL with clear positive catalysts.",
    direct_headlines: int = 2,
    rationale: str = "Buying AAPLx on earnings beat and buyback announcement.",
) -> _SentimentLLMResponse:
    """Return a mock _SentimentLLMResponse with all required fields."""
    return _SentimentLLMResponse(
        reasoning=reasoning,
        direct_headlines=direct_headlines,
        signal=signal,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# _label_headlines
# ---------------------------------------------------------------------------

def test_label_headlines_first_third_is_recent():
    headlines = ["h1", "h2", "h3"]
    labeled = _label_headlines(headlines)
    lines = labeled.splitlines()
    assert "[RECENT]" in lines[0]
    assert "[OLDER]" in lines[1]
    assert "[OLDER]" in lines[2]


def test_label_headlines_six_headlines_two_recent():
    headlines = [f"h{i}" for i in range(6)]
    labeled = _label_headlines(headlines)
    lines = labeled.splitlines()
    assert "[RECENT]" in lines[0]
    assert "[RECENT]" in lines[1]
    assert "[OLDER]" in lines[2]


def test_label_headlines_single_headline_is_recent():
    labeled = _label_headlines(["only one"])
    assert "[RECENT]" in labeled


def test_label_headlines_numbers_are_sequential():
    labeled = _label_headlines(["a", "b", "c"])
    assert "1." in labeled
    assert "2." in labeled
    assert "3." in labeled


def test_label_headlines_contains_headline_text():
    labeled = _label_headlines(["Apple earnings beat"])
    assert "Apple earnings beat" in labeled


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


def test_build_prompt_contains_recency_labels():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert "[RECENT]" in prompt
    assert "[OLDER]" in prompt


def test_build_prompt_contains_chain_of_thought_instruction():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert "chain-of-thought" in prompt.lower() or "Chain-of-thought" in prompt


def test_build_prompt_contains_confidence_calibration():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert "0.75" in prompt
    assert "0.90" in prompt


def test_build_prompt_contains_direct_headlines_field():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert '"direct_headlines"' in prompt


def test_build_prompt_contains_rationale_field():
    prompt = _build_prompt(_TICKER, _HEADLINES)
    assert '"rationale"' in prompt


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


@pytest.mark.asyncio
async def test_rationale_maps_to_report_reasoning():
    """AnalystReport.reasoning must contain the trade-log rationale, not the CoT."""
    result = _make_llm_result(rationale="Buying AAPLx on earnings beat.")
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report.reasoning == "Buying AAPLx on earnings beat."


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
# direct_headlines == 0 → None (primary gate)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_direct_headlines_returns_none():
    """Primary gate: no news directly about this ticker → discard signal."""
    result = _make_llm_result(signal="buy", confidence=0.80, direct_headlines=0)
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report is None


@pytest.mark.asyncio
async def test_zero_direct_headlines_skips_confidence_gate():
    """Even a high-confidence result is discarded when direct_headlines == 0."""
    result = _make_llm_result(confidence=0.90, direct_headlines=0)
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report is None


@pytest.mark.asyncio
async def test_one_direct_headline_passes_primary_gate():
    """direct_headlines >= 1 clears the primary gate."""
    result = _make_llm_result(confidence=0.70, direct_headlines=1)
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert isinstance(report, AnalystReport)


# ---------------------------------------------------------------------------
# below MIN_CONFIDENCE → returns None (secondary gate)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_min_confidence_returns_none():
    low_confidence_result = _make_llm_result(
        signal="sell", confidence=0.30, direct_headlines=2
    )
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=low_confidence_result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report is None


@pytest.mark.asyncio
async def test_exactly_at_min_confidence_is_returned():
    """Confidence equal to MIN_CONFIDENCE should pass (threshold is exclusive)."""
    import config
    at_threshold = _make_llm_result(
        signal="buy", confidence=config.MIN_CONFIDENCE, direct_headlines=2
    )
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=at_threshold)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert isinstance(report, AnalystReport)
    assert report.confidence == config.MIN_CONFIDENCE


# ---------------------------------------------------------------------------
# gate ordering — direct_headlines checked before confidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_direct_headlines_gate_fires_before_confidence_gate():
    """direct_headlines == 0 should return None even if confidence >= MIN_CONFIDENCE."""
    import config
    result = _make_llm_result(
        confidence=config.MIN_CONFIDENCE + 0.1,
        direct_headlines=0,
    )
    with patch("agents.sentiment_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = SentimentAnalyst()
        report = await analyst.analyze(_TICKER, _HEADLINES)

    assert report is None
