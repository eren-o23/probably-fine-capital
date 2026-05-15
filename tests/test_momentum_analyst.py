"""Tests for MomentumAnalyst and the improved prompt.

Runs without hitting any external APIs — all LLM calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.momentum_analyst import (
    MomentumAnalyst,
    _build_price_table,
    _build_prompt,
    _classify_trend,
)
from core.market_data import MomentumSignal
from core.models import AnalystReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal(
    trend: str = "up",
    short: float = 0.02,
    medium: float = 0.03,
    strength: float = 0.6,
) -> MomentumSignal:
    return MomentumSignal(
        ticker="AAPLx/USD",
        short_momentum=short,
        medium_momentum=medium,
        trend_direction=trend,  # type: ignore[arg-type]
        signal_strength=strength,
    )


def _make_history(n: int = 20, base: float = 150.0) -> list[float]:
    return [base + i * 0.5 for i in range(n)]


def _make_llm_payload(
    signal: str = "buy",
    confidence: float = 0.75,
    reasoning: str = "Strong uptrend with accelerating momentum.",
    rationale: str = "Buying AAPLx on sustained momentum breakout.",
) -> str:
    return json.dumps({
        "reasoning": reasoning,
        "signal": signal,
        "confidence": confidence,
        "rationale": rationale,
    })


# ---------------------------------------------------------------------------
# _classify_trend
# ---------------------------------------------------------------------------

def test_classify_trend_strong_uptrend():
    assert _classify_trend([1.0, 2.0, 3.0, 4.0]) == "strong_uptrend"


def test_classify_trend_strong_downtrend():
    assert _classify_trend([4.0, 3.0, 2.0, 1.0]) == "strong_downtrend"


def test_classify_trend_consolidating_mixed():
    assert _classify_trend([1.0, 3.0, 2.0, 4.0]) == "consolidating"


def test_classify_trend_too_short_returns_consolidating():
    assert _classify_trend([]) == "consolidating"
    assert _classify_trend([1.0]) == "consolidating"
    assert _classify_trend([1.0, 2.0]) == "consolidating"


def test_classify_trend_exact_three_prices():
    assert _classify_trend([1.0, 2.0, 3.0]) == "strong_uptrend"
    assert _classify_trend([3.0, 2.0, 1.0]) == "strong_downtrend"


# ---------------------------------------------------------------------------
# _build_price_table
# ---------------------------------------------------------------------------

def test_price_table_no_history_returns_unavailable():
    table = _build_price_table([])
    assert "unavailable" in table


def test_price_table_first_row_has_no_change():
    table = _build_price_table([100.0, 110.0])
    assert "—" in table


def test_price_table_computes_change_pct():
    # 100 → 110 = +10%
    table = _build_price_table([100.0, 110.0])
    assert "+10.00%" in table


def test_price_table_contains_header():
    table = _build_price_table([100.0])
    assert "| # | Price | Change % |" in table


# ---------------------------------------------------------------------------
# _build_prompt — content checks
# ---------------------------------------------------------------------------

def test_build_prompt_contains_ticker():
    signal = _make_signal()
    prompt = _build_prompt(signal, _make_history())
    assert "AAPLx/USD" in prompt


def test_build_prompt_contains_momentum_values():
    signal = _make_signal(short=0.015, medium=0.025)
    prompt = _build_prompt(signal, _make_history())
    assert "+1.5000%" in prompt
    assert "+2.5000%" in prompt


def test_build_prompt_no_history():
    signal = _make_signal()
    prompt = _build_prompt(signal, [])
    assert "unavailable" in prompt


def test_build_prompt_clips_to_12_prices():
    history = list(range(1, 30))  # 29 prices — last 12 are 18..29
    signal = _make_signal()
    prompt = _build_prompt(signal, history)
    assert "17.00" not in prompt
    assert "18.00" in prompt
    assert "29.00" in prompt


def test_build_prompt_contains_trend_classification_uptrend():
    history = [100.0, 101.0, 102.0, 103.0]
    prompt = _build_prompt(_make_signal(), history)
    assert "strong_uptrend" in prompt


def test_build_prompt_contains_trend_classification_downtrend():
    history = [103.0, 102.0, 101.0, 100.0]
    prompt = _build_prompt(_make_signal(), history)
    assert "strong_downtrend" in prompt


def test_build_prompt_contains_trend_classification_consolidating():
    history = [100.0, 102.0, 101.0, 103.0]
    prompt = _build_prompt(_make_signal(), history)
    assert "consolidating" in prompt


def test_build_prompt_contains_period_context():
    history = _make_history(20, base=100.0)
    prompt = _build_prompt(_make_signal(), history)
    assert "Period high" in prompt
    assert "Period low" in prompt
    assert "below high" in prompt
    assert "above low" in prompt


def test_build_prompt_contains_price_table_header():
    prompt = _build_prompt(_make_signal(), _make_history())
    assert "| # | Price | Change % |" in prompt


def test_build_prompt_contains_chain_of_thought_instruction():
    prompt = _build_prompt(_make_signal(), _make_history())
    assert "chain-of-thought" in prompt.lower() or "Chain-of-thought" in prompt


def test_build_prompt_contains_confidence_calibration():
    prompt = _build_prompt(_make_signal(), _make_history())
    assert "0.90" in prompt
    assert "consolidating" in prompt


def test_build_prompt_contains_rationale_in_json_schema():
    prompt = _build_prompt(_make_signal(), _make_history())
    assert '"rationale"' in prompt


# ---------------------------------------------------------------------------
# MomentumAnalyst — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_returns_analyst_report():
    mock_choice = MagicMock()
    mock_choice.message.content = _make_llm_payload(signal="buy", confidence=0.75)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        report = await analyst.analyze(_make_signal(), _make_history())

    assert isinstance(report, AnalystReport)
    assert report.ticker == "AAPLx/USD"
    assert report.signal == "buy"
    assert report.confidence == 0.75
    assert report.analyst_type == "momentum"


def test_analyze_report_reasoning_uses_rationale_not_cot():
    """AnalystReport.reasoning must contain the trade-log rationale, not the CoT."""
    # Verified structurally: analyze() sets reasoning=result.rationale
    # This is a documentation test — the real check is in test_analyze_rationale_in_report.
    from agents.momentum_analyst import _MomentumLLMResponse
    resp = _MomentumLLMResponse(
        reasoning="This is chain-of-thought.",
        signal="buy",
        confidence=0.75,
        rationale="Buying AAPLx on uptrend.",
    )
    assert resp.rationale == "Buying AAPLx on uptrend."
    assert resp.reasoning == "This is chain-of-thought."


@pytest.mark.asyncio
async def test_analyze_rationale_in_report():
    """AnalystReport.reasoning is populated from the 'rationale' LLM field."""
    mock_choice = MagicMock()
    mock_choice.message.content = _make_llm_payload(
        signal="sell",
        confidence=0.70,
        reasoning="Trend is reversing, sell signal clear.",
        rationale="Selling NVDAx on downtrend confirmation.",
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        signal = MomentumSignal(
            ticker="NVDAx/USD",
            short_momentum=-0.02,
            medium_momentum=-0.03,
            trend_direction="down",
            signal_strength=0.6,
        )
        report = await analyst.analyze(signal, _make_history())

    assert report.reasoning == "Selling NVDAx on downtrend confirmation."
    assert report.signal == "sell"


# ---------------------------------------------------------------------------
# MomentumAnalyst — LLM failure → safe hold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_returns_safe_hold_on_llm_failure():
    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection refused")
        )
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        report = await analyst.analyze(_make_signal(), _make_history())

    assert report.signal == "hold"
    assert report.confidence == 0.0
    assert report.analyst_type == "momentum"
    assert "unavailable" in report.reasoning.lower()


# ---------------------------------------------------------------------------
# MomentumAnalyst — bad JSON → retry → safe hold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_retries_on_bad_json_then_holds():
    mock_choice = MagicMock()
    mock_choice.message.content = "{ not valid json at all }"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        report = await analyst.analyze(_make_signal(), _make_history())

    assert report.signal == "hold"
    assert report.confidence == 0.0
    assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# MomentumAnalyst — missing rationale field → validation fails → safe hold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_holds_when_rationale_field_missing():
    """Response without 'rationale' fails Pydantic validation → safe hold."""
    payload = json.dumps({"signal": "buy", "confidence": 0.75, "reasoning": "strong uptrend"})
    mock_choice = MagicMock()
    mock_choice.message.content = payload
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        report = await analyst.analyze(_make_signal(), _make_history())

    assert report.signal == "hold"
    assert report.confidence == 0.0
