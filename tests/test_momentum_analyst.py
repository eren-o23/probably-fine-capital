"""Tests for MomentumAnalyst and utils/llm.py.

Runs without hitting any external APIs — all LLM calls are mocked.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.momentum_analyst import MomentumAnalyst, _build_prompt
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


# ---------------------------------------------------------------------------
# _build_prompt
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
    # 17 was excluded; 18 and 29 were included
    assert "17.00" not in prompt
    assert "18.00" in prompt
    assert "29.00" in prompt


# ---------------------------------------------------------------------------
# MomentumAnalyst — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_returns_analyst_report():
    llm_payload = json.dumps({"signal": "buy", "confidence": 0.75, "reasoning": "strong uptrend"})

    mock_choice = MagicMock()
    mock_choice.message.content = llm_payload
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
    mock_choice.message.content = "not valid json at all"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("utils.llm._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        analyst = MomentumAnalyst()
        report = await analyst.analyze(_make_signal(), _make_history())

    # Both attempts return bad JSON → should fall back to safe hold
    assert report.signal == "hold"
    assert report.confidence == 0.0
    # The API was called exactly twice (one attempt + one retry)
    assert mock_client.chat.completions.create.call_count == 2
