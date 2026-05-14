"""Tests for MacroAnalyst.

Runs without hitting any external APIs — call_llm is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.macro_analyst import (
    MacroAnalyst,
    _build_prompt,
    _pct_change,
    _trend,
    _format_anchor,
)
from core.models import AnalystReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TICKER = "NVDAx/USD"

_ALL_PRICES: dict[str, float] = {
    "AAPLx/USD": 182.50,
    "NVDAx/USD": 430.00,
    "MSFTx/USD": 310.00,
    "TSLAx/USD": 250.00,
    "AMZNx/USD": 175.00,
    "GOOGLx/USD": 140.00,
    "METAx/USD": 320.00,
    "AMDx/USD": 160.00,
    "SPYx/USD": 454.00,
    "QQQx/USD": 374.00,
}

# 48-element histories: steady climb from base to base * multiplier
def _make_history(base: float, end: float, n: int = 48) -> list[float]:
    step = (end - base) / max(n - 1, 1)
    return [round(base + i * step, 2) for i in range(n)]

_SPY_HISTORY = _make_history(400.0, 408.0)   # +2.00% over 48h
_QQQ_HISTORY = _make_history(360.0, 367.2)   # +2.00% over 48h


def _make_llm_result(signal: str = "buy", confidence: float = 0.75, reasoning: str = "markets trending up"):
    from agents.macro_analyst import _MacroLLMResponse
    return _MacroLLMResponse(signal=signal, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_pct_change_basic():
    assert abs(_pct_change([100.0, 102.0]) - 0.02) < 1e-9


def test_pct_change_insufficient_data():
    assert _pct_change([]) is None
    assert _pct_change([100.0]) is None


def test_pct_change_zero_base():
    assert _pct_change([0.0, 10.0]) is None


def test_trend_up():
    assert _trend(_make_history(100.0, 101.0)) == "up"


def test_trend_down():
    assert _trend(_make_history(101.0, 100.0)) == "down"


def test_trend_flat():
    assert _trend([100.0, 100.0]) == "flat"


def test_trend_unknown_on_empty():
    assert _trend([]) == "unknown"


def test_format_anchor_empty_history():
    block = _format_anchor("SPY", [])
    assert "insufficient macro data" in block


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_prompt_contains_ticker():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert _TICKER in prompt


def test_prompt_contains_spy_qqq_change_percentages():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    # SPY: 400→408 = +2.00%, QQQ: 360→367.2 = +2.00%
    assert "+2.00%" in prompt


def test_prompt_contains_all_ticker_prices():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    for t in _ALL_PRICES:
        assert t in prompt


def test_prompt_marks_evaluated_ticker():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert "← this ticker" in prompt


def test_prompt_notes_insufficient_data_for_empty_spy():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, [], _QQQ_HISTORY)
    assert "insufficient macro data" in prompt


# ---------------------------------------------------------------------------
# MacroAnalyst.analyze — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_returns_analyst_report():
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=_make_llm_result())):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert isinstance(report, AnalystReport)
    assert report.ticker == _TICKER
    assert report.signal == "buy"
    assert report.confidence == 0.75
    assert report.analyst_type == "macro"


# ---------------------------------------------------------------------------
# LLM failure → safe hold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_returns_safe_hold():
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=None)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert isinstance(report, AnalystReport)
    assert report.signal == "hold"
    assert report.confidence == 0.0
    assert "unavailable" in report.reasoning.lower()


# ---------------------------------------------------------------------------
# Below MIN_CONFIDENCE → None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_min_confidence_returns_none():
    weak = _make_llm_result(signal="sell", confidence=0.30)
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=weak)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert report is None


# ---------------------------------------------------------------------------
# Empty SPY history — no crash, prompt notes insufficient data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_spy_history_no_crash():
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=_make_llm_result())):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, [], _QQQ_HISTORY)

    assert isinstance(report, AnalystReport)


@pytest.mark.asyncio
async def test_both_histories_empty_no_crash():
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=_make_llm_result())):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, [], [])

    assert isinstance(report, AnalystReport)
