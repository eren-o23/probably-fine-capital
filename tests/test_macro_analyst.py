"""Tests for MacroAnalyst.

Runs without hitting any external APIs — call_llm is mocked throughout.
Includes explicit unit tests for the Python-side regime classification,
volatility labelling, and breadth computation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.macro_analyst import (
    MacroAnalyst,
    _MacroLLMResponse,
    _build_prompt,
    _classify_regime,
    _compute_breadth,
    _format_anchor,
    _pct_change,
    _spy_volatility_label,
    _trend,
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


def _make_history(base: float, end: float, n: int = 48) -> list[float]:
    step = (end - base) / max(n - 1, 1)
    return [round(base + i * step, 2) for i in range(n)]


_SPY_HISTORY = _make_history(400.0, 408.0)   # +2.00% over 48h
_QQQ_HISTORY = _make_history(360.0, 367.2)   # +2.00% over 48h


def _make_llm_result(
    signal: str = "buy",
    confidence: float = 0.75,
    reasoning: str = "Risk-on regime supports long exposure.",
    market_regime: str = "risk_on",
    rationale: str = "Buying NVDAx on risk-on macro confirmation.",
) -> _MacroLLMResponse:
    return _MacroLLMResponse(
        reasoning=reasoning,
        market_regime=market_regime,
        signal=signal,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# _pct_change (existing helper — unchanged)
# ---------------------------------------------------------------------------

def test_pct_change_basic():
    assert abs(_pct_change([100.0, 102.0]) - 0.02) < 1e-9


def test_pct_change_insufficient_data():
    assert _pct_change([]) is None
    assert _pct_change([100.0]) is None


def test_pct_change_zero_base():
    assert _pct_change([0.0, 10.0]) is None


# ---------------------------------------------------------------------------
# _trend (existing helper — unchanged)
# ---------------------------------------------------------------------------

def test_trend_up():
    assert _trend(_make_history(100.0, 101.0)) == "up"


def test_trend_down():
    assert _trend(_make_history(101.0, 100.0)) == "down"


def test_trend_flat():
    assert _trend([100.0, 100.0]) == "flat"


def test_trend_unknown_on_empty():
    assert _trend([]) == "unknown"


# ---------------------------------------------------------------------------
# _format_anchor (existing helper — unchanged)
# ---------------------------------------------------------------------------

def test_format_anchor_empty_history():
    block = _format_anchor("SPY", [])
    assert "insufficient macro data" in block


# ---------------------------------------------------------------------------
# _classify_regime (new)
# ---------------------------------------------------------------------------

def test_classify_regime_risk_on_both_strongly_up():
    assert _classify_regime(2.0, 1.5) == "risk_on"


def test_classify_regime_risk_on_boundary():
    assert _classify_regime(0.31, 0.31) == "risk_on"


def test_classify_regime_risk_off_both_strongly_down():
    assert _classify_regime(-1.0, -0.5) == "risk_off"


def test_classify_regime_risk_off_boundary():
    assert _classify_regime(-0.31, -0.31) == "risk_off"


def test_classify_regime_mixed_one_up_one_down():
    assert _classify_regime(2.0, -0.5) == "mixed"


def test_classify_regime_mixed_both_below_threshold():
    assert _classify_regime(0.1, 0.2) == "mixed"


def test_classify_regime_mixed_exactly_at_threshold():
    # Threshold is exclusive: 0.3 is NOT > 0.3
    assert _classify_regime(0.3, 0.3) == "mixed"


def test_classify_regime_none_spy():
    assert _classify_regime(None, 2.0) == "mixed"


def test_classify_regime_none_qqq():
    assert _classify_regime(2.0, None) == "mixed"


def test_classify_regime_both_none():
    assert _classify_regime(None, None) == "mixed"


# ---------------------------------------------------------------------------
# _spy_volatility_label (new)
# ---------------------------------------------------------------------------

def test_spy_volatility_label_high():
    # Large swings: pct-changes ~3%, -5.8%, +7.2%, -7.7% → std >> 1.5%
    volatile = [100.0, 103.0, 97.0, 104.0, 96.0]
    assert _spy_volatility_label(volatile) == "high"


def test_spy_volatility_label_normal():
    # Tiny moves: pct-changes ~0.1% each → std << 1.5%
    stable = [100.0, 100.1, 100.2, 100.1, 100.2]
    assert _spy_volatility_label(stable) == "normal"


def test_spy_volatility_label_unknown_empty():
    assert _spy_volatility_label([]) == "unknown"


def test_spy_volatility_label_unknown_one_price():
    assert _spy_volatility_label([100.0]) == "unknown"


def test_spy_volatility_label_unknown_two_prices():
    # Two prices → one pct-change → can't compute std
    assert _spy_volatility_label([100.0, 101.0]) == "unknown"


def test_spy_volatility_label_minimum_valid_input():
    # Three prices → two pct-changes → std computable
    result = _spy_volatility_label([100.0, 100.1, 100.2])
    assert result in ("high", "normal")


# ---------------------------------------------------------------------------
# _compute_breadth (new)
# ---------------------------------------------------------------------------

def test_compute_breadth_empty():
    assert _compute_breadth({}) == (0, 0)


def test_compute_breadth_known_split():
    # sorted values: [100, 200, 300, 400, 500], median = 300
    # above: 400, 500 → 2 advancing; below: 100, 200 → 2 declining
    prices = {"A": 100.0, "B": 200.0, "C": 300.0, "D": 400.0, "E": 500.0}
    advancing, declining = _compute_breadth(prices)
    assert advancing == 2
    assert declining == 2


def test_compute_breadth_single_ticker():
    # One ticker at median — neither advancing nor declining
    advancing, declining = _compute_breadth({"A": 100.0})
    assert advancing == 0
    assert declining == 0


def test_compute_breadth_all_different():
    prices = {"A": 10.0, "B": 20.0, "C": 30.0}
    advancing, declining = _compute_breadth(prices)
    assert advancing + declining < len(prices)  # one at median, not counted


# ---------------------------------------------------------------------------
# _build_prompt — content checks
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


def test_prompt_contains_regime_classification():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    # SPY+QQQ both +2% → risk_on
    assert "risk_on" in prompt


def test_prompt_contains_breadth_counts():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert "advancing" in prompt
    assert "declining" in prompt


def test_prompt_contains_volatility_label():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert "SPY volatility" in prompt


def test_prompt_contains_chain_of_thought_instruction():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert "chain-of-thought" in prompt.lower() or "Chain-of-thought" in prompt


def test_prompt_contains_confidence_calibration():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert "0.85" in prompt


def test_prompt_contains_market_regime_json_field():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert '"market_regime"' in prompt


def test_prompt_contains_rationale_json_field():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)
    assert '"rationale"' in prompt


def test_prompt_shows_nva_for_empty_spy_history():
    prompt = _build_prompt(_TICKER, _ALL_PRICES, [], _QQQ_HISTORY)
    assert "n/a" in prompt


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


@pytest.mark.asyncio
async def test_rationale_maps_to_report_reasoning():
    result = _make_llm_result(rationale="Buying NVDAx on risk-on confirmation.")
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=result)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert report.reasoning == "Buying NVDAx on risk-on confirmation."


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
# Invalid market_regime → None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_market_regime_returns_none():
    bad = _make_llm_result(market_regime="sideways")
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=bad)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert report is None


@pytest.mark.asyncio
async def test_empty_string_regime_returns_none():
    bad = _make_llm_result(market_regime="")
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=bad)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert report is None


# ---------------------------------------------------------------------------
# Below MIN_CONFIDENCE → None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_min_confidence_returns_none():
    weak = _make_llm_result(signal="sell", confidence=0.30, market_regime="risk_off")
    with patch("agents.macro_analyst.call_llm", new=AsyncMock(return_value=weak)):
        analyst = MacroAnalyst()
        report = await analyst.analyze(_TICKER, _ALL_PRICES, _SPY_HISTORY, _QQQ_HISTORY)

    assert report is None


# ---------------------------------------------------------------------------
# Empty histories — no crash
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
