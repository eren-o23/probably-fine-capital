"""Tests for PortfolioManager.

call_llm is mocked throughout — no API keys needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

import config
from agents.portfolio_manager import PortfolioManager, _apply_size_bounds
from core.models import AnalystReport, FundState, RiskDecision, TradeInstruction


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_report(
    ticker: str = "AAPLx/USD",
    signal: str = "buy",
    confidence: float = 0.75,
) -> AnalystReport:
    return AnalystReport(
        ticker=ticker,
        signal=signal,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="test signal",
        analyst_type="momentum",
    )


def _make_decision(
    approved: bool = True,
    ticker: str = "AAPLx/USD",
    signal: str = "buy",
    confidence: float = 0.75,
) -> RiskDecision:
    return RiskDecision(
        approved=approved,
        modified_confidence=confidence if approved else 0.0,
        veto_reason=None if approved else "test veto",
        original_report=_make_report(ticker, signal, confidence),
    )


def _make_state(cash: float = 10_000.0) -> FundState:
    """FundState with no positions so total_value == cash."""
    return FundState(
        cash=cash,
        starting_cash=10_000.0,
        peak_value=cash,
    )


def _make_llm_result(size_usd: float = 250.0, reasoning: str = "good macro setup"):
    from agents.portfolio_manager import _AllocationLLMResponse
    return _AllocationLLMResponse(
        ticker="AAPLx/USD",
        size_usd=size_usd,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# _apply_size_bounds (pure Python — no mocking needed)
# ---------------------------------------------------------------------------

def test_size_within_bounds_unchanged():
    state = _make_state(cash=10_000.0)  # total_value=10000, max_by_pct=2000
    result = _apply_size_bounds(250.0, state)
    assert result == 250.0


def test_size_above_max_trade_clamped():
    state = _make_state(cash=10_000.0)
    result = _apply_size_bounds(config.MAX_TRADE_SIZE_USD + 100, state)
    assert result == pytest.approx(min(config.MAX_TRADE_SIZE_USD, config.MAX_POSITION_PCT * 10_000.0))


def test_size_below_min_trade_raised():
    state = _make_state(cash=10_000.0)
    result = _apply_size_bounds(1.0, state)
    assert result == config.MIN_TRADE_SIZE_USD


def test_size_capped_at_max_position_pct():
    # total_value=1000 → max_by_pct = 0.20 * 1000 = 200
    # Input 400 → clamp to min(500, 400)=400, then min(400, 200)=200
    state = _make_state(cash=1_000.0)
    result = _apply_size_bounds(400.0, state)
    assert result == pytest.approx(config.MAX_POSITION_PCT * 1_000.0)


# ---------------------------------------------------------------------------
# Empty decisions → empty list, no LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_decisions_returns_empty():
    with patch("agents.portfolio_manager.call_llm") as mock_llm:
        pm = PortfolioManager()
        result = await pm.allocate([], _make_state())

    assert result == []
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Vetoed decision → skipped, LLM not called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vetoed_decision_skipped():
    vetoed = _make_decision(approved=False)
    with patch("agents.portfolio_manager.call_llm") as mock_llm:
        pm = PortfolioManager()
        result = await pm.allocate([vetoed], _make_state())

    assert result == []
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path → valid TradeInstruction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_returns_trade_instruction():
    with patch(
        "agents.portfolio_manager.call_llm",
        new=AsyncMock(return_value=_make_llm_result(size_usd=250.0)),
    ):
        pm = PortfolioManager()
        result = await pm.allocate([_make_decision()], _make_state())

    assert len(result) == 1
    instr = result[0]
    assert isinstance(instr, TradeInstruction)
    assert instr.ticker == "AAPLx/USD"
    assert instr.action == "buy"
    assert instr.size_usd == 250.0
    assert instr.rationale == "good macro setup"


# ---------------------------------------------------------------------------
# LLM returns size above MAX_TRADE_SIZE_USD → clamped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_size_above_max_clamped():
    oversized = _make_llm_result(size_usd=config.MAX_TRADE_SIZE_USD + 999)
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(return_value=oversized)):
        pm = PortfolioManager()
        result = await pm.allocate([_make_decision()], _make_state(cash=10_000.0))

    assert result[0].size_usd <= config.MAX_TRADE_SIZE_USD


# ---------------------------------------------------------------------------
# LLM returns size below MIN_TRADE_SIZE_USD → clamped up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_size_below_min_clamped():
    undersized = _make_llm_result(size_usd=0.50)
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(return_value=undersized)):
        pm = PortfolioManager()
        result = await pm.allocate([_make_decision()], _make_state())

    assert result[0].size_usd >= config.MIN_TRADE_SIZE_USD


# ---------------------------------------------------------------------------
# LLM fails → ticker skipped, no crash, others still returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_skips_ticker():
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(return_value=None)):
        pm = PortfolioManager()
        result = await pm.allocate([_make_decision()], _make_state())

    assert result == []


@pytest.mark.asyncio
async def test_llm_failure_on_one_does_not_drop_others():
    good_result = _make_llm_result(size_usd=100.0)
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return None if call_count == 1 else good_result

    decisions = [
        _make_decision(ticker="AAPLx/USD"),
        _make_decision(ticker="NVDAx/USD"),
    ]
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(side_effect=side_effect)):
        pm = PortfolioManager()
        result = await pm.allocate(decisions, _make_state())

    assert len(result) == 1
    assert result[0].ticker == "NVDAx/USD"


# ---------------------------------------------------------------------------
# Size capped at MAX_POSITION_PCT of total value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_position_pct_cap_applied():
    # total_value = 500 → max_by_pct = 0.20 * 500 = 100
    # LLM returns 300 → after trade-size clamp: min(500, 300)=300 → pct cap: 100
    state = _make_state(cash=500.0)
    oversized = _make_llm_result(size_usd=300.0)
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(return_value=oversized)):
        pm = PortfolioManager()
        result = await pm.allocate([_make_decision()], state)

    expected_cap = config.MAX_POSITION_PCT * 500.0
    assert result[0].size_usd == pytest.approx(expected_cap)


# ---------------------------------------------------------------------------
# Summary log called with correct count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_log_called():
    with patch("agents.portfolio_manager.call_llm", new=AsyncMock(return_value=_make_llm_result())):
        with patch("agents.portfolio_manager.logger") as mock_logger:
            pm = PortfolioManager()
            await pm.allocate([_make_decision(), _make_decision(ticker="NVDAx/USD")], _make_state())

    info_calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("Portfolio allocation complete" in c for c in info_calls)
    assert any("2" in c for c in info_calls if "Portfolio allocation complete" in c)
