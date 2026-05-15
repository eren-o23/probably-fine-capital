"""Tests for ResearchDesk.

All three analyst classes are mocked — no LLM calls, no network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call
import logging

import pytest

import config
from agents.research_desk import ResearchDesk
from core.market_data import MarketSnapshot, MomentumSignal
from core.models import AnalystReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(include_signals: bool = True) -> MarketSnapshot:
    """Build a MarketSnapshot covering all tradeable tickers."""
    tickers = config.TRADEABLE_TICKERS
    prices = {t: 100.0 + i for i, t in enumerate(tickers)}
    histories = {t: [100.0 + i * 0.1 for i in range(48)] for t in tickers}
    signals = {}
    if include_signals:
        signals = {
            t: MomentumSignal(
                ticker=t,
                short_momentum=0.01,
                medium_momentum=0.02,
                trend_direction="up",
                signal_strength=0.4,
            )
            for t in tickers
        }
    headlines = {t: ["headline A", "headline B"] for t in tickers}
    return MarketSnapshot(
        prices=prices,
        price_histories=histories,
        momentum_signals=signals,
        headlines=headlines,
    )


def _make_report(analyst_type: str, ticker: str = "AAPLx/USD") -> AnalystReport:
    return AnalystReport(
        ticker=ticker,
        signal="buy",
        confidence=0.75,
        reasoning="test signal",
        analyst_type=analyst_type,  # type: ignore[arg-type]
    )


def _patch_analysts(mom_return, sent_return, macro_return):
    """Context manager that patches all three analyst classes on ResearchDesk."""
    mom_mock = AsyncMock(return_value=mom_return)
    sent_mock = AsyncMock(return_value=sent_return)
    macro_mock = AsyncMock(return_value=macro_return)

    mom_inst = MagicMock()
    sent_inst = MagicMock()
    macro_inst = MagicMock()

    mom_inst.analyze = mom_mock
    sent_inst.analyze = sent_mock
    macro_inst.analyze = macro_mock

    return (
        patch("agents.research_desk.MomentumAnalyst", return_value=mom_inst),
        patch("agents.research_desk.SentimentAnalyst", return_value=sent_inst),
        patch("agents.research_desk.MacroAnalyst", return_value=macro_inst),
        mom_inst,
        sent_inst,
        macro_inst,
    )


# ---------------------------------------------------------------------------
# All analysts return reports → combined list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_analysts_return_combined_list():
    p_mom, p_sent, p_macro, mom_inst, sent_inst, macro_inst = _patch_analysts(
        _make_report("momentum"),
        _make_report("sentiment"),
        _make_report("macro"),
    )
    with p_mom, p_sent, p_macro:
        desk = ResearchDesk()
        reports = await desk.analyze(_make_snapshot())

    n_tickers = len(config.TRADEABLE_TICKERS)
    # Each ticker produces 3 reports → 30 total
    assert len(reports) == n_tickers * 3
    types = {r.analyst_type for r in reports}
    assert types == {"momentum", "sentiment", "macro"}


# ---------------------------------------------------------------------------
# One analyst returns None → filtered out
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_none_results_are_filtered_out():
    p_mom, p_sent, p_macro, _, _, _ = _patch_analysts(
        _make_report("momentum"),
        None,                        # sentiment returns None for all tickers
        _make_report("macro"),
    )
    with p_mom, p_sent, p_macro:
        desk = ResearchDesk()
        reports = await desk.analyze(_make_snapshot())

    n_tickers = len(config.TRADEABLE_TICKERS)
    # 2 analysts succeed × 18 tickers = 36
    assert len(reports) == n_tickers * 2
    assert all(r.analyst_type in ("momentum", "macro") for r in reports)


# ---------------------------------------------------------------------------
# All analysts fail → empty list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_analysts_fail_returns_empty_list():
    p_mom, p_sent, p_macro, _, _, _ = _patch_analysts(None, None, None)
    with p_mom, p_sent, p_macro:
        desk = ResearchDesk()
        reports = await desk.analyze(_make_snapshot())

    assert reports == []


# ---------------------------------------------------------------------------
# 54 tasks created — 3 analysts × 18 tickers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thirty_tasks_created():
    p_mom, p_sent, p_macro, mom_inst, sent_inst, macro_inst = _patch_analysts(
        _make_report("momentum"),
        _make_report("sentiment"),
        _make_report("macro"),
    )
    with p_mom, p_sent, p_macro:
        desk = ResearchDesk()
        await desk.analyze(_make_snapshot())

    n_tickers = len(config.TRADEABLE_TICKERS)
    assert mom_inst.analyze.call_count == n_tickers
    assert sent_inst.analyze.call_count == n_tickers
    assert macro_inst.analyze.call_count == n_tickers


# ---------------------------------------------------------------------------
# Missing momentum signals → skipped, not crashed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_momentum_signal_skipped_gracefully():
    p_mom, p_sent, p_macro, mom_inst, sent_inst, macro_inst = _patch_analysts(
        _make_report("momentum"),
        _make_report("sentiment"),
        _make_report("macro"),
    )
    with p_mom, p_sent, p_macro:
        desk = ResearchDesk()
        # Snapshot with no momentum signals at all
        reports = await desk.analyze(_make_snapshot(include_signals=False))

    # Momentum analyst never called — 2 analysts × 18 tickers = 36
    mom_inst.analyze.assert_not_called()
    assert len(reports) == len(config.TRADEABLE_TICKERS) * 2


# ---------------------------------------------------------------------------
# system_logger called with summary line
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logger_called_with_summary():
    p_mom, p_sent, p_macro, _, _, _ = _patch_analysts(
        _make_report("momentum"),
        _make_report("sentiment"),
        _make_report("macro"),
    )
    with p_mom, p_sent, p_macro:
        with patch("agents.research_desk.logger") as mock_logger:
            desk = ResearchDesk()
            reports = await desk.analyze(_make_snapshot())

    # Find the summary info call
    info_calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("Research desk complete" in c for c in info_calls)
