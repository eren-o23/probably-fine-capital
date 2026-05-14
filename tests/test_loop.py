"""Tests for TradingLoop.

All agents, get_all_market_data, and FundStateManager.save are mocked.
No CLI calls, no disk I/O, no LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.loop import TradingLoop
from core.models import AnalystReport, FundState, Position, RiskDecision, TradeInstruction


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_fund_state() -> FundState:
    return FundState(cash=10_000.0, starting_cash=10_000.0, peak_value=10_000.0)


def _make_report(ticker: str = "AAPLx/USD") -> AnalystReport:
    return AnalystReport(
        ticker=ticker,
        signal="buy",
        confidence=0.80,
        reasoning="strong momentum",
        analyst_type="momentum",
    )


def _make_decision(approved: bool = True, ticker: str = "AAPLx/USD") -> RiskDecision:
    return RiskDecision(
        approved=approved,
        modified_confidence=0.80 if approved else 0.0,
        veto_reason=None if approved else "test veto",
        original_report=_make_report(ticker),
    )


def _make_instruction(ticker: str = "AAPLx") -> TradeInstruction:
    return TradeInstruction(
        action="buy",
        ticker=ticker,
        size_usd=200.0,
        rationale="test",
    )


def _make_snapshot():
    """Return a minimal MarketSnapshot-like object (MagicMock)."""
    snap = MagicMock()
    snap.prices = {"AAPLx/USD": 150.0}
    snap.price_histories = {}
    snap.momentum_signals = {}
    snap.headlines = {}
    return snap


# ---------------------------------------------------------------------------
# Fixture — constructs TradingLoop with FundStateManager.save patched
# so no disk writes happen during any test.
# ---------------------------------------------------------------------------

@pytest.fixture
def loop():
    with patch("core.loop.FundStateManager") as MockSM:
        mock_sm = MagicMock()
        mock_sm.state = _make_fund_state()
        MockSM.return_value = mock_sm
        tl = TradingLoop()
    return tl


# ---------------------------------------------------------------------------
# TestCycle — tests for _run_cycle() called directly
# ---------------------------------------------------------------------------

class TestCycle:
    async def test_full_cycle_runs_end_to_end(self, loop: TradingLoop):
        """All pipeline steps are called in the correct order."""
        snapshot = _make_snapshot()
        reports = [_make_report()]
        decisions = [_make_decision()]
        instructions = [_make_instruction()]
        executed = ["AAPLx"]

        with patch("core.loop.get_all_market_data", new_callable=AsyncMock, return_value=snapshot) as mock_gmd, \
             patch.object(loop.research_desk, "analyze", new_callable=AsyncMock, return_value=reports) as mock_analyze, \
             patch.object(loop.risk_manager, "evaluate", return_value=decisions) as mock_eval, \
             patch.object(loop.portfolio_manager, "allocate", new_callable=AsyncMock, return_value=instructions) as mock_alloc, \
             patch.object(loop.trader, "execute", new_callable=AsyncMock, return_value=executed) as mock_exec, \
             patch.object(loop.state_manager, "save") as mock_save:

            await loop._run_cycle()

            mock_analyze.assert_awaited_once_with(snapshot)
            mock_eval.assert_called_once_with(reports, loop.state_manager.state)
            mock_alloc.assert_awaited_once_with(decisions, loop.state_manager.state)
            mock_exec.assert_awaited_once_with(instructions, loop.state_manager)
            mock_save.assert_called_once()

    async def test_cycle_count_increments_after_each_cycle(self, loop: TradingLoop):
        snapshot = _make_snapshot()

        with patch("core.loop.get_all_market_data", new_callable=AsyncMock, return_value=snapshot), \
             patch.object(loop.research_desk, "analyze", new_callable=AsyncMock, return_value=[]), \
             patch.object(loop.risk_manager, "evaluate", return_value=[]), \
             patch.object(loop.state_manager, "save"):

            assert loop.cycle_count == 0
            await loop._run_cycle()
            assert loop.cycle_count == 1
            await loop._run_cycle()
            assert loop.cycle_count == 2

    async def test_empty_decisions_skips_portfolio_and_trade(self, loop: TradingLoop):
        """When risk manager returns [], portfolio and trader must not be called."""
        snapshot = _make_snapshot()

        with patch("core.loop.get_all_market_data", new_callable=AsyncMock, return_value=snapshot), \
             patch.object(loop.research_desk, "analyze", new_callable=AsyncMock, return_value=[_make_report()]), \
             patch.object(loop.risk_manager, "evaluate", return_value=[]), \
             patch.object(loop.portfolio_manager, "allocate", new_callable=AsyncMock) as mock_alloc, \
             patch.object(loop.trader, "execute", new_callable=AsyncMock) as mock_exec, \
             patch.object(loop.state_manager, "save") as mock_save:

            await loop._run_cycle()

        mock_alloc.assert_not_awaited()
        mock_exec.assert_not_awaited()
        mock_save.assert_called_once()  # state is always saved

    async def test_state_saved_even_when_no_trades(self, loop: TradingLoop):
        """save() is called after every cycle regardless of trade count."""
        snapshot = _make_snapshot()
        decisions = [_make_decision(approved=False)]  # all vetoed

        with patch("core.loop.get_all_market_data", new_callable=AsyncMock, return_value=snapshot), \
             patch.object(loop.research_desk, "analyze", new_callable=AsyncMock, return_value=[_make_report()]), \
             patch.object(loop.risk_manager, "evaluate", return_value=decisions), \
             patch.object(loop.portfolio_manager, "allocate", new_callable=AsyncMock, return_value=[]), \
             patch.object(loop.trader, "execute", new_callable=AsyncMock, return_value=[]), \
             patch.object(loop.state_manager, "save") as mock_save:

            await loop._run_cycle()

        mock_save.assert_called_once()

    async def test_portfolio_receives_fund_state_not_manager(self, loop: TradingLoop):
        """PortfolioManager.allocate must receive FundState, not FundStateManager."""
        snapshot = _make_snapshot()
        decisions = [_make_decision()]

        with patch("core.loop.get_all_market_data", new_callable=AsyncMock, return_value=snapshot), \
             patch.object(loop.research_desk, "analyze", new_callable=AsyncMock, return_value=[_make_report()]), \
             patch.object(loop.risk_manager, "evaluate", return_value=decisions), \
             patch.object(loop.portfolio_manager, "allocate", new_callable=AsyncMock, return_value=[]) as mock_alloc, \
             patch.object(loop.trader, "execute", new_callable=AsyncMock, return_value=[]), \
             patch.object(loop.state_manager, "save"):

            await loop._run_cycle()

        _, call_kwargs = mock_alloc.call_args
        # Second positional arg must be FundState (from state_manager.state)
        state_arg = mock_alloc.call_args[0][1] if mock_alloc.call_args[0] else call_kwargs.get("fund_state")
        assert state_arg is loop.state_manager.state


# ---------------------------------------------------------------------------
# TestRun — tests for the run() outer loop
# ---------------------------------------------------------------------------

class TestRun:
    async def test_exception_in_run_cycle_caught_loop_continues(self, loop: TradingLoop):
        """An exception in _run_cycle must not abort the loop; it should continue."""
        call_count = 0

        async def mock_run_cycle():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("simulated crash")
            loop.stop()

        with patch.object(loop, "_run_cycle", side_effect=mock_run_cycle), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await loop.run()

        assert call_count == 2  # ran twice — crash on first did not stop the loop

    async def test_stop_is_respected_between_cycles(self, loop: TradingLoop):
        """stop() called from inside _run_cycle terminates the loop cleanly."""
        async def mock_run_cycle():
            loop.stop()

        with patch.object(loop, "_run_cycle", side_effect=mock_run_cycle), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await loop.run()

        # Sleep should not be awaited after stop() is set
        mock_sleep.assert_not_awaited()

    async def test_run_sets_running_true_at_start(self, loop: TradingLoop):
        async def mock_run_cycle():
            assert loop._running is True
            loop.stop()

        with patch.object(loop, "_run_cycle", side_effect=mock_run_cycle), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await loop.run()


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_sets_running_false(self, loop: TradingLoop):
        loop._running = True
        loop.stop()
        assert loop._running is False

    def test_stop_before_run_sets_flag(self, loop: TradingLoop):
        assert loop._running is False
        loop.stop()
        assert loop._running is False

    def test_initial_running_is_false(self, loop: TradingLoop):
        assert loop._running is False

    def test_initial_cycle_count_is_zero(self, loop: TradingLoop):
        assert loop.cycle_count == 0
