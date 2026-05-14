"""Tests for Trader agent.

place_order and log_trade are mocked throughout — no CLI or file I/O.
FundStateManager is mocked — no disk access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from agents.trader import Trader
from core.models import Position, TradeInstruction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instruction(
    action: str = "buy",
    ticker: str = "AAPLx",
    size_usd: float = 200.0,
) -> TradeInstruction:
    return TradeInstruction(
        action=action,  # type: ignore[arg-type]
        ticker=ticker,
        size_usd=size_usd,
        rationale="test rationale",
    )


def _make_position(current_price: float = 150.0) -> Position:
    return Position(
        ticker="AAPLx",
        size_usd=200.0,
        quantity=200.0 / current_price,
        entry_price=current_price,
        current_price=current_price,
        stop_loss_price=current_price * 0.95,
        opened_at=datetime.now(timezone.utc),
    )


def _success_result(paper: bool = True) -> dict:
    return {
        "success": True,
        "action": "buy",
        "ticker": "AAPLx/USD",
        "quantity": 1.333,
        "size_usd": 200.0,
        "price": 150.0,
        "paper_mode": paper,
        "response": {"txid": "ORDER-001"},
    }


def _failure_result() -> dict:
    return {
        "success": False,
        "action": "buy",
        "ticker": "AAPLx/USD",
        "quantity": 1.333,
        "size_usd": 200.0,
        "price": 150.0,
        "paper_mode": False,
        "error": "validation error: insufficient funds",
    }


def _mock_state_manager(position: Position | None = None) -> MagicMock:
    sm = MagicMock()
    sm.state.positions = {"AAPLx": position} if position else {}
    return sm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHold:
    async def test_hold_is_skipped_and_not_counted(self):
        trader = Trader()
        sm = _mock_state_manager()

        with patch("agents.trader.place_order") as mock_place, \
             patch("agents.trader.log_trade"):
            result = await trader.execute([_make_instruction(action="hold")], sm)

        assert result == []
        mock_place.assert_not_called()

    async def test_hold_does_not_mutate_state(self):
        trader = Trader()
        sm = _mock_state_manager()

        with patch("agents.trader.place_order"), \
             patch("agents.trader.log_trade"):
            await trader.execute([_make_instruction(action="hold")], sm)

        sm.add_position.assert_not_called()
        sm.close_position.assert_not_called()


class TestPaperBuy:
    async def test_paper_buy_calls_place_order_with_paper_mode_true(self):
        trader = Trader()
        sm = _mock_state_manager()

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result(paper=True)) as mock_place, \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            await trader.execute([_make_instruction(action="buy")], sm)

        mock_place.assert_awaited_once()
        _, kwargs = mock_place.call_args
        assert kwargs.get("paper_mode", mock_place.call_args[0][4] if len(mock_place.call_args[0]) > 4 else True) is True

    async def test_paper_buy_calls_add_position(self):
        trader = Trader()
        sm = _mock_state_manager()

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result(paper=True)), \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            result = await trader.execute([_make_instruction(action="buy", ticker="AAPLx")], sm)

        assert result == ["AAPLx"]
        sm.add_position.assert_called_once()
        pos_arg = sm.add_position.call_args[0][0]
        assert pos_arg.ticker == "AAPLx"
        assert pos_arg.size_usd == 200.0

    async def test_paper_buy_logs_trade_with_paper_status(self):
        trader = Trader()
        sm = _mock_state_manager()

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result(paper=True)), \
             patch("agents.trader.log_trade") as mock_log, \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            await trader.execute([_make_instruction(action="buy")], sm)

        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert mock_log.call_args[1].get("status") == "paper" or mock_log.call_args[0][5] == "paper"

    async def test_paper_buy_uses_fallback_price_when_no_existing_position(self):
        """No open position for a new buy → fallback price 100.0, qty = size_usd/100."""
        trader = Trader()
        sm = _mock_state_manager(position=None)

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result()) as mock_place, \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            result = await trader.execute([_make_instruction(action="buy")], sm)

        assert result == ["AAPLx"]
        args, kwargs = mock_place.call_args
        price_arg = kwargs.get("current_price") or args[3]
        assert price_arg == 100.0


class TestPaperSell:
    async def test_paper_sell_calls_place_order(self):
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result()) as mock_place, \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            await trader.execute([_make_instruction(action="sell", ticker="AAPLx")], sm)

        mock_place.assert_awaited_once()

    async def test_paper_sell_calls_close_position(self):
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result()), \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            result = await trader.execute([_make_instruction(action="sell", ticker="AAPLx")], sm)

        assert result == ["AAPLx"]
        sm.close_position.assert_called_once_with("AAPLx", 150.0)
        sm.add_position.assert_not_called()


class TestLiveMode:
    async def test_live_mode_calls_place_order_with_paper_mode_false(self):
        """In live mode, place_order is called with paper_mode=False (validates internally)."""
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))
        live_result = _success_result(paper=False)

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=live_result) as mock_place, \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", False):
            result = await trader.execute([_make_instruction(action="buy", ticker="AAPLx")], sm)

        assert result == ["AAPLx"]
        args, kwargs = mock_place.call_args
        paper_arg = kwargs.get("paper_mode") if "paper_mode" in kwargs else args[4]
        assert paper_arg is False

    async def test_live_validation_failure_skips_trade(self):
        """place_order returning success=False must not append to executed or mutate state."""
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_failure_result()), \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", False):
            result = await trader.execute([_make_instruction(action="buy", ticker="AAPLx")], sm)

        assert result == []
        sm.add_position.assert_not_called()
        sm.close_position.assert_not_called()

    async def test_live_failure_logs_trade_with_rejected_status(self):
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_failure_result()), \
             patch("agents.trader.log_trade") as mock_log, \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", False):
            await trader.execute([_make_instruction(action="buy", ticker="AAPLx")], sm)

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        status = call_args[1].get("status") or call_args[0][5]
        assert status == "rejected"


class TestIsolation:
    async def test_one_failure_does_not_abort_subsequent_trades(self):
        """If the first trade fails, the second must still execute."""
        trader = Trader()

        position = _make_position(150.0)
        sm = MagicMock()
        sm.state.positions = {"AAPLx": position, "NVDAx": position}

        instruction_a = _make_instruction(action="buy", ticker="AAPLx")
        instruction_b = _make_instruction(action="buy", ticker="NVDAx")

        # First call fails, second succeeds
        side_effects = [_failure_result(), _success_result()]

        async def fake_place_order(**kwargs):
            return side_effects.pop(0)

        with patch("agents.trader.place_order", side_effect=fake_place_order), \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", False):
            result = await trader.execute([instruction_a, instruction_b], sm)

        assert result == ["NVDAx"]

    async def test_exception_in_one_trade_does_not_abort_others(self):
        """An unexpected exception in place_order must not stop subsequent trades."""
        trader = Trader()
        sm = _mock_state_manager(position=_make_position(150.0))
        sm.state.positions = {"AAPLx": _make_position(150.0), "MSFTx": _make_position(300.0)}

        instruction_a = _make_instruction(action="buy", ticker="AAPLx")
        instruction_b = _make_instruction(action="buy", ticker="MSFTx")

        call_count = 0

        async def fake_place_order(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated crash")
            return _success_result()

        with patch("agents.trader.place_order", side_effect=fake_place_order), \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", False):
            result = await trader.execute([instruction_a, instruction_b], sm)

        assert result == ["MSFTx"]

    async def test_mixed_hold_and_buy_only_buy_counted(self):
        trader = Trader()
        sm = _mock_state_manager()

        instructions = [
            _make_instruction(action="hold", ticker="AAPLx"),
            _make_instruction(action="buy", ticker="NVDAx"),
            _make_instruction(action="hold", ticker="MSFTx"),
        ]

        with patch("agents.trader.place_order", new_callable=AsyncMock, return_value=_success_result()) as mock_place, \
             patch("agents.trader.log_trade"), \
             patch.object(__import__("agents.trader", fromlist=["config"]).config, "PAPER_TRADING", True):
            result = await trader.execute(instructions, sm)

        assert result == ["NVDAx"]
        mock_place.assert_awaited_once()
