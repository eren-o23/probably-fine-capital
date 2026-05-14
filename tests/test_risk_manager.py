"""Tests for RiskManager.

No mocks of external services — RiskManager is pure Python.
FundState and AnalystReport are constructed directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import config
from agents.risk_manager import RiskManager
from core.models import AnalystReport, FundState, Position, RiskDecision


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_report(
    signal: str = "buy",
    ticker: str = "AAPLx/USD",
    confidence: float = 0.75,
) -> AnalystReport:
    return AnalystReport(
        ticker=ticker,
        signal=signal,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="test",
        analyst_type="momentum",
    )


def _make_position(ticker: str = "AAPLx/USD", price: float = 200.0) -> Position:
    return Position(
        ticker=ticker,
        size_usd=1000.0,
        quantity=5.0,
        entry_price=price,
        current_price=price,
        stop_loss_price=price * 0.95,
        opened_at=datetime.now(timezone.utc),
    )


def _make_state(
    cash: float = 10_000.0,
    positions: dict | None = None,
    peak_value: float | None = None,
) -> FundState:
    """Build a FundState. peak_value defaults to cash so drawdown == 0."""
    pos = positions or {}
    positions_value = sum(p.current_price * p.quantity for p in pos.values())
    pv = peak_value if peak_value is not None else cash + positions_value
    return FundState(
        cash=cash,
        starting_cash=10_000.0,
        positions=pos,
        peak_value=pv,
    )


# ---------------------------------------------------------------------------
# Gate 1 — confidence below MIN_CONFIDENCE
# ---------------------------------------------------------------------------

def test_low_confidence_vetoed():
    rm = RiskManager()
    report = _make_report(confidence=config.MIN_CONFIDENCE - 0.01)
    decisions = rm.evaluate([report], _make_state())

    assert len(decisions) == 1
    d = decisions[0]
    assert d.approved is False
    assert "confidence" in d.veto_reason
    assert str(round(config.MIN_CONFIDENCE, 2)) in d.veto_reason


def test_exactly_min_confidence_passes_gate_1():
    rm = RiskManager()
    report = _make_report(confidence=config.MIN_CONFIDENCE)
    decisions = rm.evaluate([report], _make_state())

    assert decisions[0].approved is True


# ---------------------------------------------------------------------------
# Gate 2 — fund drawdown at MAX_DRAWDOWN_PCT
# ---------------------------------------------------------------------------

def test_drawdown_at_threshold_vetoes_all():
    # cash=9000, peak=10000 → drawdown = 10% = MAX_DRAWDOWN_PCT → breached
    state = _make_state(cash=9_000.0, peak_value=10_000.0)
    assert state.drawdown_pct == pytest.approx(0.10)

    rm = RiskManager()
    reports = [_make_report("buy", "AAPLx/USD"), _make_report("sell", "NVDAx/USD")]
    decisions = rm.evaluate(reports, state)

    assert all(not d.approved for d in decisions)
    assert all("drawdown" in d.veto_reason for d in decisions)


def test_drawdown_below_threshold_does_not_veto():
    # drawdown = 0
    state = _make_state(cash=10_000.0)
    rm = RiskManager()
    decisions = rm.evaluate([_make_report("buy")], state)
    assert decisions[0].approved is True


# ---------------------------------------------------------------------------
# Gate 3 — max open positions (buy only)
# ---------------------------------------------------------------------------

def test_max_positions_reached_vetoes_buy():
    tickers = config.TRADEABLE_TICKERS[: config.MAX_OPEN_POSITIONS]
    positions = {t: _make_position(t) for t in tickers}
    state = _make_state(positions=positions)

    rm = RiskManager()
    # Try to open one more buy on a ticker not yet held
    new_ticker = config.TRADEABLE_TICKERS[config.MAX_OPEN_POSITIONS]
    decisions = rm.evaluate([_make_report("buy", new_ticker)], state)

    assert decisions[0].approved is False
    assert "max open positions" in decisions[0].veto_reason


def test_max_positions_does_not_veto_sell():
    tickers = config.TRADEABLE_TICKERS[: config.MAX_OPEN_POSITIONS]
    positions = {t: _make_position(t) for t in tickers}
    state = _make_state(positions=positions)

    rm = RiskManager()
    # Sell a ticker that is held — gate 3 must not fire for sells
    held_ticker = tickers[0]
    decisions = rm.evaluate([_make_report("sell", held_ticker)], state)

    assert decisions[0].approved is True


# ---------------------------------------------------------------------------
# Gate 4 — duplicate ticker buy
# ---------------------------------------------------------------------------

def test_duplicate_buy_vetoed():
    state = _make_state(positions={"AAPLx/USD": _make_position("AAPLx/USD")})
    rm = RiskManager()
    decisions = rm.evaluate([_make_report("buy", "AAPLx/USD")], state)

    assert decisions[0].approved is False
    assert "already open" in decisions[0].veto_reason


# ---------------------------------------------------------------------------
# Gate 5 — sell with no position
# ---------------------------------------------------------------------------

def test_sell_no_position_vetoed():
    state = _make_state()  # no positions
    rm = RiskManager()
    decisions = rm.evaluate([_make_report("sell", "AAPLx/USD")], state)

    assert decisions[0].approved is False
    assert "no open position" in decisions[0].veto_reason


def test_sell_with_position_passes():
    state = _make_state(positions={"AAPLx/USD": _make_position("AAPLx/USD")})
    rm = RiskManager()
    decisions = rm.evaluate([_make_report("sell", "AAPLx/USD")], state)

    assert decisions[0].approved is True


# ---------------------------------------------------------------------------
# Gate 6 — trade size too small
# ---------------------------------------------------------------------------

def test_trade_size_too_small_vetoed():
    # cash=1.0 → estimated_size = 1.0 * MIN_CONFIDENCE = 0.60 < MIN_TRADE_SIZE_USD
    state = _make_state(cash=1.0)
    rm = RiskManager()
    decisions = rm.evaluate([_make_report("buy")], state)

    assert decisions[0].approved is False
    assert "below minimum" in decisions[0].veto_reason


# ---------------------------------------------------------------------------
# All gates pass → approved
# ---------------------------------------------------------------------------

def test_all_gates_pass_approved():
    state = _make_state(cash=10_000.0)  # drawdown=0, no positions, plenty of cash
    rm = RiskManager()
    report = _make_report("buy", "AAPLx/USD", confidence=0.75)
    decisions = rm.evaluate([report], state)

    d = decisions[0]
    assert d.approved is True
    assert d.modified_confidence == report.confidence
    assert d.veto_reason is None


# ---------------------------------------------------------------------------
# Hold signal → vetoed silently (no logger.warning call)
# ---------------------------------------------------------------------------

def test_hold_vetoed_silently():
    state = _make_state()
    rm = RiskManager()

    with patch("agents.risk_manager.logger") as mock_logger:
        decisions = rm.evaluate([_make_report("hold")], state)

    assert decisions[0].approved is False
    assert decisions[0].veto_reason == "hold signal — no action"
    # warning must not have been called for the hold veto
    for call_args in mock_logger.warning.call_args_list:
        assert "hold" not in str(call_args).lower()


# ---------------------------------------------------------------------------
# Mixed reports → correct approve/veto per report
# ---------------------------------------------------------------------------

def test_mixed_reports():
    state = _make_state(
        cash=10_000.0,
        positions={"NVDAx/USD": _make_position("NVDAx/USD")},
    )
    rm = RiskManager()
    reports = [
        _make_report("buy", "AAPLx/USD", confidence=0.80),   # should approve
        _make_report("hold", "MSFTx/USD"),                    # hold → veto
        _make_report("buy", "NVDAx/USD", confidence=0.75),   # duplicate → veto
        _make_report("sell", "NVDAx/USD", confidence=0.70),  # has position → approve
    ]
    decisions = rm.evaluate(reports, state)

    assert len(decisions) == 4
    assert decisions[0].approved is True   # AAPL buy
    assert decisions[1].approved is False  # MSFT hold
    assert decisions[2].approved is False  # NVDA duplicate buy
    assert decisions[3].approved is True   # NVDA sell


# ---------------------------------------------------------------------------
# Exception inside _evaluate_one → veto, no crash
# ---------------------------------------------------------------------------

def test_exception_in_gate_vetoes_without_crash():
    state = _make_state()
    rm = RiskManager()
    report = _make_report("buy")

    with patch.object(rm, "_evaluate_one", side_effect=RuntimeError("gate exploded")):
        decisions = rm.evaluate([report], state)

    assert len(decisions) == 1
    assert decisions[0].approved is False
    assert "internal error" in decisions[0].veto_reason


# ---------------------------------------------------------------------------
# Summary log
# ---------------------------------------------------------------------------

def test_summary_logged():
    state = _make_state()
    rm = RiskManager()
    reports = [
        _make_report("buy", "AAPLx/USD", confidence=0.75),
        _make_report("buy", "NVDAx/USD", confidence=0.20),  # below threshold → veto
    ]

    with patch("agents.risk_manager.logger") as mock_logger:
        rm.evaluate(reports, state)

    info_calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("Risk review complete" in c for c in info_calls)
