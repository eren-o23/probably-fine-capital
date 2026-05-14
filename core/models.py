"""Pydantic v2 data models for Probably Fine Capital."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class AnalystReport(BaseModel):
    """Signal output produced by a single analyst agent."""

    ticker: str
    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    analyst_type: Literal["momentum", "sentiment", "macro"]
    timestamp: datetime = Field(default_factory=_utcnow)


class RiskDecision(BaseModel):
    """Risk manager verdict on an analyst report.

    approved=False means the trade is vetoed; veto_reason must be set.
    modified_confidence may be lower than the original report's confidence
    when the risk manager has concerns but still approves.
    """

    approved: bool
    modified_confidence: float = Field(ge=0.0, le=1.0)
    veto_reason: Optional[str] = None
    original_report: AnalystReport


class TradeInstruction(BaseModel):
    """Instruction passed from the portfolio manager to the trader agent."""

    action: Literal["buy", "sell", "hold"]
    ticker: str
    size_usd: float
    rationale: str
    timestamp: datetime = Field(default_factory=_utcnow)


class Position(BaseModel):
    """A single open position held by the fund."""

    ticker: str
    size_usd: float
    quantity: float
    entry_price: float
    current_price: float
    stop_loss_price: float
    opened_at: datetime

    @computed_field
    @property
    def pnl_usd(self) -> float:
        """Unrealised PnL in USD: price delta × quantity."""
        return (self.current_price - self.entry_price) * self.quantity

    @computed_field
    @property
    def pnl_pct(self) -> float:
        """Unrealised PnL as a fraction of entry cost."""
        return (self.current_price - self.entry_price) / self.entry_price


class FundState(BaseModel):
    """Complete snapshot of the fund's state at a point in time."""

    cash: float
    starting_cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    trade_log: list[TradeInstruction] = Field(default_factory=list)
    analyst_reports: list[AnalystReport] = Field(default_factory=list)
    peak_value: float
    paper_mode: bool = True
    created_at: datetime = Field(default_factory=_utcnow)

    @computed_field
    @property
    def total_value(self) -> float:
        """Cash plus mark-to-market value of all open positions."""
        positions_value = sum(
            pos.current_price * pos.quantity for pos in self.positions.values()
        )
        return self.cash + positions_value

    @computed_field
    @property
    def total_pnl(self) -> float:
        """Net PnL versus starting cash."""
        return self.total_value - self.starting_cash

    @computed_field
    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from the fund's peak value (0.0 if no peak yet)."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def to_summary_dict(self) -> dict:
        """Return a clean, LLM-safe summary — no datetime objects, floats rounded.

        Suitable for embedding directly into an LLM prompt as JSON context.
        """
        return {
            "cash": round(self.cash, 2),
            "total_value": round(self.total_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "open_positions": len(self.positions),
            "positions": [
                {"ticker": ticker, "pnl_pct": round(pos.pnl_pct, 4)}
                for ticker, pos in self.positions.items()
            ],
        }
