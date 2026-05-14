"""Fund state manager for Probably Fine Capital.

FundStateManager wraps a FundState Pydantic model and provides the only
safe mutation path: price updates, position open/close, stop-loss checks,
and atomic checkpointing.

Checkpoint format (logs/fund_state.json):
    {
        "fund_state":       { ...FundState fields... },
        "closed_positions": [ { ticker, entry_price, exit_price,
                                quantity, pnl_usd, opened_at, closed_at } ]
    }

All public methods catch and log their own exceptions — they never raise.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    PAPER_TRADING,
    STARTING_CASH,
    STOP_LOSS_PCT,
)
from core.models import FundState, Position

logger = logging.getLogger(__name__)

_CHECKPOINT_PATH = Path("logs/fund_state.json")
_TMP_PATH = Path("logs/fund_state.json.tmp")


class FundStateManager:
    """Single source of truth for the fund's mutable state.

    Load order: restore from checkpoint if it exists, otherwise initialise
    a fresh FundState.  All writes go through save(), which is atomic.
    """

    def __init__(
        self,
        starting_cash: float = STARTING_CASH,
        paper_mode: bool = PAPER_TRADING,
    ) -> None:
        """Initialise the manager, loading the last checkpoint if available.

        Args:
            starting_cash: starting USD balance for a fresh fund.
            paper_mode: whether the fund runs in paper-trading mode.
        """
        self._closed_positions: list[dict[str, Any]] = []
        self._state: FundState = self._load_or_init(starting_cash, paper_mode)

    # ------------------------------------------------------------------
    # Public read-only accessor
    # ------------------------------------------------------------------

    @property
    def state(self) -> FundState:
        """Return the current FundState (read-only by convention)."""
        return self._state

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_init(
        self, starting_cash: float, paper_mode: bool
    ) -> FundState:
        """Load state from checkpoint or return a fresh FundState.

        Falls back to a fresh state if the file is missing, empty, or corrupt.
        """
        if _CHECKPOINT_PATH.exists():
            try:
                raw = _CHECKPOINT_PATH.read_text(encoding="utf-8").strip()
                if raw:
                    envelope = json.loads(raw)
                    state = FundState.model_validate(envelope["fund_state"])
                    self._closed_positions = envelope.get("closed_positions", [])
                    logger.info(
                        "FundStateManager: loaded checkpoint — cash=$%.2f, "
                        "%d open / %d closed positions",
                        state.cash,
                        len(state.positions),
                        len(self._closed_positions),
                    )
                    return state
            except Exception as exc:
                logger.error(
                    "FundStateManager: checkpoint corrupt, starting fresh: %s", exc
                )

        state = FundState(
            cash=starting_cash,
            starting_cash=starting_cash,
            peak_value=starting_cash,
            paper_mode=paper_mode,
        )
        logger.info(
            "FundStateManager: fresh state — cash=$%.2f, paper_mode=%s",
            starting_cash,
            paper_mode,
        )
        return state

    def save(self) -> None:
        """Persist FundState to disk atomically.

        Writes to a .tmp file then renames to the target path so the
        checkpoint is never in a partially-written state.  Creates the
        logs/ directory if it does not yet exist.
        """
        try:
            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            envelope = {
                "fund_state": self._state.model_dump(mode="json"),
                "closed_positions": self._closed_positions,
            }
            _TMP_PATH.write_text(
                json.dumps(envelope, indent=2), encoding="utf-8"
            )
            _TMP_PATH.rename(_CHECKPOINT_PATH)
            logger.debug("FundStateManager: checkpoint saved")
        except Exception as exc:
            logger.error("FundStateManager.save: failed to write checkpoint: %s", exc)

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current_price for every open position that has a new price.

        Also advances peak_value when the portfolio hits a new high.

        Args:
            prices: mapping of ticker → latest last-trade price.
        """
        try:
            for ticker, price in prices.items():
                if ticker not in self._state.positions:
                    continue
                if price <= 0:
                    logger.warning(
                        "update_prices: ignoring non-positive price for %s: %s",
                        ticker,
                        price,
                    )
                    continue
                pos = self._state.positions[ticker]
                self._state.positions[ticker] = pos.model_copy(
                    update={"current_price": price}
                )

            new_total = self._state.total_value
            if new_total > self._state.peak_value:
                self._state.peak_value = new_total
                logger.debug(
                    "update_prices: new portfolio peak $%.2f", new_total
                )
        except Exception as exc:
            logger.error("update_prices: unexpected error: %s", exc)

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def check_stop_losses(self) -> list[str]:
        """Identify positions that should be closed due to risk breaches.

        Checks two conditions:
          1. Per-position stop loss: pnl_pct <= -STOP_LOSS_PCT
          2. Fund-level max drawdown: drawdown_pct >= MAX_DRAWDOWN_PCT
             (returns all tickers — full liquidation required)

        Returns:
            List of tickers to close.  Empty if no breaches.  Never raises.
        """
        try:
            breaches: list[str] = []

            # Fund-level drawdown check first — if triggered, return everything
            drawdown = self._state.drawdown_pct
            if drawdown >= MAX_DRAWDOWN_PCT:
                logger.warning(
                    "RISK BREACH — MAX DRAWDOWN: %.1f%% >= %.1f%% threshold. "
                    "Full liquidation required.",
                    drawdown * 100,
                    MAX_DRAWDOWN_PCT * 100,
                )
                return list(self._state.positions.keys())

            # Per-position stop-loss check
            for ticker, pos in self._state.positions.items():
                if pos.pnl_pct <= -STOP_LOSS_PCT:
                    logger.warning(
                        "RISK BREACH — STOP LOSS: %s pnl=%.1f%% <= -%.1f%% threshold",
                        ticker,
                        pos.pnl_pct * 100,
                        STOP_LOSS_PCT * 100,
                    )
                    breaches.append(ticker)

            return breaches
        except Exception as exc:
            logger.error("check_stop_losses: unexpected error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """Open a new position, deducting cost from cash.

        Silently rejects (with a log) if the fund is already at MAX_OPEN_POSITIONS
        or if there is insufficient cash.

        Args:
            position: fully-formed Position to add.
        """
        try:
            if len(self._state.positions) >= MAX_OPEN_POSITIONS:
                logger.warning(
                    "add_position: rejected %s — already at MAX_OPEN_POSITIONS (%d)",
                    position.ticker,
                    MAX_OPEN_POSITIONS,
                )
                return

            cost = position.entry_price * position.quantity
            if cost > self._state.cash:
                logger.warning(
                    "add_position: rejected %s — cost $%.2f exceeds cash $%.2f",
                    position.ticker,
                    cost,
                    self._state.cash,
                )
                return

            self._state.positions[position.ticker] = position
            self._state.cash -= cost
            logger.info(
                "add_position: opened %s qty=%.4f @ $%.2f, cost=$%.2f, "
                "remaining cash=$%.2f",
                position.ticker,
                position.quantity,
                position.entry_price,
                cost,
                self._state.cash,
            )
        except Exception as exc:
            logger.error("add_position: unexpected error: %s", exc)

    def close_position(self, ticker: str, exit_price: float) -> None:
        """Close an open position, realising P&L and returning proceeds to cash.

        Args:
            ticker: the position to close.
            exit_price: price at which the position is exited.
        """
        try:
            pos = self._state.positions.get(ticker)
            if pos is None:
                logger.warning(
                    "close_position: %s not in open positions, ignoring", ticker
                )
                return

            if exit_price <= 0:
                logger.warning(
                    "close_position: invalid exit_price %.4f for %s, ignoring",
                    exit_price,
                    ticker,
                )
                return

            proceeds = exit_price * pos.quantity
            pnl = (exit_price - pos.entry_price) * pos.quantity

            self._state.cash += proceeds
            del self._state.positions[ticker]

            closed_at = datetime.now(timezone.utc).isoformat()
            self._closed_positions.append(
                {
                    "ticker": ticker,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "quantity": pos.quantity,
                    "pnl_usd": round(pnl, 4),
                    "opened_at": pos.opened_at.isoformat(),
                    "closed_at": closed_at,
                }
            )

            # Update peak after cash lands
            if self._state.total_value > self._state.peak_value:
                self._state.peak_value = self._state.total_value

            logger.info(
                "close_position: closed %s @ $%.2f, pnl=$%.2f (%.1f%%), "
                "cash=$%.2f",
                ticker,
                exit_price,
                pnl,
                (pnl / (pos.entry_price * pos.quantity)) * 100,
                self._state.cash,
            )
        except Exception as exc:
            logger.error("close_position: unexpected error: %s", exc)

    # ------------------------------------------------------------------
    # LLM-ready summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Return a plain dict suitable for embedding in an LLM prompt.

        No datetime objects.  Floats rounded to 2 d.p.

        Returns:
            Dict with keys: cash, total_value, total_pnl, drawdown_pct,
            paper_mode, open_positions (list of per-position dicts).
        """
        try:
            positions_out = [
                {
                    "ticker": ticker,
                    "size_usd": round(pos.size_usd, 2),
                    "unrealized_pnl": round(pos.pnl_usd, 2),
                    "pnl_pct": round(pos.pnl_pct * 100, 2),
                }
                for ticker, pos in self._state.positions.items()
            ]
            return {
                "cash": round(self._state.cash, 2),
                "total_value": round(self._state.total_value, 2),
                "total_pnl": round(self._state.total_pnl, 2),
                "drawdown_pct": round(self._state.drawdown_pct * 100, 2),
                "paper_mode": self._state.paper_mode,
                "open_positions": positions_out,
            }
        except Exception as exc:
            logger.error("get_portfolio_summary: unexpected error: %s", exc)
            return {}
