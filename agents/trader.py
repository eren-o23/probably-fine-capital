"""Trader agent for Probably Fine Capital.

Executes TradeInstructions via the Kraken CLI, respecting paper/live mode.
Updates FundStateManager after each successful fill.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from core.fund_state import FundStateManager
from core.models import Position, TradeInstruction
from utils.kraken_cli import place_order
from utils.logger import log_trade, system_logger as logger


class Trader:
    """Executes TradeInstructions against the Kraken CLI."""

    def __init__(self) -> None:
        """Initialise the trader with a named logger."""
        self.logger = logging.getLogger("trader")

    async def execute(
        self,
        instructions: list[TradeInstruction],
        state_manager: FundStateManager,
    ) -> list[str]:
        """Execute a batch of TradeInstructions, returning tickers that were filled.

        Each instruction is wrapped in its own try/except so a failure on one
        trade does not abort the rest. Hold instructions are skipped silently.

        Args:
            instructions:  Output from PortfolioManager.allocate().
            state_manager: Live fund state, mutated in-place after each fill.

        Returns:
            List of tickers (without /USD pair suffix) successfully executed.
        """
        executed: list[str] = []
        actionable = [i for i in instructions if i.action != "hold"]

        for instruction in instructions:
            try:
                ticker = await self._execute_one(instruction, state_manager)
                if ticker is not None:
                    executed.append(ticker)
            except Exception as exc:
                logger.error(
                    "Trader: unexpected error executing %s %s — %s",
                    instruction.action,
                    instruction.ticker,
                    exc,
                )

        logger.info(
            "Trader: %d/%d instructions filled",
            len(executed),
            len(actionable),
        )
        return executed

    async def _execute_one(
        self,
        instruction: TradeInstruction,
        state_manager: FundStateManager,
    ) -> str | None:
        """Execute one instruction. Returns the ticker on success, None otherwise."""
        if instruction.action == "hold":
            logger.debug("Trader: skipping hold for %s", instruction.ticker)
            return None

        pair = f"{instruction.ticker}/USD"
        pos = state_manager.state.positions.get(instruction.ticker)

        if pos is not None:
            current_price = pos.current_price
        else:
            # No open position yet (typical for a new buy). Use a safe fallback
            # price so place_order can compute a reasonable quantity (size/100).
            logger.warning(
                "Trader: no price in state for %s — using fallback price 100.0",
                instruction.ticker,
            )
            current_price = 100.0

        result = await place_order(
            ticker=pair,
            action=instruction.action,  # type: ignore[arg-type]
            size_usd=instruction.size_usd,
            current_price=current_price,
            paper_mode=config.PAPER_TRADING,
        )

        order_id = str(
            (result.get("response") or {}).get("txid", "paper")
        )
        status = "paper" if config.PAPER_TRADING else ("filled" if result["success"] else "rejected")

        log_trade(
            ticker=pair,
            side=instruction.action,
            size_usd=instruction.size_usd,
            price=current_price,
            order_id=order_id,
            status=status,
            error=result.get("error"),
        )

        if not result["success"]:
            logger.error(
                "Trader: order failed for %s — %s",
                instruction.ticker,
                result.get("error", "unknown error"),
            )
            return None

        quantity = instruction.size_usd / current_price

        if instruction.action == "buy":
            state_manager.add_position(
                Position(
                    ticker=instruction.ticker,
                    size_usd=instruction.size_usd,
                    quantity=quantity,
                    entry_price=current_price,
                    current_price=current_price,
                    stop_loss_price=current_price * (1.0 - config.STOP_LOSS_PCT),
                    opened_at=datetime.now(timezone.utc),
                )
            )
        else:
            state_manager.close_position(instruction.ticker, current_price)

        logger.info(
            "Trader: %s %s $%.2f @ $%.2f [%s]",
            instruction.action,
            instruction.ticker,
            instruction.size_usd,
            current_price,
            "paper" if config.PAPER_TRADING else "live",
        )
        return instruction.ticker
