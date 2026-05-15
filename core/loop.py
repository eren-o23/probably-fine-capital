"""Main trading loop for Probably Fine Capital.

Orchestrates the full agent pipeline on a 15-minute interval:
  get_all_market_data → ResearchDesk → RiskManager → PortfolioManager → Trader
Persists FundState after every cycle regardless of whether trades were made.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date

from agents.portfolio_manager import PortfolioManager
from agents.reporter import Reporter
from agents.research_desk import ResearchDesk
from agents.risk_manager import RiskManager
from agents.trader import Trader
from core.fund_state import FundStateManager
from core.market_data import get_all_market_data
from utils.logger import system_logger as logger

LOOP_INTERVAL_SECONDS: int = 900  # 15 minutes


class TradingLoop:
    """Runs the multi-agent trading pipeline on a fixed interval."""

    def __init__(self) -> None:
        """Instantiate all agents and load fund state from the last checkpoint."""
        self.state_manager = FundStateManager()
        self.research_desk = ResearchDesk()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.trader = Trader()
        self.reporter = Reporter()
        self.logger = logging.getLogger("loop")
        self.cycle_count = 0
        self._running = False
        self.last_report_date: date | None = None

    async def run(self) -> None:
        """Start the trading loop and run until stop() is called.

        Each iteration calls _run_cycle(), sleeps for the remainder of the
        15-minute interval, then repeats. Exceptions from _run_cycle are logged
        but do not abort the loop. KeyboardInterrupt triggers a clean shutdown.
        """
        self._running = True
        logger.info(
            "TradingLoop: started — cycle interval=%ds paper_mode=%s",
            LOOP_INTERVAL_SECONDS,
            self.state_manager.state.paper_mode,
        )

        try:
            while self._running:
                cycle_wall = time.monotonic()

                try:
                    await self._run_cycle()
                except Exception:
                    logger.exception("TradingLoop: unhandled error in _run_cycle — continuing")

                elapsed = time.monotonic() - cycle_wall
                sleep_for = max(0.0, LOOP_INTERVAL_SECONDS - elapsed)
                logger.debug("TradingLoop: sleeping %.1fs until next cycle", sleep_for)

                if self._running:
                    await asyncio.sleep(sleep_for)

        except KeyboardInterrupt:
            self.stop()
            logger.info("TradingLoop: interrupted by user")

    async def _run_cycle(self) -> None:
        """Execute one full trading cycle.

        Steps:
          1. Record start time
          2. Fetch market data
          3. Run research desk (all analysts in parallel)
          4. Run risk manager — skip portfolio/execute if no decisions
          5. Run portfolio manager
          6. Execute trades
          7. Persist fund state
          8. Log elapsed time
          9. Increment cycle counter
        """
        cycle_start = time.monotonic()
        logger.info("TradingLoop: cycle %d starting", self.cycle_count + 1)
        instructions = []

        # Step 2 — market data
        snapshot = await get_all_market_data()

        # Step 3 — research
        reports = await self.research_desk.analyze(snapshot)

        # Step 4 — risk gate
        decisions = self.risk_manager.evaluate(reports, self.state_manager.state)

        if not decisions:
            logger.info("TradingLoop: no approved trades this cycle")
        else:
            # Step 5 — portfolio sizing
            instructions = await self.portfolio_manager.allocate(
                decisions, self.state_manager.state
            )

            # Step 6 — trade execution
            executed = await self.trader.execute(instructions, self.state_manager)
            logger.info(
                "TradingLoop: %d/%d instructions executed",
                len(executed),
                len(instructions),
            )

        # Step 7 — persist state (always, even when no trades)
        self.state_manager.save()

        # Step 8 — log elapsed
        elapsed = time.monotonic() - cycle_start
        logger.info(
            "TradingLoop: cycle %d complete in %.1fs",
            self.cycle_count + 1,
            elapsed,
        )

        # Daily report — once per calendar day (UTC)
        today = date.today()
        if self.last_report_date != today:
            filepath, tweets = await self.reporter.run(
                self.state_manager.state,
                instructions,
                self.cycle_count,
                elapsed,
            )
            if tweets:
                await self.reporter.post_to_x(tweets)
            self.last_report_date = today

        # Step 9 — increment counter
        self.cycle_count += 1

    def stop(self) -> None:
        """Request a graceful shutdown after the current cycle completes."""
        self._running = False
        logger.info("TradingLoop: shutdown requested")
