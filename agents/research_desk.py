"""ResearchDesk orchestration layer for Probably Fine Capital.

Runs all three analyst agents in parallel for every tradeable ticker and
returns a flat, filtered list of AnalystReports ready for the risk manager.
No LLM calls happen here — this file is pure asyncio coordination.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import config
from agents.macro_analyst import MacroAnalyst
from agents.momentum_analyst import MomentumAnalyst
from agents.sentiment_analyst import SentimentAnalyst
from core.market_data import MarketSnapshot, MomentumSignal
from core.models import AnalystReport
from utils.logger import system_logger as logger


class ResearchDesk:
    """Orchestrates the three analyst agents across the full tradeable universe."""

    def __init__(self) -> None:
        """Instantiate one of each analyst agent."""
        self._momentum = MomentumAnalyst()
        self._sentiment = SentimentAnalyst()
        self._macro = MacroAnalyst()

    async def _run_momentum(
        self,
        ticker: str,
        signal: Optional[MomentumSignal],
        history: list[float],
    ) -> Optional[AnalystReport]:
        """Run the momentum analyst, or return None when no signal is available."""
        if signal is None:
            logger.warning("ResearchDesk: no momentum signal for %s — skipping", ticker)
            return None
        return await self._momentum.analyze(signal, history)

    async def analyze(self, market_data: MarketSnapshot) -> list[AnalystReport]:
        """Run all three analysts for every tradeable ticker in parallel.

        All 30 coroutines (3 analysts × 10 tickers) are submitted to a single
        asyncio.gather call so no ticker waits for another to finish.

        Nones (below-confidence discards or missing signals) are filtered out.
        Unexpected exceptions from any coroutine are logged and dropped rather
        than propagating — the fund keeps running with partial data.

        Args:
            market_data: Snapshot from get_all_market_data().

        Returns:
            Flat list of validated AnalystReports, length 0–30.
        """
        spy_history = market_data.price_histories.get("SPYx/USD", [])
        qqq_history = market_data.price_histories.get("QQQx/USD", [])

        tasks: list[asyncio.coroutines] = []
        for ticker in config.TRADEABLE_TICKERS:
            signal = market_data.momentum_signals.get(ticker)
            history = market_data.price_histories.get(ticker, [])
            headlines = market_data.headlines.get(ticker, [])

            tasks.append(self._run_momentum(ticker, signal, history))
            tasks.append(self._sentiment.analyze(ticker, headlines))
            tasks.append(
                self._macro.analyze(ticker, market_data.prices, spy_history, qqq_history)
            )

        raw = await asyncio.gather(*tasks, return_exceptions=True)

        reports: list[AnalystReport] = []
        for result in raw:
            if isinstance(result, AnalystReport):
                reports.append(result)
            elif isinstance(result, BaseException):
                logger.error("ResearchDesk: unexpected exception from analyst: %s", result)
            # None (below-confidence or skipped) is silently dropped

        logger.info(
            "Research desk complete: %d signals from %d tickers",
            len(reports),
            len(config.TRADEABLE_TICKERS),
        )
        return reports
