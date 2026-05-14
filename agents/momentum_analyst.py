"""MomentumAnalyst agent for Probably Fine Capital.

Receives a pre-computed MomentumSignal from market_data and uses an LLM
to reason about it, returning an AnalystReport with a signal and confidence.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from core.market_data import MomentumSignal
from core.models import AnalystReport
from utils.llm import call_llm

logger = logging.getLogger(__name__)

_PRICE_SNIPPET_LEN: int = 12  # number of recent closes shown in the prompt


class _MomentumLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _build_prompt(signal: MomentumSignal, price_history: list[float]) -> str:
    """Build the LLM prompt from a MomentumSignal and recent price history."""
    snippet = price_history[-_PRICE_SNIPPET_LEN:] if price_history else []
    snippet_str = ", ".join(f"{p:.2f}" for p in snippet) if snippet else "unavailable"

    return f"""You are analysing price momentum for a tokenised stock on Kraken.

Ticker: {signal.ticker}
Short momentum  (12-period rate of change): {signal.short_momentum:+.4%}
Medium momentum (24-period rate of change): {signal.medium_momentum:+.4%}
Trend direction: {signal.trend_direction}
Signal strength (0 = flat, 1 = strong): {signal.signal_strength:.4f}
Recent closing prices (oldest → newest): {snippet_str}

Based solely on these momentum metrics, decide whether to BUY, SELL, or HOLD.
Assign confidence above 0.80 only when the signal is unusually clear and consistent.

Respond with valid JSON only:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "reasoning": "one concise sentence"
}}"""


class MomentumAnalyst:
    """Analyst agent that evaluates price momentum signals via LLM reasoning."""

    async def analyze(
        self,
        signal: MomentumSignal,
        price_history: list[float],
    ) -> AnalystReport:
        """Evaluate a momentum signal and return an AnalystReport.

        On LLM failure returns a safe hold at zero confidence.

        Args:
            signal: Pre-computed momentum signal for one ticker.
            price_history: Recent closing prices, oldest first.

        Returns:
            A validated AnalystReport with analyst_type="momentum".
        """
        prompt = _build_prompt(signal, price_history)
        result = await call_llm(prompt, _MomentumLLMResponse)

        if result is None:
            logger.warning(
                "MomentumAnalyst: LLM call failed for %s — returning safe hold",
                signal.ticker,
            )
            return AnalystReport(
                ticker=signal.ticker,
                signal="hold",
                confidence=0.0,
                reasoning="LLM unavailable — defaulting to hold",
                analyst_type="momentum",
            )

        logger.info(
            "MomentumAnalyst: %s → %s (confidence=%.2f)",
            signal.ticker,
            result.signal,
            result.confidence,
        )
        return AnalystReport(
            ticker=signal.ticker,
            signal=result.signal,
            confidence=result.confidence,
            reasoning=result.reasoning,
            analyst_type="momentum",
        )
