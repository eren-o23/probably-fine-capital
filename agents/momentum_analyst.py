"""MomentumAnalyst agent for Probably Fine Capital.

Receives a pre-computed MomentumSignal from market_data and uses an LLM
to reason about it, returning an AnalystReport with a signal and confidence.

The prompt uses structured chain-of-thought reasoning, a per-candle price
table, period high/low context, and calibrated confidence rules to produce
balanced, higher-quality signals.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from core.market_data import MomentumSignal
from core.models import AnalystReport
from utils.llm import call_llm

logger = logging.getLogger(__name__)

_PRICE_SNIPPET_LEN: int = 12

_SYSTEM_PROMPT = (
    "You are a quantitative momentum analyst at a hedge fund. "
    "Your job is to identify high-conviction directional trades based on price momentum. "
    "You are balanced — you seek strong returns but avoid overconfident signals. "
    "When in doubt, hold. Respond in valid JSON only. No markdown, no explanation outside the JSON."
)


class _MomentumLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    reasoning: str
    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def _classify_trend(prices: list[float]) -> Literal["strong_uptrend", "strong_downtrend", "consolidating"]:
    """Classify the recent trend from the last three closing prices."""
    if len(prices) < 3:
        return "consolidating"
    if prices[-1] > prices[-2] > prices[-3]:
        return "strong_uptrend"
    if prices[-1] < prices[-2] < prices[-3]:
        return "strong_downtrend"
    return "consolidating"


def _build_price_table(snippet: list[float]) -> str:
    """Format a price snippet as a markdown table with per-candle change %."""
    if not snippet:
        return "| # | Price | Change % |\n|---|---|---|\n| - | unavailable | - |"
    rows = ["| # | Price | Change % |", "|---|---|---|"]
    for i, price in enumerate(snippet):
        if i == 0:
            change_str = "—"
        else:
            prev = snippet[i - 1]
            pct = ((price - prev) / prev * 100) if prev != 0 else 0.0
            change_str = f"{pct:+.2f}%"
        rows.append(f"| {i + 1} | {price:.2f} | {change_str} |")
    return "\n".join(rows)


def _build_prompt(signal: MomentumSignal, price_history: list[float]) -> str:
    """Build the chain-of-thought LLM prompt from a MomentumSignal and price history."""
    snippet = price_history[-_PRICE_SNIPPET_LEN:] if price_history else []
    current_price = price_history[-1] if price_history else None
    trend_class = _classify_trend(price_history)
    price_table = _build_price_table(snippet)

    current_str = f"{current_price:.2f}" if current_price is not None else "unavailable"

    if price_history and current_price is not None:
        period_high = max(price_history)
        period_low = min(price_history)
        pct_below_high = (period_high - current_price) / period_high * 100
        pct_above_low = (
            (current_price - period_low) / period_low * 100 if period_low != 0 else 0.0
        )
        context_str = (
            f"Period high : {period_high:.2f}  "
            f"(current is {pct_below_high:.2f}% below high)\n"
            f"Period low  : {period_low:.2f}  "
            f"(current is {pct_above_low:.2f}% above low)"
        )
    else:
        context_str = "Period high/low: unavailable"

    return f"""Analyse the momentum signal for this tokenised stock and produce a trade decision.

Ticker         : {signal.ticker}
Current price  : {current_str}

--- Momentum metrics ---
Short momentum  (12-period rate of change): {signal.short_momentum:+.4%}
Medium momentum (24-period rate of change): {signal.medium_momentum:+.4%}
Trend direction : {signal.trend_direction}
Signal strength : {signal.signal_strength:.4f}  (0 = flat, 1 = strong)

--- Python-classified trend ---
Trend class: {trend_class}

--- Recent price history (last {len(snippet)} candles, oldest to newest) ---
{price_table}

--- Period context ---
{context_str}

--- Confidence calibration rules ---
- Only exceed 0.80 confidence if trend is strong AND price is not extended
  (for buys: current price should be near the period low, not near the high)
- Default to hold (confidence ≈ 0.50) when trend is consolidating
- Never output confidence above 0.90 — no signal is certain
- Prefer missing a trade to making a bad one

--- Chain-of-thought instruction ---
First, analyse the trend direction and strength.
Second, assess whether current price is extended or has room to run.
Third, consider if momentum is accelerating or decelerating.
Then output your JSON decision.

Respond with valid JSON only:
{{
  "reasoning": "2-3 sentences of chain-of-thought analysis",
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "rationale": "one sentence suitable for a trade log"
}}"""


class MomentumAnalyst:
    """Analyst agent that evaluates price momentum signals via LLM reasoning."""

    async def analyze(
        self,
        signal: MomentumSignal,
        price_history: list[float],
    ) -> AnalystReport:
        """Evaluate a momentum signal and return an AnalystReport.

        On LLM failure returns a safe hold at zero confidence. Never raises.

        Args:
            signal: Pre-computed momentum signal for one ticker.
            price_history: Recent closing prices, oldest first.

        Returns:
            A validated AnalystReport with analyst_type="momentum".
        """
        prompt = _build_prompt(signal, price_history)
        result = await call_llm(prompt, _MomentumLLMResponse, system_prompt=_SYSTEM_PROMPT)

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

        logger.debug(
            "MomentumAnalyst: chain-of-thought for %s — %s",
            signal.ticker,
            result.reasoning,
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
            reasoning=result.rationale,
            analyst_type="momentum",
        )
