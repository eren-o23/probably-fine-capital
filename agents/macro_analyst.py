"""MacroAnalyst agent for Probably Fine Capital.

Evaluates broad market conditions using SPY and QQQ as macro anchors,
returning an AnalystReport or None when confidence is too low to act on.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import config
from core.models import AnalystReport
from utils.llm import call_llm
from utils.logger import system_logger as logger

_FLAT_THRESHOLD: float = 0.005  # ±0.5% treated as flat, mirrors market_data.py
_LAST_N_CLOSES: int = 5


class _MacroLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Pure-Python helpers — no I/O, no LLM
# ---------------------------------------------------------------------------

def _pct_change(history: list[float]) -> Optional[float]:
    """Return first-to-last rate of change, or None when data is insufficient."""
    if len(history) < 2 or history[0] == 0.0:
        return None
    return (history[-1] - history[0]) / history[0]


def _trend(history: list[float]) -> str:
    """Classify first-to-last price movement as up / down / flat / unknown."""
    change = _pct_change(history)
    if change is None:
        return "unknown"
    if change > _FLAT_THRESHOLD:
        return "up"
    if change < -_FLAT_THRESHOLD:
        return "down"
    return "flat"


def _format_anchor(name: str, history: list[float]) -> str:
    """Format one macro anchor block for the prompt."""
    if not history:
        return f"{name}:\n  Status: insufficient macro data"
    change = _pct_change(history)
    change_str = f"{change:+.2%}" if change is not None else "n/a"
    last5 = history[-_LAST_N_CLOSES:]
    closes_str = ", ".join(f"{p:.2f}" for p in last5)
    return (
        f"{name}:\n"
        f"  48h change : {change_str}\n"
        f"  Trend      : {_trend(history)}\n"
        f"  Last {_LAST_N_CLOSES} closes: {closes_str}"
    )


def _format_prices(ticker: str, all_prices: dict[str, float]) -> str:
    """Format the full universe price table, marking the evaluated ticker."""
    lines = []
    for t, price in all_prices.items():
        marker = "  ← this ticker" if t == ticker else ""
        lines.append(f"  {t:<14}: ${price:.2f}{marker}")
    return "\n".join(lines)


def _build_prompt(
    ticker: str,
    all_prices: dict[str, float],
    spy_history: list[float],
    qqq_history: list[float],
) -> str:
    """Build the LLM prompt from macro anchors and universe prices."""
    spy_block = _format_anchor("SPY (S&P 500 proxy)", spy_history)
    qqq_block = _format_anchor("QQQ (Nasdaq proxy)", qqq_history)
    prices_block = _format_prices(ticker, all_prices)

    return f"""You are evaluating broad market conditions to assess a tokenised stock on Kraken.

Ticker under evaluation: {ticker}

--- MACRO ANCHORS (48-hour window) ---
{spy_block}

{qqq_block}

--- CURRENT UNIVERSE PRICES ---
{prices_block}

Given these macro conditions, decide whether to BUY, SELL, or HOLD the ticker.
Assign confidence above 0.80 only when macro conditions give a clear directional edge.
Your reasoning must reference the macro context, not the ticker's individual history.

Respond with valid JSON only:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "reasoning": "one concise sentence referencing macro context"
}}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MacroAnalyst:
    """Analyst agent that evaluates broad market conditions via LLM reasoning."""

    async def analyze(
        self,
        ticker: str,
        all_prices: dict[str, float],
        spy_history: list[float],
        qqq_history: list[float],
    ) -> Optional[AnalystReport]:
        """Evaluate macro conditions for a ticker and return an AnalystReport.

        Returns None when the LLM's confidence is below MIN_CONFIDENCE.
        Returns a hold at confidence=0.0 (not None) when the LLM is unavailable.
        Proceeds with whatever history is available — empty lists are handled
        gracefully by noting "insufficient macro data" in the prompt.

        Args:
            ticker:      xStock pair being evaluated, e.g. "AAPLx/USD".
            all_prices:  Current prices for all tickers in the universe.
            spy_history: Last 48 hourly closes for SPYx/USD. May be empty.
            qqq_history: Last 48 hourly closes for QQQx/USD. May be empty.

        Returns:
            A validated AnalystReport with analyst_type="macro",
            or None if confidence is below config.MIN_CONFIDENCE.
        """
        prompt = _build_prompt(ticker, all_prices, spy_history, qqq_history)
        result = await call_llm(prompt, _MacroLLMResponse)

        if result is None:
            logger.warning(
                "MacroAnalyst: LLM call failed for %s — returning safe hold",
                ticker,
            )
            return AnalystReport(
                ticker=ticker,
                signal="hold",
                confidence=0.0,
                reasoning="LLM unavailable — defaulting to hold",
                analyst_type="macro",
            )

        if result.confidence < config.MIN_CONFIDENCE:
            logger.info(
                "MacroAnalyst: %s signal below MIN_CONFIDENCE (%.2f < %.2f) — discarding",
                ticker,
                result.confidence,
                config.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "MacroAnalyst: %s → %s (confidence=%.2f)",
            ticker,
            result.signal,
            result.confidence,
        )
        return AnalystReport(
            ticker=ticker,
            signal=result.signal,
            confidence=result.confidence,
            reasoning=result.reasoning,
            analyst_type="macro",
        )
