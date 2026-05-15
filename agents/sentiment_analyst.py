"""SentimentAnalyst agent for Probably Fine Capital.

Receives a list of news headlines from market_data and uses an LLM to score
sentiment, returning an AnalystReport or None when the signal is too weak.

Two gates before returning a signal:
  1. Primary  — direct_headlines == 0: no actionable news about this ticker
  2. Secondary — confidence < MIN_CONFIDENCE: signal strength too low

Empty headlines always return a hold at confidence 0.0, never None.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import config
from core.models import AnalystReport
from utils.llm import call_llm
from utils.logger import system_logger as logger

_SYSTEM_PROMPT = (
    "You are a sentiment analyst at a hedge fund. "
    "You assess how recent news headlines will affect a stock's price in the next 1-4 hours. "
    "You are disciplined — you only signal when headlines contain clear, actionable information. "
    "Ambiguous or stale news means hold. "
    "Respond with a single raw JSON object only. No markdown, no code fences, no prose before or after. "
    "Your entire response must be valid JSON."
)


class _SentimentLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    reasoning: str
    direct_headlines: int = Field(ge=0)
    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def _label_headlines(headlines: list[str]) -> str:
    """Format headlines as a numbered list with [RECENT] / [OLDER] labels.

    Index 0 is treated as the most recent. The first third of the list
    (at least 1) is labelled [RECENT]; the rest [OLDER].
    """
    n_recent = max(1, len(headlines) // 3)
    lines = []
    for i, h in enumerate(headlines):
        label = "[RECENT]" if i < n_recent else "[OLDER]"
        lines.append(f"  {i + 1}. {label} {h!r}")
    return "\n".join(lines)


def _build_prompt(ticker: str, headlines: list[str]) -> str:
    """Build the chain-of-thought LLM prompt from a ticker and news headlines."""
    labeled = _label_headlines(headlines)
    return f"""Analyse the news sentiment for this tokenised stock and produce a trade signal.

Ticker: {ticker}

News headlines (index 0 = most recent, [RECENT] = first third, [OLDER] = rest):
{labeled}

--- Chain-of-thought instruction ---
First, identify which headlines are directly about {ticker} vs general market news.
Second, assess whether the news is actionable (earnings, product launches, regulatory
decisions, analyst upgrades/downgrades) or noise (general commentary).
Third, weight recent headlines 2x vs older ones.
Then output your JSON decision.

--- Confidence calibration rules ---
- Only signal buy/sell if at least 1 headline is directly about the ticker (not general market news)
- Confidence above 0.75 requires 2+ direct headlines agreeing
- Earnings surprises, FDA decisions, major contract wins warrant up to 0.85 confidence
- General market sentiment alone: max confidence 0.60
- When headlines are mixed or contradictory: hold
- Never exceed 0.90 confidence

Respond with valid JSON only:
{{
  "reasoning": "2-3 sentences of analysis",
  "direct_headlines": 0,
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "rationale": "one sentence for trade log"
}}"""


class SentimentAnalyst:
    """Analyst agent that scores news sentiment via LLM reasoning."""

    async def analyze(
        self,
        ticker: str,
        headlines: list[str],
    ) -> Optional[AnalystReport]:
        """Score news sentiment for a ticker and return an AnalystReport.

        Returns None when:
          - direct_headlines == 0 (no news directly about this ticker), or
          - confidence < MIN_CONFIDENCE (secondary strength gate).
        Returns a hold at confidence=0.0 (not None) when there are no
        headlines or the LLM is unavailable.

        Args:
            ticker:    xStock pair, e.g. "AAPLx/USD".
            headlines: Recent headlines from get_news_headlines(). May be empty.

        Returns:
            A validated AnalystReport with analyst_type="sentiment",
            or None if the signal is too weak to act on.
        """
        if not headlines:
            logger.info("SentimentAnalyst: no headlines for %s — holding", ticker)
            return AnalystReport(
                ticker=ticker,
                signal="hold",
                confidence=0.0,
                reasoning="no headlines available",
                analyst_type="sentiment",
            )

        prompt = _build_prompt(ticker, headlines)
        result = await call_llm(prompt, _SentimentLLMResponse, system_prompt=_SYSTEM_PROMPT)

        if result is None:
            logger.warning(
                "SentimentAnalyst: LLM call failed for %s — returning safe hold",
                ticker,
            )
            return AnalystReport(
                ticker=ticker,
                signal="hold",
                confidence=0.0,
                reasoning="LLM unavailable — defaulting to hold",
                analyst_type="sentiment",
            )

        logger.debug(
            "SentimentAnalyst: chain-of-thought for %s — %s",
            ticker,
            result.reasoning,
        )

        # Primary gate — no headlines directly about this ticker
        if result.direct_headlines == 0:
            logger.debug(
                "SentimentAnalyst: no direct headlines for %s, skipping",
                ticker,
            )
            return None

        # Secondary gate — signal too weak
        if result.confidence < config.MIN_CONFIDENCE:
            logger.info(
                "SentimentAnalyst: %s signal below MIN_CONFIDENCE (%.2f < %.2f) — discarding",
                ticker,
                result.confidence,
                config.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "SentimentAnalyst: %s → %s (confidence=%.2f, direct=%d)",
            ticker,
            result.signal,
            result.confidence,
            result.direct_headlines,
        )
        return AnalystReport(
            ticker=ticker,
            signal=result.signal,
            confidence=result.confidence,
            reasoning=result.rationale,
            analyst_type="sentiment",
        )
