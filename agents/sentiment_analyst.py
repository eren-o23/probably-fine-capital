"""SentimentAnalyst agent for Probably Fine Capital.

Receives a list of news headlines from market_data and uses an LLM to score
overall sentiment, returning an AnalystReport or None when confidence is too
low to act on.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import config
from core.models import AnalystReport
from utils.llm import call_llm
from utils.logger import system_logger as logger


class _SentimentLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _build_prompt(ticker: str, headlines: list[str]) -> str:
    """Build the LLM prompt from a ticker and list of news headlines."""
    numbered = "\n".join(f"  {i + 1}. {h!r}" for i, h in enumerate(headlines))
    return f"""You are analysing news sentiment for a tokenised stock on Kraken.

Ticker: {ticker}

News headlines (most recent first):
{numbered}

Based solely on these headlines, decide the overall sentiment: BUY, SELL, or HOLD.
Assign confidence above 0.80 only when sentiment is overwhelmingly clear and one-sided.

Respond with valid JSON only:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "reasoning": "one concise sentence"
}}"""


class SentimentAnalyst:
    """Analyst agent that scores news sentiment via LLM reasoning."""

    async def analyze(
        self,
        ticker: str,
        headlines: list[str],
    ) -> Optional[AnalystReport]:
        """Score news sentiment for a ticker and return an AnalystReport.

        Returns None when the LLM's confidence is below MIN_CONFIDENCE —
        signalling that the signal is too weak to act on.
        Returns a hold at confidence=0.0 (not None) when there are no
        headlines or the LLM is unavailable.

        Args:
            ticker:    xStock pair, e.g. "AAPLx/USD".
            headlines: Recent headlines from get_news_headlines(). May be empty.

        Returns:
            A validated AnalystReport with analyst_type="sentiment",
            or None if confidence is below config.MIN_CONFIDENCE.
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
        result = await call_llm(prompt, _SentimentLLMResponse)

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

        if result.confidence < config.MIN_CONFIDENCE:
            logger.info(
                "SentimentAnalyst: %s signal below MIN_CONFIDENCE (%.2f < %.2f) — discarding",
                ticker,
                result.confidence,
                config.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "SentimentAnalyst: %s → %s (confidence=%.2f)",
            ticker,
            result.signal,
            result.confidence,
        )
        return AnalystReport(
            ticker=ticker,
            signal=result.signal,
            confidence=result.confidence,
            reasoning=result.reasoning,
            analyst_type="sentiment",
        )
