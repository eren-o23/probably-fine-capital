"""PortfolioManager agent for Probably Fine Capital.

Takes approved RiskDecisions and uses an LLM to size each position,
then enforces hard size bounds in Python before returning TradeInstructions.
LLM calls for all approved decisions run in parallel.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from pydantic import BaseModel, Field

import config
from core.models import AnalystReport, FundState, RiskDecision, TradeInstruction
from utils.llm import call_llm
from utils.logger import system_logger as logger


class _AllocationLLMResponse(BaseModel):
    """Internal model for parsing the LLM's position-sizing response."""

    ticker: str
    size_usd: float = Field(gt=0.0)
    reasoning: str


def _build_prompt(
    report: AnalystReport,
    confidence: float,
    fund_state: FundState,
) -> str:
    """Build the LLM prompt for position sizing."""
    summary = fund_state.to_summary_dict()
    max_by_pct = config.MAX_POSITION_PCT * fund_state.total_value

    return f"""You are a portfolio manager sizing a trade for an AI hedge fund.

Ticker    : {report.ticker}
Signal    : {report.signal}
Analyst   : {report.analyst_type}
Confidence: {confidence:.2f}

Current portfolio:
{json.dumps(summary, indent=2)}

Hard constraints (you must stay within all of these):
  MAX_POSITION_PCT   : {config.MAX_POSITION_PCT:.0%} of total fund value (max ${max_by_pct:.2f})
  MIN_TRADE_SIZE_USD : ${config.MIN_TRADE_SIZE_USD:.2f}
  MAX_TRADE_SIZE_USD : ${config.MAX_TRADE_SIZE_USD:.2f}
  Available cash     : ${fund_state.cash:.2f}

Determine the appropriate position size in USD.
Higher confidence justifies a larger allocation within the constraints.

Respond with valid JSON only:
{{
  "ticker": "{report.ticker}",
  "size_usd": 0.00,
  "reasoning": "one concise sentence"
}}"""


def _apply_size_bounds(llm_size: float, fund_state: FundState) -> float:
    """Clamp LLM-suggested size to hard Python limits. Never trust the LLM alone."""
    # Step 1: clamp between absolute trade-size floor and ceiling
    size = max(config.MIN_TRADE_SIZE_USD, min(config.MAX_TRADE_SIZE_USD, llm_size))
    # Step 2: cap at the maximum position size as a fraction of total fund value
    max_by_pct = config.MAX_POSITION_PCT * fund_state.total_value
    return min(size, max_by_pct)


class PortfolioManager:
    """Agent that converts approved RiskDecisions into sized TradeInstructions."""

    async def allocate(
        self,
        decisions: list[RiskDecision],
        fund_state: FundState,
    ) -> list[TradeInstruction]:
        """Size each approved decision via LLM and return validated TradeInstructions.

        Vetoed decisions are ignored entirely. Approved decisions are processed in
        parallel. If the LLM fails for any single ticker it is skipped; the rest
        are still returned.

        Args:
            decisions:  Output of RiskManager.evaluate().
            fund_state: Current fund snapshot for sizing context.

        Returns:
            List of TradeInstructions, one per successfully sized approved decision.
        """
        approved = [d for d in decisions if d.approved]
        if not approved:
            logger.info("Portfolio allocation complete: 0 instructions generated")
            return []

        tasks = [self._allocate_one(d, fund_state) for d in approved]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        instructions: list[TradeInstruction] = []
        for result in raw:
            if isinstance(result, TradeInstruction):
                instructions.append(result)
            elif isinstance(result, BaseException):
                logger.error(
                    "PortfolioManager: unexpected exception in allocation: %s", result
                )
            # None = LLM failed, already logged in _allocate_one

        logger.info(
            "Portfolio allocation complete: %d instructions generated",
            len(instructions),
        )
        return instructions

    async def _allocate_one(
        self,
        decision: RiskDecision,
        fund_state: FundState,
    ) -> Optional[TradeInstruction]:
        """Size one approved decision. Returns None if the LLM is unavailable."""
        report = decision.original_report
        prompt = _build_prompt(report, decision.modified_confidence, fund_state)
        result = await call_llm(prompt, _AllocationLLMResponse)

        if result is None:
            logger.warning(
                "PortfolioManager: LLM failed for %s — skipping",
                report.ticker,
            )
            return None

        final_size = _apply_size_bounds(result.size_usd, fund_state)

        logger.info(
            "PortfolioManager: %s %s → size=$%.2f (LLM suggested $%.2f)",
            report.ticker,
            report.signal,
            final_size,
            result.size_usd,
        )
        return TradeInstruction(
            action=report.signal,
            ticker=report.ticker,
            size_usd=final_size,
            rationale=result.reasoning,
        )
