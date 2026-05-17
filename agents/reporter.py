"""Reporter agent for Probably Fine Capital.

Generates a daily markdown report (and an X thread draft) summarising
fund performance, trades, and market observations.

Two LLM calls per run:
  1. Narrative summary  → headline + 5 narrative sections (JSON)
  2. X thread draft      → 5-tweet thread (JSON list)

Both calls are best-effort: a failed narrative falls back to a minimal
report with raw fund state; a failed thread call drops the X section.
run() never raises — reporter failure must not crash the trading loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
import tweepy

import config
from core.models import FundState, TradeInstruction
from utils.llm import call_llm

_TWEET_MAX_CHARS: int = 280
_EXPECTED_TWEETS: int = 5


class _NarrativeResponse(BaseModel):
    """Parsed JSON from the narrative LLM call."""

    headline: str = Field(max_length=200)
    performance_summary: str
    trade_recap: str
    market_observations: str
    outlook: str
    risk_flags: str


class _ThreadResponse(BaseModel):
    """Parsed JSON from the X-thread LLM call.

    Wrapping the array in a model lets us reuse call_llm() which expects
    a Pydantic BaseModel response type.
    """

    tweets: list[str]


def _truncate_at_word_boundary(text: str, max_chars: int = _TWEET_MAX_CHARS) -> str:
    """Truncate `text` to at most `max_chars`, breaking at a word boundary.

    Appends "…" if truncation actually occurred. Reserves one char for the
    ellipsis so the result is always <= max_chars.
    """
    if len(text) <= max_chars:
        return text
    cutoff = max_chars - 1
    truncated = text[:cutoff]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def _format_position(ticker: str, pos) -> dict:
    """Return a JSON-safe dict for one position.

    Maps spec field names to actual Position fields:
      average_cost  → Position.entry_price
      unrealised_pnl → Position.pnl_usd (computed)
    """
    return {
        "ticker": ticker,
        "quantity": round(pos.quantity, 4),
        "average_cost": round(pos.entry_price, 2),
        "current_price": round(pos.current_price, 2),
        "unrealised_pnl": round(pos.pnl_usd, 2),
    }


def _format_instructions(instructions: list[TradeInstruction]) -> list[dict]:
    """Return JSON-safe trade dicts for the prompt."""
    return [
        {
            "ticker": i.ticker,
            "action": i.action,
            "size_usd": round(i.size_usd, 2),
            "rationale": i.rationale,
        }
        for i in instructions
    ]


class Reporter:
    """Daily report generator. Writes markdown to logs/reports/."""

    def __init__(self) -> None:
        """Configure the named logger and ensure the output directory exists."""
        self.logger = logging.getLogger("reporter")
        self.reports_dir = Path("logs/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        fund_state: FundState,
        instructions: list[TradeInstruction],
        cycle_count: int,
        elapsed_seconds: float,
    ) -> tuple[str, list[str] | None]:
        """Generate the daily report and return the filepath and tweet list.

        Intended integration in core/loop.py:
            On TradingLoop add `last_report_date: date | None = None`.
            After each cycle, if `date.today() != self.last_report_date`,
            filepath, tweets = await self.reporter.run(...)
            if tweets: await self.reporter.post_to_x(tweets)
            then set `self.last_report_date = date.today()`.

        Args:
            fund_state:      Snapshot of fund state at report time.
            instructions:    TradeInstructions issued in the reporting window.
            cycle_count:     Total cycles run since startup.
            elapsed_seconds: Total fund uptime in seconds.

        Returns:
            (filepath, tweets) — filepath is the written markdown path as a string;
            tweets is the truncated tweet list, or None if thread generation failed.
            Never raises — on LLM failure, a minimal report is written instead.
        """
        now = datetime.now(timezone.utc)
        filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.reports_dir / filename

        try:
            narrative = await self._call_narrative(
                fund_state, instructions, cycle_count, elapsed_seconds, now
            )
        except Exception as exc:
            self.logger.error("Reporter: narrative call raised: %s", exc)
            narrative = None

        if narrative is None:
            self.logger.warning("Reporter: narrative LLM failed — writing minimal report")
            self._write_minimal_report(filepath, fund_state, instructions, cycle_count, elapsed_seconds, now)
            return str(filepath), None

        try:
            thread = await self._call_thread(narrative)
        except Exception as exc:
            self.logger.error("Reporter: thread call raised: %s", exc)
            thread = None

        tweets: Optional[list[str]] = None
        if thread is not None:
            tweets = [_truncate_at_word_boundary(t) for t in thread.tweets]
        else:
            self.logger.warning("Reporter: thread LLM failed — writing report without X section")

        self._write_full_report(
            filepath, fund_state, narrative, tweets, cycle_count, elapsed_seconds, now
        )
        self.logger.info("Reporter: wrote daily report to %s", filepath)
        return str(filepath), tweets

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _call_narrative(
        self,
        fund_state: FundState,
        instructions: list[TradeInstruction],
        cycle_count: int,
        elapsed_seconds: float,
        now: datetime,
    ) -> Optional[_NarrativeResponse]:
        """Build the narrative prompt and call the LLM."""
        positions = [_format_position(t, p) for t, p in fund_state.positions.items()]
        trades = _format_instructions(instructions)

        prompt = f"""You are the chief writer for an AI hedge fund called Probably Fine Capital.
Generate a daily report from the data below.

Date (UTC)        : {now.strftime('%Y-%m-%d %H:%M:%S')}
Cycle count       : {cycle_count}
Uptime (hours)    : {elapsed_seconds / 3600:.2f}
Cash available    : ${fund_state.cash:,.2f}
Total portfolio   : ${fund_state.total_value:,.2f}
Total PnL         : ${fund_state.total_pnl:,.2f}
Drawdown          : {fund_state.drawdown_pct * 100:.2f}%

Open positions ({len(positions)}):
{positions if positions else 'none'}

Trades issued today ({len(trades)}):
{trades if trades else 'none'}

Respond with a single raw JSON object only. No markdown, no code fences, no prose before or after. Your entire response must be valid JSON, using this exact schema:
{{
  "headline": "one punchy sentence, max 100 chars",
  "performance_summary": "2-3 sentences on P&L and positioning",
  "trade_recap": "2-3 sentences on what was traded and why",
  "market_observations": "2-3 sentences on macro/momentum themes seen today",
  "outlook": "1-2 sentences forward looking",
  "risk_flags": "one sentence, 'None' if nothing notable"
}}"""

        return await call_llm(prompt, _NarrativeResponse)

    async def _call_thread(self, narrative: _NarrativeResponse) -> Optional[_ThreadResponse]:
        """Build the X-thread prompt from a parsed narrative and call the LLM."""
        prompt = f"""You write social-media threads for Probably Fine Capital, an AI hedge fund.
Turn the narrative below into a 5-tweet thread for X (Twitter).

Headline             : {narrative.headline}
Performance summary  : {narrative.performance_summary}
Trade recap          : {narrative.trade_recap}
Market observations  : {narrative.market_observations}
Outlook              : {narrative.outlook}
Risk flags           : {narrative.risk_flags}

Rules:
  - Each tweet MUST be under 280 characters.
  - Tweet 1: hook + headline
  - Tweet 2: performance + positions
  - Tweet 3: trade recap
  - Tweet 4: outlook + risk flags
  - Tweet 5: sign-off tagging @krakenfx @lablabai @Surgexyz_ with #ProbablyFineCapital #xStocks — keep under 280 chars

Respond with a single raw JSON object only. No markdown, no code fences, no prose before or after. Your entire response must be valid JSON:
{{
  "tweets": ["tweet 1...", "tweet 2...", "tweet 3...", "tweet 4...", "tweet 5..."]
}}"""

        return await call_llm(prompt, _ThreadResponse)

    # ------------------------------------------------------------------
    # Markdown writers
    # ------------------------------------------------------------------

    def _write_full_report(
        self,
        filepath: Path,
        fund_state: FundState,
        narrative: _NarrativeResponse,
        tweets: Optional[list[str]],
        cycle_count: int,
        elapsed_seconds: float,
        now: datetime,
    ) -> None:
        """Write the full markdown report — narrative plus optional X thread."""
        lines: list[str] = [
            "# Probably Fine Capital — Daily Report",
            f"**Date:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Cycle:** {cycle_count} | **Uptime:** {elapsed_seconds / 3600:.1f}h",
            "",
            f"> {narrative.headline}",
            "",
            "## Performance",
            narrative.performance_summary,
            "",
            "## Trades Today",
            narrative.trade_recap,
            "",
            "## Market Observations",
            narrative.market_observations,
            "",
            "## Outlook",
            narrative.outlook,
            "",
            "## Risk Flags",
            narrative.risk_flags,
            "",
        ]

        if tweets:
            lines.extend(["---", "## X Thread Draft"])
            for i, tweet in enumerate(tweets, start=1):
                lines.append(f"{i}. {tweet}")
            lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _write_minimal_report(
        self,
        filepath: Path,
        fund_state: FundState,
        instructions: list[TradeInstruction],
        cycle_count: int,
        elapsed_seconds: float,
        now: datetime,
    ) -> None:
        """Write a fallback markdown report from raw fund state — no LLM content."""
        lines: list[str] = [
            "# Probably Fine Capital — Daily Report (raw fallback)",
            f"**Date:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Cycle:** {cycle_count} | **Uptime:** {elapsed_seconds / 3600:.1f}h",
            "",
            "> Narrative generation failed — showing raw state.",
            "",
            "## State",
            f"- Cash available : ${fund_state.cash:,.2f}",
            f"- Total value    : ${fund_state.total_value:,.2f}",
            f"- Total PnL      : ${fund_state.total_pnl:,.2f}",
            f"- Drawdown       : {fund_state.drawdown_pct * 100:.2f}%",
            f"- Open positions : {len(fund_state.positions)}",
            "",
            "## Open Positions",
        ]
        if fund_state.positions:
            for ticker, pos in fund_state.positions.items():
                lines.append(
                    f"- {ticker}: qty={pos.quantity:.4f} "
                    f"entry=${pos.entry_price:.2f} "
                    f"now=${pos.current_price:.2f} "
                    f"pnl=${pos.pnl_usd:.2f}"
                )
        else:
            lines.append("- none")

        lines.extend(["", "## Trades Today"])
        if instructions:
            for inst in instructions:
                lines.append(
                    f"- {inst.action} {inst.ticker} ${inst.size_usd:.2f} — {inst.rationale}"
                )
        else:
            lines.append("- none")
        lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # X posting (stub)
    # ------------------------------------------------------------------

    async def post_to_x(self, tweets: list[str]) -> bool:
        """Post thread to X using tweepy.

        Requires env vars: X_API_KEY, X_API_SECRET,
        X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
        Install: pip install tweepy
        Returns True on success, False on failure.
        Currently stubbed — set X_ENABLED=true in .env to activate.
        """
        if not config.X_ENABLED:
            self.logger.info("X posting disabled — set X_ENABLED=true to activate")
            return False
        consumer_key        = os.getenv("X_CONSUMER_KEY")
        consumer_secret     = os.getenv("X_CONSUMER_SECRET")
        access_token        = os.getenv("X_ACCESS_TOKEN")
        access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            self.logger.warning("X posting skipped — missing credentials")
            return False

        try:
            client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
            )
            previous_id = None
            for tweet in tweets:
                if previous_id is None:
                    response = client.create_tweet(text=tweet)
                else:
                    response = client.create_tweet(
                        text=tweet,
                        in_reply_to_tweet_id=previous_id,
                    )
                previous_id = response.data["id"]
                await asyncio.sleep(1)
            self.logger.info("X thread posted (%d tweets)", len(tweets))
            return True
        except Exception as exc:
            self.logger.error("X posting failed: %s", exc)
            return False
