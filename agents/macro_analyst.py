"""MacroAnalyst agent for Probably Fine Capital.

Evaluates broad market conditions using SPY and QQQ as macro anchors,
returning an AnalystReport or None when confidence is too low to act on.

New in improved version:
  - Python-side regime classification (risk_on / risk_off / mixed)
  - Breadth proxy from universe snapshot prices
  - SPY volatility label computed before the LLM call
  - Chain-of-thought prompt with calibrated confidence rules
  - Explicit validation of the market_regime field after parsing
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import config
from core.models import AnalystReport
from utils.llm import call_llm
from utils.logger import system_logger as logger

_FLAT_THRESHOLD: float = 0.005      # ±0.5% treated as flat, mirrors market_data.py
_LAST_N_CLOSES: int = 5
_REGIME_THRESHOLD: float = 0.3     # ±0.3% threshold for risk_on / risk_off
_VOLATILITY_THRESHOLD: float = 1.5 # std > 1.5% of hourly pct-changes = "high"

_VALID_REGIMES: frozenset[str] = frozenset({"risk_on", "risk_off", "mixed"})

_SYSTEM_PROMPT = (
    "You are a macro analyst at a hedge fund. "
    "You assess broad market conditions to determine whether the environment "
    "favours risk-on or risk-off positioning. "
    "You look at index behaviour to guide individual stock decisions. "
    "You are balanced — you avoid overriding strong stock-specific signals "
    "with weak macro views. "
    "The signal field must be exactly one of buy, sell, or hold — never a placeholder, ellipsis, or any other value. "
    "Respond with a single raw JSON object only. No markdown, no code fences, no prose before or after. "
    "Your entire response must be valid JSON."
)


class _MacroLLMResponse(BaseModel):
    """Internal model for parsing the LLM's JSON response."""

    reasoning: str
    market_regime: str          # validated explicitly after parsing
    signal: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------------------
# Pure-Python helpers — no I/O, no LLM
# ---------------------------------------------------------------------------

def _pct_change(history: list[float]) -> Optional[float]:
    """Return first-to-last rate of change as a fraction, or None when insufficient."""
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
    """Format one macro anchor block for the prompt (kept for backward compat)."""
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


def _classify_regime(
    spy_chg_pct: Optional[float],
    qqq_chg_pct: Optional[float],
) -> str:
    """Classify market regime from SPY and QQQ period changes (in percent, not fraction).

    Args:
        spy_chg_pct: SPY period change as a percentage (e.g. 2.0 means +2%).
        qqq_chg_pct: QQQ period change as a percentage.

    Returns:
        "risk_on"  — both indices up > 0.3%
        "risk_off" — both indices down > 0.3%
        "mixed"    — everything else, including missing data
    """
    if spy_chg_pct is None or qqq_chg_pct is None:
        return "mixed"
    if spy_chg_pct > _REGIME_THRESHOLD and qqq_chg_pct > _REGIME_THRESHOLD:
        return "risk_on"
    if spy_chg_pct < -_REGIME_THRESHOLD and qqq_chg_pct < -_REGIME_THRESHOLD:
        return "risk_off"
    return "mixed"


def _spy_volatility_label(spy_history: list[float]) -> str:
    """Classify SPY intra-period volatility from hourly pct-changes.

    Returns "high" if std of hourly pct-changes > 1.5%, "normal" otherwise,
    or "unknown" when there is insufficient data (fewer than 3 prices).
    """
    if len(spy_history) < 3:
        return "unknown"
    pct_changes = [
        (spy_history[i] - spy_history[i - 1]) / spy_history[i - 1] * 100
        for i in range(1, len(spy_history))
        if spy_history[i - 1] != 0
    ]
    if len(pct_changes) < 2:
        return "unknown"
    n = len(pct_changes)
    mean = sum(pct_changes) / n
    variance = sum((x - mean) ** 2 for x in pct_changes) / n
    std = variance ** 0.5
    return "high" if std > _VOLATILITY_THRESHOLD else "normal"


def _compute_breadth(all_prices: dict[str, float]) -> tuple[int, int]:
    """Return (advancing, declining) as a snapshot-based relative breadth proxy.

    Since all_prices is a current-price snapshot (no per-ticker histories),
    tickers above the universe median price are counted as advancing; those
    below are counted as declining. Tickers exactly at the median are neutral.

    Returns:
        Tuple of (advancing_count, declining_count).
    """
    if not all_prices:
        return 0, 0
    values = sorted(all_prices.values())
    median = values[len(values) // 2]
    advancing = sum(1 for p in all_prices.values() if p > median)
    declining = sum(1 for p in all_prices.values() if p < median)
    return advancing, declining


def _build_prompt(
    ticker: str,
    all_prices: dict[str, float],
    spy_history: list[float],
    qqq_history: list[float],
) -> str:
    """Build the chain-of-thought LLM prompt from pre-computed macro metrics."""
    # Compute Python-side metrics
    spy_frac = _pct_change(spy_history)
    qqq_frac = _pct_change(qqq_history)
    spy_chg_pct = spy_frac * 100 if spy_frac is not None else None
    qqq_chg_pct = qqq_frac * 100 if qqq_frac is not None else None

    regime = _classify_regime(spy_chg_pct, qqq_chg_pct)
    vol_label = _spy_volatility_label(spy_history)
    advancing, declining = _compute_breadth(all_prices)
    total = len(all_prices)

    spy_str = f"{spy_chg_pct:+.2f}%" if spy_chg_pct is not None else "n/a"
    qqq_str = f"{qqq_chg_pct:+.2f}%" if qqq_chg_pct is not None else "n/a"

    # Ticker's relative position vs universe
    ticker_price = all_prices.get(ticker)
    ticker_str = f"${ticker_price:.2f}" if ticker_price is not None else "n/a"
    if ticker_price is not None and all_prices:
        values = sorted(all_prices.values())
        median = values[len(values) // 2]
        if ticker_price > median:
            relative_pos = "showing relative strength (above universe median)"
        elif ticker_price < median:
            relative_pos = "showing relative weakness (below universe median)"
        else:
            relative_pos = "in line with universe median"
    else:
        relative_pos = "data unavailable"

    prices_block = _format_prices(ticker, all_prices)

    return f"""Evaluate broad market conditions and provide a macro-based trade signal.

Ticker under evaluation: {ticker}  (current price: {ticker_str})
Ticker position: {relative_pos}

--- MARKET REGIME ---
SPY period change : {spy_str}
QQQ period change : {qqq_str}
Regime classified : {regime}
SPY volatility    : {vol_label}

--- BREADTH (snapshot proxy) ---
{advancing} advancing, {declining} declining out of {total} tracked tickers

--- CURRENT UNIVERSE PRICES ---
{prices_block}

--- Chain-of-thought instruction ---
First, assess the overall market regime.
Second, consider whether the breadth supports or contradicts the index move.
Third, determine if this ticker is showing relative strength or weakness vs the market.
Then output your decision.

--- Confidence calibration rules ---
- risk_on regime + ticker showing relative strength: up to 0.78
- risk_off regime + ticker showing relative weakness: up to 0.78
- mixed regime: max confidence 0.62, prefer hold
- High volatility: reduce confidence by 0.10 regardless of regime
- Never exceed 0.85 — macro is a supporting signal, not primary
- Macro should confirm stock-specific signals, not override them

Respond with valid JSON only:
{{
  "reasoning": "2-3 sentences of macro analysis",
  "market_regime": "risk_on" | "risk_off" | "mixed",
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.00,
  "rationale": "one sentence for trade log"
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

        Returns None when:
          - market_regime field is not a valid value, or
          - confidence < MIN_CONFIDENCE.
        Returns a hold at confidence=0.0 (not None) when the LLM is unavailable.

        Args:
            ticker:      xStock pair being evaluated, e.g. "AAPLx/USD".
            all_prices:  Current prices for all tickers in the universe.
            spy_history: Last 48 hourly closes for SPYx/USD. May be empty.
            qqq_history: Last 48 hourly closes for QQQx/USD. May be empty.

        Returns:
            A validated AnalystReport with analyst_type="macro",
            or None if the signal is too weak or the regime is unrecognised.
        """
        prompt = _build_prompt(ticker, all_prices, spy_history, qqq_history)
        result = await call_llm(prompt, _MacroLLMResponse, system_prompt=_SYSTEM_PROMPT)

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

        logger.debug(
            "MacroAnalyst: chain-of-thought for %s — %s",
            ticker,
            result.reasoning,
        )

        if result.market_regime not in _VALID_REGIMES:
            logger.warning(
                "MacroAnalyst: invalid market_regime '%s' for %s — discarding",
                result.market_regime,
                ticker,
            )
            return None

        if result.confidence < config.MIN_CONFIDENCE:
            logger.info(
                "MacroAnalyst: %s signal below MIN_CONFIDENCE (%.2f < %.2f) — discarding",
                ticker,
                result.confidence,
                config.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "MacroAnalyst: %s → %s (confidence=%.2f, regime=%s)",
            ticker,
            result.signal,
            result.confidence,
            result.market_regime,
        )
        return AnalystReport(
            ticker=ticker,
            signal=result.signal,
            confidence=result.confidence,
            reasoning=result.rationale,
            analyst_type="macro",
        )
