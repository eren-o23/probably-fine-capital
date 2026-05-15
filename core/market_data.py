"""Market data aggregation layer for Probably Fine Capital.

Provides prices, price histories, pure-Python momentum signals, and news
headlines for every tradeable xStock ticker. All public functions return
partial data on failure — they never raise.

News headlines are cached per-ticker for 60 minutes to stay within
Alpaca free-tier limits.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import aiohttp
from pydantic import BaseModel, Field

from config import ACTIVE_TICKERS, ALPACA_API_KEY, ALPACA_API_SECRET, TRADEABLE_TICKERS
from utils.kraken_cli import get_price_history, get_prices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# News cache — module-level, survives across calls within a process
# ---------------------------------------------------------------------------

_NEWS_CACHE_TTL: timedelta = timedelta(minutes=60)
_news_cache: dict[str, tuple[datetime, list[str]]] = {}

_ALPACA_NEWS_URL: str = "https://data.alpaca.markets/v1beta1/news"
_NEWS_PAGE_SIZE: int = 10

# ---------------------------------------------------------------------------
# Momentum thresholds
# ---------------------------------------------------------------------------

_FLAT_THRESHOLD: float = 0.005   # < ±0.5% change treated as flat
_STRENGTH_CAP: float = 0.05      # 5% move = max signal strength of 1.0

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MomentumSignal(BaseModel):
    """Pure-Python momentum signal derived from OHLC closing prices.

    Consumed by MomentumAnalyst on Day 2 — no LLM involved in its computation.
    """

    ticker: str
    short_momentum: float     # 12-period (~12 h) rate of change as fraction
    medium_momentum: float    # 24-period (~1 day) rate of change as fraction
    trend_direction: Literal["up", "down", "flat"]
    signal_strength: float    # abs(medium_momentum) / 0.05, clamped to [0, 1]


class MarketSnapshot(BaseModel):
    """Complete market state at one point in time, passed to analyst agents."""

    prices: dict[str, float]
    price_histories: dict[str, list[float]]
    momentum_signals: dict[str, MomentumSignal]
    headlines: dict[str, list[str]]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ticker_to_symbol(ticker: str) -> str:
    """Convert xStock pair to bare equity symbol.

    Example: "AAPLx/USD" → "AAPL"
    """
    base = ticker.split("/")[0]   # "AAPLx"
    return base[:-1]              # strip the trailing "x"


def _compute_roc(prices: list[float], lookback: int) -> float:
    """Rate of change from `lookback` periods ago to now.

    Returns 0.0 when there are not enough data points or the base price is zero.
    """
    if len(prices) < lookback + 1:
        return 0.0
    base = prices[-(lookback + 1)]
    if base == 0.0:
        return 0.0
    return (prices[-1] / base) - 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_current_prices() -> dict[str, float]:
    """Fetch the last trade price for every tradeable xStock ticker.

    Delegates to kraken_cli.get_prices which is already concurrent and
    skips individual failures. This wrapper adds a top-level safety net.

    Returns:
        Mapping of ticker → price for every ticker that succeeded.
        Empty dict on total failure.
    """
    try:
        prices = await get_prices(ACTIVE_TICKERS)
        if not prices:
            logger.warning("get_current_prices: all tickers failed, returned empty dict")
        return prices
    except Exception as exc:
        logger.error("get_current_prices: unexpected error: %s", exc)
        return {}


async def get_price_histories(periods: int = 48) -> dict[str, list[float]]:
    """Fetch 60-min OHLC closing prices for every tradeable ticker concurrently.

    Args:
        periods: number of 60-min candles to return per ticker (48 = 2 days).

    Returns:
        Mapping of ticker → list of closes, oldest first.
        Tickers that fail are omitted rather than raising.
    """
    raw = await asyncio.gather(
        *(get_price_history(t, interval=60, periods=periods) for t in ACTIVE_TICKERS),
        return_exceptions=True,
    )
    histories: dict[str, list[float]] = {}
    for ticker, result in zip(ACTIVE_TICKERS, raw):
        if isinstance(result, BaseException):
            logger.warning("get_price_histories: skipping %s — %s", ticker, result)
        else:
            histories[ticker] = result
    return histories


def calculate_momentum_signals(
    price_histories: dict[str, list[float]],
) -> dict[str, MomentumSignal]:
    """Compute momentum signals from OHLC closing prices. No LLM, no I/O.

    Signals per ticker:
      short_momentum:  12-period (~12 h) rate of change
      medium_momentum: 24-period (~1 day) rate of change
      trend_direction: "up" if medium > +0.5%, "down" if < -0.5%, else "flat"
      signal_strength: abs(medium_momentum) / 0.05 clamped to [0.0, 1.0]

    Args:
        price_histories: mapping of ticker → closes list, oldest first.

    Returns:
        Mapping of ticker → MomentumSignal. Skips tickers with fewer than
        2 data points. Never raises.
    """
    signals: dict[str, MomentumSignal] = {}
    for ticker, prices in price_histories.items():
        try:
            if len(prices) < 2:
                logger.debug(
                    "calculate_momentum_signals: too few prices for %s (%d), skipping",
                    ticker,
                    len(prices),
                )
                continue

            short_mom = _compute_roc(prices, lookback=12)
            medium_mom = _compute_roc(prices, lookback=24)

            if medium_mom > _FLAT_THRESHOLD:
                direction: Literal["up", "down", "flat"] = "up"
            elif medium_mom < -_FLAT_THRESHOLD:
                direction = "down"
            else:
                direction = "flat"

            strength = _clamp(abs(medium_mom) / _STRENGTH_CAP, 0.0, 1.0)

            signals[ticker] = MomentumSignal(
                ticker=ticker,
                short_momentum=round(short_mom, 6),
                medium_momentum=round(medium_mom, 6),
                trend_direction=direction,
                signal_strength=round(strength, 4),
            )
        except Exception as exc:
            logger.warning("calculate_momentum_signals: skipping %s — %s", ticker, exc)

    return signals


async def get_news_headlines(
    ticker: str,
    session: aiohttp.ClientSession | None = None,
) -> list[str]:
    """Fetch recent news headlines for a single ticker from Alpaca Markets News API.

    Results are cached for 60 minutes per ticker. Returns an empty list on
    any failure — never raises.

    Args:
        ticker: xStock pair, e.g. "AAPLx/USD". Converted to bare symbol internally.
        session: optional aiohttp session to reuse. A new one is created (and
            closed) when not provided.

    Returns:
        Up to 10 headline strings, most recent first. Empty list on failure.
    """
    now = datetime.now(timezone.utc)
    cached = _news_cache.get(ticker)
    if cached is not None:
        cached_at, headlines = cached
        if now - cached_at < _NEWS_CACHE_TTL:
            logger.debug("get_news_headlines: cache hit for %s", ticker)
            return headlines

    symbol = _ticker_to_symbol(ticker)
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    params = {
        "symbols": symbol,
        "limit": _NEWS_PAGE_SIZE,
        "sort": "desc",
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        async with session.get(
            _ALPACA_NEWS_URL,
            headers=headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "get_news_headlines: Alpaca HTTP %d for %s", resp.status, symbol
                )
                return []
            data = await resp.json()
            headlines = [
                a["headline"] for a in data.get("news", []) if a.get("headline")
            ]
            _news_cache[ticker] = (now, headlines)
            logger.debug(
                "get_news_headlines: fetched %d headlines for %s", len(headlines), symbol
            )
            return headlines
    except Exception as exc:
        logger.warning("get_news_headlines: failed for %s — %s", symbol, exc)
        return []
    finally:
        if own_session:
            await session.close()


async def get_all_market_data(periods: int = 48) -> MarketSnapshot:
    """Fetch prices, histories, momentum signals, and headlines for all tickers.

    Execution plan:
      1. Prices + histories fetched concurrently (both hit Kraken).
      2. Momentum signals derived from histories (CPU-only, immediate).
      3. News headlines fetched concurrently sharing one aiohttp session.

    Any sub-failure yields partial data — this function never raises.

    Args:
        periods: number of 60-min candles for price history (default 48 = 2 days).

    Returns:
        MarketSnapshot with whatever data was available at call time.
    """
    prices_task = asyncio.create_task(get_current_prices())
    histories_task = asyncio.create_task(get_price_histories(periods))

    prices_result, histories_result = await asyncio.gather(
        prices_task, histories_task, return_exceptions=True
    )

    if isinstance(prices_result, BaseException):
        logger.error("get_all_market_data: prices unavailable: %s", prices_result)
        prices_result = {}
    if isinstance(histories_result, BaseException):
        logger.error("get_all_market_data: histories unavailable: %s", histories_result)
        histories_result = {}

    momentum = calculate_momentum_signals(histories_result)

    headlines: dict[str, list[str]] = {}
    async with aiohttp.ClientSession() as session:
        news_results = await asyncio.gather(
            *(get_news_headlines(t, session=session) for t in ACTIVE_TICKERS),
            return_exceptions=True,
        )
    for ticker, result in zip(ACTIVE_TICKERS, news_results):
        headlines[ticker] = result if not isinstance(result, BaseException) else []

    return MarketSnapshot(
        prices=prices_result,
        price_histories=histories_result,
        momentum_signals=momentum,
        headlines=headlines,
    )
