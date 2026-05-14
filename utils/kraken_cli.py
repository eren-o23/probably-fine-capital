"""Async wrapper around the `kraken` CLI binary.

Every CLI call follows the contract from kraken-docs/CONTEXT.md:

    kraken <command> [args...] -o json 2>/dev/null

- stdout is the only machine-data channel; stderr is diagnostics and discarded.
- exit code 0 means success; non-zero means stdout holds a JSON error envelope.
- Error routing keys on the envelope's `error` field (a stable category), never
  on `message` (human-readable, not stable).

All subprocess logic lives in `_run_kraken_command`. No other function in this
module spawns a process directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Literal

from config import KRAKEN_CLI_PATH, XSTOCK_ASSET_CLASS

logger = logging.getLogger(__name__)

_COMMAND_TIMEOUT_S: float = 30.0
_MAX_NETWORK_RETRIES: int = 3
_NETWORK_BACKOFF_BASE_S: float = 2.0      # delays: 2s, 4s, 8s
_RATE_LIMIT_WAIT_S: float = 5.0


class KrakenCLIError(Exception):
    """Raised when a kraken CLI command fails and cannot be recovered.

    Attributes:
        category: the error category from the CLI envelope (auth, api, validation,
            network, rate_limit, parse, config, io, websocket, or "unknown").
        message: the human-readable detail (do not parse it programmatically).
    """

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(f"[{category}] {message}")


async def _run_kraken_command(args: list[str]) -> dict:
    """Run `kraken <args> -o json`, route errors by category, retry per policy.

    Args:
        args: CLI arguments after the binary name and before `-o json`
            (e.g. ["ticker", "AAPLx/USD", "--asset-class", "tokenized_asset"]).

    Returns:
        The parsed JSON object from stdout on success.

    Raises:
        KrakenCLIError: on any non-recoverable failure. `rate_limit` is retried
            once after waiting; `network` is retried up to 3 times with
            exponential backoff; `auth`/`validation`/`api`/everything else raises
            immediately.
    """
    network_retries = 0
    rate_limit_retried = False

    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                KRAKEN_CLI_PATH,
                *args,
                "-o",
                "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise KrakenCLIError(
                "io",
                f"kraken CLI binary not found at '{KRAKEN_CLI_PATH}'. "
                f"Install it or set KRAKEN_CLI_PATH. ({exc})",
            ) from exc

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_COMMAND_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            # Treated as a network-class failure so the retry policy applies.
            if network_retries >= _MAX_NETWORK_RETRIES:
                raise KrakenCLIError(
                    "network", f"kraken {' '.join(args)} timed out after {_COMMAND_TIMEOUT_S}s"
                ) from exc
            network_retries += 1
            delay = _NETWORK_BACKOFF_BASE_S**network_retries
            logger.warning(
                "kraken command timed out (retry %d/%d in %.0fs): %s",
                network_retries,
                _MAX_NETWORK_RETRIES,
                delay,
                " ".join(args),
            )
            await asyncio.sleep(delay)
            continue

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()

        try:
            data: dict = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError as exc:
            raise KrakenCLIError(
                "parse", f"could not parse kraken output as JSON: {exc}; raw={stdout!r}"
            ) from exc

        if proc.returncode == 0:
            return data

        category = str(data.get("error", "unknown"))
        message = str(data.get("message", "no message"))

        if category == "rate_limit":
            if rate_limit_retried:
                logger.error("kraken rate_limit, already retried once — giving up: %s", message)
                raise KrakenCLIError(category, message)
            rate_limit_retried = True
            suggestion = data.get("suggestion", "wait and retry")
            docs_url = data.get("docs_url", "")
            logger.warning(
                "kraken rate_limit — waiting %.0fs then retrying once. suggestion: %s %s",
                _RATE_LIMIT_WAIT_S,
                suggestion,
                f"(docs: {docs_url})" if docs_url else "",
            )
            await asyncio.sleep(_RATE_LIMIT_WAIT_S)
            continue

        if category == "network":
            if network_retries >= _MAX_NETWORK_RETRIES:
                logger.error(
                    "kraken network error, exhausted %d retries — giving up: %s",
                    _MAX_NETWORK_RETRIES,
                    message,
                )
                raise KrakenCLIError(category, message)
            network_retries += 1
            delay = _NETWORK_BACKOFF_BASE_S**network_retries
            logger.warning(
                "kraken network error (retry %d/%d in %.0fs): %s",
                network_retries,
                _MAX_NETWORK_RETRIES,
                delay,
                message,
            )
            await asyncio.sleep(delay)
            continue

        if category in ("auth", "validation"):
            logger.error("kraken %s error (not retrying): %s", category, message)
            raise KrakenCLIError(category, message)

        if category == "api":
            logger.error("kraken api error — full response: %s", data)
            raise KrakenCLIError(category, message)

        # config, io, parse, websocket, unknown — log full response and raise.
        logger.error("kraken %s error: %s — full response: %s", category, message, data)
        raise KrakenCLIError(category, message)


def _extract_pair_payload(data: dict, ticker: str, skip_keys: tuple[str, ...] = ()) -> object:
    """Pull the payload for `ticker` out of a CLI response.

    Kraken sometimes normalizes the pair key in responses, so fall back to the
    first non-skipped key if an exact match is absent.
    """
    if ticker in data:
        return data[ticker]
    for key, value in data.items():
        if key in skip_keys:
            continue
        return value
    raise KrakenCLIError("parse", f"no payload for {ticker} in response: {data}")


async def test_connection() -> bool:
    """Run a public ticker call for AAPLx/USD to verify the CLI works.

    Returns:
        True if the call succeeded; False otherwise. Logs a clear status either way.
    """
    try:
        price = await get_price("AAPLx/USD")
        logger.info("Kraken CLI connection OK — AAPLx/USD last trade: $%.2f", price)
        return True
    except (KrakenCLIError, ValueError) as exc:
        logger.error("Kraken CLI connection FAILED: %s", exc)
        return False


async def get_price(ticker: str) -> float:
    """Return the last trade price for an xStock pair.

    Args:
        ticker: pair in xStock format, e.g. "AAPLx/USD".

    Returns:
        The last trade price as a float.

    Raises:
        KrakenCLIError: if the CLI call fails or the response lacks price data.
    """
    data = await _run_kraken_command(["ticker", ticker, "--asset-class", XSTOCK_ASSET_CLASS])
    payload = _extract_pair_payload(data, ticker)
    if not isinstance(payload, dict) or "c" not in payload:
        raise KrakenCLIError("parse", f"no last-trade ('c') field for {ticker}: {data}")
    try:
        return float(payload["c"][0])
    except (TypeError, ValueError, IndexError) as exc:
        raise KrakenCLIError("parse", f"malformed price for {ticker}: {payload.get('c')!r}") from exc


async def get_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch last trade prices for many pairs concurrently.

    A failure on any single ticker is logged and skipped — partial data beats
    no data. Never raises for individual ticker failures.

    Args:
        tickers: list of xStock pairs.

    Returns:
        Mapping of ticker -> price for every ticker that succeeded.
    """
    results = await asyncio.gather(*(get_price(t) for t in tickers), return_exceptions=True)
    prices: dict[str, float] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, BaseException):
            logger.warning("Skipping %s — price fetch failed: %s", ticker, result)
            continue
        prices[ticker] = result
    return prices


async def get_price_history(ticker: str, interval: int = 60, periods: int = 48) -> list[float]:
    """Fetch OHLC candles for a pair and return closing prices, oldest first.

    Args:
        ticker: xStock pair, e.g. "AAPLx/USD".
        interval: candle interval in minutes (1, 5, 15, 30, 60, 240, 1440, ...).
        periods: how many of the most recent closes to return.

    Returns:
        List of closing prices, oldest first, length <= `periods`.

    Raises:
        KrakenCLIError: if the CLI call fails or no candle data is present.
    """
    data = await _run_kraken_command(
        ["ohlc", ticker, "--interval", str(interval), "--asset-class", XSTOCK_ASSET_CLASS]
    )
    candles: list | None = None
    for key, value in data.items():
        if key == "last":
            continue
        if isinstance(value, list):
            candles = value
            break
    if candles is None:
        raise KrakenCLIError("parse", f"no OHLC candle list for {ticker}: {data}")
    try:
        closes = [float(candle[4]) for candle in candles]
    except (TypeError, ValueError, IndexError) as exc:
        raise KrakenCLIError("parse", f"malformed OHLC candles for {ticker}") from exc
    return closes[-periods:]


async def get_balance() -> dict[str, float]:
    """Fetch all cash balances.

    Returns:
        Mapping of asset code -> amount as float.

    Raises:
        KrakenCLIError: if the CLI call fails (e.g. auth) or a value isn't numeric.
    """
    data = await _run_kraken_command(["balance"])
    balances: dict[str, float] = {}
    for asset, amount in data.items():
        try:
            balances[asset] = float(amount)
        except (TypeError, ValueError):
            logger.warning("Skipping non-numeric balance entry %s=%r", asset, amount)
    return balances


async def get_open_positions() -> dict[str, dict]:
    """Fetch current open margin positions, re-keyed by trading pair.

    Returns:
        Mapping of pair -> raw position dict. Empty dict if there are none.

    Raises:
        KrakenCLIError: if the CLI call fails.
    """
    data = await _run_kraken_command(["positions", "--show-pnl"])
    positions: dict[str, dict] = {}
    for txid, pos in data.items():
        if not isinstance(pos, dict):
            continue
        pair = str(pos.get("pair", txid))
        positions[pair] = pos
    return positions


async def init_paper_account(starting_cash: float) -> bool:
    """Initialize the native Kraken paper-trading account.

    Args:
        starting_cash: starting USD balance for the paper account.

    Returns:
        True if the account was initialized; False on failure (error is logged).
    """
    try:
        data = await _run_kraken_command(["paper", "init", "--balance", str(starting_cash)])
        logger.info("Paper account initialized with $%.2f — response: %s", starting_cash, data)
        return True
    except KrakenCLIError as exc:
        logger.error("Failed to initialize paper account: %s", exc)
        return False


async def place_order(
    ticker: str,
    action: Literal["buy", "sell"],
    size_usd: float,
    current_price: float,
    paper_mode: bool = True,
) -> dict:
    """Place a market order, in paper or live mode.

    Quantity is derived as `size_usd / current_price`. In live mode the order is
    validated with `--validate` first and only executed if validation succeeds.

    Args:
        ticker: xStock pair, e.g. "AAPLx/USD".
        action: "buy" or "sell".
        size_usd: notional size in USD.
        current_price: latest price, used to convert notional -> quantity.
        paper_mode: if True, route through `kraken paper buy/sell`; if False,
            route through `kraken order buy/sell --asset-class tokenized_asset
            --type market` (with a validate step first).

    Returns:
        A dict with keys: action, ticker, quantity, size_usd, price, timestamp,
        paper_mode, success — plus `response` on success or `error` on failure.

    Raises:
        ValueError: if `current_price` or `size_usd` is not positive (caller bug,
            not a CLI failure).
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")
    if size_usd <= 0:
        raise ValueError(f"size_usd must be positive, got {size_usd}")

    quantity = size_usd / current_price
    qty_str = f"{quantity:.8f}"
    result: dict = {
        "action": action,
        "ticker": ticker,
        "quantity": quantity,
        "size_usd": size_usd,
        "price": current_price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_mode": paper_mode,
        "success": False,
    }

    try:
        if paper_mode:
            data = await _run_kraken_command(["paper", action, ticker, qty_str])
        else:
            # Step 1 — validate. If this raises, we never execute.
            await _run_kraken_command(
                [
                    "order",
                    action,
                    ticker,
                    qty_str,
                    "--asset-class",
                    XSTOCK_ASSET_CLASS,
                    "--type",
                    "market",
                    "--validate",
                ]
            )
            # Step 2 — execute.
            data = await _run_kraken_command(
                [
                    "order",
                    action,
                    ticker,
                    qty_str,
                    "--asset-class",
                    XSTOCK_ASSET_CLASS,
                    "--type",
                    "market",
                ]
            )
        result["success"] = True
        result["response"] = data
        logger.info(
            "Order placed [%s]: %s %s %s @ $%.2f (≈$%.2f)",
            "PAPER" if paper_mode else "LIVE",
            action,
            qty_str,
            ticker,
            current_price,
            size_usd,
        )
    except KrakenCLIError as exc:
        result["error"] = str(exc)
        logger.error(
            "Order FAILED [%s]: %s %s %s — %s",
            "PAPER" if paper_mode else "LIVE",
            action,
            qty_str,
            ticker,
            exc,
        )

    return result
