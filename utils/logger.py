"""Structured logging for Probably Fine Capital.

Three destinations:
  log_decision(...)  → logs/decisions.jsonl   (one JSON object per line)
  log_trade(...)     → logs/trades.jsonl      (one JSON object per line)
  system_logger      → stdout + logs/fund.log (human-readable, LOG_LEVEL from config)

The logs/ directory is created at import time.
JSONL handles are opened once and kept open; each write is flushed immediately.
All timestamps are UTC ISO 8601.
No function in this module raises — write failures are reported to stderr only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import LOG_FILE, LOG_LEVEL

# ---------------------------------------------------------------------------
# Directory + file handles — created once at import
# ---------------------------------------------------------------------------

_LOGS_DIR = Path("logs")
_DECISIONS_PATH = _LOGS_DIR / "decisions.jsonl"
_TRADES_PATH = _LOGS_DIR / "trades.jsonl"

_LOGS_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

_decisions_fh = _DECISIONS_PATH.open("a", encoding="utf-8", buffering=1)
_trades_fh = _TRADES_PATH.open("a", encoding="utf-8", buffering=1)


# ---------------------------------------------------------------------------
# system_logger — Python Logger, UTC timestamps, two handlers
# ---------------------------------------------------------------------------

class _UTCFormatter(logging.Formatter):
    """Logging formatter that forces UTC in asctime."""
    converter = time.gmtime


def _build_system_logger() -> logging.Logger:
    """Create and configure the system logger (idempotent)."""
    lg = logging.getLogger("pfc.system")

    if lg.handlers:
        return lg

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    lg.setLevel(level)

    fmt = _UTCFormatter(
        fmt="[%(asctime)s UTC] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    lg.addHandler(stdout_handler)
    lg.addHandler(file_handler)
    lg.propagate = False

    return lg


system_logger: logging.Logger = _build_system_logger()


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def log_decision(
    agent: str,
    ticker: str,
    action: str,
    confidence: float,
    reasoning: str,
) -> None:
    """Append one decision record to logs/decisions.jsonl.

    Args:
        agent:      name of the analyst or agent making the decision.
        ticker:     xStock pair, e.g. "AAPLx/USD".
        action:     "buy", "sell", or "hold".
        confidence: analyst confidence score, 0.0–1.0.
        reasoning:  free-text explanation from the agent.
    """
    try:
        entry = {
            "timestamp": _utcnow(),
            "agent": agent,
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        _decisions_fh.write(json.dumps(entry) + "\n")
        _decisions_fh.flush()
    except Exception as exc:
        print(f"log_decision failed: {exc}", file=sys.stderr)


def log_trade(
    ticker: str,
    side: str,
    size_usd: float,
    price: float,
    order_id: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Append one trade record to logs/trades.jsonl.

    Args:
        ticker:   xStock pair, e.g. "AAPLx/USD".
        side:     "buy" or "sell".
        size_usd: notional trade size in USD.
        price:    execution price.
        order_id: Kraken order ID or paper order ID.
        status:   "filled", "rejected", "paper", etc.
        error:    error string if the trade failed, else None.
    """
    try:
        entry = {
            "timestamp": _utcnow(),
            "ticker": ticker,
            "side": side,
            "size_usd": size_usd,
            "price": price,
            "order_id": order_id,
            "status": status,
            "error": error,
        }
        _trades_fh.write(json.dumps(entry) + "\n")
        _trades_fh.flush()
    except Exception as exc:
        print(f"log_trade failed: {exc}", file=sys.stderr)
