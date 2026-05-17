"""Central configuration for Probably Fine Capital.

All constants, env-loaded secrets, and risk limits live here.
Import this module anywhere — it loads .env once on first import
and raises immediately if required variables are missing.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. ENV LOADING — fail fast on missing required vars
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    """Return the value of a required environment variable or raise clearly."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in all required values."
        )
    return value


def _optional(name: str, default: str) -> str:
    """Return the value of an optional environment variable, or its default."""
    return os.getenv(name, default)


# Secrets — loaded from .env, never hardcoded
FIREWORKS_API_KEY: str = _require("FIREWORKS_API_KEY")
KRAKEN_API_KEY: str = _require("KRAKEN_API_KEY")
KRAKEN_API_SECRET: str = _require("KRAKEN_API_SECRET")
ALPACA_API_KEY: str = _require("ALPACA_API_KEY")
ALPACA_API_SECRET: str = _require("ALPACA_API_SECRET")

# Optional with defaults
_paper_raw: str = _optional("PAPER_TRADING", "true")
PAPER_TRADING: bool = _paper_raw.strip().lower() in ("true", "1", "yes")

STARTING_CASH: float = float(_optional("STARTING_CASH", "10000.0"))
FUND_NAME: str = _optional("FUND_NAME", "Probably Fine Capital")
LOG_LEVEL: str = _optional("LOG_LEVEL", "INFO").upper()
LOG_FILE: str = _optional("LOG_FILE", "logs/fund.log")

# ---------------------------------------------------------------------------
# 2. TRADING UNIVERSE
# ---------------------------------------------------------------------------

TRADEABLE_TICKERS: list[str] = [
    "AAPLx/USD",
    "NVDAx/USD",
    "MSFTx/USD",
    "TSLAx/USD",
    "AMZNx/USD",
    "GOOGLx/USD",
    "METAx/USD",
    "AMDx/USD",
    "SPYx/USD",
    "QQQx/USD",
    "JPMx/USD",
    "GLDx/USD",
    "SGOVx/USD",
    "JNJx/USD",
    "PGx/USD",
    "COINx/USD",
    "PLTRx/USD",
    "LLYx/USD",
]

ACTIVE_TICKERS: list[str] = TRADEABLE_TICKERS[:6]

XSTOCK_ASSET_CLASS: str = "tokenized_asset"

# ---------------------------------------------------------------------------
# 3. HARD RISK LIMITS — enforced in Python, never delegated to an LLM
# ---------------------------------------------------------------------------

MAX_POSITION_PCT: float = 0.20      # max 20% of portfolio in one position
STOP_LOSS_PCT: float = 0.05         # sell if position is down 5%
MAX_DRAWDOWN_PCT: float = 0.10      # pause all trading if portfolio is down 10%
MIN_CONFIDENCE: float = 0.60        # minimum analyst confidence to consider a trade
MAX_OPEN_POSITIONS: int = 8         # maximum simultaneous open positions
MIN_TRADE_SIZE_USD: float = 10.0    # smallest allowed trade
MAX_TRADE_SIZE_USD: float = 500.0   # largest allowed trade

# ---------------------------------------------------------------------------
# 4. LOOP SETTINGS
# ---------------------------------------------------------------------------

LOOP_INTERVAL_MINS: int = 15        # how often the main trading loop runs
REPORT_INTERVAL_MINS: int = 60      # how often the reporter agent fires

# ---------------------------------------------------------------------------
# 5. LLM SETTINGS
# ---------------------------------------------------------------------------

FIREWORKS_BASE_URL: str = _optional(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_MODEL: str = _optional(
    "FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6"
)
LLM_MAX_TOKENS: int = 2048
LLM_TEMPERATURE: float = 0.1

# ---------------------------------------------------------------------------
# 6. KRAKEN SETTINGS
# ---------------------------------------------------------------------------

KRAKEN_CLI_PATH: str = _optional("KRAKEN_CLI_PATH", "kraken")

# ---------------------------------------------------------------------------
# 7. SOCIAL POSTING — optional X (Twitter) integration
# ---------------------------------------------------------------------------

X_ENABLED: bool = _optional("X_ENABLED", "false").strip().lower() == "true"

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Validate all config values are within sensible ranges and log a startup summary.

    Raises ValueError if any value is out of range.
    Call this once at application startup before entering the trading loop.
    """
    errors: list[str] = []

    if STARTING_CASH <= 0:
        errors.append(f"STARTING_CASH must be positive, got {STARTING_CASH}")

    if not (0 < MAX_POSITION_PCT <= 1.0):
        errors.append(f"MAX_POSITION_PCT must be in (0, 1], got {MAX_POSITION_PCT}")

    if not (0 < STOP_LOSS_PCT < 1.0):
        errors.append(f"STOP_LOSS_PCT must be in (0, 1), got {STOP_LOSS_PCT}")

    if not (0 < MAX_DRAWDOWN_PCT < 1.0):
        errors.append(f"MAX_DRAWDOWN_PCT must be in (0, 1), got {MAX_DRAWDOWN_PCT}")

    if not (0 < MIN_CONFIDENCE <= 1.0):
        errors.append(f"MIN_CONFIDENCE must be in (0, 1], got {MIN_CONFIDENCE}")

    if MAX_OPEN_POSITIONS < 1:
        errors.append(f"MAX_OPEN_POSITIONS must be >= 1, got {MAX_OPEN_POSITIONS}")

    if MIN_TRADE_SIZE_USD <= 0:
        errors.append(f"MIN_TRADE_SIZE_USD must be positive, got {MIN_TRADE_SIZE_USD}")

    if MAX_TRADE_SIZE_USD <= MIN_TRADE_SIZE_USD:
        errors.append(
            f"MAX_TRADE_SIZE_USD ({MAX_TRADE_SIZE_USD}) must exceed "
            f"MIN_TRADE_SIZE_USD ({MIN_TRADE_SIZE_USD})"
        )

    if not TRADEABLE_TICKERS:
        errors.append("TRADEABLE_TICKERS must not be empty")

    if LLM_TEMPERATURE < 0 or LLM_TEMPERATURE > 2.0:
        errors.append(f"LLM_TEMPERATURE must be in [0, 2], got {LLM_TEMPERATURE}")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    # Ensure the summary is visible even when called before logging is configured.
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    mode = "PAPER" if PAPER_TRADING else "LIVE"
    logger.info("=" * 50)
    logger.info("Probably Fine Capital — config validated")
    logger.info("  Mode          : %s", mode)
    logger.info("  Starting cash : $%.2f", STARTING_CASH)
    logger.info("  Tickers       : %d (%s)", len(TRADEABLE_TICKERS), ", ".join(TRADEABLE_TICKERS))
    logger.info("  LLM model     : %s", FIREWORKS_MODEL)
    logger.info("=" * 50)
