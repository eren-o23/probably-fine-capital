"""Entry point for Probably Fine Capital.

Usage:
  python main.py                    # mode and keys from .env
  python main.py --paper            # force paper mode regardless of .env
  python main.py --once             # run one cycle then exit (smoke test)
  python main.py --log-level DEBUG  # verbose output
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

# --paper must set the env var before `import config` runs, because
# config.PAPER_TRADING is computed from os.environ at import time.
if "--paper" in sys.argv:
    os.environ["PAPER_TRADING"] = "true"

import config
from core.loop import TradingLoop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="probably-fine-capital",
        description="AI-native hedge fund trading tokenised xStocks on Kraken 24/7.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper-trading mode (overrides PAPER_TRADING in .env).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single trading cycle then exit (smoke test).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        metavar="LEVEL",
        help="Root log level: DEBUG | INFO | WARNING  (default: INFO).",
    )
    return parser.parse_args()


def _print_banner() -> None:
    mode = "PAPER" if config.PAPER_TRADING else "LIVE"
    tickers = "  ".join(config.TRADEABLE_TICKERS)
    print("=" * 52)
    print("  Probably Fine Capital")
    print("=" * 52)
    print(f"  Mode          : {mode}")
    print(f"  Starting cash : ${config.STARTING_CASH:,.2f}")
    print(f"  Universe      : {tickers}")
    print("=" * 52)
    print()


def main() -> None:
    args = _parse_args()

    # Redundant but explicit: ensure the env var is set for anything that
    # reads it after this point (e.g. FundStateManager default arg).
    if args.paper:
        os.environ["PAPER_TRADING"] = "true"

    # Raises immediately on missing required vars or bad risk constants.
    config.validate_config()

    # Configure the root logger so module-level loggers (agents, core.*)
    # that propagate to root pick up the chosen level.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s UTC] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _print_banner()

    loop = TradingLoop()

    # SIGINT and SIGTERM both request a clean shutdown via stop().
    # Using signal.signal() (not asyncio) so it works whether or not
    # an event loop is running at signal delivery time.
    def _handle_signal(signum: int, frame: object) -> None:
        loop.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.once:
        asyncio.run(loop._run_cycle())
    else:
        asyncio.run(loop.run())


if __name__ == "__main__":
    main()
