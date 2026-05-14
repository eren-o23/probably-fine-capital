# Probably Fine Capital — Session Handoff

## Goal

Build an AI-native hedge fund that trades tokenized US stocks (xStocks) on Kraken
24/7 using a multi-agent system. Submitted to the lablab.ai x Kraken hackathon.

The final system has specialist agents (Momentum, Sentiment, Macro analysts → Risk
Manager → Portfolio Manager → Trader → Reporter) that run on a 15-minute loop.
Paper mode uses native Kraken paper commands; live mode executes real orders.

See `claude.md` for the full architecture, agent list, and build schedule.

---

## Current State — Day 1 Foundation Complete

All 7 Day 1 files are built and tested. Nothing is half-finished.

| File | What it does |
|---|---|
| `pyproject.toml` | hatchling build, deps: aiohttp, pydantic>=2.6, openai, python-dotenv |
| `core/models.py` | Pydantic v2 models: AnalystReport, RiskDecision, TradeInstruction, Position, FundState |
| `config.py` | Env loading (fail-fast on missing required vars), risk constants, LLM settings, validate_config() |
| `utils/kraken_cli.py` | Async subprocess wrapper for `kraken` CLI; full error routing by category; paper + live order flow |
| `core/market_data.py` | Prices, OHLC histories, pure-Python momentum signals, NewsAPI headlines with 60-min cache |
| `core/fund_state.py` | FundStateManager: price updates, stop-loss checks, position open/close, atomic JSON checkpoint |
| `utils/logger.py` | system_logger (stdout + file), log_decision() → decisions.jsonl, log_trade() → trades.jsonl |

Install deps: `pip install -e ".[dev]"`  
Copy env: `cp .env.example .env` and fill in keys  
Smoke test: `FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s NEWS_API_KEY=n python -c "import config; config.validate_config()"`

---

## Files Actively Being Edited

None. Day 1 is complete. The next session starts fresh files.

---

## What Failed or Needed Fixing

**config.py / .env.example naming mismatch** — `.env.example` used `INITIAL_CAPITAL_USD`
but `config.py` loaded `STARTING_CASH`. Fixed by standardising on `STARTING_CASH`.

**Risk limits leaking into .env.example** — `MAX_POSITION_SIZE_PCT` and `MAX_DRAWDOWN_PCT`
appeared in `.env.example` with different values than the Python constants (0.10 vs 0.15,
0.20 vs 0.10). This is dangerous: users editing `.env` would think they're changing limits
but the values are never read. Both removed from `.env.example`.

**validate_config() was silent** — the startup summary used `logger.info()` which is silent
if no handlers are configured yet (the common case at import time). Fixed by adding a
`logging.basicConfig` guard inside `validate_config()` that only fires if the root logger
has no handlers.

**Wrong Kraken command** — the spec said `kraken open-positions` but the actual CLI command
is `kraken positions --show-pnl` (confirmed in tool-catalog.json). Fixed in kraken_cli.py.

**KRAKEN_CLI_PATH default wrong** — `.env.example` had `kraken-cli` (the installer package
name) but the binary is named `kraken`. Fixed to match config.py default.

**FIREWORKS_MODEL stale** — was `llama-v3p1-70b-instruct`, updated to `llama-v3p3-70b-instruct`
in both config.py and .env.example.

---

## Next Step — Day 2: Research Desk

Build three analyst agents in `agents/`. Each takes a `MarketSnapshot` and returns an
`AnalystReport`. All follow the LLM usage pattern in `claude.md`:
build prompt → JSON-only response → validate as Pydantic model → return or safe default.

**agents/momentum_analyst.py**
- Input: `MomentumSignal` from `market_data.calculate_momentum_signals()`
- The signal is already computed; the LLM's job is to reason about it and assign confidence
- Prompt should include: ticker, short/medium momentum, trend direction, price history snippet

**agents/sentiment_analyst.py**
- Input: headlines list from `market_data.get_news_headlines()`
- Returns buy/sell/hold with confidence based on headline sentiment
- Prompt should include: ticker, 5 most recent headlines, current price for context

**agents/macro_analyst.py**
- Input: SPYx/USD and QQQx/USD signals (broad market proxies already in TRADEABLE_TICKERS)
- Returns a market-wide bias that influences all other signals

**Shared LLM call helper** — before building the three agents, write `utils/llm.py`:
a single `async call_llm(prompt: str, response_model: type[BaseModel]) -> BaseModel` that
handles the Fireworks API call, JSON parsing, Pydantic validation, one retry on parse
failure, and the safe-default return. All three analysts will use it.

**Run all three in parallel** via `asyncio.gather` in an orchestration function
`agents/research_desk.py` → returns `list[AnalystReport]`.

The Fireworks client uses the `openai` package pointed at `FIREWORKS_BASE_URL` with
`FIREWORKS_API_KEY`. See `config.py` for the exact values.
