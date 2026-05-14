# Probably Fine Capital — Session Handoff

## Goal

Build an AI-native hedge fund that trades tokenised US stocks (xStocks) on Kraken
24/7 using a multi-agent system. Submitted to the lablab.ai x Kraken hackathon.

The final system has specialist agents (Momentum, Sentiment, Macro analysts → Risk
Manager → Portfolio Manager → Trader → Reporter) that run on a 15-minute loop.
Paper mode uses native Kraken paper commands; live mode executes real orders.

See `CLAUDE.md` for the full architecture, agent list, and build schedule.

---

## Current State — Day 1 + Day 2 + Day 3 Complete

67 tests, all passing. Nothing is half-finished.

### Day 1 — Foundation

| File | What it does |
|---|---|
| `pyproject.toml` | hatchling build, deps: aiohttp, pydantic>=2.6, openai, python-dotenv |
| `core/models.py` | Pydantic v2 models: AnalystReport, RiskDecision, TradeInstruction, Position, FundState |
| `config.py` | Env loading (fail-fast on missing required vars), risk constants, LLM settings, validate_config() |
| `utils/kraken_cli.py` | Async subprocess wrapper for `kraken` CLI; full error routing by category; paper + live order flow |
| `core/market_data.py` | Prices, OHLC histories, pure-Python momentum signals, NewsAPI headlines with 60-min cache |
| `core/fund_state.py` | FundStateManager: price updates, stop-loss checks, position open/close, atomic JSON checkpoint |
| `utils/logger.py` | system_logger (stdout + file), log_decision() → decisions.jsonl, log_trade() → trades.jsonl |

### Day 2 — Research Desk

| File | What it does |
|---|---|
| `utils/llm.py` | Shared async LLM helper: Fireworks API via openai package, JSON parse, Pydantic validation, 1 retry, returns None on failure |
| `agents/momentum_analyst.py` | MomentumAnalyst: takes MomentumSignal + price history → AnalystReport. Always returns (safe hold on failure). |
| `agents/sentiment_analyst.py` | SentimentAnalyst: takes headlines → AnalystReport or None. Empty headlines → hold at 0.0. Filters below MIN_CONFIDENCE → None. |
| `agents/macro_analyst.py` | MacroAnalyst: takes SPY/QQQ histories + universe prices → AnalystReport or None. Filters below MIN_CONFIDENCE → None. |
| `agents/research_desk.py` | ResearchDesk: runs all 3 analysts × 10 tickers in one asyncio.gather (30 tasks). Flattens + filters Nones. Logs summary. |

### Day 3 — Risk + Portfolio

| File | What it does |
|---|---|
| `agents/risk_manager.py` | RiskManager: pure Python, 6 gates in order, synchronous. Hold signals vetoed silently. Per-report try/except. |
| `agents/portfolio_manager.py` | PortfolioManager: LLM sizes each approved decision in parallel. Python enforces size bounds after LLM. |

Install deps: `pip install -e ".[dev]"`
Copy env: `cp .env.example .env` and fill in keys
Smoke test: `FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s NEWS_API_KEY=n python -c "import config; config.validate_config()"`
Run all tests: `FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s NEWS_API_KEY=n python -m pytest tests/ -q`

---

## Files Actively Being Edited

None. Day 1–3 are complete. The next session starts fresh files.

---

## What Failed or Needed Fixing

### Day 1 (carried forward)

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

### Day 2–3

**Test substring false failure** — `test_build_prompt_clips_to_12_prices` checked
`"1.00" not in prompt` but `"1.00"` is a substring of `"21.00"`. Fixed to check
`"17.00" not in prompt` (the first excluded price) and `"18.00" in prompt`.

**Spec field name mismatches** — the original specs used informal names that didn't match
actual Pydantic model fields. Corrected silently in each file:
- `fund_state.cash_usd` → `fund_state.cash`
- `fund_state.open_positions` → `fund_state.positions`
- `report.action` → `report.signal` (in risk_manager context)
- `TradeInstruction(signal=...)` → `TradeInstruction(action=...)`
- `TradeInstruction(reasoning=...)` → `TradeInstruction(rationale=...)`
- `decision.signal` → `decision.original_report.signal`

**get_portfolio_summary() lives on FundStateManager, not FundState** — the spec said
"call get_portfolio_summary()" but PortfolioManager receives a `FundState`, not a
`FundStateManager`. Used `FundState.to_summary_dict()` instead, which was built for
exactly this purpose.

**SentimentAnalyst and MacroAnalyst return None below MIN_CONFIDENCE** — MomentumAnalyst
always returns an AnalystReport (safe hold on failure). Sentiment and Macro return
`Optional[AnalystReport]` because they apply the MIN_CONFIDENCE filter before returning.
This asymmetry is intentional: momentum always has something to say (the signal is
pre-computed); sentiment and macro may legitimately have nothing actionable.

---

## Next Step — Day 4: Trader + Reporter

### agents/trader.py
- Input: `list[TradeInstruction]`, `FundStateManager`
- For each instruction: check paper mode from config
- Paper mode: call `kraken paper buy/sell` via kraken_cli, log via log_trade()
- Live mode: call `kraken order buy/sell --validate` first, then execute if valid
- Update FundStateManager after each fill (add_position / close_position)
- Return list of executed ticker strings

### agents/reporter.py
- Input: `FundState`, `list[TradeInstruction]`, elapsed time
- LLM agent: generates a human-readable hourly summary
- Writes to logs/reports/ as timestamped markdown files
- Optionally posts to a configured webhook (X / Slack)
- Should use call_llm() from utils/llm.py

### After trader + reporter: main.py
Wire everything together in a 15-minute async loop:
1. `get_all_market_data()`
2. `ResearchDesk.analyze(snapshot)`
3. `RiskManager.evaluate(reports, fund_state)`
4. `PortfolioManager.allocate(decisions, fund_state)`
5. `Trader.execute(instructions, fund_state_manager)`
6. Every 4th loop: `Reporter.run(fund_state, trades)`
7. `FundStateManager.save()`
8. Sleep until next interval
