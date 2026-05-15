# Probably Fine Capital — Session Handoff

## Goal

Build an AI-native hedge fund that trades tokenised US stocks (xStocks) on Kraken
24/7 using a multi-agent system. Submitted to the lablab.ai x Kraken hackathon.

The final system has specialist agents (Momentum, Sentiment, Macro analysts → Risk
Manager → Portfolio Manager → Trader → Reporter) that run on a 15-minute loop.
Paper mode uses native Kraken paper commands; live mode executes real orders.

See `CLAUDE.md` for the full architecture, agent list, and build schedule.

---

## Current State — Days 1–4 Complete + Agent Prompt Rewrites

172 tests, all passing. Nothing is half-finished.

### Day 1 — Foundation

| File | What it does |
|---|---|
| `pyproject.toml` | hatchling build, deps: aiohttp, pydantic>=2.6, openai, python-dotenv |
| `core/models.py` | Pydantic v2 models: AnalystReport, RiskDecision, TradeInstruction, Position, FundState |
| `config.py` | Env loading (fail-fast on missing required vars), risk constants, LLM settings, validate_config(), X_ENABLED flag |
| `utils/kraken_cli.py` | Async subprocess wrapper for `kraken` CLI; full error routing by category; paper + live order flow |
| `core/market_data.py` | Prices, OHLC histories, pure-Python momentum signals, NewsAPI headlines with 60-min cache |
| `core/fund_state.py` | FundStateManager: price updates, stop-loss checks, position open/close, atomic JSON checkpoint |
| `utils/logger.py` | system_logger (stdout + file), log_decision() → decisions.jsonl, log_trade() → trades.jsonl |

### Day 2 — Research Desk (prompts rewritten this session)

| File | What it does |
|---|---|
| `utils/llm.py` | Shared async LLM helper: Fireworks API via openai package, JSON parse, Pydantic validation, 1 retry, returns None on failure |
| `agents/momentum_analyst.py` | MomentumAnalyst: price table + trend classification + CoT prompt. `rationale` field → AnalystReport.reasoning. Always returns. |
| `agents/sentiment_analyst.py` | SentimentAnalyst: [RECENT]/[OLDER] labels, direct_headlines primary gate, CoT prompt. Returns None or hold. |
| `agents/macro_analyst.py` | MacroAnalyst: Python-side regime/breadth/volatility, CoT prompt, market_regime validation. Returns None or hold. |
| `agents/research_desk.py` | ResearchDesk: runs all 3 analysts × 10 tickers in one asyncio.gather (30 tasks). Flattens + filters Nones. |

### Day 3 — Risk + Portfolio

| File | What it does |
|---|---|
| `agents/risk_manager.py` | RiskManager: pure Python, 6 gates in order, synchronous. Hold signals vetoed silently. Per-report try/except. |
| `agents/portfolio_manager.py` | PortfolioManager: LLM sizes each approved decision in parallel. Python enforces size bounds after LLM. |

### Day 4 — Execution + Reporting + Orchestration

| File | What it does |
|---|---|
| `agents/trader.py` | Trader: executes TradeInstructions via place_order(). Updates FundStateManager after each fill. Per-instruction try/except for isolation. |
| `agents/reporter.py` | Reporter: two LLM calls (NarrativeResponse + ThreadResponse). Writes timestamped markdown to logs/reports/. Tweet truncation at word boundary ≤280 chars. post_to_x() stubbed behind X_ENABLED. |
| `core/loop.py` | TradingLoop: 15-min cycle (market data → analysts → risk → portfolio → trader → save). Calls Reporter once per calendar day. KeyboardInterrupt-safe. |
| `main.py` | CLI entry: --paper, --once, --log-level. Pre-checks --paper BEFORE importing config. SIGINT/SIGTERM wired to loop.stop(). |

### Agent Prompt Rewrites (all completed this session)

**MomentumAnalyst** — added `_classify_trend()` (strong_uptrend / strong_downtrend /
consolidating based on last 3 closes), `_build_price_table()` (markdown table with
per-candle Change %), period high/low context, chain-of-thought instruction, confidence
calibration rules. `rationale` field → AnalystReport.reasoning; `reasoning` → DEBUG only.

**SentimentAnalyst** — added `_label_headlines()` (first `max(1, n//3)` headlines get
[RECENT], rest [OLDER]), `direct_headlines: int` as primary gate (== 0 → None before
confidence check), `rationale` field. Gate order: direct_headlines → MIN_CONFIDENCE.

**MacroAnalyst** — three Python-side helpers computed before any LLM call:
- `_classify_regime(spy_chg_pct, qqq_chg_pct)` → risk_on / risk_off / mixed (±0.3% threshold)
- `_spy_volatility_label(spy_history)` → high / normal / unknown (pop. std of hourly
  pct-changes, 1.5% threshold, requires ≥3 prices)
- `_compute_breadth(all_prices)` → (advancing, declining) via universe-median proxy

`market_regime` is `str` (not Literal) so invalid values trigger a targeted
`logger.warning` before returning None, rather than a silent Pydantic parse failure.

---

## Quick Start

```
pip install -e ".[dev]"
cp .env.example .env          # fill in FIREWORKS_API_KEY, KRAKEN_API_KEY, etc.

# smoke test
FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s NEWS_API_KEY=n \
  python -c "import config; config.validate_config()"

# full test suite (172 tests)
FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s NEWS_API_KEY=n \
  python -m pytest tests/ -q

# run paper trading loop
python main.py --paper

# run one cycle and exit (for debugging)
python main.py --paper --once
```

---

## Files Actively Being Edited

None. Days 1–4 are complete. The next session starts fresh work.

---

## What Failed or Needed Fixing

### Day 1 (carried forward)

**config.py / .env.example naming mismatch** — `.env.example` used `INITIAL_CAPITAL_USD`
but `config.py` loaded `STARTING_CASH`. Fixed by standardising on `STARTING_CASH`.

**Risk limits leaking into .env.example** — `MAX_POSITION_SIZE_PCT` and `MAX_DRAWDOWN_PCT`
appeared with different values than the Python constants. Users editing `.env` would think
they're changing limits that are never read. Both removed from `.env.example`.

**validate_config() was silent** — used `logger.info()` before any handlers configured.
Fixed with a `logging.basicConfig` guard inside `validate_config()`.

**Wrong Kraken command** — spec said `kraken open-positions`, actual CLI is
`kraken positions --show-pnl`. Fixed in kraken_cli.py.

**FIREWORKS_MODEL stale** — was `llama-v3p1-70b-instruct`, updated to
`llama-v3p3-70b-instruct` in both config.py and .env.example.

### Day 2–3 (carried forward)

**Test substring false failure** — `test_build_prompt_clips_to_12_prices` checked
`"1.00" not in prompt` but `"1.00"` is a substring of `"21.00"`. Fixed to check
`"17.00" not in prompt` (first excluded price).

**Spec field name mismatches** — corrected silently everywhere:
- `fund_state.cash_usd` → `fund_state.cash`
- `fund_state.open_positions` → `fund_state.positions`
- `report.action` → `report.signal`
- `TradeInstruction(signal=...)` → `TradeInstruction(action=...)`
- `TradeInstruction(reasoning=...)` → `TradeInstruction(rationale=...)`
- `decision.signal` → `decision.original_report.signal`
- `Position.average_cost` → `Position.entry_price`
- `Position.unrealised_pnl` → `Position.pnl_usd`

**get_portfolio_summary() doesn't exist on FundState** — spec called this method but
PortfolioManager receives `FundState`, not `FundStateManager`. Used
`FundState.to_summary_dict()` instead.

### Day 4 + rewrites (this session)

**`--paper` flag timing** — `config.PAPER_TRADING` is computed at import time.
argparse runs after import, so the flag would be silently ignored. Fixed with a
pre-check `if "--paper" in sys.argv: os.environ["PAPER_TRADING"] = "true"` at the
top of main.py before any imports.

**`utils/logger.py` does not export `get_logger()`** — used `logging.getLogger("loop")`
directly, same pattern as trader.py.

**`FundStateManager.save()` is synchronous** — spec said `await state_manager.save()`.
save() has no async I/O; removed the await.

**`FundStateManager` has no `load()` method** — loading happens in `__init__`. No
explicit call needed.

**`KrakenCLI` class doesn't exist** — the module exposes module-level async functions.
Used `place_order()` directly.

**loop.py `instructions` scoping** — `instructions = []` was inside the `if decisions:`
block, so it was undefined when decisions was empty and Reporter tried to reference it.
Fixed by initialising `instructions = []` before the decisions check.

**Test assertions outside `with` block** — in test_loop.py, assertions on mocks placed
after the `with` block closed (restoring originals), causing AttributeError. Fixed by
capturing mock references with `as` clauses and keeping assertions inside the block.

**`all_prices` is a snapshot, not per-ticker histories** — spec said "compute first vs
last price for each ticker" for breadth, but `all_prices: dict[str, float]` has one
price per ticker. Used universe-median proxy and documented it clearly.

**`_pct_change` returns a fraction, not a percentage** — `_classify_regime` expects
percentage values (2.0 for +2%). Existing helper returns fractions (0.02). Multiplied
by 100 in `_build_prompt` before passing to `_classify_regime`.

**`call_llm()` expects a Pydantic model, not a raw type** — Reporter's X thread response
is a JSON array. Wrapped as `_ThreadResponse(tweets: list[str])` to satisfy call_llm's
`response_model` parameter.

---

## Next Step — Day 5: Go Live + Cloud Deploy

### Pre-flight checks
- Run one full paper cycle manually and inspect logs/decisions.jsonl and logs/trades.jsonl
- Confirm Kraken CLI is authenticated: `kraken balances -o json`
- Confirm paper account is initialised: `kraken paper status -o json`

### Cloud deploy (pick one)
- **Railway** — push repo, set env vars in dashboard, `Procfile: web: python main.py --paper`
- **Fly.io** — `fly launch`, set secrets, deploy
- **VPS** — copy repo, `pip install -e .`, run via nohup or systemd

### What live mode needs
- Set `PAPER_TRADING=false` in environment
- Confirm Kraken account has xStock (tokenized_asset) trading enabled
- Monitor logs for first live cycle before walking away

### Day 6–7: Iterate + Submission Polish
- Review logs/reports/ to assess signal quality across a few cycles
- Tune MIN_CONFIDENCE, regime thresholds, or position sizing if signals look weak/noisy
- Write submission README for lablab.ai portal
- Record a 2–3 minute demo video showing the live loop and one executed trade
