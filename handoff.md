# Probably Fine Capital — Session Handoff

## Goal

Build an AI-native hedge fund that trades tokenised US stocks (xStocks) on Kraken
24/7 using a multi-agent system. Submitted to the lablab.ai x Kraken hackathon.

The final system has specialist agents (Momentum, Sentiment, Macro analysts → Risk
Manager → Portfolio Manager → Trader → Reporter) that run on a 15-minute loop.
Paper mode uses native Kraken paper commands; live mode executes real orders.

See `CLAUDE.md` for the full architecture, agent list, and build schedule.

---

## Current State — Days 1–4 Complete + Post-Build Polish

180 tests, all passing. Nothing is half-finished.

### Day 1 — Foundation

| File | What it does |
|---|---|
| `pyproject.toml` | hatchling build, deps: aiohttp, pydantic>=2.6, openai, python-dotenv, tweepy>=4.0 |
| `core/models.py` | Pydantic v2 models: AnalystReport, RiskDecision, TradeInstruction, Position, FundState |
| `config.py` | Env loading (fail-fast on missing required vars), risk constants, LLM settings, validate_config(), X_ENABLED flag |
| `utils/kraken_cli.py` | Async subprocess wrapper for `kraken` CLI; full error routing by category; paper + live order flow |
| `core/market_data.py` | Prices, OHLC histories, pure-Python momentum signals, Alpaca News API headlines with 60-min cache |
| `core/fund_state.py` | FundStateManager: price updates, stop-loss checks, position open/close, atomic JSON checkpoint |
| `utils/logger.py` | system_logger (stdout + file), log_decision() → decisions.jsonl, log_trade() → trades.jsonl |

### Day 2 — Research Desk

| File | What it does |
|---|---|
| `utils/llm.py` | Shared async LLM helper: Fireworks API via openai package, JSON parse, Pydantic validation, 3 attempts, semaphore(1) + 7s sleep for rate limiting, 429 retry with backoff |
| `agents/momentum_analyst.py` | MomentumAnalyst: price table + trend classification + CoT prompt |
| `agents/sentiment_analyst.py` | SentimentAnalyst: [RECENT]/[OLDER] labels, direct_headlines primary gate, CoT prompt |
| `agents/macro_analyst.py` | MacroAnalyst: Python-side regime/breadth/volatility, CoT prompt |
| `agents/research_desk.py` | ResearchDesk: runs all 3 analysts × ACTIVE_TICKERS in one asyncio.gather. Flattens + filters Nones. |

### Day 3 — Risk + Portfolio

| File | What it does |
|---|---|
| `agents/risk_manager.py` | RiskManager: pure Python, 6 gates in order, synchronous. Hold signals vetoed silently. Per-report try/except. |
| `agents/portfolio_manager.py` | PortfolioManager: LLM sizes each approved decision in parallel. Python enforces size bounds after LLM. |

### Day 4 — Execution + Reporting + Orchestration

| File | What it does |
|---|---|
| `agents/trader.py` | Trader: executes TradeInstructions via place_order(). Updates FundStateManager after each fill. Per-instruction try/except for isolation. |
| `agents/reporter.py` | Reporter: two LLM calls (NarrativeResponse + ThreadResponse). Writes timestamped markdown to logs/reports/. post_to_x() fully wired via tweepy. run() returns (filepath, tweets) tuple. |
| `core/loop.py` | TradingLoop: 15-min cycle (market data → analysts → risk → portfolio → trader → save). Calls Reporter once per calendar day, then post_to_x() if tweets returned. KeyboardInterrupt-safe. |
| `main.py` | CLI entry: --paper, --once, --log-level. Pre-checks --paper BEFORE importing config. SIGINT/SIGTERM wired to loop.stop(). |

### Post-Build Polish (this session)

**Ticker universe** — TRADEABLE_TICKERS expanded to 18: added JPMx, GLDx, SGOVx,
JNJx, PGx, COINx, PLTRx, LLYx. ACTIVE_TICKERS = TRADEABLE_TICKERS[:6] (first 6)
is the live trading subset used by market_data and research_desk. TRADEABLE_TICKERS
is retained as the full defined universe.

**X / Twitter posting wired** — `post_to_x()` now uses tweepy.Client with
OAuth 1.0a (consumer_key/secret + access_token/secret read from env at call time).
Credentials checked before instantiating client. Retry thread posts as reply chain
using in_reply_to_tweet_id. `loop.py` calls `post_to_x(tweets)` after each daily
report if tweets is not None.

**reporter.run() return type** — changed from `str` to `tuple[str, list[str] | None]`
so the loop can pass tweets directly to post_to_x() without re-running the LLM.

**Rate limiting** — `_LLM_SEMAPHORE = asyncio.Semaphore(1)` + `await asyncio.sleep(7.0)`
inside the semaphore block after each successful HTTP call. With ACTIVE_TICKERS=6
and 2 analysts (momentum + macro) = 12 calls × ~8s = ~96s per cycle, well within
the 15-minute window and safely under 10 RPM.

**429 retry** — `call_llm()` retries up to 3 attempts total. On RateLimitError:
waits 2^(attempt+1) seconds (2s, 4s) then retries. Non-429 errors fail immediately.
JSON/validation errors also retry up to 3 times.

**News provider** — replaced NewsAPI.org with Alpaca Markets News API. Endpoint:
`GET https://data.alpaca.markets/v1beta1/news`. Auth via request headers
(APCA-API-KEY-ID, APCA-API-SECRET-KEY). Response key changed from `articles[].title`
to `news[].headline`. Page size increased from 5 → 10. Config keys renamed from
NEWS_API_KEY to ALPACA_API_KEY + ALPACA_API_SECRET.

**Tweet 5 sign-off** — updated to tag @krakenfx @lablabai @Surgexyz_ with
#ProbablyFineCapital #xStocks.

---

## Quick Start

```
pip install -e ".[dev]"
cp .env.example .env    # fill in keys (see env vars below)

# smoke test
FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s \
  ALPACA_API_KEY=ak ALPACA_API_SECRET=as \
  python -c "import config; config.validate_config()"

# full test suite (180 tests)
FIREWORKS_API_KEY=fw KRAKEN_API_KEY=k KRAKEN_API_SECRET=s \
  ALPACA_API_KEY=ak ALPACA_API_SECRET=as \
  python -m pytest tests/ -q

# run paper trading loop
python main.py --paper

# run one cycle and exit (for debugging)
python main.py --paper --once
```

### Required env vars

```
FIREWORKS_API_KEY=       # Fireworks AI LLM key
KRAKEN_API_KEY=          # Kraken trading key
KRAKEN_API_SECRET=       # Kraken trading secret
ALPACA_API_KEY=          # Alpaca Markets paper account key
ALPACA_API_SECRET=       # Alpaca Markets paper account secret
```

### Optional env vars (for X posting)

```
X_ENABLED=true
X_CONSUMER_KEY=
X_CONSUMER_SECRET=
X_ACCESS_TOKEN=
X_ACCESS_TOKEN_SECRET=
```

---

## Files Actively Being Edited

None. All changes this session are complete and tested.

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

### Day 4 + rewrites (carried forward)

**`--paper` flag timing** — `config.PAPER_TRADING` is computed at import time.
argparse runs after import, so the flag would be silently ignored. Fixed with a
pre-check `if "--paper" in sys.argv: os.environ["PAPER_TRADING"] = "true"` at the
top of main.py before any imports.

**loop.py `instructions` scoping** — `instructions = []` was inside the `if decisions:`
block, so it was undefined when decisions was empty and Reporter tried to reference it.
Fixed by initialising `instructions = []` before the decisions check.

**`call_llm()` expects a Pydantic model, not a raw type** — Reporter's X thread response
is a JSON array. Wrapped as `_ThreadResponse(tweets: list[str])` to satisfy call_llm's
`response_model` parameter.

### This session

**Stub test broken by tweepy wiring** — `test_post_to_x_returns_false_when_enabled_but_stubbed`
was written to assert the old TODO stub always returned False. After wiring tweepy,
the method started returning True (real credentials present in .env). Deleted — the
behaviour it tested no longer exists; `test_reporter_x.py` covers the live path.

**reporter.run() return type mismatch** — previous session left a partial edit where
the early return (`narrative is None` path) had been updated to `return str(filepath), None`
but the normal return was still `return str(filepath)`. Fixed the final return to
`return str(filepath), tweets`.

**research_desk tests counted against TRADEABLE_TICKERS after ACTIVE_TICKERS split** —
four tests in `test_research_desk.py` used `len(config.TRADEABLE_TICKERS)` as multiplier
after research_desk was changed to iterate ACTIVE_TICKERS (6). Updated to
`len(config.ACTIVE_TICKERS)`.

**momentum_analyst retry count test** — `test_analyze_retries_on_bad_json_then_holds`
asserted `call_count == 2` (old range(2)). After bumping to 3 attempts, updated to
`call_count == 3`.

---

## Next Step — Day 5: Go Live + Cloud Deploy

### Pre-flight checks
- Run one full paper cycle manually: `python main.py --paper --once`
- Inspect `logs/decisions.jsonl` and `logs/trades.jsonl` after the cycle
- Confirm Kraken CLI is authenticated: `kraken balances -o json`
- Confirm paper account is initialised: `kraken paper status -o json`
- Confirm Alpaca keys work: `curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" "https://data.alpaca.markets/v1beta1/news?symbols=AAPL&limit=3"`

### Cloud deploy (pick one)
- **Railway** — push repo, set env vars in dashboard, `Procfile: web: python main.py --paper`
- **Fly.io** — `fly launch`, set secrets, deploy
- **VPS** — copy repo, `pip install -e .`, run via nohup or systemd

### What live mode needs
- Set `PAPER_TRADING=false` in environment
- Confirm Kraken account has xStock (tokenized_asset) trading enabled
- Monitor logs for first live cycle before walking away

### Day 6–7: Iterate + Submission Polish
- Review `logs/reports/` to assess signal quality across a few cycles
- Consider expanding ACTIVE_TICKERS beyond 6 once the rate-limit budget is confirmed
- Tune MIN_CONFIDENCE, regime thresholds, or position sizing if signals look weak/noisy
- Write submission README for lablab.ai portal
- Record a 2–3 minute demo video showing the live loop and one executed trade
