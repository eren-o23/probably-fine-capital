# Probably Fine Capital — Claude Code Context

## What This Project Is
An AI-native hedge fund that trades tokenized US stocks 
(xStocks) on Kraken 24/7. Built for the lablab.ai x Kraken 
hackathon. Uses a multi-agent system where specialist AI 
agents research, manage risk, and execute trades autonomously.

## Stack
- Python 3.11+
- Fireworks AI API (OpenAI-compatible) for all LLM calls
- Kraken CLI for trade execution (subprocess calls)
- Pydantic v2 for all data models
- python-dotenv for environment variables
- aiohttp for HTTP requests
- asyncio throughout — all I/O is async

## Project Structure
```
probably-fine-capital/
├── agents/          # all trading agents
├── core/            # orchestration, state, market data
├── utils/           # kraken CLI wrapper, logger, prompts
├── tests/           # test files
├── logs/            # runtime logs (gitignored)
├── main.py          # entry point
├── config.py        # all constants and settings
```

## Agent Architecture
```
Market Data → Research Analysts (parallel)
           → Risk Manager
           → Portfolio Manager
           → Trader (Kraken CLI)
           → Reporter
```

### The Agents
- **MomentumAnalyst** — price trend signals
- **SentimentAnalyst** — news headline signals  
- **MacroAnalyst** — broad market signals
- **RiskManager** — gates all trades, hard Python limits
- **PortfolioManager** — sizing and allocation decisions
- **Trader** — executes via Kraken CLI, respects paper mode
- **Reporter** — hourly summaries and social media posts

## Coding Conventions
- All I/O functions are async
- All structured data uses Pydantic models
- All agent outputs are validated Pydantic objects 
  before being passed to the next agent
- LLM calls always request JSON output and parse it safely
- Hard risk limits (stop-loss, position size) are always 
  Python logic — never delegated to an LLM
- Every function has a docstring
- Every external call has error handling and one retry
- Use logging module throughout — never bare print statements
- Type hints on every function signature

## Environment Variables
```
FIREWORKS_API_KEY=        # Fireworks AI API key
KRAKEN_API_KEY=           # Kraken trading key
KRAKEN_API_SECRET=        # Kraken trading secret
PAPER_TRADING=true        # true = log only, false = live
NEWS_API_KEY=             # NewsAPI.org free tier key
```

## Hard Risk Rules (Never Override)
These are enforced in Python, never by LLM:
- MAX_POSITION_PCT = 0.20 (max 20% in one position)
- STOP_LOSS_PCT = 0.05 (sell if down 5%)
- MAX_DRAWDOWN_PCT = 0.10 (pause if portfolio down 10%)
- MIN_CONFIDENCE = 0.60 (minimum signal confidence to trade)
- MAX_OPEN_POSITIONS = 8
- MIN_TRADE_SIZE_USD = 10.0
- MAX_TRADE_SIZE_USD = 500.0

## LLM Usage Pattern
All LLM calls follow this pattern:
1. Build a focused prompt with only relevant context
2. Instruct the model to respond in JSON only
3. Parse and validate response as a Pydantic model
4. If parsing fails, log the error and return a safe default
Never pass raw LLM output to another agent unvalidated.

## Paper Mode
Controlled by PAPER_TRADING env var.
- Paper mode: log trade intentions, never call Kraken CLI
- Live mode: execute real trades via Kraken CLI
Every trade function checks this before executing.

## Tradeable Universe
xStock tickers on Kraken:
AAPLx/USD, NVDAx/USD, MSFTx/USD, TSLAx/USD, AMZNx/USD,
GOOGLx/USD, METAx/USD, AMDx/USD, SPYx/USD, QQQx/USD

## Current Build Status
- [ ] Day 1: Foundation (scaffold, models, config, 
              kraken CLI, market data, state, logger)
- [ ] Day 2: Research Desk (3 analyst agents)
- [ ] Day 3: Risk Manager + Portfolio Manager
- [ ] Day 4: Trader + Reporter
- [ ] Day 5: Go live + cloud deploy
- [ ] Day 6-7: Iterate + submission polish

## Session Rules
- Build one file at a time
- Show the complete file before moving on
- Wait for confirmation before continuing
- Always show the test command for each file
- Never skip error handling
- Ask before making assumptions about business logic

## Kraken CLI Critical Rules

### xStock Ticker Format
- Tickers use `x` suffix: AAPLx, TSLAx, NVDAx
- Pair format: AAPLx/USD, TSLAx/USD
- All xStock commands need: --asset-class tokenized_asset

### Invocation Pattern
Always: kraken <command> -o json 2>/dev/null
- stdout = machine data only
- stderr = diagnostics, ignore
- exit code 0 = success

### Paper Trading
Use native Kraken paper mode:
  kraken paper init --balance 10000 -o json
  kraken paper buy AAPLx/USD 0.1 -o json
  kraken paper status -o json

### Order Safety
Always validate before executing:
  kraken order buy AAPLx/USD 0.1 --validate -o json
  # get approval, then:
  kraken order buy AAPLx/USD 0.1 -o json

### Error Routing
Route on error.error field:
  rate_limit → backoff using suggestion field
  network    → exponential backoff, retry
  auth       → re-authenticate, stop trading
  validation → do NOT retry, fix inputs
  api        → inspect parameters

  ## Design Decisions
- RiskDecision.veto_reason is not validated at model level
  Enforcement happens in RiskManager agent logic
- to_summary_dict() omits paper_mode by design
  Add it if agents ever need to reason about live vs paper
- get_all_market_data() returns MarketSnapshot (Pydantic model),
  not a plain dict — all agents type their input accordingly
- FundState fields: cash (not cash_usd), positions 
  (not open_positions)
- AnalystReport fields: signal (not action)