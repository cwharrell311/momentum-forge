# CLAUDE.md — Instructions for Claude Code

## Project Overview

Vicuna (formerly Confluence Engine) is a personal trading platform that identifies high-probability trade setups by detecting when multiple independent signal layers align on the same ticker. It is NOT a toy project — this will be used to trade real money.

## Who Is Building This

Chris — a financial controller / accountant with 20 years of finance experience. He understands markets, options, and financial data deeply but is learning to code via "vibe coding" with Claude. Explain technical decisions clearly. Don't assume deep software engineering knowledge, but don't dumb down the finance or trading concepts. Chris prefers step-by-step explanations built in large chunks.

**IMPORTANT: Update this CLAUDE.md file at the end of every session with current state.**

## Tech Stack

- **Backend:** Python 3.12+, FastAPI, SQLAlchemy (async), Pydantic v2
- **Frontend:** Single HTML dashboard (vanilla JS + Tailwind CDN) served from FastAPI — rebranded to "Vicuna"
- **Database:** PostgreSQL 16+ (via Docker), Redis 7+ (via Docker)
- **Task runner:** APScheduler for periodic scans
- **Broker:** Alpaca API (paper trading first, then live)
- **Data sources:** Alpaca (momentum bars), FMP API (VIX, insider), Unusual Whales API (options flow, GEX, vol surface, dark pool, short interest)
- **Containerization:** Docker Compose for full stack (PostgreSQL + Redis + API)

## Current State (v0.3.3 — Feb 13, 2025)

**All 8 signal layers are fully implemented:**

| # | Layer | Source | Weight | Status |
|---|-------|--------|--------|--------|
| 1 | Options Flow | Unusual Whales | 0.25 | ACTIVE (requires UW key) |
| 2 | GEX / Dealer Positioning | Unusual Whales | 0.18 | ACTIVE (requires UW key) |
| 3 | Dark Pool Activity | Unusual Whales | 0.15 | ACTIVE (requires UW key) |
| 4 | Volatility Surface | Unusual Whales | 0.12 | ACTIVE (requires UW key) |
| 5 | Momentum / Technical | Alpaca | 0.12 | ACTIVE (free) |
| 6 | Insider Cluster Buying | FMP / UW | 0.10 | ACTIVE (free) |
| 7 | Short Interest | Unusual Whales | 0.08 | ACTIVE (requires UW key) |
| 8 | VIX Regime Filter | FMP | modifier | ACTIVE (free, not scored) |

**Infrastructure:**
- Dashboard: 5 tabs (Screener, Journal, Performance, Broker, System) — Watchlist tab removed
- Flow gate: options_flow MUST agree with direction + GEX or volatility confirm before trade-worthy
- Caching: In-memory result cache (zero API calls per page load)
- Scheduler: 15-min scan interval (configurable via SCAN_INTERVAL) + 2-hour signal grading job
- Auto-journal: logs qualifying setups (60%+ conviction, 3+ layers) automatically
- Signal tracker: forward-tests every qualifying signal (40%+ conviction, 2+ layers) against actual price outcomes
- Trade journal: CRUD with P&L tracking
- Broker: Alpaca paper trading integration (account, positions, orders)
- System: FMP quota tracking, signal layer status panel
- DB: Auto-creates tables on startup, graceful fallback if PostgreSQL not running
- Universe: 35 default tickers in watchlist.yaml (mega-cap, high-momentum, blue chip, sector ETFs, memes)
- Index filtering: SPXW, SPX, VIX etc. filtered from universe discovery
- Direction-aware UI: green=bullish, red=bearish, yellow=dark pool conflict

**Key changes from Feb 13 session (v0.3.3):**
- Built signal forward-testing scorecard system — answers "Are these signals actually predictive?"
- New `signal_history` table records every qualifying signal with price at fire time
- `signal_tracker.py` service: records signals after each scan, grades past signals against T+1/5/10/20 outcomes
- Grading job runs every 2 hours, looks up actual Alpaca closing prices and computes hit rates + returns
- New Performance API: `/api/v1/performance/scorecard`, `/by-layer`, `/signals`, `/grade`, `/stats`
- Performance tab redesigned: signal scorecard (hit rates by horizon, conviction buckets, trade-worthy comparison, layer accuracy bars, signal browser table) + existing trade P&L stats
- Lower recording threshold (40% conviction, 2 layers) than auto-journal (60%, 3) to capture more data for analysis
- One signal per ticker per day deduplication

**Key changes from earlier Feb 13 session (v0.3.2):**
- Fixed intraday charts not rendering — root cause: charts were initializing via setTimeout during renderDetail() while the detail panel was still hidden (max-height:0). The dataset.loaded flag then prevented re-initialization when the panel opened.
- Fix: moved chart loading to toggleDetail() so it fires when the panel opens and container has visible dimensions
- Added autoSize:true to lightweight-charts options for proper container-aware sizing

**Key changes from v0.3.0 (Feb 12 session):**
- Fixed UW 429 rate limit cascade — root cause: 5 parallel processors × 85 tickers overwhelming shared rate limiter
- UW rate limiter tuned: 1.5 req/sec burst 3 → 1.8 req/sec burst 8
- Added asyncio.Semaphore(3) to cap concurrent UW requests across all processors
- Added daily quota guard: skips calls when within 500 of 15K/day limit
- Added 30s global cooldown after 5 consecutive 429s (lets UW minute window reset)
- UW client now reads rate limit headers (x-uw-daily-req-count, x-uw-req-per-minute-remaining)
- 429 retries reduced from 2→1, backoff increased from 2s/4s→5s/10s
- UW rate_limited_count now tracked and shown in System tab
- universe_max_tickers default reduced from 200→50 (keeps daily calls ~9K, under 15K limit)
- Added diagnostic endpoint: GET /api/v1/system/debug/bars/{ticker} — shows raw Alpaca bar data to verify prices

**Known issue (in progress):**
- Dashboard shows WRONG stock prices for some tickers (QQQ shows $821.87, real price ~$613; PLTR $146.59 vs ~$136; TSLA ~correct)
- Price chain traced: Alpaca bars → momentum.py `closes[-1]` → metadata["price"] → dashboard
- Code logic is correct — suspect Alpaca is returning incorrect bar data for some tickers
- Debug endpoint added to inspect raw bars, but Chris hasn't been able to hit it yet (port conflict between Docker API container and local uvicorn on port 8000)
- **Next step:** Chris needs to `git pull` latest code, stop one of the two port-8000 processes, then test `http://localhost:8000/api/v1/system/debug/bars/QQQ` to see raw Alpaca data

**Key changes from v0.2.0:**
- Momentum layer switched from FMP to Alpaca (no quota limit, better data)
- All 5 UW layers fully implemented (no longer stubs)
- Frontend rebranded from "Confluence" to "Vicuna"
- Watchlist tab removed — let the market show what's trading
- Flow gate added — prevents trading against smart money
- Auto-journal service added
- Universe discovery with index product filtering
- Direction-aware coloring throughout UI
- Dark pool conflict warnings in flow gate details
- UW 429 rate limiting handled with retry/backoff

## Code Standards

- Type hints everywhere. Use `from __future__ import annotations`.
- Pydantic models for all API request/response schemas.
- Async by default for I/O (httpx for HTTP, asyncpg for DB).
- Each signal processor implements `SignalProcessor` ABC from `src/signals/base.py`.
- All external API calls go through rate-limited clients in `src/utils/data_providers.py`.
- Environment variables via `.env` file, loaded with `pydantic-settings`.
- Never hardcode API keys, tickers, or thresholds.

## Key Architecture Rules

1. **Signal processors are independent.** They don't import from each other. They all return `SignalResult` objects.
2. **The confluence scorer is the only thing that combines signals.** It lives in `src/services/confluence.py`.
3. **The VIX regime filter modifies weights, not scores.** It adjusts how much each signal layer matters based on market conditions.
4. **Paper trading by default.** `LIVE_TRADING_ENABLED` must be explicitly `true` for real orders.
5. **All scores are 0.0 to 1.0.** Direction is a separate enum (BULLISH/BEARISH/NEUTRAL). Never mix score and direction.
6. **FMP free tier: 250 calls/day.** Use the /stable/ endpoints (not legacy /api/v3/). Cache aggressively.
7. **Dashboard reads from cache, not live API.** The scheduler populates the cache; the dashboard reads it instantly.
8. **Flow gate controls trade-worthiness.** Options flow must agree with dominant direction, plus at least one of GEX/volatility must confirm. Prevents trading against smart money.
9. **All 8 processors run in parallel** via `asyncio.gather()`. If one fails, others complete unaffected.
10. **Signal weights and thresholds live in `config/signals.yaml`**, not hardcoded.

## File Structure

```
confluence-engine/
├── CLAUDE.md                   # This file — UPDATE EVERY SESSION
├── docker-compose.yml          # PostgreSQL 16 + Redis 7 + API service
├── Dockerfile                  # Python 3.12 slim, uvicorn
├── start.sh                    # One-command Docker Compose startup
├── .env                        # API keys (gitignored)
├── .env.example                # Template for .env
├── config/
│   ├── signals.yaml            # Weights, confluence multipliers, regime mods, flow gate
│   └── watchlist.yaml          # Default watchlist (35 tickers)
├── scripts/
│   ├── seed_watchlist.py       # Seed watchlist into DB
│   └── migrate.py              # Auto-migrations
├── src/
│   ├── config.py               # pydantic-settings config from .env
│   ├── api/
│   │   ├── main.py             # FastAPI app, lifespan, route mounting
│   │   ├── dependencies.py     # Singleton init: FMP, Alpaca, UW, processors, cache
│   │   └── routes/
│   │       ├── confluence.py   # GET /confluence (cache), POST /confluence/scan, GET /confluence/{ticker}
│   │       ├── signals.py      # GET /signals/{ticker} (individual layer results)
│   │       ├── watchlist.py    # CRUD for watchlist
│   │       ├── trades.py       # Trade journal with P&L
│   │       ├── broker.py       # Alpaca: account, positions, orders
│   │       ├── system.py       # Status, quota, signal layers, diagnostics
│   │       ├── regime.py       # GET /regime (VIX regime)
│   │       └── performance.py  # Signal scorecard: hit rates, layer accuracy, signal browser
│   ├── services/
│   │   ├── confluence.py       # Core scoring engine (weighted + confluence multiplier + flow gate)
│   │   ├── cache.py            # In-memory result cache (ResultCache, thread-safe)
│   │   ├── scheduler.py        # APScheduler: periodic scans -> cache + DB + auto-journal + signal tracking
│   │   ├── auto_journal.py     # Auto-log trade setups when signals fire
│   │   └── signal_tracker.py   # Forward testing: record signals + grade against T+1/5/10/20 outcomes
│   ├── signals/
│   │   ├── base.py             # ABC, SignalResult, Direction, Regime, ConfluenceScore
│   │   ├── momentum.py         # ACTIVE: Alpaca bars → MA alignment, golden cross, volume
│   │   ├── vix_regime.py       # ACTIVE: FMP VIX → regime classification (modifier only)
│   │   ├── insider.py          # ACTIVE: FMP/UW Form 4 → cluster buying, C-suite weighting
│   │   ├── options_flow.py     # ACTIVE: UW → sweep/block premium, bid/ask sentiment
│   │   ├── gex.py              # ACTIVE: UW → dealer gamma exposure positioning
│   │   ├── volatility.py       # ACTIVE: UW → IV rank, skew, term structure
│   │   ├── dark_pool.py        # ACTIVE: UW → FINRA ATS accumulation/distribution
│   │   └── short_interest.py   # ACTIVE: UW → SI%, days to cover, squeeze potential
│   ├── models/
│   │   ├── base.py             # SQLAlchemy declarative base
│   │   └── tables.py           # Watchlist, Signal, ConfluenceScoreRecord, Trade, Alert
│   ├── utils/
│   │   ├── data_providers.py   # FMPClient, AlpacaClient, UnusualWhalesClient (all rate-limited)
│   │   ├── rate_limiter.py     # Token bucket rate limiter
│   │   └── db.py               # Async engine, session factory, create_tables()
│   └── ui/
│       └── dashboard.html      # Single-page dashboard (vanilla JS + Tailwind) — "Vicuna" branded
```

## Common Commands

```bash
# Start full stack (PostgreSQL + Redis + API)
docker compose up -d

# Run backend with hot reload (from confluence-engine directory)
uvicorn src.api.main:app --reload --port 8000

# Or use the one-command startup script
./start.sh

# Seed watchlist (creates tables + inserts tickers)
python -m scripts.seed_watchlist

# View dashboard
open http://localhost:8000

# View API docs
open http://localhost:8000/docs
```

## Configuration

**Environment variables (.env):**
- `FMP_API_KEY` — Financial Modeling Prep (free tier, 250 calls/day)
- `UW_API_KEY` — Unusual Whales (powers 5 of 8 layers)
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — Alpaca paper/live trading
- `ALPACA_BASE_URL` — `https://paper-api.alpaca.markets` (paper) or `https://api.alpaca.markets` (live)
- `SCAN_INTERVAL` — seconds between scans (default 900 = 15 min)
- `AUTO_TRADE_ENABLED` — auto-journal qualifying signals (default true)
- `AUTO_TRADE_MIN_CONVICTION` — threshold for auto-journal (default 60)
- `AUTO_TRADE_MIN_LAYERS` — min agreeing layers for auto-journal (default 3)
- `LIVE_TRADING_ENABLED` — must be explicitly `true` for real orders (default false)

**Signal config (config/signals.yaml):**
- Signal weights (options_flow highest at 0.25, short_interest lowest at 0.08)
- Confluence multiplier (1 layer=0.80x discount → 5 layers=1.50x max)
- Regime modifiers (calm=trust bullish, crisis=favor bearish)
- Flow gate config (primary=options_flow, secondary=gex+volatility)
- Alert thresholds (conviction, premium, cluster window, squeeze detection)
- Refresh intervals per layer

## Important Context

- The "confluence multiplier" is central: more independent layers agreeing = higher conviction. This is the core value proposition.
- Chris is the only user. Don't over-engineer auth, multi-tenancy, or scale concerns.
- Options flow data is the most valuable single layer (requires Unusual Whales subscription).
- The VIX regime filter is NOT a trade signal — it's a modifier that adjusts signal weights.
- Cloud sandbox blocks outbound HTTP (FMP returns 403) — must test locally on Chris's Mac.
- FMP free tier gets 402 on paid-only endpoints. The quote endpoint includes priceAvg50/priceAvg200.
- Momentum uses Alpaca bars (not FMP) — no quota limit, better historical data.
- UW returns 429 when rate limited — handled with retry + exponential backoff.
- When in doubt about finance/trading concepts, ask Chris — he knows finance better than most developers.
