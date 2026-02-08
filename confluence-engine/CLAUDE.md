# CLAUDE.md — Instructions for Claude Code

## Project Overview

Confluence Engine is a personal trading platform that identifies high-probability trade setups by detecting when multiple independent signal layers align on the same ticker. It is NOT a toy project — this will be used to trade real money.

## Who Is Building This

Chris — a financial controller / accountant with 20 years of finance experience. He understands markets, options, and financial data deeply but is learning to code via "vibe coding" with Claude. Explain technical decisions clearly. Don't assume deep software engineering knowledge, but don't dumb down the finance or trading concepts. Chris prefers step-by-step explanations built in large chunks.

## Tech Stack

- **Backend:** Python 3.12+, FastAPI, SQLAlchemy (async), Pydantic v2
- **Frontend:** Single HTML dashboard (vanilla JS + Tailwind CDN) served from FastAPI
- **Database:** PostgreSQL 16+ (via Docker), Redis 7+ (via Docker)
- **Task runner:** APScheduler for periodic scans
- **Broker:** Alpaca API (paper trading first, then live)
- **Data sources:** FMP API (free tier), Unusual Whales API (Phase 2), SEC EDGAR
- **Containerization:** Docker Compose for PostgreSQL + Redis

## Current State (v0.2.0)

**Active signal layers: 3 of 8**
1. Momentum / Technical (FMP free) — MA alignment, golden/death cross, trend strength, 52-week position, relative volume
2. VIX Regime Filter (FMP free) — modifies signal weights, NOT scored directly
3. Insider Trading (FMP free) — SEC Form 4 filings, cluster buying detection, C-suite weighting

**Stub layers (need subscriptions):**
4. Options Flow (Unusual Whales ~$50/mo)
5. GEX / Dealer Positioning (Unusual Whales)
6. Volatility Surface (Unusual Whales)
7. Dark Pool Activity (Unusual Whales)
8. Short Interest (Unusual Whales)

**Infrastructure:**
- Dashboard: 6 tabs (Screener, Journal, Performance, Watchlist, Broker, System)
- Caching: In-memory result cache (zero API calls per page load)
- Scheduler: 15-min scan interval (conserves FMP free tier quota)
- Trade journal: CRUD with P&L tracking
- Broker: Alpaca paper trading integration (account, positions, orders)
- System: FMP quota tracking, signal layer status panel
- DB: Auto-creates tables on startup, graceful fallback if PostgreSQL not running

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

## File Structure

```
confluence-engine/
├── CLAUDE.md                   # This file
├── docker-compose.yml          # PostgreSQL 16 + Redis 7
├── .env                        # API keys (gitignored)
├── .env.example                # Template for .env
├── config/
│   └── watchlist.yaml          # Default watchlist (14 tickers)
├── scripts/
│   └── seed_watchlist.py       # Seed watchlist into DB
├── src/
│   ├── config.py               # pydantic-settings config from .env
│   ├── api/
│   │   ├── main.py             # FastAPI app, lifespan, route mounting
│   │   ├── dependencies.py     # Singleton init: FMP, Alpaca, processors, cache
│   │   └── routes/
│   │       ├── confluence.py   # GET /confluence (from cache), GET /confluence/{ticker}
│   │       ├── signals.py      # GET /signals/{ticker} (individual layer results)
│   │       ├── watchlist.py    # CRUD for watchlist
│   │       ├── trades.py       # Trade journal with P&L
│   │       ├── broker.py       # Alpaca: account, positions, orders
│   │       ├── system.py       # Status, quota, signal layers API
│   │       └── regime.py       # GET /regime (VIX regime)
│   ├── services/
│   │   ├── confluence.py       # Core scoring engine (weighted + confluence multiplier)
│   │   ├── cache.py            # In-memory result cache (ResultCache)
│   │   └── scheduler.py        # APScheduler: periodic scans -> cache + DB history
│   ├── signals/
│   │   ├── base.py             # ABC, SignalResult, Direction, Regime, ConfluenceScore
│   │   ├── momentum.py         # ACTIVE: price/volume technicals via FMP
│   │   ├── vix_regime.py       # ACTIVE: VIX regime classification
│   │   ├── insider.py          # ACTIVE: SEC Form 4 insider trading
│   │   ├── options_flow.py     # STUB: needs Unusual Whales
│   │   ├── gex.py              # STUB: needs Unusual Whales
│   │   ├── volatility.py       # STUB: needs Unusual Whales
│   │   ├── dark_pool.py        # STUB: needs Unusual Whales
│   │   └── short_interest.py   # STUB: needs Unusual Whales
│   ├── models/
│   │   └── tables.py           # SQLAlchemy models: Watchlist, Signal, Trade, etc.
│   ├── utils/
│   │   ├── data_providers.py   # FMPClient, AlpacaClient, UnusualWhalesClient
│   │   ├── rate_limiter.py     # Token bucket rate limiter
│   │   └── db.py               # Async engine, session factory, create_tables()
│   └── ui/
│       └── dashboard.html      # Single-page dashboard (vanilla JS + Tailwind)
```

## Common Commands

```bash
# Start infrastructure (PostgreSQL + Redis)
docker compose up -d

# Run backend (from confluence-engine directory)
uvicorn src.api.main:app --reload --port 8000

# Seed watchlist (creates tables + inserts tickers)
python -m scripts.seed_watchlist

# View dashboard
open http://localhost:8000

# View API docs
open http://localhost:8000/docs
```

## Important Context

- The "confluence multiplier" is central: more independent layers agreeing = higher conviction. This is the core value proposition.
- Chris is the only user. Don't over-engineer auth, multi-tenancy, or scale concerns.
- Options flow data is the most valuable single layer (requires Unusual Whales subscription).
- The VIX regime filter is NOT a trade signal — it's a modifier that adjusts signal weights.
- Cloud sandbox blocks outbound HTTP (FMP returns 403) — must test locally on Chris's Mac.
- FMP free tier gets 402 on paid-only endpoints (technical indicators). The quote endpoint includes priceAvg50/priceAvg200, which is enough for momentum scoring.
- When in doubt about finance/trading concepts, ask Chris — he knows finance better than most developers.
