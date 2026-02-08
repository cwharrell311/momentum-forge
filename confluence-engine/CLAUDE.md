# CLAUDE.md — Instructions for Claude Code

## Project Overview

Confluence Engine is a personal trading platform that identifies high-probability trade setups by detecting when multiple independent signal layers align on the same ticker. It is NOT a toy project — this will be used to trade real money.

## Who Is Building This

Chris — a financial controller / accountant with 20 years of finance experience. He understands markets, options, and financial data deeply but is learning to code via "vibe coding" with Claude. Explain technical decisions clearly. Don't assume deep software engineering knowledge, but don't dumb down the finance or trading concepts.

## Tech Stack (Mandatory)

- **Backend:** Python 3.12+, FastAPI, SQLAlchemy (async), Pydantic v2
- **Frontend:** React (Vite), TailwindCSS, Recharts for charts
- **Database:** PostgreSQL 16+ (main store), Redis 7+ (caching/real-time)
- **Task runner:** APScheduler for periodic data pulls
- **Broker:** Alpaca API (paper trading first)
- **Primary data:** Financial Modeling Prep (FMP) API, Unusual Whales API, SEC EDGAR
- **Containerization:** Docker Compose for local dev

## Code Standards

- Type hints everywhere in Python. Use `from __future__ import annotations`.
- Pydantic models for all API request/response schemas.
- Async by default for I/O operations (httpx for HTTP calls, asyncpg for DB).
- Each signal processor must implement the `SignalProcessor` abstract base class in `src/signals/base.py`.
- All external API calls must go through rate-limited clients in `src/utils/`.
- Environment variables via `.env` file, loaded with `pydantic-settings`.
- Never hardcode API keys, tickers, or thresholds — always configurable.

## Key Architecture Rules

1. **Signal processors are independent.** They don't import from each other. They all return `SignalResult` objects.
2. **The confluence scorer is the only thing that combines signals.** It lives in `src/services/confluence.py`.
3. **The VIX regime filter modifies weights, not scores.** It adjusts how much each signal layer matters based on market conditions.
4. **Paper trading by default.** The `LIVE_TRADING_ENABLED` env var must be explicitly set to `true` to send real orders. Default is `false`.
5. **All scores are 0.0 to 1.0.** Direction is a separate enum (BULLISH/BEARISH/NEUTRAL). Never mix score and direction.

## File Structure

```
src/
├── api/
│   ├── main.py              # FastAPI app, middleware, lifespan
│   └── routes/
│       ├── confluence.py     # GET /confluence, GET /confluence/{ticker}
│       ├── signals.py        # GET /signals/{ticker}, GET /signals/layer/{layer}
│       ├── watchlist.py      # CRUD for watchlist
│       ├── trades.py         # Trade journal
│       └── alerts.py         # Alert management
├── services/
│   ├── confluence.py         # Core scoring engine
│   ├── alerting.py           # Alert generation and delivery
│   └── scheduler.py          # APScheduler job definitions
├── signals/
│   ├── base.py               # ABC + SignalResult + Direction
│   ├── options_flow.py       # Layer 1: Unusual Whales flow data
│   ├── gex.py                # Layer 2: Dealer positioning
│   ├── volatility.py         # Layer 3: IV surface analysis
│   ├── dark_pool.py          # Layer 4: Dark pool prints
│   ├── insider.py            # Layer 5: SEC Form 4 cluster buying
│   ├── short_interest.py     # Layer 6: SI + borrow cost
│   ├── vix_regime.py         # Layer 7: VIX regime classification
│   └── momentum.py           # Layer 8: Price/volume technicals
├── utils/
│   ├── data_providers.py     # HTTP clients for FMP, UW, EDGAR, etc.
│   ├── rate_limiter.py       # Token bucket rate limiter
│   └── db.py                 # SQLAlchemy async engine + session
└── ui/                       # React frontend (Vite)
```

## Development Workflow

1. Always start by checking `ROADMAP.md` for current phase priorities.
2. Build signal processors one at a time. Get each one working and tested before moving to the next.
3. Use `docker compose up -d db redis` to run infrastructure locally.
4. Run backend with `uvicorn src.api.main:app --reload`.
5. Run frontend with `cd src/ui && npm run dev`.
6. Test API endpoints with the built-in FastAPI Swagger UI at `/docs`.

## Common Commands

```bash
# Start infrastructure
docker compose up -d db redis

# Run backend
cd src && uvicorn api.main:app --reload --port 8000

# Run frontend
cd src/ui && npm run dev

# Run tests
pytest tests/ -v

# Database migrations (if using alembic)
alembic upgrade head

# Seed watchlist
python scripts/seed_watchlist.py
```

## Important Context

- The "confluence multiplier" concept is central: the more independent signal layers that agree, the higher the conviction score. This is the core value proposition.
- Chris will be the only user initially. Don't over-engineer auth, multi-tenancy, or scale concerns. Keep it simple and functional.
- Options flow data is the most valuable single layer. If forced to prioritize, always prioritize getting options flow working well.
- The VIX regime filter is NOT a trade signal. It's a modifier that adjusts how other signals are weighted. Don't score it the same way.
- When in doubt about a financial/trading concept, ask Chris — he knows finance better than most developers.
