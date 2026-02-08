# Confluence Engine

A personal trading platform that identifies high-probability trade setups by detecting **confluence** — multiple independent signal layers aligning on the same ticker at the same time. Built for options and equity swing trading.

## Core Thesis

No single signal is a holy grail. But when 4-5 independent data layers agree simultaneously, the probability of a successful trade increases dramatically. This platform synthesizes those layers into a single actionable dashboard.

## Signal Layers (Priority Order)

| # | Layer | What It Tells You | Data Source | Cost |
|---|-------|-------------------|-------------|------|
| 1 | **Options Flow** | What informed money is betting on (sweeps, blocks, unusual OI) | Unusual Whales API / Tradier / CBOE | $$ |
| 2 | **GEX / Dealer Positioning** | Where market makers will amplify or dampen moves | SpotGamma API or calculated from options chain | $-$$ |
| 3 | **Volatility Surface** | Is IV cheap/expensive? Skew shifts? | Options chain data (Tradier/CBOE) | $ |
| 4 | **Dark Pool Prints** | Where institutions are accumulating stock positions | FINRA ADF / FlowAlgo API | $$ |
| 5 | **Insider Cluster Buying** | C-suite executives buying their own stock (Form 4) | SEC EDGAR (free) / OpenInsider | Free |
| 6 | **Short Interest + Borrow Cost** | Squeeze setup detection | FINRA / Ortex / S3 Partners | $-$$ |
| 7 | **VIX Regime Filter** | Market context — calm vs. stressed | CBOE VIX futures (free via FMP) | Free |
| 8 | **Momentum / Technicals** | Price structure confirmation | FMP / Twelve Data / calculated | $ |

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (React)                   │
│  Dashboard │ Screener │ Trade Panel │ Alerts         │
└──────────────────────┬──────────────────────────────┘
                       │ REST + WebSocket
┌──────────────────────┴──────────────────────────────┐
│                  API LAYER (FastAPI)                  │
│  /signals │ /confluence │ /trades │ /alerts          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│              CONFLUENCE SCORING ENGINE                │
│  Combines signal layers → Ranked ticker list         │
│  Weights configurable │ Regime-aware                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│               SIGNAL PROCESSORS (8 layers)           │
│  Each processor: fetch → normalize → score → emit    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│                DATA LAYER                            │
│  PostgreSQL (positions, history, alerts)              │
│  Redis (real-time signal cache, rate limiting)        │
│  Scheduled jobs (cron for batch data pulls)           │
└─────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│              EXTERNAL DATA SOURCES                   │
│  FMP │ Unusual Whales │ EDGAR │ Tradier │ Alpaca    │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

- **Backend**: Python 3.12+ / FastAPI
- **Frontend**: React (Vite) + TailwindCSS + Recharts
- **Database**: PostgreSQL (persistent) + Redis (cache/real-time)
- **Broker Integration**: Alpaca API (paper trading first, then live)
- **Task Scheduling**: APScheduler or Celery for batch data pulls
- **Deployment**: Docker Compose (local dev) → VPS or Railway (prod)

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 20+
- PostgreSQL 16+
- Redis 7+
- Docker & Docker Compose (recommended)

### Quick Start
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/confluence-engine.git
cd confluence-engine

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start with Docker Compose
docker compose up -d

# Or run locally:
# Backend
cd src && pip install -r requirements.txt && uvicorn api.main:app --reload

# Frontend
cd ui && npm install && npm run dev
```

### API Keys Needed (MVP)
| Service | Purpose | Cost | Sign Up |
|---------|---------|------|---------|
| Financial Modeling Prep | Price data, fundamentals, technicals | Free tier available | https://financialmodelingprep.com |
| Unusual Whales | Options flow, dark pool, GEX | ~$50/mo | https://unusualwhales.com |
| SEC EDGAR | Insider buying (Form 4) | Free | https://www.sec.gov/edgar |
| Alpaca | Paper/live trading execution | Free | https://alpaca.markets |

## Project Structure
```
confluence-engine/
├── README.md
├── ARCHITECTURE.md          # Detailed technical architecture
├── ROADMAP.md               # Phased build plan
├── CLAUDE.md                # Instructions for Claude Code
├── docker-compose.yml
├── .env.example
├── config/
│   ├── signals.yaml         # Signal weights and thresholds
│   └── watchlist.yaml       # Tickers to track
├── src/
│   ├── api/                 # FastAPI routes
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware/
│   ├── services/            # Business logic
│   │   ├── confluence.py    # Core scoring engine
│   │   └── alerting.py
│   ├── signals/             # One module per signal layer
│   │   ├── base.py          # Abstract signal processor
│   │   ├── options_flow.py
│   │   ├── gex.py
│   │   ├── volatility.py
│   │   ├── dark_pool.py
│   │   ├── insider.py
│   │   ├── short_interest.py
│   │   ├── vix_regime.py
│   │   └── momentum.py
│   ├── utils/
│   │   ├── data_providers.py
│   │   └── rate_limiter.py
│   └── ui/                  # React frontend
│       ├── package.json
│       └── src/
├── scripts/
│   ├── seed_watchlist.py
│   └── backtest.py
└── tests/
```

## License

Private / Personal Use
