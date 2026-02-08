# Roadmap: Confluence Engine

## Phase 1: Foundation + First Signal (Weeks 1-2)
**Goal: See your first confluence data on screen.**

- [ ] Project scaffolding (FastAPI + React + Docker Compose)
- [ ] PostgreSQL + Redis setup
- [ ] `.env` config management
- [ ] Watchlist management (CRUD, seed with 50-100 liquid optionable tickers)
- [ ] **Signal Layer 1: Momentum/Technicals** (FMP API)
  - RSI, MACD, moving averages, relative volume
  - Score normalization (0.0-1.0 strength + direction)
- [ ] **Signal Layer 2: VIX Regime Filter** (FMP API)
  - VIX current level + term structure (contango/backwardation)
  - Regime classification: calm / elevated / stressed / crisis
- [ ] Basic API: `/confluence`, `/signals/{ticker}`, `/regime`
- [ ] Minimal React dashboard: table of tickers + scores

**Why these two first:** Both use the free FMP API, so zero incremental cost. Gets the full pipeline working end-to-end before adding paid data.

---

## Phase 2: Options Flow — The Big One (Weeks 3-4)
**Goal: Integrate the highest-value signal layer.**

- [ ] Unusual Whales API integration
- [ ] **Signal Layer 3: Options Flow**
  - Sweep detection (size, premium, aggressor side)
  - Block trade tracking
  - Unusual OI / volume ratio alerts
  - Opening vs. closing classification
  - Premium-weighted directional scoring
- [ ] **Signal Layer 4: GEX / Dealer Positioning**
  - GEX calculation from options chain (or UW if available)
  - Key levels: GEX flip, call wall, put wall
  - Overlay data for frontend charts
- [ ] **Signal Layer 5: Volatility Surface**
  - IV Rank / IV Percentile
  - Skew analysis (put/call IV ratio by strike)
  - IV vs. historical realized vol comparison
- [ ] Confluence scoring engine v1 (combine all 5 layers)
- [ ] Enhanced dashboard with layer heatmap
- [ ] Ticker deep-dive view

---

## Phase 3: Institutional + Structural Layers (Weeks 5-6)
**Goal: Add the confirmation layers that separate noise from signal.**

- [ ] **Signal Layer 6: Insider Buying** (SEC EDGAR)
  - Form 4 parser
  - Cluster detection (multiple insiders buying within window)
  - Filter: buys only, exclude automatic/planned transactions
- [ ] **Signal Layer 7: Dark Pool Prints** (UW higher tier or FlowAlgo)
  - Large block detection at price levels
  - Accumulation pattern recognition
- [ ] **Signal Layer 8: Short Interest + Borrow Cost**
  - Short interest % of float
  - Days to cover
  - Cost to borrow spikes
  - Squeeze score calculation
- [ ] Full 8-layer confluence scoring
- [ ] Alert system (high conviction notifications)
- [ ] Trade journal (manual logging with confluence context)

---

## Phase 4: Execution + Paper Trading (Weeks 7-8)
**Goal: Trade directly from the dashboard — paper money first.**

- [ ] Alpaca API integration (paper trading account)
- [ ] Position sizing calculator (Kelly criterion or fixed risk %)
- [ ] One-click trade from confluence signals
- [ ] Active position monitoring with live P&L
- [ ] Trade journal auto-populated from executions
- [ ] Performance analytics (win rate, avg P&L, by signal layer)

---

## Phase 5: Intelligence + Refinement (Month 3+)
**Goal: Make the system smarter over time.**

- [ ] Claude API integration for earnings call NLP
- [ ] Backtest framework: replay historical signals against actual price outcomes
- [ ] Signal weight optimization based on backtest results
- [ ] Correlation analysis: which signal combinations actually predict best?
- [ ] Custom alerting rules (Slack, SMS, push)
- [ ] Mobile-responsive UI for monitoring on the go
- [ ] Live trading toggle (switch from paper to real)

---

## Deferred / Future Ideas

- Multi-timeframe analysis (intraday + swing + position)
- Credit market leading indicators (HY spread, CDS)
- Satellite / alternative data integration
- Social sentiment layer (Reddit, Twitter/X, StockTwits)
- Automated strategy execution (fully algo — requires heavy backtesting first)
- SaaS productization (if you decide to sell access)

---

## MVP Definition

**You can start trading from this platform when Phase 2 is complete.** That gives you:
- Momentum confirmation (free)
- VIX regime awareness (free)
- Options flow signals (~$50/mo)
- GEX levels (~included with UW)
- Volatility analysis (included)
- 5-layer confluence scoring

Phases 3-5 are about making it *better*, not making it *usable*.
