# Architecture: Confluence Engine

## Design Principles

1. **Each signal layer is independent.** A signal processor fetches, normalizes, and scores data for its domain without knowing about other layers. This means you can add/remove layers without breaking anything.

2. **The confluence scorer is the brain.** It pulls scores from all active signal processors and combines them into a single conviction score per ticker. Weights are configurable.

3. **Regime-aware scoring.** The VIX regime filter modifies how other signals are weighted. In a stressed market (VIX backwardation), bullish flow signals are discounted and bearish signals amplified. In calm markets, the opposite.

4. **Separation of signal from execution.** The system tells you *what* looks good. You decide whether to trade it. Execution (broker API) is a separate, opt-in module.

---

## Signal Processor Interface

Every signal layer implements the same interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class SignalResult:
    ticker: str
    layer: str                    # e.g., "options_flow"
    direction: Direction
    strength: float               # 0.0 to 1.0
    confidence: float             # 0.0 to 1.0
    timestamp: datetime
    metadata: dict                # Layer-specific details
    explanation: str              # Human-readable summary

class SignalProcessor(ABC):
    @abstractmethod
    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        """Scan a list of tickers and return signals."""
        pass

    @abstractmethod
    async def scan_single(self, ticker: str) -> SignalResult | None:
        """Deep scan a single ticker."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def refresh_interval_seconds(self) -> int:
        """How often this layer should be refreshed."""
        pass
```

---

## Confluence Scoring Algorithm

```python
@dataclass
class ConfluenceScore:
    ticker: str
    direction: Direction
    conviction: float             # 0.0 to 1.0 — the "holy grail" score
    active_layers: int            # How many layers are firing
    total_layers: int             # How many layers have data
    signals: list[SignalResult]   # Individual layer results
    regime: str                   # Current VIX regime
    timestamp: datetime

def calculate_confluence(signals: list[SignalResult], regime: str) -> ConfluenceScore:
    """
    Core algorithm:
    1. Group signals by direction (bullish vs bearish)
    2. Apply regime-adjusted weights to each layer
    3. Calculate weighted average strength for dominant direction
    4. Boost score based on number of agreeing layers (confluence multiplier)
    5. Penalize if any strong signal contradicts the dominant direction
    """
```

### Default Signal Weights

These are starting points — tune based on personal backtesting:

```yaml
signal_weights:
  options_flow: 0.25         # Highest weight — most predictive single layer
  gex_positioning: 0.18      # Strong structural edge
  volatility_surface: 0.12   # Important for trade sizing and entry timing
  dark_pool: 0.15            # Excellent confirmation of institutional intent
  insider_buying: 0.10       # Strong but slow-moving
  short_interest: 0.08       # Useful for squeeze setups
  vix_regime: 0.00           # Not weighted directly — modifies other weights
  momentum: 0.12             # Trend confirmation

regime_modifiers:
  calm:                      # VIX < 15, contango
    bullish_boost: 1.1
    bearish_discount: 0.8
  elevated:                  # VIX 15-25
    bullish_boost: 1.0
    bearish_discount: 1.0
  stressed:                  # VIX 25-35, backwardation
    bullish_boost: 0.7
    bearish_discount: 1.3
  crisis:                    # VIX > 35, deep backwardation
    bullish_boost: 0.4
    bearish_discount: 1.5

confluence_multiplier:
  # Bonus applied when multiple layers agree
  2_layers: 1.0             # No bonus — need more confirmation
  3_layers: 1.15            # Moderate boost
  4_layers: 1.30            # Strong boost
  5_plus_layers: 1.50       # Maximum conviction
```

---

## Data Provider Strategy

### Tier 1: MVP (Get trading within 2-4 weeks)

| Layer | Data Source | Method | Cost |
|-------|-----------|--------|------|
| Options Flow | Unusual Whales API | REST, poll every 5 min | ~$50/mo |
| GEX | Unusual Whales (included) or manual calc from options chain | REST | Included |
| Volatility | Options chain from FMP or Tradier | REST | Free-$30/mo |
| Momentum | FMP (price, volume, technicals) | REST | Free tier |
| VIX Regime | FMP (VIX price + futures) | REST | Free tier |
| Insider Buying | SEC EDGAR XBRL | REST (free, rate-limited) | Free |

**MVP cost: ~$50-80/month in data**

### Tier 2: Enhanced (Month 2-3)

| Layer | Data Source | Method | Cost |
|-------|-----------|--------|------|
| Dark Pool | Unusual Whales (higher tier) or FlowAlgo | REST | +$30-50/mo |
| Short Interest | FINRA (delayed free) or Ortex | REST | Free-$50/mo |
| Execution | Alpaca (paper first) | REST + WebSocket | Free |

### Tier 3: Power User (Month 4+)

- Real-time options stream via Tradier or CBOE
- Alternative data feeds (satellite, web traffic)
- Claude API for earnings call NLP analysis
- Multi-broker execution (IBKR for more option strategies)

---

## Database Schema (Core Tables)

```sql
-- Tickers we're actively tracking
CREATE TABLE watchlist (
    ticker VARCHAR(10) PRIMARY KEY,
    added_at TIMESTAMP DEFAULT NOW(),
    sector VARCHAR(50),
    avg_volume BIGINT,
    market_cap BIGINT,
    active BOOLEAN DEFAULT TRUE
);

-- Individual signal events
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    layer VARCHAR(30) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    strength FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    metadata JSONB,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_signals_ticker_time ON signals(ticker, created_at DESC);
CREATE INDEX idx_signals_layer ON signals(layer, created_at DESC);

-- Confluence scores (computed periodically)
CREATE TABLE confluence_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    conviction FLOAT NOT NULL,
    active_layers INT NOT NULL,
    regime VARCHAR(20),
    signal_ids INT[],
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_confluence_conviction ON confluence_scores(conviction DESC, created_at DESC);

-- Trade journal (manual or automated)
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,        -- 'long' or 'short'
    instrument VARCHAR(20) NOT NULL,  -- 'equity', 'call', 'put', 'spread'
    entry_price FLOAT,
    exit_price FLOAT,
    quantity INT,
    confluence_score_id INT REFERENCES confluence_scores(id),
    entry_at TIMESTAMP,
    exit_at TIMESTAMP,
    pnl FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    alert_type VARCHAR(30),          -- 'confluence_high', 'flow_spike', etc.
    message TEXT,
    conviction FLOAT,
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## API Endpoints (FastAPI)

```
GET  /api/v1/confluence              # Top confluence scores (the main screener)
GET  /api/v1/confluence/{ticker}     # Deep dive on single ticker
GET  /api/v1/signals/{ticker}        # All active signals for a ticker
GET  /api/v1/signals/layer/{layer}   # All signals from one layer
GET  /api/v1/regime                  # Current VIX regime status
GET  /api/v1/watchlist               # Current watchlist
POST /api/v1/watchlist               # Add/remove tickers
GET  /api/v1/alerts                  # Unacknowledged alerts
POST /api/v1/trades                  # Log a trade
GET  /api/v1/trades                  # Trade journal
GET  /api/v1/trades/performance      # P&L analytics
POST /api/v1/execute                 # Send order to broker (opt-in)
```

---

## Frontend Views

### 1. Confluence Dashboard (Home)
The main screen. A ranked table of tickers sorted by conviction score, with:
- Ticker, direction (bull/bear icon), conviction score (0-100), active layers count
- Sparkline of recent price action
- Color-coded heatmap cells for each signal layer
- Click to expand → shows individual signal details and explanations
- Current regime badge in the header

### 2. Ticker Deep Dive
Single-ticker view showing:
- All 8 signal layers with detailed scores and metadata
- Options flow feed (recent sweeps, blocks)
- GEX levels overlaid on price chart
- IV rank / IV percentile gauge
- Insider activity timeline
- Short interest trend

### 3. Trade Panel
- One-click paper trade from any confluence signal
- Position sizing calculator (based on account size + conviction)
- Active positions with live P&L
- Trade journal with confluence score at entry

### 4. Alerts
- Push notifications when a ticker crosses a conviction threshold
- Configurable per-layer alerts (e.g., "notify me on any sweep > $1M premium")

---

## Security Notes

- All API keys stored in `.env`, never committed
- Broker API uses paper trading by default; live trading requires explicit config flag
- Rate limiting on all external API calls to avoid bans
- No PII stored — this is a personal tool
