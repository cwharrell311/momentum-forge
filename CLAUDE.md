# CLAUDE.md - MomentumForge

## Project Overview

MomentumForge is a stock momentum screener that scans NASDAQ/NYSE stocks for multi-factor momentum signals. It combines technical analysis, fundamental data, options flow, and social sentiment into a weighted 0-100 momentum score. The application has a Python backend (Flask API + screener engine) and a single-file React frontend.

## Architecture

```
momentum-forge/
├── screener.py      # Core screener engine (~970 lines)
├── server.py        # Flask REST API server (~194 lines)
├── index.html       # Self-contained React 18 frontend (~1230 lines)
└── requirements.txt # Python dependencies
```

**All source code lives at the repository root.** There are no `src/`, `tests/`, or `config/` directories.

### Component Responsibilities

- **screener.py**: `MomentumScreener` class (stock data fetching, technical/fundamental/options/social analysis, scoring) and `SocialSentimentScanner` class (StockTwits API, web search sentiment, keyword-based NLP). Data models use Python `@dataclass`: `StockSignal` and `SocialSentiment`.
- **server.py**: Flask app with 7 REST endpoints. Runs scans in background threads via `threading.Thread`. Stores results in global in-memory state (no database).
- **index.html**: Single-file React SPA loaded via CDN (React 18.2.0 + Babel 7.23.5 for in-browser JSX transpilation). Contains all CSS, JS, and markup. Communicates with the backend via `fetch()` and polls `/api/status` every 1 second during scans.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend language | Python 3 |
| Web framework | Flask 3.0+ with flask-cors |
| Data fetching | yfinance, requests |
| Data processing | pandas, numpy |
| Frontend framework | React 18 (CDN, not bundled) |
| JSX transpilation | Babel (in-browser) |
| External APIs | Yahoo Finance (free), Financial Modeling Prep (optional, needs API key), StockTwits (free) |

## Setup and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server (default: http://0.0.0.0:5000)
python server.py

# Or run CLI screener directly
python screener.py --fmp-key <API_KEY> --min-cap 1.0 --output results.json --format json

# Open frontend
# Serve index.html or open it in a browser pointing at http://localhost:5000
```

### Environment Variables

- `FMP_API_KEY` - Financial Modeling Prep API key (optional; without it, earnings/revenue data comes from Yahoo Finance fallback)
- `PORT` - Server port (default: 5000)

### CLI Arguments (screener.py)

- `--fmp-key` - FMP API key
- `--min-cap` - Minimum market cap in billions (default: 1.0)
- `--output` - Output file path (default: results.json)
- `--format` - `json` or `csv`
- `--no-social` - Skip social sentiment scanning

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/status` | GET | Current scan progress |
| `/api/scan` | POST | Start async scan (body: `{min_market_cap, include_social}`) |
| `/api/results` | GET | Filtered results (query: `min_score`, `sector`, `signal`, `limit`) |
| `/api/stock/<ticker>` | GET | Single stock detail (analyzes on-demand if not in cache) |
| `/api/sectors` | GET | Unique sectors from results |
| `/api/stats` | GET | Aggregate statistics |

All responses are JSON. Scans run in a background thread; the frontend polls `/api/status` for progress.

## Key Code Patterns

### Data Models (screener.py)

- `SocialSentiment` dataclass: score (-1 to +1), volume, sources, summary, trending, key_topics
- `StockSignal` dataclass: ~30 fields covering price, technicals, fundamentals, options, social, computed score/signals. Has `.to_dict()` for JSON serialization that flattens nested `SocialSentiment`.

### Analysis Pipeline (MomentumScreener.analyze_stock)

1. `get_stock_data()` - Yahoo Finance: price, moving averages (MA20/50/200), volume spike %, sector, analyst data
2. **Technical filter**: stock must be above ALL three MAs (MA20, MA50, MA200) or it's filtered out
3. `get_options_data()` - Yahoo Finance options chain: call/put ratio, unusual OTM call activity
4. `get_earnings_data_fmp()` - FMP API: earnings surprise %, YoY revenue growth, revenue acceleration
5. `SocialSentimentScanner.get_sentiment()` - StockTwits + optional web search
6. `calculate_momentum_score()` - Weighted composite score (0-100)
7. `determine_signals()` - Rules-based signal detection (earnings, revenue, volume, options, social)
8. Stock must have at least one signal or it's filtered out

### Momentum Scoring Breakdown (100 points max)

| Category | Max Points | Thresholds |
|----------|-----------|------------|
| Technical (above MAs) | 35 | MA20: 8, MA50: 12, MA200: 15 |
| Volume spike | 15 | 200%+: 15, 150%+: 10, 120%+: 5 |
| Earnings surprise | 12 | 20%+: 12, 10%+: 8, 5%+: 4 |
| Revenue growth | 13 | 30%+: 8, 15%+: 5, 5%+: 3, accelerating: +5 |
| Options flow | 12 | Unusual activity: 8, ratio>1.5: 4 |
| Social sentiment | 13 | Score>0.5: 8, >0.2: 5, >0: 2, trending+bullish: +5, bearish penalty: -3 |

### Concurrency

- `ThreadPoolExecutor` scans the 150-stock universe in parallel
- 10 workers for technical-only scans, 5 workers when social scanning is enabled (rate limit avoidance)
- Background thread in server.py for non-blocking API scans

### Stock Universe

Hardcoded list of ~150 liquid NASDAQ/NYSE stocks in `get_universe()` (screener.py:397-439), organized by category: mega caps, large cap tech, high growth, financials, healthcare, consumer, industrial, energy.

## Testing

No test framework or test files are configured. Validation is done manually through the web UI and CLI output.

## Linting / Formatting

No linting or formatting tools (flake8, black, prettier, eslint) are configured.

## Important Conventions

- **Flat file structure**: All code at repository root. Do not introduce subdirectories without good reason.
- **Dataclasses for models**: Use Python `@dataclass` with type hints and `to_dict()` methods for serialization.
- **Logging**: Uses Python `logging` module (`logger = logging.getLogger(__name__)`). Log at INFO for progress, WARNING for recoverable errors, DEBUG for filtered-out stocks.
- **Error handling**: All external API calls are wrapped in try/except with fallback to `None` or default values. Never let a single stock failure crash the scan.
- **API key optionality**: The screener works without FMP_API_KEY (falls back to Yahoo Finance data). Social sentiment works without web_search_func (falls back to StockTwits only).
- **In-memory state**: No database. Scan results live in global `scan_results` list in server.py. Data is lost on restart.
- **Frontend is a single HTML file**: All CSS, React components, and Babel config in one file. React and Babel loaded from CDN, not npm.
- **Type hints**: Python code uses type hints (`Optional`, `List`, `Dict`, `Callable`) from the `typing` module.
