"""
Multi-asset data feeds for backtesting.

Fetches historical OHLCV data from free sources:
- Stocks: yfinance (Yahoo Finance — free, no API key)
- Crypto: ccxt (Binance public API — free, no key for historical)
- Polymarket: CLOB API (free, public prediction market data)

All feeds return a standardized DataFrame with columns:
    timestamp, open, high, low, close, volume

No paid API keys required for historical data.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger("forge.data_feeds")


class AssetClass(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    POLYMARKET = "polymarket"


@dataclass
class OHLCV:
    """Standardized bar data across all asset classes."""
    df: pd.DataFrame        # timestamp, open, high, low, close, volume
    symbol: str
    asset_class: AssetClass
    timeframe: str           # "1m", "5m", "15m", "1h", "4h", "1d"
    source: str              # "yfinance", "binance", "polymarket"


# ── Stock Data (yfinance) ──


def fetch_stock_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    include_context: bool = False,
) -> OHLCV:
    """
    Fetch stock OHLCV from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g., "SPY", "AAPL", "QQQ")
        period: How far back ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        interval: Bar size ("1m", "5m", "15m", "1h", "1d", "1wk")
            Note: intraday data (1m-1h) limited to last 60 days on free tier.
        include_context: If True, automatically fetch and merge market context
            data (VIX, DXY, yields) into the OHLCV DataFrame. Only applies to
            daily bars — intraday bars skip context to avoid misaligned joins.

    Returns:
        OHLCV with standardized DataFrame. When include_context=True, the
        DataFrame also contains: vix, dxy, yield_10y, yield_3m, yield_spread.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} (period={period}, interval={interval})")

    # Standardize column names
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    # Ensure timezone-aware UTC timestamps
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")

    # Optionally merge cross-asset market context (VIX, DXY, yields)
    if include_context and interval in ("1d", "1wk"):
        try:
            context = fetch_market_context(period=period)
            df = add_context_to_ohlcv(df, context)
            log.info("Stock %s: merged market context (%d context rows)", symbol, len(context))
        except Exception as e:
            log.warning("Stock %s: failed to merge market context: %s", symbol, e)

    log.info("Stock %s: %d bars (%s, %s)", symbol, len(df), period, interval)
    return OHLCV(df=df, symbol=symbol, asset_class=AssetClass.STOCK, timeframe=interval, source="yfinance")


def fetch_stock_intraday(
    symbol: str,
    days_back: int = 30,
    interval: str = "5m",
) -> OHLCV:
    """
    Fetch intraday stock data. Yahoo limits intraday to ~60 days.

    For day trading backtests — 5m bars are the sweet spot for stocks.
    1m data creates too much noise; 15m misses entries.
    """
    import yfinance as yf

    # yfinance intraday limits: 1m=7d, 5m=60d, 15m=60d, 1h=730d
    period_map = {"1m": 7, "5m": 60, "15m": 60, "1h": 730}
    max_days = period_map.get(interval, 60)
    actual_days = min(days_back, max_days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{actual_days}d", interval=interval)

    if df.empty:
        raise ValueError(f"No intraday data for {symbol}")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")

    log.info("Stock intraday %s: %d bars (%dd, %s)", symbol, len(df), actual_days, interval)
    return OHLCV(df=df, symbol=symbol, asset_class=AssetClass.STOCK, timeframe=interval, source="yfinance")


# ── Cross-Asset Context Data ──


def fetch_market_context(period: str = "2y") -> pd.DataFrame:
    """
    Fetch macro market context: VIX, Dollar Index, Treasury yields.

    This provides regime-level information that individual asset OHLCV cannot:
    - VIX: volatility regime (calm < 15, elevated 15-25, crisis > 25)
    - DXY: dollar strength (risk-on/off proxy)
    - 10Y yield: rate environment (rising = tightening)
    - 3M yield: risk-free rate proxy for Sharpe calculations
    - Yield spread (10Y - 3M): inversion signals recession risk

    All data is free from yfinance — no API keys needed.

    Args:
        period: How far back ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")

    Returns:
        DataFrame indexed by date with columns:
        vix, dxy, yield_10y, yield_3m, yield_spread
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "DX-Y.NYB": "dxy",
        "^TNX": "yield_10y",
        "^IRX": "yield_3m",
    }

    frames: dict[str, pd.Series] = {}

    for yf_symbol, col_name in tickers.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                log.warning("Market context: no data for %s, skipping", yf_symbol)
                continue
            series = hist["Close"].copy()
            series.index = series.index.tz_localize(None) if series.index.tz is None else series.index.tz_convert("UTC").tz_localize(None)
            series.name = col_name
            frames[col_name] = series
        except Exception as e:
            log.warning("Market context: failed to fetch %s (%s): %s", yf_symbol, col_name, e)

    if not frames:
        raise ValueError("Failed to fetch any market context data (VIX, DXY, yields)")

    # Combine all series into one DataFrame, aligned by date
    context_df = pd.DataFrame(frames)
    context_df.index.name = "date"
    context_df = context_df.sort_index()

    # Forward-fill gaps (weekends, holidays) within each column
    context_df = context_df.ffill()

    # Compute yield spread (10Y - 3M) — classic recession indicator
    if "yield_10y" in context_df.columns and "yield_3m" in context_df.columns:
        context_df["yield_spread"] = context_df["yield_10y"] - context_df["yield_3m"]
    else:
        context_df["yield_spread"] = np.nan

    log.info(
        "Market context: %d rows, columns=%s, range=%s to %s",
        len(context_df),
        list(context_df.columns),
        context_df.index.min().strftime("%Y-%m-%d") if len(context_df) > 0 else "N/A",
        context_df.index.max().strftime("%Y-%m-%d") if len(context_df) > 0 else "N/A",
    )
    return context_df


def add_context_to_ohlcv(ohlcv_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cross-asset market context into an asset's OHLCV DataFrame.

    Left-joins context data (VIX, DXY, yields) onto the asset's OHLCV by date,
    then forward-fills missing context values so weekends/holidays don't leave NaNs.

    After merging, strategies can use the context columns as features for
    regime detection (e.g., "don't go long when VIX > 30 and yield curve inverted").

    Args:
        ohlcv_df: Asset OHLCV DataFrame (indexed by timestamp, may be tz-aware).
        context_df: Market context DataFrame from fetch_market_context()
            (indexed by tz-naive date).

    Returns:
        Copy of ohlcv_df with added columns: vix, dxy, yield_10y, yield_3m,
        yield_spread. Missing values are forward-filled.
    """
    result = ohlcv_df.copy()

    # Normalize the OHLCV index to tz-naive dates for joining
    if result.index.tz is not None:
        join_dates = result.index.tz_convert("UTC").tz_localize(None).normalize()
    else:
        join_dates = result.index.normalize()

    # Build a date-keyed lookup from context
    context_cols = [c for c in context_df.columns if c in ("vix", "dxy", "yield_10y", "yield_3m", "yield_spread")]

    for col in context_cols:
        if col in context_df.columns:
            # Map context values onto OHLCV dates
            mapping = context_df[col]
            result[col] = join_dates.map(mapping)

    # Forward-fill to cover any dates where context was missing (holidays etc.)
    for col in context_cols:
        if col in result.columns:
            result[col] = result[col].ffill()

    log.info(
        "Context merge: added %d context columns to %d OHLCV rows",
        len(context_cols),
        len(result),
    )
    return result


def fetch_multi_timeframe(
    symbol: str,
    period: str = "2y",
    asset_class: str = "stock",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV and compute weekly aggregate features for multi-timeframe analysis.

    Many strategies benefit from knowing the macro trend direction — e.g.,
    "only take long day-trades when the weekly trend is bullish." This function
    provides that by resampling daily data to weekly and merging key weekly
    indicators back onto the daily DataFrame.

    Added columns:
    - close_weekly_ma: 10-period weekly moving average of close (smoothed macro trend)
    - weekly_momentum: weekly close percentage change (1-week return)
    - weekly_trend: True when weekly close > weekly 10-period MA (bullish macro trend)

    Args:
        symbol: Ticker symbol (e.g., "SPY", "AAPL", "BTC/USDT")
        period: How far back ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        asset_class: "stock" or "crypto" — determines which fetcher to use

    Returns:
        Daily OHLCV DataFrame with weekly trend columns added.
    """
    # Fetch daily data using the appropriate fetcher
    if asset_class == "crypto":
        # Convert period string to days_back for crypto fetcher
        period_to_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
        }
        days_back = period_to_days.get(period, 730)
        ohlcv = fetch_crypto_data(symbol=symbol, timeframe="1d", days_back=days_back)
    else:
        ohlcv = fetch_stock_data(symbol=symbol, period=period, interval="1d")

    daily = ohlcv.df.copy()

    # Resample to weekly bars (week ending Friday)
    try:
        weekly = daily.resample("W-FRI").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
    except Exception as e:
        log.warning("Multi-timeframe %s: weekly resample failed: %s", symbol, e)
        # Return daily data without weekly features rather than crashing
        daily["close_weekly_ma"] = np.nan
        daily["weekly_momentum"] = np.nan
        daily["weekly_trend"] = False
        return daily

    # Compute weekly indicators
    weekly["close_weekly_ma"] = weekly["close"].rolling(window=10, min_periods=1).mean()
    weekly["weekly_momentum"] = weekly["close"].pct_change()
    weekly["weekly_trend"] = weekly["close"] > weekly["close_weekly_ma"]

    # Map weekly values back to daily rows (each day gets its week's values)
    # Use reindex + ffill so every daily row gets the most recent weekly value
    weekly_features = weekly[["close_weekly_ma", "weekly_momentum", "weekly_trend"]].copy()

    # Reindex weekly features to daily dates and forward-fill
    daily_index = daily.index
    weekly_features_reindexed = weekly_features.reindex(daily_index, method="ffill")

    daily["close_weekly_ma"] = weekly_features_reindexed["close_weekly_ma"]
    daily["weekly_momentum"] = weekly_features_reindexed["weekly_momentum"]
    daily["weekly_trend"] = weekly_features_reindexed["weekly_trend"].fillna(False)

    log.info(
        "Multi-timeframe %s: %d daily bars + weekly features (weekly_trend True %.0f%% of days)",
        symbol,
        len(daily),
        daily["weekly_trend"].mean() * 100 if len(daily) > 0 else 0,
    )
    return daily


# ── Crypto Data (ccxt with exchange fallback + yfinance) ──

# Exchanges to try in order — Binance is blocked in the US (HTTP 451),
# so we fall through to Kraken, Coinbase, then yfinance.
_CCXT_EXCHANGE_CHAIN = ["kraken", "coinbasepro", "binance"]

# Map ccxt pair format (BTC/USDT) to yfinance ticker (BTC-USD)
_CRYPTO_YF_MAP = {
    "BTC/USDT": "BTC-USD", "BTC/USD": "BTC-USD",
    "ETH/USDT": "ETH-USD", "ETH/USD": "ETH-USD",
    "SOL/USDT": "SOL-USD", "SOL/USD": "SOL-USD",
    "DOGE/USDT": "DOGE-USD", "DOGE/USD": "DOGE-USD",
    "ADA/USDT": "ADA-USD", "ADA/USD": "ADA-USD",
    "AVAX/USDT": "AVAX-USD", "AVAX/USD": "AVAX-USD",
    "LINK/USDT": "LINK-USD", "LINK/USD": "LINK-USD",
    "DOT/USDT": "DOT-USD", "DOT/USD": "DOT-USD",
    "MATIC/USDT": "MATIC-USD", "MATIC/USD": "MATIC-USD",
    "XRP/USDT": "XRP-USD", "XRP/USD": "XRP-USD",
    "BNB/USDT": "BNB-USD", "BNB/USD": "BNB-USD",
}


def _fetch_crypto_ccxt(
    symbol: str,
    timeframe: str,
    days_back: int,
) -> pd.DataFrame | None:
    """Try fetching crypto data from ccxt exchanges (Kraken → Coinbase → Binance)."""
    try:
        import ccxt
    except ImportError:
        log.warning("ccxt not installed, skipping exchange fetch")
        return None

    since = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    for exchange_id in _CCXT_EXCHANGE_CHAIN:
        try:
            exchange_cls = getattr(ccxt, exchange_id, None)
            if not exchange_cls:
                continue
            exchange = exchange_cls({"enableRateLimit": True})

            all_candles = []
            fetch_since = since
            limit = 1000

            while True:
                candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                fetch_since = candles[-1][0] + 1
                if len(candles) < limit:
                    break

            if all_candles:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
                df = df.dropna()
                log.info("Crypto %s via %s: %d bars", symbol, exchange_id, len(df))
                return df

        except Exception as e:
            log.debug("Exchange %s failed for %s: %s", exchange_id, symbol, e)
            continue

    return None


def _fetch_crypto_yfinance(symbol: str, days_back: int) -> pd.DataFrame | None:
    """Fallback: fetch crypto via yfinance (BTC-USD, ETH-USD, etc.)."""
    yf_ticker = _CRYPTO_YF_MAP.get(symbol)
    if not yf_ticker:
        # Try auto-converting: "BTC/USDT" -> "BTC-USD"
        base = symbol.split("/")[0] if "/" in symbol else symbol
        yf_ticker = f"{base}-USD"

    try:
        import yfinance as yf

        if days_back > 1800:
            period = "10y"
        elif days_back > 730:
            period = "5y"
        elif days_back > 365:
            period = "2y"
        elif days_back > 180:
            period = "1y"
        else:
            period = "6mo"
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            return None

        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.index.name = "timestamp"
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")

        log.info("Crypto %s via yfinance (%s): %d bars", symbol, yf_ticker, len(df))
        return df

    except Exception as e:
        log.warning("yfinance fallback failed for %s (%s): %s", symbol, yf_ticker, e)
        return None


def fetch_crypto_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    days_back: int = 730,
) -> OHLCV:
    """
    Fetch crypto OHLCV with exchange fallback chain.

    Tries: Kraken → Coinbase → Binance → yfinance.
    This ensures data works from US-based GitHub Actions runners
    (Binance blocks US IPs with HTTP 451).

    Args:
        symbol: Trading pair (e.g., "BTC/USDT", "ETH/USDT", "SOL/USDT")
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
        days_back: How many days of history to fetch.
    """
    # Try ccxt exchanges first
    df = _fetch_crypto_ccxt(symbol, timeframe, days_back)
    source = "ccxt"

    # Fallback to yfinance for daily data
    if df is None or df.empty:
        df = _fetch_crypto_yfinance(symbol, days_back)
        source = "yfinance"

    if df is None or df.empty:
        raise ValueError(f"No crypto data for {symbol} — all sources failed (ccxt + yfinance)")

    log.info("Crypto %s: %d bars (%dd, %s) via %s", symbol, len(df), days_back, timeframe, source)
    return OHLCV(df=df, symbol=symbol, asset_class=AssetClass.CRYPTO, timeframe=timeframe, source=source)


def fetch_crypto_intraday(
    symbol: str = "BTC/USDT",
    days_back: int = 30,
    timeframe: str = "5m",
) -> OHLCV:
    """
    Fetch intraday crypto data. Crypto trades 24/7 so we get dense data.

    5m bars for day trading, 1h for swing analysis.
    """
    return fetch_crypto_data(symbol=symbol, timeframe=timeframe, days_back=days_back)


# ── Polymarket Data ──


def fetch_polymarket_data(
    condition_id: str | None = None,
    slug: str | None = None,
    days_back: int = 90,
) -> OHLCV:
    """
    Fetch Polymarket prediction market price history.

    Polymarket uses a CLOB (Central Limit Order Book) API.
    Markets are binary (YES/NO) with prices between $0-$1.

    Args:
        condition_id: The Polymarket condition ID for the market.
        slug: Market slug for discovery (e.g., "will-btc-hit-100k-2025").
        days_back: How many days of history.

    Note: Polymarket's public CLOB API provides market data freely.
    We fetch from their Gamma API for historical pricing.
    """
    import httpx

    # Polymarket Gamma API for historical data
    gamma_url = "https://gamma-api.polymarket.com"

    # First, find markets if no condition_id provided
    if not condition_id and slug:
        resp = httpx.get(f"{gamma_url}/markets", params={"slug": slug}, timeout=30)
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            raise ValueError(f"No Polymarket market found for slug: {slug}")
        market = markets[0] if isinstance(markets, list) else markets
        condition_id = market.get("conditionId") or market.get("condition_id")

    if not condition_id:
        raise ValueError("Must provide either condition_id or slug")

    # Fetch price history from CLOB API
    clob_url = "https://clob.polymarket.com"
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())

    resp = httpx.get(
        f"{clob_url}/prices-history",
        params={
            "market": condition_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": 60,  # 1-hour candles
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    history = data.get("history", [])
    if not history:
        raise ValueError(f"No price history for Polymarket condition {condition_id}")

    # Convert to OHLCV-like format (prediction markets have price, not OHLCV)
    records = []
    for point in history:
        ts = pd.to_datetime(point["t"], unit="s", utc=True)
        price = float(point["p"])
        records.append({
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,  # Volume not in price history endpoint
        })

    df = pd.DataFrame(records).set_index("timestamp")
    df = df.sort_index()

    log.info("Polymarket %s: %d data points (%dd)", condition_id[:12], len(df), days_back)
    return OHLCV(
        df=df,
        symbol=condition_id[:16],
        asset_class=AssetClass.POLYMARKET,
        timeframe="1h",
        source="polymarket",
    )


def discover_polymarket_markets(
    query: str = "",
    active: bool = True,
    limit: int = 20,
) -> list[dict]:
    """
    Discover active Polymarket markets.

    Returns list of markets with: question, slug, conditionId, volume, liquidity.
    """
    import httpx

    resp = httpx.get(
        "https://gamma-api.polymarket.com/markets",
        params={
            "active": str(active).lower(),
            "limit": limit,
            **({"tag": query} if query else {}),
        },
        timeout=30,
    )
    resp.raise_for_status()
    markets = resp.json()

    results = []
    for m in markets:
        results.append({
            "question": m.get("question", ""),
            "slug": m.get("slug", ""),
            "condition_id": m.get("conditionId", ""),
            "volume": m.get("volume", 0),
            "liquidity": m.get("liquidity", 0),
            "end_date": m.get("endDate", ""),
            "outcomes": m.get("outcomes", []),
        })

    return results


# ── Economic Calendar ──


def fetch_economic_calendar(days_ahead: int = 7) -> pd.DataFrame:
    """
    Fetch economic calendar events from free sources.

    Returns events with: date, time, event, impact, forecast, previous.
    High-impact events: FOMC, CPI, NFP, GDP, PPI.

    Uses investpy / yfinance economic calendar or free API.
    Falls back to a static calendar of known recurring events.
    """
    # Static high-impact events that recur monthly/quarterly
    # These are the events that move markets and should halt trading
    HIGH_IMPACT_EVENTS = [
        "FOMC Rate Decision",
        "Non-Farm Payrolls",
        "CPI (Consumer Price Index)",
        "PPI (Producer Price Index)",
        "GDP (Gross Domestic Product)",
        "PCE Price Index",
        "Initial Jobless Claims",
        "Retail Sales",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "Federal Reserve Chair Speech",
        "FOMC Minutes",
        "ECB Rate Decision",
    ]

    try:
        import yfinance as yf

        # Try to get earnings calendar as a proxy for high-vol events
        # yfinance doesn't have a direct economic calendar
        # Fall back to static data
        raise NotImplementedError("Using static calendar")
    except Exception:
        pass

    # Generate static recurring events for the next N days
    now = datetime.now(timezone.utc)
    events = []

    # First Friday of each month = NFP
    for month_offset in range(0, max(1, days_ahead // 30) + 2):
        first_day = (now + timedelta(days=30 * month_offset)).replace(day=1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        nfp_date = first_day + timedelta(days=days_until_friday)
        if now <= nfp_date <= now + timedelta(days=days_ahead):
            events.append({
                "date": nfp_date.strftime("%Y-%m-%d"),
                "time": "08:30",
                "event": "Non-Farm Payrolls",
                "impact": "HIGH",
                "currency": "USD",
            })

    # Mid-month CPI (usually around 13th)
    for month_offset in range(0, max(1, days_ahead // 30) + 2):
        cpi_date = (now + timedelta(days=30 * month_offset)).replace(day=13)
        if now <= cpi_date <= now + timedelta(days=days_ahead):
            events.append({
                "date": cpi_date.strftime("%Y-%m-%d"),
                "time": "08:30",
                "event": "CPI (Consumer Price Index)",
                "impact": "HIGH",
                "currency": "USD",
            })

    # FOMC (roughly every 6 weeks, 8 meetings/year) — known 2025-2026 dates
    fomc_dates = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ]
    for d in fomc_dates:
        fomc_dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if now <= fomc_dt <= now + timedelta(days=days_ahead):
            events.append({
                "date": d,
                "time": "14:00",
                "event": "FOMC Rate Decision",
                "impact": "HIGH",
                "currency": "USD",
            })

    if events:
        return pd.DataFrame(events).sort_values("date")
    return pd.DataFrame(columns=["date", "time", "event", "impact", "currency"])


# ── Convenience Multi-Asset Fetcher ──


def fetch_universe(
    stocks: list[str] | None = None,
    crypto: list[str] | None = None,
    polymarket_slugs: list[str] | None = None,
    stock_period: str = "2y",
    crypto_days: int = 730,
    poly_days: int = 90,
    timeframe: str = "1d",
) -> dict[str, OHLCV]:
    """
    Fetch data for an entire trading universe across asset classes.

    Returns dict mapping symbol -> OHLCV.
    Failures are logged and skipped (won't crash the whole run).
    """
    universe: dict[str, OHLCV] = {}

    for sym in (stocks or []):
        try:
            universe[sym] = fetch_stock_data(sym, period=stock_period, interval=timeframe)
        except Exception as e:
            log.warning("Failed to fetch stock %s: %s", sym, e)

    for pair in (crypto or []):
        try:
            universe[pair] = fetch_crypto_data(pair, timeframe=timeframe, days_back=crypto_days)
        except Exception as e:
            log.warning("Failed to fetch crypto %s: %s", pair, e)

    for slug in (polymarket_slugs or []):
        try:
            universe[slug] = fetch_polymarket_data(slug=slug, days_back=poly_days)
        except Exception as e:
            log.warning("Failed to fetch Polymarket %s: %s", slug, e)

    log.info("Universe loaded: %d assets (%d stocks, %d crypto, %d polymarket)",
             len(universe),
             len(stocks or []),
             len(crypto or []),
             len(polymarket_slugs or []))

    return universe
