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
) -> OHLCV:
    """
    Fetch stock OHLCV from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g., "SPY", "AAPL", "QQQ")
        period: How far back ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        interval: Bar size ("1m", "5m", "15m", "1h", "1d", "1wk")
            Note: intraday data (1m-1h) limited to last 60 days on free tier.

    Returns:
        OHLCV with standardized DataFrame.
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


# ── Crypto Data (ccxt / Binance) ──


def fetch_crypto_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    days_back: int = 730,
) -> OHLCV:
    """
    Fetch crypto OHLCV from Binance via ccxt.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT", "ETH/USDT", "SOL/USDT")
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
        days_back: How many days of history to fetch.

    No API key needed for public historical data.
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})

    since = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    all_candles = []
    limit = 1000  # Binance max per request

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1  # Next ms after last candle
        if len(candles) < limit:
            break

    if not all_candles:
        raise ValueError(f"No crypto data for {symbol}")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.dropna()

    log.info("Crypto %s: %d bars (%dd, %s)", symbol, len(df), days_back, timeframe)
    return OHLCV(df=df, symbol=symbol, asset_class=AssetClass.CRYPTO, timeframe=timeframe, source="binance")


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
