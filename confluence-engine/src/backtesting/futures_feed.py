"""
Futures data feed via Interactive Brokers (IBKR) TWS/Gateway.

Provides historical OHLCV data for index futures (ES, NQ, YM, RTY),
commodity futures, and treasury futures via the ib_insync library.

Setup:
    1. Install IB Gateway or TWS (Trader Workstation)
    2. Enable API connections in Gateway/TWS config
    3. pip install ib_insync
    4. Set env vars: IBKR_HOST (default 127.0.0.1), IBKR_PORT (default 4002)

Note: IBKR requires a funded account for live data. Paper trading accounts
get delayed data. For backtesting, we can also fall back to yfinance for
futures ETF proxies (ES -> SPY, NQ -> QQQ).

ib_insync docs: https://ib-insync.readthedocs.io/
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.backtesting.data_feeds import OHLCV, AssetClass

log = logging.getLogger("forge.futures_feed")

# Standard futures contracts
FUTURES_CONTRACTS = {
    "ES": {"exchange": "CME", "name": "E-mini S&P 500", "multiplier": 50, "tick_size": 0.25},
    "NQ": {"exchange": "CME", "name": "E-mini Nasdaq 100", "multiplier": 20, "tick_size": 0.25},
    "YM": {"exchange": "CBOT", "name": "E-mini Dow", "multiplier": 5, "tick_size": 1.0},
    "RTY": {"exchange": "CME", "name": "E-mini Russell 2000", "multiplier": 50, "tick_size": 0.10},
    "MES": {"exchange": "CME", "name": "Micro E-mini S&P 500", "multiplier": 5, "tick_size": 0.25},
    "MNQ": {"exchange": "CME", "name": "Micro E-mini Nasdaq 100", "multiplier": 2, "tick_size": 0.25},
    "GC": {"exchange": "COMEX", "name": "Gold Futures", "multiplier": 100, "tick_size": 0.10},
    "CL": {"exchange": "NYMEX", "name": "Crude Oil Futures", "multiplier": 1000, "tick_size": 0.01},
    "ZB": {"exchange": "CBOT", "name": "30-Year Treasury Bond", "multiplier": 1000, "tick_size": 1/32},
    "ZN": {"exchange": "CBOT", "name": "10-Year Treasury Note", "multiplier": 1000, "tick_size": 1/64},
}

# ETF proxies for when IBKR is unavailable
FUTURES_ETF_PROXY = {
    "ES": "SPY",
    "NQ": "QQQ",
    "YM": "DIA",
    "RTY": "IWM",
    "MES": "SPY",
    "MNQ": "QQQ",
    "GC": "GLD",
    "CL": "USO",
    "ZB": "TLT",
    "ZN": "IEF",
}


class IBKRClient:
    """
    Interactive Brokers client using ib_insync.

    Connects to TWS or IB Gateway for historical data and trading.
    Falls back to ETF proxies if IBKR is unavailable.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int = 1,
    ):
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("IBKR_PORT", "4002"))
        self.client_id = client_id
        self._ib = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to TWS/Gateway. Returns True if successful."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            log.info("Connected to IBKR at %s:%d", self.host, self.port)
            return True
        except Exception as e:
            log.warning("IBKR connection failed: %s (will use ETF proxies)", e)
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False

    def fetch_futures_history(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
    ) -> pd.DataFrame:
        """
        Fetch historical futures data from IBKR.

        Args:
            symbol: Futures symbol (e.g., "ES", "NQ")
            duration: How far back ("1 D", "1 W", "1 M", "1 Y", "2 Y")
            bar_size: Bar granularity ("1 min", "5 mins", "1 hour", "1 day")
            what_to_show: "TRADES", "MIDPOINT", "BID", "ASK"

        Returns:
            DataFrame with OHLCV columns.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        from ib_insync import Future

        contract_info = FUTURES_CONTRACTS.get(symbol)
        if not contract_info:
            raise ValueError(f"Unknown futures symbol: {symbol}")

        # Create continuous futures contract
        contract = Future(
            symbol=symbol,
            exchange=contract_info["exchange"],
            currency="USD",
        )

        # Qualify the contract (resolves to front month)
        contracts = self._ib.qualifyContracts(contract)
        if not contracts:
            raise ValueError(f"Could not qualify contract for {symbol}")

        contract = contracts[0]
        log.info("Qualified contract: %s %s (expiry: %s)",
                 contract.symbol, contract.exchange, contract.lastTradeDateOrContractMonth)

        # Request historical data
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            raise ValueError(f"No historical data for {symbol}")

        # Convert to DataFrame
        rows = []
        for bar in bars:
            rows.append({
                "timestamp": pd.Timestamp(bar.date).tz_localize("UTC"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp")

        log.info("IBKR %s: %d bars (%s, %s)", symbol, len(df), duration, bar_size)
        return df


def _fetch_futures_via_proxy(
    symbol: str,
    period: str = "2y",
) -> pd.DataFrame:
    """
    Fetch futures-equivalent data using ETF proxies via yfinance.

    Used as fallback when IBKR is not available.
    """
    proxy = FUTURES_ETF_PROXY.get(symbol)
    if not proxy:
        raise ValueError(f"No ETF proxy for futures symbol: {symbol}")

    import yfinance as yf
    ticker = yf.Ticker(proxy)
    df = ticker.history(period=period, interval="1d")

    if df.empty:
        raise ValueError(f"No proxy data for {symbol} (proxy={proxy})")

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")

    # Scale prices by contract multiplier for realistic P&L
    contract_info = FUTURES_CONTRACTS.get(symbol, {})
    multiplier = contract_info.get("multiplier", 1)

    # Store multiplier in DataFrame attrs for use by strategies
    df.attrs["futures_symbol"] = symbol
    df.attrs["futures_multiplier"] = multiplier
    df.attrs["proxy_etf"] = proxy
    df.attrs["is_proxy"] = True

    log.info("Futures proxy %s -> %s: %d bars (multiplier=%d)",
             symbol, proxy, len(df), multiplier)
    return df


def fetch_futures_data(
    symbol: str,
    days_back: int = 365,
    bar_size: str = "1 day",
    ibkr_client: IBKRClient | None = None,
) -> OHLCV:
    """
    Fetch futures data, trying IBKR first, falling back to ETF proxy.

    Args:
        symbol: Futures symbol ("ES", "NQ", "GC", etc.)
        days_back: Days of history
        bar_size: Bar size for IBKR ("1 day", "1 hour", "5 mins")
        ibkr_client: Optional IBKR client instance

    Returns:
        OHLCV with AssetClass.FUTURES
    """
    symbol = symbol.upper()

    # Try IBKR first
    if ibkr_client is not None:
        if not ibkr_client._connected:
            ibkr_client.connect()

        if ibkr_client._connected:
            try:
                # Convert days_back to IBKR duration string
                if days_back <= 1:
                    duration = "1 D"
                elif days_back <= 7:
                    duration = f"{days_back} D"
                elif days_back <= 365:
                    months = max(1, days_back // 30)
                    duration = f"{months} M"
                else:
                    years = max(1, days_back // 365)
                    duration = f"{years} Y"

                df = ibkr_client.fetch_futures_history(
                    symbol, duration=duration, bar_size=bar_size,
                )
                return OHLCV(
                    df=df, symbol=symbol, asset_class=AssetClass.FUTURES,
                    timeframe=bar_size, source="ibkr",
                )
            except Exception as e:
                log.warning("IBKR fetch failed for %s: %s (falling back to proxy)", symbol, e)

    # Fallback to ETF proxy
    period_map = {
        365: "1y", 730: "2y", 1825: "5y", 3650: "10y",
    }
    period = "2y"
    for d, p in sorted(period_map.items()):
        if days_back <= d:
            period = p
            break

    df = _fetch_futures_via_proxy(symbol, period=period)

    return OHLCV(
        df=df, symbol=symbol, asset_class=AssetClass.FUTURES,
        timeframe="1d", source=f"proxy:{FUTURES_ETF_PROXY.get(symbol, '?')}",
    )
