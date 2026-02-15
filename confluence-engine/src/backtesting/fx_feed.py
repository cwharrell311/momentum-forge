"""
FX data feed via OANDA v20 REST API.

Provides historical OHLCV candlestick data for major and minor FX pairs.
OANDA offers free practice accounts with full API access — no paid
subscription required for historical data.

Supported pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF,
NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, and ~60 more.

Setup:
    1. Create free practice account at https://www.oanda.com
    2. Generate API token in account settings
    3. Set env vars: OANDA_API_TOKEN, OANDA_ACCOUNT_ID

API docs: https://developer.oanda.com/rest-live-v20/instrument-ep/
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.backtesting.data_feeds import OHLCV, AssetClass

log = logging.getLogger("forge.fx_feed")

# OANDA API endpoints
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"

# Default FX pairs universe
DEFAULT_FX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "EUR_AUD", "EUR_CAD", "EUR_CHF",
]

# Map display format to OANDA format
def _to_oanda_pair(pair: str) -> str:
    """Convert 'EUR/USD' or 'EURUSD' to 'EUR_USD'."""
    pair = pair.upper().replace("/", "_")
    if "_" not in pair and len(pair) == 6:
        pair = pair[:3] + "_" + pair[3:]
    return pair


def _to_display_pair(oanda_pair: str) -> str:
    """Convert 'EUR_USD' to 'EUR/USD'."""
    return oanda_pair.replace("_", "/")


class OandaClient:
    """
    OANDA v20 REST API client for FX data.

    Fetches historical candles and account info. Practice accounts
    have full API access with no rate limits for historical data.
    """

    def __init__(
        self,
        api_token: str | None = None,
        account_id: str | None = None,
        practice: bool = True,
    ):
        self.api_token = api_token or os.environ.get("OANDA_API_TOKEN", "")
        self.account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self.base_url = OANDA_PRACTICE_URL if practice else OANDA_LIVE_URL

        if not self.api_token:
            log.warning("OANDA_API_TOKEN not set — FX data will be unavailable")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    def fetch_candles(
        self,
        pair: str,
        granularity: str = "D",
        count: int = 500,
        from_time: str | None = None,
        to_time: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from OANDA.

        Args:
            pair: Instrument name (e.g., "EUR_USD")
            granularity: "S5","S10","S15","S30","M1","M2","M4","M5","M10",
                        "M15","M30","H1","H2","H3","H4","H6","H8","H12","D","W","M"
            count: Number of candles (max 5000)
            from_time: RFC3339 start time
            to_time: RFC3339 end time

        Returns:
            DataFrame with open, high, low, close, volume columns.
        """
        import httpx

        pair = _to_oanda_pair(pair)
        url = f"{self.base_url}/v3/instruments/{pair}/candles"

        params: dict = {
            "granularity": granularity,
            "price": "M",  # mid prices
        }
        if from_time and to_time:
            params["from"] = from_time
            params["to"] = to_time
        else:
            params["count"] = min(count, 5000)

        try:
            resp = httpx.get(url, params=params, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise ValueError(f"OANDA API error for {pair}: {e}")

        candles = data.get("candles", [])
        if not candles:
            raise ValueError(f"No candles returned for {pair}")

        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": pd.Timestamp(c["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")

        log.info("OANDA %s: %d candles (%s)", pair, len(df), granularity)
        return df

    def fetch_history(
        self,
        pair: str,
        days_back: int = 365,
        granularity: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch extended history by paginating through OANDA's 5000-candle limit.

        Args:
            pair: FX pair (e.g., "EUR_USD")
            days_back: How many days of history
            granularity: Bar size

        Returns:
            Concatenated DataFrame of all candles.
        """
        pair = _to_oanda_pair(pair)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)

        # Estimate candles needed
        granularity_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D": 1440, "W": 10080, "M": 43200,
        }
        mins = granularity_minutes.get(granularity, 1440)
        total_candles = int((days_back * 24 * 60) / mins)

        all_dfs = []
        current_start = start

        while current_start < end:
            chunk_end = min(
                current_start + timedelta(minutes=mins * 4999),
                end,
            )

            try:
                df = self.fetch_candles(
                    pair,
                    granularity=granularity,
                    from_time=current_start.isoformat(),
                    to_time=chunk_end.isoformat(),
                )
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                log.warning("OANDA chunk fetch failed: %s", e)

            current_start = chunk_end

        if not all_dfs:
            raise ValueError(f"No OANDA data for {pair} over {days_back} days")

        result = pd.concat(all_dfs)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        log.info("OANDA %s: %d total candles over %d days", pair, len(result), days_back)
        return result


def fetch_fx_data(
    pair: str,
    days_back: int = 365,
    granularity: str = "D",
    client: OandaClient | None = None,
) -> OHLCV:
    """
    Fetch FX pair data and return standardized OHLCV.

    Args:
        pair: FX pair ("EUR/USD", "EURUSD", or "EUR_USD")
        days_back: Days of history
        granularity: OANDA granularity string

    Returns:
        OHLCV with AssetClass.FOREX
    """
    if client is None:
        client = OandaClient()

    oanda_pair = _to_oanda_pair(pair)
    display_pair = _to_display_pair(oanda_pair)

    df = client.fetch_history(oanda_pair, days_back=days_back, granularity=granularity)

    return OHLCV(
        df=df,
        symbol=display_pair,
        asset_class=AssetClass.FOREX,
        timeframe=granularity,
        source="oanda",
    )
