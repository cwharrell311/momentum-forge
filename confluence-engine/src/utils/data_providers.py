"""
Centralized HTTP clients for external data sources.

Every external API call in the app goes through one of these clients.
This gives us:
1. Rate limiting — never exceed API limits
2. Error handling — consistent retry/fallback behavior
3. Single place to update if an API changes

Usage:
    fmp = FMPClient(api_key="your_key")
    quote = await fmp.get_quote("AAPL")
    rsi = await fmp.get_rsi("AAPL")
    await fmp.close()
"""

from __future__ import annotations

import httpx

from src.utils.rate_limiter import RateLimiter


class FMPClient:
    """
    Financial Modeling Prep API client.

    FMP provides: stock quotes, technical indicators (RSI, MACD, SMA),
    company profiles, and VIX data. Free tier allows ~250 requests/day.

    Docs: https://site.financialmodelingprep.com/developer/docs
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        # FMP free tier: ~5 requests/sec is safe
        self._limiter = RateLimiter(rate=4.0, max_tokens=8)

    async def _get(self, path: str, params: dict | None = None) -> list | dict | None:
        """Make a rate-limited GET request to FMP."""
        await self._limiter.acquire()

        url = f"{self.BASE_URL}/{path}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        try:
            resp = await self._client.get(url, params=all_params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"FMP API error ({e.response.status_code}): {path}")
            return None
        except httpx.RequestError as e:
            print(f"FMP request failed: {e}")
            return None

    async def get_quote(self, ticker: str) -> dict | None:
        """Get current quote (price, volume, change, 52w range)."""
        data = await self._get(f"quote/{ticker}")
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_rsi(self, ticker: str, period: int = 14) -> dict | None:
        """Get RSI technical indicator."""
        data = await self._get(
            f"technical_indicator/daily/{ticker}",
            params={"period": period, "type": "rsi"},
        )
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_macd(self, ticker: str) -> dict | None:
        """Get MACD technical indicator (line, signal, histogram)."""
        data = await self._get(
            f"technical_indicator/daily/{ticker}",
            params={"type": "macd"},
        )
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_sma(self, ticker: str, period: int = 20) -> dict | None:
        """Get Simple Moving Average for a given period."""
        data = await self._get(
            f"technical_indicator/daily/{ticker}",
            params={"period": period, "type": "sma"},
        )
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_sma_bundle(self, ticker: str) -> dict:
        """Get SMA 20, 50, and 200 in one call bundle."""
        result = {}
        for period in [20, 50, 200]:
            data = await self.get_sma(ticker, period)
            if data:
                result[f"sma_{period}"] = data.get("sma")
        return result

    async def get_vix_quote(self) -> dict | None:
        """Get current VIX level."""
        data = await self._get("quote/%5EVIX")
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_company_profile(self, ticker: str) -> dict | None:
        """Get company profile (sector, market cap, etc.)."""
        data = await self._get(f"profile/{ticker}")
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()


class UnusualWhalesClient:
    """
    Unusual Whales API client — Phase 2.

    Will provide: options flow (sweeps, blocks, unusual OI),
    GEX/dealer positioning, dark pool prints.

    Placeholder until you have a Platinum subscription.
    """

    BASE_URL = "https://api.unusualwhales.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._limiter = RateLimiter(rate=2.0, max_tokens=5)

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        """Make a rate-limited GET request to Unusual Whales."""
        if not self.api_key:
            return None

        await self._limiter.acquire()

        try:
            resp = await self._client.get(
                f"{self.BASE_URL}/{path}",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"UW API error ({e.response.status_code}): {path}")
            return None
        except httpx.RequestError as e:
            print(f"UW request failed: {e}")
            return None

    async def close(self) -> None:
        await self._client.aclose()
