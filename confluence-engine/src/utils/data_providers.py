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

from datetime import date

import httpx

from src.utils.rate_limiter import RateLimiter


def _today() -> str:
    """Current date as YYYY-MM-DD string for quota tracking."""
    return date.today().isoformat()


class FMPClient:
    """
    Financial Modeling Prep API client.

    Uses the new /stable/ endpoints (FMP retired the legacy /api/v3/ URLs
    in August 2025). Free tier allows ~250 requests/day.

    Tracks API call counts so you can monitor quota usage.

    Docs: https://site.financialmodelingprep.com/developer/docs
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    DAILY_QUOTA = 250  # FMP free tier limit

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        # FMP free tier: ~5 requests/sec is safe
        self._limiter = RateLimiter(rate=4.0, max_tokens=8)
        # Quota tracking
        self._call_count = 0
        self._error_count = 0
        self._rate_limited_count = 0
        self._last_reset_date: str = _today()

    def _check_date_reset(self) -> None:
        """Reset counters if it's a new day (FMP quotas reset daily)."""
        today = _today()
        if today != self._last_reset_date:
            self._call_count = 0
            self._error_count = 0
            self._rate_limited_count = 0
            self._last_reset_date = today

    @property
    def quota_status(self) -> dict:
        """Get current quota usage stats."""
        self._check_date_reset()
        return {
            "calls_today": self._call_count,
            "errors_today": self._error_count,
            "rate_limited_today": self._rate_limited_count,
            "quota_limit": self.DAILY_QUOTA,
            "quota_remaining": max(0, self.DAILY_QUOTA - self._call_count),
            "quota_pct_used": round(self._call_count / self.DAILY_QUOTA * 100, 1),
            "date": self._last_reset_date,
        }

    async def _get(self, path: str, params: dict | None = None) -> list | dict | None:
        """Make a rate-limited GET request to FMP."""
        self._check_date_reset()
        await self._limiter.acquire()

        url = f"{self.BASE_URL}/{path}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        try:
            self._call_count += 1
            resp = await self._client.get(url, params=all_params)
            if resp.status_code == 429:
                self._rate_limited_count += 1
                print(f"FMP RATE LIMITED (429): {path} — quota likely exhausted")
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._error_count += 1
            print(f"FMP API error ({e.response.status_code}): {path}")
            return None
        except httpx.RequestError as e:
            self._error_count += 1
            print(f"FMP request failed: {e}")
            return None

    async def get_quote(self, ticker: str) -> dict | None:
        """Get current quote (price, volume, change, 52w range)."""
        data = await self._get("quote", params={"symbol": ticker})
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_rsi(self, ticker: str, period: int = 14) -> dict | None:
        """Get RSI technical indicator."""
        data = await self._get(
            "technical-indicators/rsi",
            params={"symbol": ticker, "periodLength": period, "timeframe": "1day"},
        )
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_macd(self, ticker: str) -> dict | None:
        """Get MACD technical indicator (line, signal, histogram)."""
        data = await self._get(
            "technical-indicators/macd",
            params={"symbol": ticker, "timeframe": "1day"},
        )
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_sma(self, ticker: str, period: int = 20) -> dict | None:
        """Get Simple Moving Average for a given period."""
        data = await self._get(
            "technical-indicators/sma",
            params={"symbol": ticker, "periodLength": period, "timeframe": "1day"},
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
        data = await self._get("quote", params={"symbol": "^VIX"})
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_company_profile(self, ticker: str) -> dict | None:
        """Get company profile (sector, market cap, etc.)."""
        data = await self._get("profile", params={"symbol": ticker})
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_insider_trading(self, ticker: str, limit: int = 50) -> list[dict]:
        """
        Get recent insider transactions (SEC Form 4 filings).

        Returns a list of transactions with fields like:
        - transactionType: "P-Purchase" or "S-Sale"
        - securitiesTransacted: number of shares
        - price: transaction price
        - reportingName: insider name
        - typeOfOwner: "officer", "director", "10 percent owner"
        - transactionDate: date of the transaction

        FMP stable endpoint: /insider-trading?symbol=AAPL
        """
        data = await self._get(
            "insider-trading",
            params={"symbol": ticker, "limit": limit},
        )
        if isinstance(data, list):
            return data
        return []

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()


class AlpacaClient:
    """
    Alpaca Trading API client.

    Supports both paper trading and live trading. Paper trading is FREE
    and uses a separate base URL (paper-api.alpaca.markets).

    Features:
    - Get account info (buying power, equity, cash)
    - List current positions
    - Place market/limit orders
    - Get order status and cancel orders

    Docs: https://docs.alpaca.markets/reference
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )
        self._limiter = RateLimiter(rate=3.0, max_tokens=5)

    @property
    def is_configured(self) -> bool:
        """Check if Alpaca API keys are set."""
        return bool(self.api_key and self.secret_key)

    @property
    def is_paper(self) -> bool:
        """Check if we're using paper trading."""
        return "paper" in self.base_url

    async def _request(self, method: str, path: str, json: dict | None = None) -> dict | list | None:
        """Make an authenticated request to Alpaca."""
        if not self.is_configured:
            return None

        await self._limiter.acquire()

        try:
            resp = await self._client.request(
                method,
                f"{self.base_url}/v2/{path}",
                json=json,
            )
            resp.raise_for_status()
            if resp.status_code == 204:
                return {}
            return resp.json()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.json().get("message", "")
            except Exception:
                body = e.response.text[:200]
            print(f"Alpaca API error ({e.response.status_code}): {path} — {body}")
            return None
        except httpx.RequestError as e:
            print(f"Alpaca request failed: {e}")
            return None

    async def get_account(self) -> dict | None:
        """Get account details: equity, buying power, cash, status."""
        return await self._request("GET", "account")

    async def get_positions(self) -> list | None:
        """Get all open positions."""
        return await self._request("GET", "positions")

    async def get_position(self, ticker: str) -> dict | None:
        """Get position for a specific ticker."""
        return await self._request("GET", f"positions/{ticker}")

    async def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,          # "buy" or "sell"
        order_type: str = "market",  # "market" or "limit"
        limit_price: float | None = None,
        time_in_force: str = "day",  # "day", "gtc", "ioc"
    ) -> dict | None:
        """
        Place an order on Alpaca.

        For market orders: just ticker, qty, side.
        For limit orders: also pass limit_price.
        """
        payload: dict = {
            "symbol": ticker.upper(),
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price is not None:
            payload["limit_price"] = str(limit_price)

        return await self._request("POST", "orders", json=payload)

    async def get_orders(self, status: str = "open", limit: int = 50) -> list | None:
        """Get orders. Status: open, closed, all."""
        return await self._request("GET", f"orders?status={status}&limit={limit}")

    async def cancel_order(self, order_id: str) -> dict | None:
        """Cancel an open order."""
        return await self._request("DELETE", f"orders/{order_id}")

    async def close_position(self, ticker: str) -> dict | None:
        """Close an entire position for a ticker (sell all shares)."""
        return await self._request("DELETE", f"positions/{ticker}")

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
