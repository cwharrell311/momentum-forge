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
    Unusual Whales API client.

    Provides access to institutional-grade options flow, GEX/dealer positioning,
    dark pool prints, short interest, and options chain data.

    API Docs: https://api.unusualwhales.com/docs
    OpenAPI Spec: https://api.unusualwhales.com/api/openapi

    Basic tier ($150/mo): 120 req/min, 15K req/day, REST only.

    Rate limit headers returned by UW:
    - x-uw-daily-req-count: calls made today
    - x-uw-token-req-limit: daily limit
    - x-uw-minute-req-counter: calls this minute
    - x-uw-req-per-minute-remaining: remaining this minute
    """

    BASE_URL = "https://api.unusualwhales.com/api"
    DAILY_QUOTA = 15_000   # Basic tier
    PER_MINUTE = 120       # Basic tier

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # UW Basic: 120 req/min = 2 req/sec steady state
        self._limiter = RateLimiter(rate=2.0, max_tokens=5)
        # Quota tracking
        self._call_count = 0
        self._error_count = 0
        self._last_reset_date: str = _today()

    def _check_date_reset(self) -> None:
        """Reset counters if it's a new day."""
        today = _today()
        if today != self._last_reset_date:
            self._call_count = 0
            self._error_count = 0
            self._last_reset_date = today

    @property
    def is_configured(self) -> bool:
        """Check if UW API key is set."""
        return bool(self.api_key)

    @property
    def quota_status(self) -> dict:
        """Get current UW quota usage stats."""
        self._check_date_reset()
        return {
            "calls_today": self._call_count,
            "errors_today": self._error_count,
            "quota_limit": self.DAILY_QUOTA,
            "quota_remaining": max(0, self.DAILY_QUOTA - self._call_count),
            "quota_pct_used": round(self._call_count / self.DAILY_QUOTA * 100, 1),
            "date": self._last_reset_date,
        }

    async def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        """Make a rate-limited GET request to Unusual Whales."""
        if not self.api_key:
            return None

        self._check_date_reset()
        await self._limiter.acquire()

        try:
            self._call_count += 1
            resp = await self._client.get(
                f"{self.BASE_URL}/{path}",
                params=params,
            )
            if resp.status_code == 429:
                print(f"UW RATE LIMITED (429): {path}")
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._error_count += 1
            print(f"UW API error ({e.response.status_code}): {path}")
            return None
        except httpx.RequestError as e:
            self._error_count += 1
            print(f"UW request failed: {e}")
            return None

    # ── Options Flow ─────────────────────────────────────────────

    async def get_flow_alerts(self, ticker: str) -> list[dict]:
        """
        Get options flow alerts for a specific ticker.

        Returns recent flow alerts including sweeps, blocks, golden sweeps,
        and unusual activity. Each alert includes:
        - alert_rule: sweep, block, golden_sweep, unusual_vol, etc.
        - sentiment: bullish, bearish, neutral
        - total_premium: dollar value of the trade
        - volume, open_interest: for vol/OI analysis
        - option_type: call or put
        - strike, expiry: contract details
        - bid_ask_side: at_bid, at_ask, mid (critical for true sentiment)

        UW endpoint: /stock/{ticker}/flow
        """
        data = await self._get(f"stock/{ticker}/flow")
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_market_flow(self, limit: int = 200) -> list[dict]:
        """
        Get market-wide options flow alerts (all tickers).

        Useful for discovering new tickers with unusual activity.

        UW endpoint: /flow/alerts
        """
        data = await self._get("flow/alerts", params={"limit": limit})
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_flow_by_expiry(self, ticker: str, expiry: str) -> list[dict]:
        """Get flow for a specific ticker and expiration date (YYYY-MM-DD)."""
        data = await self._get(f"stock/{ticker}/flow", params={"expiry": expiry})
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    # ── Greek Exposure (GEX) ─────────────────────────────────────

    async def get_greek_exposure(self, ticker: str) -> dict | None:
        """
        Get gamma/delta exposure (GEX/DEX) data for a ticker.

        GEX = Gamma x OI x 100 x Spot Price
        - Positive GEX: dealers long gamma -> mean-reverting
        - Negative GEX: dealers short gamma -> trending

        Returns strike-level gamma exposure and key levels (gamma walls).

        UW endpoint: /stock/{ticker}/greek-exposure
        """
        data = await self._get(f"stock/{ticker}/greek-exposure")
        if isinstance(data, dict):
            return data
        return None

    async def get_greek_exposure_by_expiry(self, ticker: str, expiry: str) -> dict | None:
        """Get GEX data filtered by a specific expiration date."""
        data = await self._get(
            f"stock/{ticker}/greek-exposure",
            params={"expiry": expiry},
        )
        if isinstance(data, dict):
            return data
        return None

    # ── Options Chain (for IV/Volatility analysis) ───────────────

    async def get_option_chain(self, ticker: str) -> list[dict]:
        """
        Get the full options chain for a ticker.

        Each contract includes: strike, expiry, option_type, bid, ask,
        implied_volatility, volume, open_interest, delta, gamma, theta, vega.

        Used for: IV rank, volatility surface, put/call skew, term structure.

        UW endpoint: /stock/{ticker}/option-contracts
        """
        data = await self._get(f"stock/{ticker}/option-contracts")
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_option_chain_by_expiry(self, ticker: str, expiry: str) -> list[dict]:
        """Get options chain for a specific expiration date."""
        data = await self._get(
            f"stock/{ticker}/option-contracts",
            params={"expiry": expiry},
        )
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    # ── Dark Pool ────────────────────────────────────────────────

    async def get_dark_pool_flow(self, ticker: str) -> list[dict]:
        """
        Get dark pool (off-exchange) prints for a ticker.

        FINRA ATS data showing block trades and large off-exchange prints.
        Block trades: >10K shares or >$200K notional.

        Each print includes: volume, price, notional_value, trade_count,
        dark_pool_short_volume, date.

        UW endpoint: /darkpool/{ticker}
        """
        data = await self._get(f"darkpool/{ticker}")
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    # ── Short Interest ───────────────────────────────────────────

    async def get_short_interest(self, ticker: str) -> dict | None:
        """
        Get short interest data for a ticker.

        Includes: short_interest (shares), short_interest_pct_float,
        days_to_cover, short_interest_change_pct, date.

        Updated bi-weekly by FINRA.

        UW endpoint: /stock/{ticker}/short-interest
        """
        data = await self._get(f"stock/{ticker}/short-interest")
        if isinstance(data, dict):
            return data
        return None

    # ── Volume/OI Overview ───────────────────────────────────────

    async def get_volume_oi(self, ticker: str) -> dict | None:
        """
        Get options volume and open interest overview.

        Returns aggregate put/call volume, put/call OI, put/call ratio.

        UW endpoint: /stock/{ticker}/volume-oi
        """
        data = await self._get(f"stock/{ticker}/volume-oi")
        if isinstance(data, dict):
            return data
        return None

    # ── Ticker Overview ──────────────────────────────────────────

    async def get_ticker_overview(self, ticker: str) -> dict | None:
        """
        Get a comprehensive overview of a ticker from UW.

        UW endpoint: /stock/{ticker}
        """
        data = await self._get(f"stock/{ticker}")
        if isinstance(data, dict):
            return data
        return None

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()
