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

import asyncio
import logging
import time
from datetime import date

import httpx

from src.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)


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

    async def _get(self, path: str, params: dict | None = None, retries: int = 2) -> list | dict | None:
        """Make a rate-limited GET request to FMP with retry on transient errors."""
        self._check_date_reset()

        # Guard: skip calls when quota is nearly exhausted.
        # Reserve 20 calls for VIX/regime (critical for dashboard).
        if self._call_count >= self.DAILY_QUOTA - 20:
            log.warning(
                "FMP quota nearly exhausted: %d/%d used — skipping %s. "
                "Increase SCAN_INTERVAL or upgrade FMP plan.",
                self._call_count, self.DAILY_QUOTA, path,
            )
            return None

        url = f"{self.BASE_URL}/{path}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        for attempt in range(retries + 1):
            await self._limiter.acquire()
            try:
                self._call_count += 1
                resp = await self._client.get(url, params=all_params)
                if resp.status_code == 429:
                    self._rate_limited_count += 1
                    log.warning("FMP rate limited (429): %s — quota likely exhausted", path)
                    return None
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                self._error_count += 1
                log.error("FMP API error (%d): %s", e.response.status_code, path)
                return None  # Don't retry HTTP errors (4xx/5xx are usually permanent)
            except httpx.RequestError as e:
                self._error_count += 1
                if attempt < retries:
                    wait = 2 ** attempt  # 1s, 2s
                    log.warning("FMP request failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, retries + 1, e, wait)
                    await asyncio.sleep(wait)
                else:
                    log.error("FMP request failed after %d attempts: %s", retries + 1, e)
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
    Alpaca Trading + Market Data API client.

    Supports both paper trading and live trading. Paper trading is FREE
    and uses a separate base URL (paper-api.alpaca.markets).

    Market data uses data.alpaca.markets (IEX free, SIP with funded account).
    No daily quota limit — replaces FMP for price/volume/technical data.

    Features:
    - Get account info (buying power, equity, cash)
    - List current positions
    - Place market/limit orders
    - Get order status and cancel orders
    - Get historical bars (daily, hourly, minute)
    - Get latest snapshots (price, quote, volume)

    Docs: https://docs.alpaca.markets/reference
    """

    DATA_URL = "https://data.alpaca.markets"

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        _auth_headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers=_auth_headers,
        )
        # Separate client for market data API (different base URL)
        self._data_client = httpx.AsyncClient(
            timeout=30.0,
            headers=_auth_headers,
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
            log.error("Alpaca API error (%d): %s — %s", e.response.status_code, path, body)
            return None
        except httpx.RequestError as e:
            log.error("Alpaca request failed: %s", e)
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

    # ── Market Data API ──────────────────────────────────────────

    async def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 252,
    ) -> list[dict] | None:
        """
        Get historical price bars from Alpaca Market Data API.

        Uses SIP feed (all US exchanges) for daily/historical bars — free
        accounts can access SIP data that's >15 minutes old. Daily bars
        are always historical, so SIP works without a paid subscription.

        This replaced the earlier IEX-only approach which only covered ~2-8%
        of market volume, causing most tickers to return zero bars.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            timeframe: Bar size — "1Day", "1Hour", "1Min", etc.
            limit: Number of bars to fetch (max 10000, default 252 = 1 year)

        Returns:
            List of bars with: t (timestamp), o (open), h (high), l (low),
            c (close), v (volume), n (trade count), vw (VWAP).
        """
        if not self.is_configured:
            log.warning("Alpaca bars %s: skipped — client not configured", ticker)
            return None

        await self._limiter.acquire()

        # Explicit start date ensures Alpaca returns recent bars.
        from datetime import datetime, timedelta, timezone

        start_date = (datetime.now(timezone.utc) - timedelta(days=int(limit * 1.5))).strftime("%Y-%m-%dT00:00:00Z")

        # Try without explicit feed first (uses account default),
        # then fall back to iex if empty. Paper accounts may not have SIP access.
        #
        # IMPORTANT: sort=desc + reverse ensures we always get the MOST RECENT
        # bars. With sort=asc and limit=252, Alpaca returns the oldest 252 bars
        # from start_date — silently dropping the newest data when the date
        # range contains more than 252 trading days. By sorting desc first,
        # the limit keeps the most recent bars, then we reverse to chronological
        # order for SMA calculations.
        for feed in (None, "iex"):
            try:
                params: dict = {
                    "timeframe": timeframe,
                    "limit": limit,
                    "start": start_date,
                    "sort": "desc",
                    "adjustment": "split",
                }
                if feed:
                    params["feed"] = feed

                resp = await self._data_client.get(
                    f"{self.DATA_URL}/v2/stocks/{ticker}/bars",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                bars = data.get("bars") or []
                if bars:
                    # Reverse to chronological order (oldest first) for SMA/scoring
                    bars = list(reversed(bars))
                    log.debug("Alpaca bars %s: %d bars (feed=%s)", ticker, len(bars), feed or "default")
                    return bars
                log.debug("Alpaca bars %s: empty with feed=%s, trying next", ticker, feed or "default")
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text[:200]
                except Exception:
                    pass
                log.warning("Alpaca bars %s feed=%s: HTTP %d — %s", ticker, feed or "default", e.response.status_code, body)
                continue
            except httpx.RequestError as e:
                log.error("Alpaca bars request failed for %s: %s", ticker, e)
                return None

        log.info("Alpaca bars %s: no data from any feed", ticker)
        return None

    async def get_snapshot(self, ticker: str) -> dict | None:
        """
        Get latest snapshot for a ticker (most recent trade, quote, bars).

        Returns: latestTrade, latestQuote, minuteBar, dailyBar, prevDailyBar.
        Note: Without Algo Trader Plus subscription, real-time SIP snapshots
        may not work. Falls back gracefully if the endpoint errors.
        """
        if not self.is_configured:
            return None

        await self._limiter.acquire()

        try:
            resp = await self._data_client.get(
                f"{self.DATA_URL}/v2/stocks/{ticker}/snapshot",
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error("Alpaca snapshot error (%d): %s", e.response.status_code, ticker)
            return None
        except httpx.RequestError as e:
            log.error("Alpaca snapshot request failed for %s: %s", ticker, e)
            return None

    async def close(self) -> None:
        """Shut down both HTTP clients (trading + market data)."""
        await self._client.aclose()
        await self._data_client.aclose()


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
        # UW Basic: 120 req/min = 2 req/sec steady state.
        # Burst of 8 absorbs initial parallel startup without immediately
        # queueing. Rate of 1.8/sec stays safely under 120/min (108/min actual).
        self._limiter = RateLimiter(rate=1.8, max_tokens=8)
        # Concurrency control: 5 signal processors fire via asyncio.gather(),
        # each calling UW sequentially per ticker. Without a semaphore, all 5
        # queue up on the rate limiter simultaneously, building a backlog that
        # triggers 429 cascades. Semaphore of 3 ensures smooth pacing.
        self._semaphore = asyncio.Semaphore(3)
        # Quota tracking
        self._call_count = 0
        self._error_count = 0
        self._rate_limited_count = 0
        self._last_reset_date: str = _today()
        # Cooldown: activated when 429s cascade, pauses all calls
        self._cooldown_until: float = 0.0
        self._consecutive_429s: int = 0

    def _check_date_reset(self) -> None:
        """Reset counters if it's a new day."""
        today = _today()
        if today != self._last_reset_date:
            self._call_count = 0
            self._error_count = 0
            self._rate_limited_count = 0
            self._consecutive_429s = 0
            self._cooldown_until = 0.0
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
            "rate_limited_today": self._rate_limited_count,
            "quota_limit": self.DAILY_QUOTA,
            "quota_remaining": max(0, self.DAILY_QUOTA - self._call_count),
            "quota_pct_used": round(self._call_count / self.DAILY_QUOTA * 100, 1),
            "date": self._last_reset_date,
        }

    def _read_rate_headers(self, resp: httpx.Response) -> None:
        """Read UW rate limit headers to self-correct our pacing."""
        try:
            remaining = resp.headers.get("x-uw-req-per-minute-remaining")
            if remaining is not None:
                remaining_int = int(remaining)
                if remaining_int < 10:
                    log.info(
                        "UW per-minute remaining: %d — approaching limit", remaining_int
                    )
            # Daily count from UW (more accurate than our local counter)
            daily = resp.headers.get("x-uw-daily-req-count")
            if daily is not None:
                server_count = int(daily)
                # Sync if our counter has drifted significantly
                if abs(server_count - self._call_count) > 50:
                    log.debug(
                        "UW daily count drift: ours=%d server=%d — syncing",
                        self._call_count, server_count,
                    )
                    self._call_count = server_count
        except (ValueError, TypeError):
            pass

    async def _get(self, path: str, params: dict | None = None, retries: int = 1) -> dict | list | None:
        """
        Make a rate-limited GET request to Unusual Whales.

        Protections against 429 cascades:
        1. Daily quota guard — stops calls when near 15K/day limit
        2. Global cooldown — pauses all calls after repeated 429s
        3. Semaphore — limits concurrent in-flight requests to 3
        4. Rate limiter — token bucket at 1.8 req/sec, burst of 8
        5. Backoff — 5s/10s wait on 429 (longer than the old 2s/4s)
        """
        if not self.api_key:
            return None

        self._check_date_reset()

        # Guard: skip calls when daily quota is nearly exhausted.
        # Reserve 500 calls for critical operations (flow discovery, VIX).
        if self._call_count >= self.DAILY_QUOTA - 500:
            log.warning(
                "UW quota nearly exhausted: %d/%d used — skipping %s. "
                "Increase SCAN_INTERVAL or reduce UNIVERSE_MAX_TICKERS.",
                self._call_count, self.DAILY_QUOTA, path,
            )
            return None

        # Global cooldown: after a burst of 429s, pause all requests
        # to let UW's per-minute counter reset.
        now = time.monotonic()
        if self._cooldown_until > now:
            log.debug(
                "UW in cooldown (%.0fs left) — skipping %s",
                self._cooldown_until - now, path,
            )
            return None

        url = f"{self.BASE_URL}/{path}"

        async with self._semaphore:
            for attempt in range(retries + 1):
                await self._limiter.acquire()
                try:
                    self._call_count += 1
                    resp = await self._client.get(url, params=params)

                    # Read rate limit headers for self-correction
                    self._read_rate_headers(resp)

                    if resp.status_code == 429:
                        self._rate_limited_count += 1
                        self._consecutive_429s += 1

                        # After 5 consecutive 429s, activate 30s global cooldown
                        # so UW's per-minute window can fully reset
                        if self._consecutive_429s >= 5:
                            self._cooldown_until = time.monotonic() + 30.0
                            log.warning(
                                "UW 429 flood (%d consecutive) on %s — "
                                "entering 30s global cooldown",
                                self._consecutive_429s, path,
                            )
                            return None

                        if attempt < retries:
                            wait = 5 * (attempt + 1)  # 5s, 10s
                            log.warning(
                                "UW rate limited (429): %s — waiting %ds "
                                "(attempt %d/%d)",
                                path, wait, attempt + 1, retries + 1,
                            )
                            await asyncio.sleep(wait)
                            continue

                        log.warning(
                            "UW rate limited (429): %s — skipping", path
                        )
                        return None

                    resp.raise_for_status()
                    self._consecutive_429s = 0  # Reset streak on success
                    data = resp.json()
                    # Log response shape for debugging
                    if isinstance(data, list):
                        log.debug("UW %s → %d items", path, len(data))
                    elif isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        log.debug("UW %s → dict keys: %s", path, keys)
                    return data
                except httpx.HTTPStatusError as e:
                    self._error_count += 1
                    log.error("UW API error (%d): %s", e.response.status_code, path)
                    return None  # Don't retry HTTP errors
                except httpx.RequestError as e:
                    self._error_count += 1
                    if attempt < retries:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        log.warning(
                            "UW request failed (attempt %d/%d): %s — retrying in %ds",
                            attempt + 1, retries + 1, e, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        log.error(
                            "UW request failed after %d attempts: %s",
                            retries + 1, e,
                        )
                        return None

    # ── Options Flow ─────────────────────────────────────────────

    async def get_flow_alerts(self, ticker: str, limit: int = 200) -> list[dict]:
        """
        Get unusual options flow alerts for a specific ticker.

        Returns aggregated flow alerts (sweeps, blocks, golden sweeps)
        with sentiment, premium, and alert classification.

        UW endpoint: /option-trades/flow-alerts?ticker_symbol={ticker}
        """
        data = await self._get(
            "option-trades/flow-alerts",
            params={"ticker_symbol": ticker, "limit": limit},
        )
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_flow_recent(self, ticker: str) -> list[dict]:
        """
        Get raw recent option trades for a ticker (individual fills).

        UW endpoint: /stock/{ticker}/flow-recent
        """
        data = await self._get(f"stock/{ticker}/flow-recent")
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_market_flow(self, limit: int = 200) -> list[dict]:
        """
        Get market-wide options flow alerts (all tickers).

        Useful for discovering new tickers with unusual activity.

        UW endpoint: /option-trades/flow-alerts
        """
        data = await self._get(
            "option-trades/flow-alerts",
            params={"limit": limit},
        )
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def get_flow_by_expiry(self, ticker: str, expiry: str) -> list[dict]:
        """Get flow for a specific ticker and expiration date (YYYY-MM-DD)."""
        data = await self._get(
            f"stock/{ticker}/flow-recent",
            params={"expiry": expiry},
        )
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    # ── Greek Exposure (GEX) ─────────────────────────────────────

    async def get_greek_exposure(self, ticker: str) -> dict | None:
        """
        Get spot gamma exposure (GEX) data by strike for a ticker.

        GEX = Gamma x OI x 100 x Spot Price
        - Positive GEX: dealers long gamma -> mean-reverting
        - Negative GEX: dealers short gamma -> trending

        Returns strike-level gamma exposure and key levels (gamma walls).

        UW endpoint: /stock/{ticker}/spot-exposures/strike
        """
        data = await self._get(f"stock/{ticker}/spot-exposures/strike")
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"data": data}
        return None

    async def get_greek_exposure_by_strike(self, ticker: str) -> dict | None:
        """
        Get static (non-spot) Greek exposure by strike.

        UW endpoint: /stock/{ticker}/greek-exposure/strike
        """
        data = await self._get(f"stock/{ticker}/greek-exposure/strike")
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

        UW endpoint: /shorts/{ticker}/data
        """
        data = await self._get(f"shorts/{ticker}/data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return {"data": data}
        return None

    # ── Insider Transactions ────────────────────────────────────

    async def get_insider_transactions(self, ticker: str, limit: int = 100) -> list[dict]:
        """
        Get insider transactions (SEC Form 4 filings) from UW.

        UW endpoint: /insider/transactions?ticker_symbol={ticker}
        """
        data = await self._get(
            "insider/transactions",
            params={"ticker_symbol": ticker, "limit": limit},
        )
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    # ── Volume/OI Overview ───────────────────────────────────────

    async def get_volume_oi(self, ticker: str) -> dict | None:
        """
        Get options volume and open interest overview.

        Returns aggregate put/call volume, put/call OI, put/call ratio.

        UW endpoint: /stock/{ticker}/options-volume
        """
        data = await self._get(f"stock/{ticker}/options-volume")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return {"data": data}
        return None

    # ── Interpolated IV ──────────────────────────────────────────

    async def get_interpolated_iv(self, ticker: str) -> dict | None:
        """
        Get interpolated IV data and percentiles for a ticker.

        Includes: IV rank, IV percentile, current IV, historical IV.
        Used by the volatility signal processor.

        UW endpoint: /stock/{ticker}/interpolated-iv
        """
        data = await self._get(f"stock/{ticker}/interpolated-iv")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return {"data": data}
        return None

    # ── Net Premium Ticks ────────────────────────────────────────

    async def get_net_prem_ticks(self, ticker: str) -> list[dict]:
        """
        Get net premium tick data for a ticker.

        UW endpoint: /stock/{ticker}/net-prem-ticks
        """
        data = await self._get(f"stock/{ticker}/net-prem-ticks")
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()
