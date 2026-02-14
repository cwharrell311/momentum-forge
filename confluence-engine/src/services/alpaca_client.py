"""
Alpaca Broker Client — paper and live trading via Alpaca API.

Wraps alpaca-py (the official SDK) into the async interface that
AutonomousTrader expects. All methods return plain dicts so the
rest of the codebase never imports alpaca-py directly.

Usage:
    client = AlpacaClient.from_env()          # reads ALPACA_KEY / ALPACA_SECRET
    account = await client.get_account()       # {'equity': '100000', 'cash': '98500', ...}
    bars = await client.get_bars("SPY", "1Day", limit=252)
    order = await client.submit_order("SPY", 10, "buy", "market", "day")
    positions = await client.get_positions()

Set ALPACA_PAPER=true (default) for paper trading.
Set ALPACA_PAPER=false for live trading (real money).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import partial

log = logging.getLogger("forge.alpaca")


class AlpacaClient:
    """
    Async wrapper around alpaca-py SDK.

    The alpaca-py SDK is synchronous, so we run blocking calls
    in a thread pool via asyncio.to_thread().
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self.paper = paper
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)

        mode = "PAPER" if paper else "LIVE"
        log.info("Alpaca client initialized (%s)", mode)

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        """Create client from environment variables."""
        api_key = os.environ.get("ALPACA_KEY", os.environ.get("ALPACA_API_KEY", ""))
        secret_key = os.environ.get("ALPACA_SECRET", os.environ.get("ALPACA_SECRET_KEY", ""))
        paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_KEY and ALPACA_SECRET env vars. "
                "Get free keys at https://app.alpaca.markets"
            )

        return cls(api_key, secret_key, paper=paper)

    # ── Account ──

    async def get_account(self) -> dict:
        """Get account info (equity, cash, buying power, etc.)."""
        account = await asyncio.to_thread(self._trading.get_account)
        return {
            "id": str(account.id),
            "equity": str(account.equity),
            "cash": str(account.cash),
            "buying_power": str(account.buying_power),
            "portfolio_value": str(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
            "status": str(account.status),
        }

    # ── Positions ──

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = await asyncio.to_thread(self._trading.get_all_positions)
        return [
            {
                "symbol": str(p.symbol),
                "qty": str(p.qty),
                "side": str(p.side),
                "avg_entry_price": str(p.avg_entry_price),
                "market_value": str(p.market_value),
                "cost_basis": str(p.cost_basis),
                "unrealized_pl": str(p.unrealized_pl),
                "unrealized_plpc": str(p.unrealized_plpc),
                "current_price": str(p.current_price),
                "change_today": str(p.change_today),
            }
            for p in positions
        ]

    async def get_position(self, symbol: str) -> dict | None:
        """Get position for a single symbol."""
        try:
            p = await asyncio.to_thread(self._trading.get_open_position, symbol)
            return {
                "symbol": str(p.symbol),
                "qty": str(p.qty),
                "side": str(p.side),
                "avg_entry_price": str(p.avg_entry_price),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl),
                "unrealized_plpc": str(p.unrealized_plpc),
                "current_price": str(p.current_price),
            }
        except Exception:
            return None

    async def close_position(self, symbol: str) -> dict:
        """Close all shares of a position."""
        result = await asyncio.to_thread(self._trading.close_position, symbol)
        return {"symbol": symbol, "status": "closed", "order_id": str(getattr(result, "id", ""))}

    async def close_all_positions(self) -> list[dict]:
        """Close all open positions (emergency liquidation)."""
        results = await asyncio.to_thread(self._trading.close_all_positions, cancel_orders=True)
        return [{"status": "closed_all", "count": len(results) if results else 0}]

    # ── Orders ──

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict:
        """
        Submit an order to Alpaca.

        Args:
            symbol: Ticker (e.g., "SPY")
            qty: Number of shares
            side: "buy" or "sell"
            type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Required for limit/stop_limit orders
            stop_price: Required for stop/stop_limit orders
        """
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }.get(time_in_force.lower(), TimeInForce.DAY)

        if type == "market":
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
        elif type == "limit":
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
        elif type == "stop":
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                stop_price=stop_price,
            )
        elif type == "stop_limit":
            req = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        else:
            raise ValueError(f"Unknown order type: {type}")

        order = await asyncio.to_thread(self._trading.submit_order, req)

        mode = "PAPER" if self.paper else "LIVE"
        log.info(
            "ORDER [%s]: %s %d %s @ %s (id=%s)",
            mode, side.upper(), qty, symbol, type, order.id,
        )

        return {
            "id": str(order.id),
            "symbol": str(order.symbol),
            "qty": str(order.qty),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
        }

    async def get_orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        """Get orders filtered by status."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        req = GetOrdersRequest(
            status=status_map.get(status, QueryOrderStatus.OPEN),
            limit=limit,
        )
        orders = await asyncio.to_thread(self._trading.get_orders, req)
        return [
            {
                "id": str(o.id),
                "symbol": str(o.symbol),
                "qty": str(o.qty),
                "side": str(o.side),
                "type": str(o.type),
                "status": str(o.status),
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
            }
            for o in orders
        ]

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        result = await asyncio.to_thread(self._trading.cancel_orders)
        return {"status": "cancelled_all", "count": len(result) if result else 0}

    # ── Market Data ──

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 252,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict]:
        """
        Get historical OHLCV bars from Alpaca.

        Args:
            symbol: Ticker
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            limit: Max bars to return
            start: Start datetime (default: limit days ago)
            end: End datetime (default: now)
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf_map = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, TimeFrame.Minute) if hasattr(TimeFrame, '__call__') else TimeFrame.Minute,
            "15min": TimeFrame(15, TimeFrame.Minute) if hasattr(TimeFrame, '__call__') else TimeFrame.Minute,
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe.lower(), TimeFrame.Day)

        if not end:
            end = datetime.now(timezone.utc)
        if not start:
            # Estimate start based on limit and timeframe
            if "day" in timeframe.lower():
                start = end - timedelta(days=int(limit * 1.5))  # Extra for weekends
            elif "hour" in timeframe.lower():
                start = end - timedelta(hours=limit * 2)
            else:
                start = end - timedelta(days=limit)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )

        try:
            bars_data = await asyncio.to_thread(self._data.get_stock_bars, req)
            bars = bars_data[symbol] if symbol in bars_data else []

            result = [
                {
                    "t": str(bar.timestamp),
                    "o": float(bar.open),
                    "h": float(bar.high),
                    "l": float(bar.low),
                    "c": float(bar.close),
                    "v": int(bar.volume),
                }
                for bar in bars
            ]

            return result[-limit:]  # Trim to requested limit

        except Exception as e:
            log.warning("Failed to get bars for %s: %s", symbol, e)
            return []

    async def get_latest_quote(self, symbol: str) -> dict | None:
        """Get the latest quote for a symbol."""
        from alpaca.data.requests import StockLatestQuoteRequest

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = await asyncio.to_thread(self._data.get_stock_latest_quote, req)
            quote = quotes.get(symbol)
            if not quote:
                return None
            return {
                "symbol": symbol,
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size),
                "timestamp": str(quote.timestamp),
            }
        except Exception as e:
            log.warning("Failed to get quote for %s: %s", symbol, e)
            return None

    async def get_latest_trade(self, symbol: str) -> dict | None:
        """Get the latest trade for a symbol."""
        from alpaca.data.requests import StockLatestTradeRequest

        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trades = await asyncio.to_thread(self._data.get_stock_latest_trade, req)
            trade = trades.get(symbol)
            if not trade:
                return None
            return {
                "symbol": symbol,
                "price": float(trade.price),
                "size": int(trade.size),
                "timestamp": str(trade.timestamp),
            }
        except Exception as e:
            log.warning("Failed to get trade for %s: %s", symbol, e)
            return None

    # ── Utility ──

    async def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = await asyncio.to_thread(self._trading.get_clock)
        return clock.is_open

    async def get_clock(self) -> dict:
        """Get market clock info."""
        clock = await asyncio.to_thread(self._trading.get_clock)
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }

    def __repr__(self) -> str:
        mode = "paper" if self.paper else "live"
        return f"AlpacaClient(mode={mode})"
