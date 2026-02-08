"""
Broker API routes — Alpaca paper/live trading.

Enables trade execution directly from the dashboard. Uses the Alpaca
Trade API v2, which supports both paper trading (free) and live trading.

IMPORTANT: Paper trading is the default. Live trading requires:
1. Setting LIVE_TRADING_ENABLED=true in .env
2. Changing ALPACA_BASE_URL to https://api.alpaca.markets

Safety features:
- Paper trading by default (can't lose real money by accident)
- Position size limits (configurable)
- All orders logged with confluence context
- Live trading requires explicit .env flag

GET  /api/v1/broker/account    → Account info (equity, buying power)
GET  /api/v1/broker/positions  → All open positions
POST /api/v1/broker/orders     → Place an order
GET  /api/v1/broker/orders     → List orders
DEL  /api/v1/broker/orders/:id → Cancel an order
DEL  /api/v1/broker/positions/:ticker → Close a position
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Request/Response Schemas ──

class OrderRequest(BaseModel):
    ticker: str
    qty: int
    side: str           # "buy" or "sell"
    order_type: str = "market"   # "market" or "limit"
    limit_price: float | None = None
    time_in_force: str = "day"   # "day", "gtc", "ioc"


class AccountResponse(BaseModel):
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    status: str
    trading_blocked: bool
    is_paper: bool


class PositionResponse(BaseModel):
    ticker: str
    qty: float
    side: str
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class BrokerStatus(BaseModel):
    configured: bool
    is_paper: bool
    connected: bool


# ── Helpers ──

def _get_client():
    from src.api.dependencies import get_alpaca_client
    client = get_alpaca_client()
    if not client or not client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Alpaca not configured. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file.",
        )
    return client


# ── Endpoints ──

@router.get("/status", response_model=BrokerStatus)
async def broker_status():
    """Check if Alpaca is configured and connected."""
    from src.api.dependencies import get_alpaca_client
    client = get_alpaca_client()

    if not client or not client.is_configured:
        return BrokerStatus(configured=False, is_paper=True, connected=False)

    account = await client.get_account()
    return BrokerStatus(
        configured=True,
        is_paper=client.is_paper,
        connected=account is not None,
    )


@router.get("/account", response_model=AccountResponse)
async def get_account():
    """Get Alpaca account info: equity, buying power, cash."""
    client = _get_client()
    account = await client.get_account()

    if not account:
        raise HTTPException(status_code=502, detail="Failed to connect to Alpaca")

    return AccountResponse(
        equity=float(account.get("equity", 0)),
        buying_power=float(account.get("buying_power", 0)),
        cash=float(account.get("cash", 0)),
        portfolio_value=float(account.get("portfolio_value", 0)),
        status=account.get("status", "unknown"),
        trading_blocked=account.get("trading_blocked", False),
        is_paper=client.is_paper,
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions():
    """Get all open Alpaca positions."""
    client = _get_client()
    positions = await client.get_positions()

    if positions is None:
        raise HTTPException(status_code=502, detail="Failed to fetch positions")

    return [
        PositionResponse(
            ticker=p.get("symbol", ""),
            qty=float(p.get("qty", 0)),
            side=p.get("side", "long"),
            avg_entry=float(p.get("avg_entry_price", 0)),
            current_price=float(p.get("current_price", 0)),
            market_value=float(p.get("market_value", 0)),
            unrealized_pnl=float(p.get("unrealized_pl", 0)),
            unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)) * 100,
        )
        for p in positions
    ]


@router.post("/orders")
async def place_order(order: OrderRequest):
    """
    Place a buy/sell order via Alpaca.

    Defaults to paper trading. Market orders execute immediately
    during market hours. Limit orders wait for the target price.
    """
    client = _get_client()

    # Validate side
    if order.side.lower() not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")

    if order.qty <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    if order.order_type == "limit" and order.limit_price is None:
        raise HTTPException(status_code=400, detail="Limit orders require a limit_price")

    result = await client.place_order(
        ticker=order.ticker,
        qty=order.qty,
        side=order.side,
        order_type=order.order_type,
        limit_price=order.limit_price,
        time_in_force=order.time_in_force,
    )

    if not result:
        raise HTTPException(status_code=502, detail="Order rejected by Alpaca. Check account status and market hours.")

    return {
        "order_id": result.get("id"),
        "status": result.get("status"),
        "ticker": result.get("symbol"),
        "side": result.get("side"),
        "qty": result.get("qty"),
        "type": result.get("type"),
        "filled_avg_price": result.get("filled_avg_price"),
        "created_at": result.get("created_at"),
    }


@router.get("/orders")
async def list_orders(status: str = "open"):
    """List orders. Status: open, closed, all."""
    client = _get_client()
    orders = await client.get_orders(status=status)

    if orders is None:
        raise HTTPException(status_code=502, detail="Failed to fetch orders")

    return [
        {
            "order_id": o.get("id"),
            "ticker": o.get("symbol"),
            "side": o.get("side"),
            "qty": o.get("qty"),
            "type": o.get("type"),
            "status": o.get("status"),
            "filled_avg_price": o.get("filled_avg_price"),
            "limit_price": o.get("limit_price"),
            "created_at": o.get("created_at"),
            "filled_at": o.get("filled_at"),
        }
        for o in orders
    ]


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an open order."""
    client = _get_client()
    result = await client.cancel_order(order_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Order not found or already filled")
    return {"status": "cancelled", "order_id": order_id}


@router.delete("/positions/{ticker}")
async def close_position(ticker: str):
    """Close an entire position (sell all shares)."""
    client = _get_client()
    result = await client.close_position(ticker.upper())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No open position for {ticker}")
    return {"status": "closing", "ticker": ticker.upper()}
