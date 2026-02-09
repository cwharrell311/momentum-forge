"""
Broker API routes — Alpaca paper/live trading.

Enables trade execution directly from the dashboard. Uses the Alpaca
Trade API v2, which supports both paper trading (free) and live trading.

IMPORTANT: Paper trading is the default. Live trading requires:
1. Setting LIVE_TRADING_ENABLED=true in .env
2. Changing ALPACA_BASE_URL to https://api.alpaca.markets

Safety features:
- Paper trading by default (can't lose real money by accident)
- Live trading blocked unless LIVE_TRADING_ENABLED=true
- Position size limits based on account equity
- All orders auto-logged to trade journal with Alpaca order ID
- Conviction-based position sizing helper

GET  /api/v1/broker/status               → Is Alpaca configured?
GET  /api/v1/broker/account              → Account info (equity, buying power)
GET  /api/v1/broker/positions            → All open positions
POST /api/v1/broker/orders               → Place an order
GET  /api/v1/broker/orders               → List orders
DEL  /api/v1/broker/orders/:id           → Cancel an order
DEL  /api/v1/broker/positions/:ticker    → Close a position
POST /api/v1/broker/position-size        → Calculate position size
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.models.tables import Trade
from src.utils.db import get_session

router = APIRouter()
log = logging.getLogger(__name__)


# ── Request/Response Schemas ──

class OrderRequest(BaseModel):
    ticker: str
    qty: int
    side: str           # "buy" or "sell"
    order_type: str = "market"   # "market" or "limit"
    limit_price: float | None = None
    time_in_force: str = "day"   # "day", "gtc", "ioc"
    confluence_score_id: int | None = None  # Link order to the score that triggered it
    notes: str | None = None


class AccountResponse(BaseModel):
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    day_trade_count: int
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
    change_today_pct: float


class BrokerStatus(BaseModel):
    configured: bool
    is_paper: bool
    connected: bool
    live_trading_enabled: bool


class PositionSizeRequest(BaseModel):
    """Calculate how many shares to buy given risk parameters."""
    ticker: str
    conviction: float = Field(ge=0.0, le=1.0, description="Confluence conviction 0.0-1.0")
    max_risk_pct: float = Field(
        default=2.0, ge=0.1, le=10.0,
        description="Max % of equity to risk on this trade (default 2%)",
    )
    stop_loss_pct: float = Field(
        default=5.0, ge=0.5, le=50.0,
        description="Stop loss % below entry (default 5%)",
    )


class PositionSizeResponse(BaseModel):
    ticker: str
    current_price: float
    account_equity: float
    conviction: float
    conviction_pct: int
    risk_per_trade: float       # Dollar amount at risk
    suggested_qty: int          # Shares to buy
    position_value: float       # Total cost
    position_pct: float         # % of equity this position uses
    max_loss: float             # If stop loss hits
    rationale: str


class OrderResponse(BaseModel):
    order_id: str | None
    status: str | None
    ticker: str | None
    side: str | None
    qty: str | None
    type: str | None
    filled_avg_price: str | None
    created_at: str | None
    trade_journal_id: int | None = None  # Auto-logged trade journal entry


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


def _check_live_trading_allowed(client) -> None:
    """
    Safety gate: block live trading unless explicitly enabled.

    Paper trading always allowed. Live trading requires
    LIVE_TRADING_ENABLED=true in .env to prevent accidental
    real-money trades.
    """
    if not client.is_paper:
        settings = get_settings()
        if not settings.live_trading_enabled:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Live trading is DISABLED. You are connected to the live Alpaca API "
                    "but LIVE_TRADING_ENABLED is false in your .env. Set it to true "
                    "ONLY when you are ready to trade real money."
                ),
            )


async def _log_trade_to_journal(
    ticker: str,
    side: str,
    qty: int,
    alpaca_order_id: str | None,
    entry_price: float | None,
    confluence_score_id: int | None,
    notes: str | None,
) -> int | None:
    """Auto-log a broker order to the trade journal for P&L tracking."""
    try:
        trade = Trade(
            ticker=ticker.upper(),
            side="long" if side == "buy" else "short",
            instrument="equity",
            entry_price=entry_price,
            quantity=qty,
            confluence_score_id=confluence_score_id,
            alpaca_order_id=alpaca_order_id,
            entry_at=datetime.utcnow(),
            notes=notes or f"Auto-logged from Alpaca order {alpaca_order_id}",
        )
        async with get_session() as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            log.info("Trade journal entry #%d created for %s %s %d shares", trade.id, side, ticker, qty)
            return trade.id
    except Exception as e:
        # Don't fail the order if journal logging fails (DB might be down)
        log.warning("Failed to auto-log trade to journal: %s", e)
        return None


# ── Endpoints ──

@router.get("/status", response_model=BrokerStatus)
async def broker_status():
    """Check if Alpaca is configured and connected."""
    from src.api.dependencies import get_alpaca_client
    client = get_alpaca_client()
    settings = get_settings()

    if not client or not client.is_configured:
        return BrokerStatus(
            configured=False,
            is_paper=True,
            connected=False,
            live_trading_enabled=settings.live_trading_enabled,
        )

    account = await client.get_account()
    return BrokerStatus(
        configured=True,
        is_paper=client.is_paper,
        connected=account is not None,
        live_trading_enabled=settings.live_trading_enabled,
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
        day_trade_count=int(account.get("daytrade_count", 0)),
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
            change_today_pct=float(p.get("change_today", 0)) * 100,
        )
        for p in positions
    ]


@router.post("/orders", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    """
    Place a buy/sell order via Alpaca.

    Safety:
    - Paper trading always allowed
    - Live trading blocked unless LIVE_TRADING_ENABLED=true
    - All orders auto-logged to the trade journal
    """
    client = _get_client()
    _check_live_trading_allowed(client)

    # Validate inputs
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
        raise HTTPException(
            status_code=502,
            detail="Order rejected by Alpaca. Check account status and market hours.",
        )

    alpaca_order_id = result.get("id")
    filled_price = result.get("filled_avg_price")
    entry_price = float(filled_price) if filled_price else order.limit_price

    # Auto-log to trade journal
    trade_id = await _log_trade_to_journal(
        ticker=order.ticker,
        side=order.side,
        qty=order.qty,
        alpaca_order_id=alpaca_order_id,
        entry_price=entry_price,
        confluence_score_id=order.confluence_score_id,
        notes=order.notes,
    )

    return OrderResponse(
        order_id=alpaca_order_id,
        status=result.get("status"),
        ticker=result.get("symbol"),
        side=result.get("side"),
        qty=result.get("qty"),
        type=result.get("type"),
        filled_avg_price=result.get("filled_avg_price"),
        created_at=result.get("created_at"),
        trade_journal_id=trade_id,
    )


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
    _check_live_trading_allowed(client)

    result = await client.cancel_order(order_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Order not found or already filled")
    return {"status": "cancelled", "order_id": order_id}


@router.delete("/positions/{ticker}")
async def close_position(ticker: str):
    """Close an entire position (sell all shares)."""
    client = _get_client()
    _check_live_trading_allowed(client)

    result = await client.close_position(ticker.upper())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No open position for {ticker}")
    return {"status": "closing", "ticker": ticker.upper()}


@router.post("/position-size", response_model=PositionSizeResponse)
async def calculate_position_size(req: PositionSizeRequest):
    """
    Calculate suggested position size based on conviction and risk.

    Uses the Kelly-inspired formula:
    1. Start with max_risk_pct of equity (default 2%)
    2. Scale by conviction (high conviction → larger position)
    3. Cap at max_risk_pct / stop_loss_pct to limit max loss

    Example: $100K equity, 75% conviction, 2% risk, 5% stop loss
    → Risk $2,000 × 0.75 conviction scale = $1,500 at risk
    → $1,500 / (5% stop × $180 price) = 166 shares of AAPL
    → Position value: ~$29,880 (29.9% of equity)
    → Max loss if stopped out: $1,500
    """
    client = _get_client()
    account = await client.get_account()
    if not account:
        raise HTTPException(status_code=502, detail="Failed to connect to Alpaca")

    equity = float(account.get("equity", 0))
    if equity <= 0:
        raise HTTPException(status_code=400, detail="Account equity is zero")

    # Get current price from Alpaca position or use FMP
    current_price = None

    # Try getting price from an existing position first
    position = await client.get_position(req.ticker.upper())
    if position and isinstance(position, dict):
        current_price = float(position.get("current_price", 0))

    # Fallback: use FMP quote
    if not current_price:
        from src.api.dependencies import get_fmp_client
        fmp = get_fmp_client()
        if fmp:
            quote = await fmp.get_quote(req.ticker.upper())
            if quote:
                current_price = float(quote.get("price", 0))

    if not current_price or current_price <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot determine current price for {req.ticker}",
        )

    # Scale risk by conviction: higher conviction → use more of the risk budget
    # At 50% conviction: use 50% of risk budget
    # At 80% conviction: use 80% of risk budget
    # At 100% conviction: use full risk budget
    conviction_scale = req.conviction

    # Dollar amount at risk = equity × risk% × conviction scale
    risk_dollars = equity * (req.max_risk_pct / 100) * conviction_scale

    # Position size = risk dollars / (stop loss % × price)
    stop_loss_per_share = current_price * (req.stop_loss_pct / 100)
    raw_qty = risk_dollars / stop_loss_per_share if stop_loss_per_share > 0 else 0
    suggested_qty = max(1, math.floor(raw_qty))

    position_value = suggested_qty * current_price
    position_pct = (position_value / equity) * 100
    max_loss = suggested_qty * stop_loss_per_share

    # Build rationale
    if req.conviction >= 0.70:
        confidence_label = "high conviction"
    elif req.conviction >= 0.50:
        confidence_label = "moderate conviction"
    else:
        confidence_label = "low conviction"

    rationale = (
        f"{confidence_label} ({round(req.conviction * 100)}%) → "
        f"risking ${risk_dollars:,.0f} ({req.max_risk_pct * conviction_scale:.1f}% of equity) "
        f"with {req.stop_loss_pct}% stop loss → "
        f"{suggested_qty} shares at ${current_price:,.2f}"
    )

    return PositionSizeResponse(
        ticker=req.ticker.upper(),
        current_price=round(current_price, 2),
        account_equity=round(equity, 2),
        conviction=req.conviction,
        conviction_pct=round(req.conviction * 100),
        risk_per_trade=round(risk_dollars, 2),
        suggested_qty=suggested_qty,
        position_value=round(position_value, 2),
        position_pct=round(position_pct, 2),
        max_loss=round(max_loss, 2),
        rationale=rationale,
    )
