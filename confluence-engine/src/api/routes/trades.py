"""
Trade journal API routes.

Log trades with the confluence score that triggered them, track
entry/exit prices, and review your P&L. This is how you measure
whether the engine is actually making you money.

POST   /api/v1/trades              -> Log a new trade (entry)
GET    /api/v1/trades              -> List all trades (journal view)
GET    /api/v1/trades/{id}         -> Get a single trade
PATCH  /api/v1/trades/{id}         -> Update a trade (close it, add notes)
GET    /api/v1/trades/stats/summary -> P&L summary stats
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from src.models.tables import Trade
from src.utils.db import get_session

router = APIRouter()


# ── Request/Response Schemas ──

class TradeCreateRequest(BaseModel):
    """Log a new trade entry."""
    ticker: str
    side: str = "long"               # 'long' or 'short'
    instrument: str = "equity"        # 'equity', 'call', 'put', 'spread'
    entry_price: float
    quantity: int = 1
    confluence_score_id: int | None = None
    notes: str | None = None


class TradeUpdateRequest(BaseModel):
    """Update an existing trade (e.g., close it with exit price)."""
    exit_price: float | None = None
    exit_at: str | None = None        # ISO format datetime
    notes: str | None = None


class TradeResponse(BaseModel):
    id: int
    ticker: str
    side: str
    instrument: str
    entry_price: float | None
    exit_price: float | None
    quantity: int | None
    pnl: float | None
    pnl_pct: float | None = None
    confluence_score_id: int | None
    entry_at: str | None
    exit_at: str | None
    notes: str | None
    status: str                       # 'open' or 'closed'
    created_at: str | None


class TradeSummary(BaseModel):
    total_trades: int
    open_trades: int
    closed_trades: int
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float                   # 0.0 to 1.0
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float


# ── Helpers ──

def _format_trade(t: Trade) -> TradeResponse:
    """Convert a Trade ORM object to API response."""
    is_closed = t.exit_price is not None
    pnl = t.pnl
    pnl_pct = None

    # Compute P&L if we have entry + exit
    if is_closed and t.entry_price and t.exit_price:
        if t.side == "long":
            pnl = (t.exit_price - t.entry_price) * (t.quantity or 1)
        else:
            pnl = (t.entry_price - t.exit_price) * (t.quantity or 1)

        if t.entry_price > 0:
            raw_pct = (t.exit_price - t.entry_price) / t.entry_price * 100
            pnl_pct = round(raw_pct if t.side == "long" else -raw_pct, 2)

    return TradeResponse(
        id=t.id,
        ticker=t.ticker,
        side=t.side,
        instrument=t.instrument,
        entry_price=t.entry_price,
        exit_price=t.exit_price,
        quantity=t.quantity,
        pnl=round(pnl, 2) if pnl is not None else None,
        pnl_pct=pnl_pct,
        confluence_score_id=t.confluence_score_id,
        entry_at=t.entry_at.isoformat() if t.entry_at else None,
        exit_at=t.exit_at.isoformat() if t.exit_at else None,
        notes=t.notes,
        status="closed" if is_closed else "open",
        created_at=t.created_at.isoformat() if t.created_at else None,
    )


# ── Endpoints ──

@router.post("", response_model=TradeResponse, status_code=201)
async def create_trade(request: TradeCreateRequest):
    """
    Log a new trade entry.

    Record the ticker, side, instrument, entry price, and quantity.
    Optionally link it to the confluence score that triggered the trade.
    """
    trade = Trade(
        ticker=request.ticker.upper(),
        side=request.side.lower(),
        instrument=request.instrument.lower(),
        entry_price=request.entry_price,
        quantity=request.quantity,
        confluence_score_id=request.confluence_score_id,
        entry_at=datetime.utcnow(),
        notes=request.notes,
    )

    async with get_session() as session:
        session.add(trade)
        await session.commit()
        await session.refresh(trade)
        return _format_trade(trade)


@router.get("", response_model=list[TradeResponse])
async def list_trades(
    status: str | None = Query(None, description="Filter: 'open' or 'closed'"),
    ticker: str | None = Query(None, description="Filter by ticker"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    List trades from your journal.

    Most recent trades first. Filter by status (open/closed) or ticker.
    """
    async with get_session() as session:
        query = select(Trade).order_by(desc(Trade.created_at)).limit(limit)

        if status == "open":
            query = query.where(Trade.exit_price.is_(None))
        elif status == "closed":
            query = query.where(Trade.exit_price.isnot(None))

        if ticker:
            query = query.where(Trade.ticker == ticker.upper())

        result = await session.execute(query)
        trades = result.scalars().all()
        return [_format_trade(t) for t in trades]


@router.get("/stats/summary", response_model=TradeSummary)
async def trade_summary():
    """
    Get P&L summary statistics across all closed trades.

    Shows win rate, average win/loss, best/worst trade, and total P&L.
    This is the scorecard that tells you if the engine is working.
    """
    async with get_session() as session:
        # Count all trades
        total_result = await session.execute(select(func.count(Trade.id)))
        total_trades = total_result.scalar() or 0

        # Count open trades
        open_result = await session.execute(
            select(func.count(Trade.id)).where(Trade.exit_price.is_(None))
        )
        open_trades = open_result.scalar() or 0

        # Get all closed trades for P&L calcs
        closed_result = await session.execute(
            select(Trade).where(Trade.exit_price.isnot(None))
        )
        closed_trades_list = closed_result.scalars().all()
        closed_count = len(closed_trades_list)

        # Calculate P&L for each closed trade
        pnls = []
        for t in closed_trades_list:
            if t.entry_price and t.exit_price:
                if t.side == "long":
                    pnl = (t.exit_price - t.entry_price) * (t.quantity or 1)
                else:
                    pnl = (t.entry_price - t.exit_price) * (t.quantity or 1)
                pnls.append(pnl)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return TradeSummary(
            total_trades=total_trades,
            open_trades=open_trades,
            closed_trades=closed_count,
            total_pnl=round(sum(pnls), 2) if pnls else 0.0,
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / len(pnls), 3) if pnls else 0.0,
            avg_win=round(sum(wins) / len(wins), 2) if wins else 0.0,
            avg_loss=round(sum(losses) / len(losses), 2) if losses else 0.0,
            best_trade=round(max(pnls), 2) if pnls else 0.0,
            worst_trade=round(min(pnls), 2) if pnls else 0.0,
        )


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(trade_id: int):
    """Get a single trade by ID."""
    async with get_session() as session:
        trade = await session.get(Trade, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return _format_trade(trade)


@router.patch("/{trade_id}", response_model=TradeResponse)
async def update_trade(trade_id: int, request: TradeUpdateRequest):
    """
    Update a trade — typically to close it with an exit price.

    When you set an exit_price, the P&L is automatically calculated.
    You can also update notes at any time (trade journal annotations).
    """
    async with get_session() as session:
        trade = await session.get(Trade, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

        if request.exit_price is not None:
            trade.exit_price = request.exit_price
            trade.exit_at = (
                datetime.fromisoformat(request.exit_at)
                if request.exit_at
                else datetime.utcnow()
            )
            # Calculate P&L
            if trade.entry_price:
                if trade.side == "long":
                    trade.pnl = (request.exit_price - trade.entry_price) * (trade.quantity or 1)
                else:
                    trade.pnl = (trade.entry_price - request.exit_price) * (trade.quantity or 1)

        if request.notes is not None:
            trade.notes = request.notes

        await session.commit()
        await session.refresh(trade)
        return _format_trade(trade)
