"""
Watchlist API routes.

Manage which tickers the engine scans. This is your universe of
stocks. Add tickers you want to track, remove ones you don't.

GET    /api/v1/watchlist       → List all watched tickers
POST   /api/v1/watchlist       → Add a ticker
DELETE /api/v1/watchlist/{ticker} → Remove a ticker
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from src.models.tables import Watchlist
from src.utils.db import get_session

router = APIRouter()


class WatchlistItem(BaseModel):
    ticker: str
    sector: str | None = None
    active: bool = True


class WatchlistResponse(BaseModel):
    ticker: str
    sector: str | None = None
    active: bool
    added_at: str | None = None


class AddTickerRequest(BaseModel):
    ticker: str
    sector: str | None = None


@router.get("", response_model=list[WatchlistResponse])
async def get_watchlist():
    """List all tickers in the watchlist."""
    async with get_session() as session:
        result = await session.execute(
            select(Watchlist).where(Watchlist.active.is_(True)).order_by(Watchlist.ticker)
        )
        rows = result.scalars().all()
        return [
            WatchlistResponse(
                ticker=r.ticker,
                sector=r.sector,
                active=r.active,
                added_at=r.added_at.isoformat() if r.added_at else None,
            )
            for r in rows
        ]


@router.post("", response_model=WatchlistResponse, status_code=201)
async def add_ticker(request: AddTickerRequest):
    """
    Add a ticker to the watchlist.

    If the ticker already exists but was deactivated, it gets reactivated.
    """
    ticker = request.ticker.upper()

    async with get_session() as session:
        # Check if already exists
        existing = await session.get(Watchlist, ticker)
        if existing:
            if existing.active:
                raise HTTPException(status_code=409, detail=f"{ticker} already in watchlist")
            existing.active = True
            await session.commit()
            return WatchlistResponse(
                ticker=existing.ticker,
                sector=existing.sector,
                active=existing.active,
                added_at=existing.added_at.isoformat() if existing.added_at else None,
            )

        # Create new entry
        item = Watchlist(ticker=ticker, sector=request.sector)
        session.add(item)
        await session.commit()

        return WatchlistResponse(
            ticker=item.ticker,
            sector=item.sector,
            active=item.active,
            added_at=item.added_at.isoformat() if item.added_at else None,
        )


@router.delete("/{ticker}", status_code=204)
async def remove_ticker(ticker: str):
    """
    Remove a ticker from the watchlist.

    Soft-deletes by setting active=False rather than truly deleting,
    so you keep the history of what you've tracked.
    """
    ticker = ticker.upper()

    async with get_session() as session:
        existing = await session.get(Watchlist, ticker)
        if not existing:
            raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")

        existing.active = False
        await session.commit()
