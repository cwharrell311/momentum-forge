"""
Confluence API routes.

These are the most important endpoints in the app — they return
the ranked list of tickers sorted by conviction score. This is
what you trade from.

GET /api/v1/confluence         → Top confluence scores (the screener)
GET /api/v1/confluence/{ticker} → Deep dive on a single ticker
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Response Schemas ──
# Pydantic models define the exact shape of API responses.
# This gives you auto-generated docs and type safety.

class SignalResponse(BaseModel):
    layer: str
    direction: str
    strength: float
    confidence: float
    explanation: str
    metadata: dict = {}


class ConfluenceResponse(BaseModel):
    ticker: str
    direction: str
    conviction: int          # 0-100 for display
    active_layers: int
    total_layers: int
    regime: str
    signals: list[SignalResponse] = []


class ConfluenceListResponse(BaseModel):
    regime: str
    scores: list[ConfluenceResponse]
    scanned_tickers: int


# ── Endpoints ──

@router.get("", response_model=ConfluenceListResponse)
async def get_confluence():
    """
    Get ranked confluence scores for all watchlist tickers.

    This is the main screener — it returns tickers sorted by
    conviction score (highest first). Only tickers with at least
    one active signal are included.
    """
    # Import here to avoid circular imports at module load time
    from src.api.dependencies import get_engine, get_watchlist_tickers

    engine = get_engine()
    tickers = await get_watchlist_tickers()

    if not tickers:
        return ConfluenceListResponse(regime="unknown", scores=[], scanned_tickers=0)

    scores = await engine.scan_all(tickers)
    regime = await engine.get_current_regime()

    return ConfluenceListResponse(
        regime=regime.value,
        scanned_tickers=len(tickers),
        scores=[
            ConfluenceResponse(
                ticker=s.ticker,
                direction=s.direction.value,
                conviction=s.conviction_pct,
                active_layers=s.active_layers,
                total_layers=s.total_layers,
                regime=s.regime.value,
                signals=[
                    SignalResponse(
                        layer=sig.layer,
                        direction=sig.direction.value,
                        strength=round(sig.strength, 3),
                        confidence=round(sig.confidence, 3),
                        explanation=sig.explanation,
                        metadata=sig.metadata,
                    )
                    for sig in s.signals
                ],
            )
            for s in scores
        ],
    )


@router.get("/{ticker}", response_model=ConfluenceResponse)
async def get_confluence_ticker(ticker: str):
    """
    Deep dive on a single ticker.

    Returns detailed confluence score with all individual signal
    layer results and their metadata.
    """
    from src.api.dependencies import get_engine

    engine = get_engine()
    score = await engine.scan_single(ticker.upper())

    if not score:
        raise HTTPException(
            status_code=404,
            detail=f"No signals found for {ticker.upper()}",
        )

    return ConfluenceResponse(
        ticker=score.ticker,
        direction=score.direction.value,
        conviction=score.conviction_pct,
        active_layers=score.active_layers,
        total_layers=score.total_layers,
        regime=score.regime.value,
        signals=[
            SignalResponse(
                layer=sig.layer,
                direction=sig.direction.value,
                strength=round(sig.strength, 3),
                confidence=round(sig.confidence, 3),
                explanation=sig.explanation,
                metadata=sig.metadata,
            )
            for sig in score.signals
        ],
    )
