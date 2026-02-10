"""
Confluence API routes.

These are the most important endpoints in the app — they return
the ranked list of tickers sorted by conviction score. This is
what you trade from.

The main GET /confluence endpoint reads from the CACHE, not from
live API calls. The background scheduler handles scanning on a
timer and storing results in the cache. This means:
- Dashboard loads instantly (no waiting for API)
- Page refreshes don't burn API quota
- You always see the latest scan results

GET  /api/v1/confluence         → Top confluence scores (from cache)
POST /api/v1/confluence/scan    → Manually trigger a full scan
GET  /api/v1/confluence/{ticker} → Deep dive on a single ticker (live scan)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ── Response Schemas ──

class SignalResponse(BaseModel):
    layer: str
    direction: str
    strength: float
    confidence: float
    explanation: str
    metadata: dict = Field(default_factory=dict)


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
    cache_age_seconds: int = 0


# ── Helper ──

def _format_score(s) -> ConfluenceResponse:
    """Convert a ConfluenceScore object to API response format."""
    return ConfluenceResponse(
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


# ── Endpoints ──

@router.get("", response_model=ConfluenceListResponse)
async def get_confluence():
    """
    Get ranked confluence scores for all watchlist tickers.

    Reads from the cache (populated by the background scheduler).
    Zero API calls — loads instantly.
    """
    from src.api.dependencies import get_cache

    cache = get_cache()
    result = await cache.latest()

    if not result:
        return ConfluenceListResponse(
            regime="unknown",
            scores=[],
            scanned_tickers=0,
            cache_age_seconds=0,
        )

    return ConfluenceListResponse(
        regime=result.regime.value,
        scanned_tickers=result.scanned_tickers,
        cache_age_seconds=int(cache.age_seconds),
        scores=[_format_score(s) for s in result.scores],
    )


@router.post("/scan")
async def trigger_scan():
    """
    Manually trigger a full confluence scan.

    Runs immediately instead of waiting for the next scheduled scan.
    Returns the number of tickers scanned and top conviction score.
    """
    from src.services.scheduler import run_confluence_scan

    await run_confluence_scan()

    from src.api.dependencies import get_cache
    cache = get_cache()
    result = await cache.latest()

    if not result or not result.scores:
        return {"status": "complete", "scanned": 0, "top_conviction": 0}

    top = max(s.conviction_pct for s in result.scores)
    return {"status": "complete", "scanned": len(result.scores), "top_conviction": top}


@router.get("/{ticker}", response_model=ConfluenceResponse)
async def get_confluence_ticker(ticker: str):
    """
    Deep dive on a single ticker.

    First checks the cache. If not found, does a live scan
    (costs API calls but only for one ticker).
    """
    from src.api.dependencies import get_cache, get_engine

    ticker = ticker.upper()

    # Try cache first
    cache = get_cache()
    result = await cache.latest()
    if result:
        for s in result.scores:
            if s.ticker == ticker:
                return _format_score(s)

    # Cache miss — do a live scan for this one ticker
    engine = get_engine()
    score = await engine.scan_single(ticker)

    if not score:
        raise HTTPException(
            status_code=404,
            detail=f"No signals found for {ticker}",
        )

    return _format_score(score)
