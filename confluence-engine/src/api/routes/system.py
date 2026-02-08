"""
System status API route.

Provides operational status of the engine — FMP API quota usage,
cache health, and database connectivity.

GET /api/v1/system/status → Full system status
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QuotaStatus(BaseModel):
    calls_today: int
    errors_today: int
    rate_limited_today: int
    quota_limit: int
    quota_remaining: int
    quota_pct_used: float
    date: str


class SystemStatus(BaseModel):
    fmp_quota: QuotaStatus
    cache_has_data: bool
    cache_age_seconds: float | None
    db_connected: bool


@router.get("/status", response_model=SystemStatus)
async def system_status():
    """
    Get system health and quota usage.

    Shows FMP API quota remaining, cache freshness, and DB connectivity.
    Check this if signals aren't updating — usually means quota is exhausted.
    """
    from src.api.dependencies import get_cache, get_fmp_client

    # FMP quota
    fmp = get_fmp_client()
    quota = fmp.quota_status if fmp else {
        "calls_today": 0, "errors_today": 0, "rate_limited_today": 0,
        "quota_limit": 250, "quota_remaining": 250, "quota_pct_used": 0,
        "date": "",
    }

    # Cache status
    cache = get_cache()
    cache_age = cache.age_seconds if cache.has_data else None

    # DB connectivity
    db_ok = False
    try:
        from src.utils.db import get_session
        from sqlalchemy import text

        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    return SystemStatus(
        fmp_quota=QuotaStatus(**quota),
        cache_has_data=cache.has_data,
        cache_age_seconds=round(cache_age, 1) if cache_age is not None else None,
        db_connected=db_ok,
    )
