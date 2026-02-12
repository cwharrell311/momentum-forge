"""
VIX Regime API route.

Returns the current market regime classification.
This is shown as a badge on the dashboard header so you
always know what kind of market you're operating in.

GET /api/v1/regime → Current regime + VIX level
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Cache regime for 5 minutes — VIX doesn't move fast enough
# to justify burning FMP quota on every dashboard refresh
_regime_cache: dict = {}
_regime_cache_ts: float = 0
_REGIME_TTL = 300  # seconds


class RegimeResponse(BaseModel):
    regime: str
    vix_level: float | None = None
    vix_change: float | None = None
    description: str


@router.get("", response_model=RegimeResponse)
async def get_regime():
    """Get current VIX regime classification (cached 5 min)."""
    global _regime_cache, _regime_cache_ts

    now = time.time()
    if _regime_cache and (now - _regime_cache_ts) < _REGIME_TTL:
        return RegimeResponse(**_regime_cache)

    from src.api.dependencies import get_vix_processor

    vix = get_vix_processor()
    if not vix:
        return RegimeResponse(
            regime="unknown",
            description="VIX processor not available",
        )

    result = await vix.scan_single("VIX")
    if not result:
        return RegimeResponse(
            regime="unknown",
            description="Could not fetch VIX data",
        )

    resp = {
        "regime": result.metadata.get("regime", "unknown"),
        "vix_level": result.metadata.get("vix_level"),
        "vix_change": result.metadata.get("vix_change"),
        "description": result.explanation,
    }
    _regime_cache = resp
    _regime_cache_ts = now

    return RegimeResponse(**resp)
