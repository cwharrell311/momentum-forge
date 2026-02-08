"""
VIX Regime API route.

Returns the current market regime classification.
This is shown as a badge on the dashboard header so you
always know what kind of market you're operating in.

GET /api/v1/regime → Current regime + VIX level
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RegimeResponse(BaseModel):
    regime: str
    vix_level: float | None = None
    vix_change: float | None = None
    description: str


@router.get("", response_model=RegimeResponse)
async def get_regime():
    """Get current VIX regime classification."""
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

    return RegimeResponse(
        regime=result.metadata.get("regime", "unknown"),
        vix_level=result.metadata.get("vix_level"),
        vix_change=result.metadata.get("vix_change"),
        description=result.explanation,
    )
