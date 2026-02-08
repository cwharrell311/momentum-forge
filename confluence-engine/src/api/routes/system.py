"""
System status API route.

Provides operational status of the engine — FMP API quota usage,
cache health, database connectivity, and signal layer status.

GET /api/v1/system/status → Full system status
GET /api/v1/system/layers → All 8 signal layers with active/stub status
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


class LayerStatus(BaseModel):
    name: str
    display_name: str
    status: str           # "active", "stub"
    phase: int            # 1, 2, 3
    data_source: str      # "FMP", "Unusual Whales", "Alpaca"
    weight: float         # Default weight in confluence scoring
    description: str
    refresh_seconds: int


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


# All 8 signal layers with their metadata
LAYER_REGISTRY: list[dict] = [
    {
        "name": "momentum",
        "display_name": "Momentum / Technical",
        "phase": 1,
        "data_source": "FMP (free tier)",
        "description": "Price trend via MA alignment, relative volume, 52-week position, golden/death cross detection",
        "refresh_seconds": 300,
    },
    {
        "name": "vix_regime",
        "display_name": "VIX Regime Filter",
        "phase": 1,
        "data_source": "FMP (free tier)",
        "description": "Market stress classification (Calm/Elevated/Stressed/Crisis). Modifies signal weights, not scored directly",
        "refresh_seconds": 900,
    },
    {
        "name": "options_flow",
        "display_name": "Options Flow",
        "phase": 2,
        "data_source": "Unusual Whales ($50/mo)",
        "description": "Detects unusual sweeps, large blocks, and abnormal open interest changes — smart money footprints",
        "refresh_seconds": 120,
    },
    {
        "name": "gex",
        "display_name": "GEX / Dealer Positioning",
        "phase": 2,
        "data_source": "Unusual Whales ($50/mo)",
        "description": "Gamma exposure levels that reveal where dealers are forced to hedge — identifies magnetic price levels",
        "refresh_seconds": 300,
    },
    {
        "name": "volatility",
        "display_name": "Volatility Surface",
        "phase": 2,
        "data_source": "Unusual Whales ($50/mo)",
        "description": "IV rank, skew analysis, and term structure — detects when options are pricing in a big move",
        "refresh_seconds": 300,
    },
    {
        "name": "dark_pool",
        "display_name": "Dark Pool Activity",
        "phase": 3,
        "data_source": "Unusual Whales ($50/mo)",
        "description": "Off-exchange block prints that reveal institutional accumulation or distribution",
        "refresh_seconds": 600,
    },
    {
        "name": "insider",
        "display_name": "Insider Trading",
        "phase": 3,
        "data_source": "FMP (free tier)",
        "description": "SEC Form 4 filings — cluster buying by C-suite is one of the strongest long-term signals",
        "refresh_seconds": 86400,
    },
    {
        "name": "short_interest",
        "display_name": "Short Interest",
        "phase": 3,
        "data_source": "Unusual Whales ($50/mo)",
        "description": "Short interest ratio, days to cover, and cost to borrow — identifies squeeze potential",
        "refresh_seconds": 3600,
    },
]


@router.get("/layers", response_model=list[LayerStatus])
async def get_signal_layers():
    """
    Get status of all 8 signal layers.

    Shows which layers are active (producing signals) vs. still stubs
    (waiting for data source subscriptions). Includes phase, data source,
    default weight, and description for each layer.
    """
    from src.services.confluence import DEFAULT_WEIGHTS

    # Check which processors are actually active (not raising NotImplementedError)
    active_names = set()
    try:
        from src.api.dependencies import get_processors
        for p in get_processors():
            active_names.add(p.name)
    except Exception:
        pass

    layers = []
    for layer in LAYER_REGISTRY:
        weight = DEFAULT_WEIGHTS.get(layer["name"], 0.0)
        # vix_regime has weight 0 — it modifies other weights
        if layer["name"] == "vix_regime":
            weight = 0.0

        layers.append(LayerStatus(
            name=layer["name"],
            display_name=layer["display_name"],
            status="active" if layer["name"] in active_names else "stub",
            phase=layer["phase"],
            data_source=layer["data_source"],
            weight=weight,
            description=layer["description"],
            refresh_seconds=layer["refresh_seconds"],
        ))

    return layers
