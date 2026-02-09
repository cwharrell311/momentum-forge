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


class ApiQuotaStatus(BaseModel):
    provider: str
    calls_today: int
    errors_today: int
    quota_limit: int
    quota_remaining: int
    quota_pct_used: float
    configured: bool


class AutoJournalStatus(BaseModel):
    enabled: bool
    min_conviction: int
    min_layers: int
    logged_this_session: list[str]  # Tickers already logged


class SystemStatus(BaseModel):
    fmp_quota: QuotaStatus
    uw_quota: ApiQuotaStatus | None
    cache_has_data: bool
    cache_age_seconds: float | None
    db_connected: bool
    active_layers: int
    total_layers: int
    auto_journal: AutoJournalStatus | None = None


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
    from src.api.dependencies import get_cache, get_fmp_client, get_processors, get_uw_client

    # FMP quota
    fmp = get_fmp_client()
    quota = fmp.quota_status if fmp else {
        "calls_today": 0, "errors_today": 0, "rate_limited_today": 0,
        "quota_limit": 250, "quota_remaining": 250, "quota_pct_used": 0,
        "date": "",
    }

    # UW quota
    uw = get_uw_client()
    uw_quota = None
    if uw and uw.is_configured:
        uw_status = uw.quota_status
        uw_quota = ApiQuotaStatus(
            provider="Unusual Whales",
            calls_today=uw_status["calls_today"],
            errors_today=uw_status["errors_today"],
            quota_limit=uw_status["quota_limit"],
            quota_remaining=uw_status["quota_remaining"],
            quota_pct_used=uw_status["quota_pct_used"],
            configured=True,
        )

    # Cache status
    cache = get_cache()
    cache_age = cache.age_seconds if cache.has_data else None

    # Layer counts
    processors = get_processors()
    total_layers = len(LAYER_REGISTRY)
    active_layers = len(processors)

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

    # Auto-journal status
    from src.config import get_settings as _get_settings
    from src.services.auto_journal import get_session_log

    _settings = _get_settings()
    auto_journal = AutoJournalStatus(
        enabled=_settings.auto_trade_enabled,
        min_conviction=_settings.auto_trade_min_conviction,
        min_layers=_settings.auto_trade_min_layers,
        logged_this_session=get_session_log(),
    )

    return SystemStatus(
        fmp_quota=QuotaStatus(**quota),
        uw_quota=uw_quota,
        cache_has_data=cache.has_data,
        cache_age_seconds=round(cache_age, 1) if cache_age is not None else None,
        db_connected=db_ok,
        active_layers=active_layers,
        total_layers=total_layers,
        auto_journal=auto_journal,
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
        "data_source": "Unusual Whales (Basic $150/mo)",
        "description": "Institutional flow: golden sweeps, blocks, bid/ask sentiment analysis. Highest-weighted signal (0.25)",
        "refresh_seconds": 300,
    },
    {
        "name": "gex",
        "display_name": "GEX / Dealer Positioning",
        "phase": 2,
        "data_source": "Unusual Whales (Basic $150/mo)",
        "description": "Gamma exposure reveals dealer hedging: positive GEX = mean-reverting, negative GEX = trending. Identifies gamma walls",
        "refresh_seconds": 300,
    },
    {
        "name": "volatility",
        "display_name": "Volatility Surface",
        "phase": 2,
        "data_source": "Unusual Whales (Basic $150/mo)",
        "description": "IV rank, put/call skew, term structure. Detects when options are cheap/expensive and hidden fear/greed",
        "refresh_seconds": 600,
    },
    {
        "name": "dark_pool",
        "display_name": "Dark Pool Activity",
        "phase": 2,
        "data_source": "Unusual Whales (Basic $150/mo)",
        "description": "FINRA ATS data: institutional accumulation/distribution patterns, block trades, short volume ratio",
        "refresh_seconds": 900,
    },
    {
        "name": "insider",
        "display_name": "Insider Trading",
        "phase": 1,
        "data_source": "FMP (free tier)",
        "description": "SEC Form 4 filings — cluster buying by C-suite is one of the strongest long-term signals",
        "refresh_seconds": 86400,
    },
    {
        "name": "short_interest",
        "display_name": "Short Interest",
        "phase": 2,
        "data_source": "Unusual Whales (Basic $150/mo)",
        "description": "SI% of float, days to cover, squeeze potential. Contrarian signal — high SI = squeeze fuel",
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


# ── Auto-Journal Control ──


class AutoJournalToggle(BaseModel):
    enabled: bool
    min_conviction: int | None = None  # Optional: update threshold
    min_layers: int | None = None      # Optional: update threshold


@router.post("/auto-journal", response_model=AutoJournalStatus)
async def toggle_auto_journal(req: AutoJournalToggle):
    """
    Toggle auto-journal on/off and optionally update thresholds.

    When enabled, the scanner automatically logs trade journal entries
    for any ticker that crosses the conviction threshold. This lets you
    track signals while you're away from the dashboard during market hours.
    """
    from src.config import get_settings as _get_settings
    from src.services.auto_journal import get_session_log

    settings = _get_settings()

    # Update the runtime settings (in-memory, survives until restart)
    settings.auto_trade_enabled = req.enabled
    if req.min_conviction is not None:
        settings.auto_trade_min_conviction = req.min_conviction
    if req.min_layers is not None:
        settings.auto_trade_min_layers = req.min_layers

    return AutoJournalStatus(
        enabled=settings.auto_trade_enabled,
        min_conviction=settings.auto_trade_min_conviction,
        min_layers=settings.auto_trade_min_layers,
        logged_this_session=get_session_log(),
    )


@router.post("/auto-journal/clear")
async def clear_auto_journal_session():
    """
    Clear the auto-journal session log.

    Resets the deduplication tracker so tickers can be re-logged.
    Call this at the start of each trading day.
    """
    from src.services.auto_journal import clear_session_log

    clear_session_log()
    return {"status": "cleared"}
