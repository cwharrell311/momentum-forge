"""
Shared application dependencies.

This module holds the singleton instances of all data clients,
signal processors, confluence engine, and result cache. They're
created once at startup (in main.py lifespan) and shared across
all routes.

Why singletons? Because:
1. Data clients have rate limiters — one shared instance per API
   ensures we never exceed limits across endpoints.
2. Creating new HTTP connections for every request is wasteful.
3. The result cache stores scan results so the dashboard loads
   instantly without burning API calls.
"""

from __future__ import annotations

from src.services.cache import ResultCache
from src.services.confluence import ConfluenceEngine
from src.signals.base import SignalProcessor
from src.signals.dark_pool import DarkPoolProcessor
from src.signals.gex import GexProcessor
from src.signals.insider import InsiderProcessor
from src.signals.momentum import MomentumProcessor
from src.signals.options_flow import OptionsFlowProcessor
from src.signals.short_interest import ShortInterestProcessor
from src.signals.volatility import VolatilityProcessor
from src.signals.vix_regime import VixRegimeProcessor
from src.utils.data_providers import AlpacaClient, FMPClient, UnusualWhalesClient

# These get populated by init_app() during startup
_fmp_client: FMPClient | None = None
_uw_client: UnusualWhalesClient | None = None
_alpaca_client: AlpacaClient | None = None
_processors: list[SignalProcessor] = []
_engine: ConfluenceEngine | None = None
_vix_processor: VixRegimeProcessor | None = None
_cache: ResultCache = ResultCache()


def init_app(
    fmp_api_key: str,
    uw_api_key: str = "",
    alpaca_key: str = "",
    alpaca_secret: str = "",
    alpaca_base_url: str = "",
) -> None:
    """
    Initialize all shared dependencies.

    Called once during FastAPI lifespan startup. Creates:
    - FMP client (free tier: momentum, VIX, insider)
    - UW client (Basic $150/mo: options flow, GEX, volatility, dark pool, SI)
    - Alpaca client (paper/live trading)
    - All 8 signal processors (UW-powered ones gracefully skip if no API key)
    - Confluence engine (combines all processors with parallel scanning)
    """
    global _fmp_client, _uw_client, _alpaca_client, _processors, _engine, _vix_processor

    _fmp_client = FMPClient(api_key=fmp_api_key)

    # UW client — all 5 UW-powered processors check is_configured before calling
    _uw_client = UnusualWhalesClient(api_key=uw_api_key)

    # Alpaca client — works even without keys (is_configured returns False)
    _alpaca_client = AlpacaClient(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
        base_url=alpaca_base_url or "https://paper-api.alpaca.markets",
    )

    # ── FMP-powered processors (free tier, always active) ──
    momentum = MomentumProcessor(fmp_client=_fmp_client)
    vix = VixRegimeProcessor(fmp_client=_fmp_client)
    insider = InsiderProcessor(fmp_client=_fmp_client)
    _vix_processor = vix

    # ── UW-powered processors (activate when UW_API_KEY is set) ──
    options_flow = OptionsFlowProcessor(uw_client=_uw_client)
    gex = GexProcessor(uw_client=_uw_client)
    volatility = VolatilityProcessor(uw_client=_uw_client)
    dark_pool = DarkPoolProcessor(uw_client=_uw_client)
    short_interest = ShortInterestProcessor(uw_client=_uw_client)

    # All 8 processors — UW ones return empty results if not configured
    _processors = [
        momentum,           # FMP free tier
        vix,                # FMP free tier (regime filter, not scored)
        insider,            # FMP free tier
        options_flow,       # UW Basic ($150/mo) — highest weight
        gex,                # UW Basic
        volatility,         # UW Basic
        dark_pool,          # UW Basic
        short_interest,     # UW Basic
    ]

    _engine = ConfluenceEngine(processors=_processors)


async def cleanup_app() -> None:
    """Close all shared resources. Called during shutdown."""
    global _fmp_client, _uw_client, _alpaca_client
    if _fmp_client:
        await _fmp_client.close()
    if _uw_client:
        await _uw_client.close()
    if _alpaca_client:
        await _alpaca_client.close()


def get_engine() -> ConfluenceEngine:
    """Get the shared confluence engine instance."""
    if _engine is None:
        raise RuntimeError("App not initialized — call init_app() first")
    return _engine


def get_processors() -> list[SignalProcessor]:
    """Get all registered signal processors."""
    return _processors


def get_vix_processor() -> VixRegimeProcessor | None:
    """Get the VIX regime processor."""
    return _vix_processor


def get_fmp_client() -> FMPClient | None:
    """Get the shared FMP client instance (for quota tracking)."""
    return _fmp_client


def get_uw_client() -> UnusualWhalesClient | None:
    """Get the shared Unusual Whales client instance (for quota tracking)."""
    return _uw_client


def get_alpaca_client() -> AlpacaClient | None:
    """Get the shared Alpaca client instance."""
    return _alpaca_client


def get_cache() -> ResultCache:
    """Get the shared result cache."""
    return _cache


async def get_watchlist_tickers() -> list[str]:
    """
    Load active watchlist tickers from the database.

    Falls back to the YAML config if the database isn't available
    (e.g., during initial setup before migrations are run).
    """
    try:
        from sqlalchemy import select

        from src.models.tables import Watchlist
        from src.utils.db import get_session

        async with get_session() as session:
            result = await session.execute(
                select(Watchlist.ticker).where(Watchlist.active.is_(True))
            )
            tickers = [row[0] for row in result.all()]
            if tickers:
                return tickers
    except Exception:
        pass

    # Fallback: load from YAML config
    try:
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parent.parent.parent / "config" / "watchlist.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data.get("tickers", [])
    except Exception:
        # Last resort: a few liquid names to get started
        return ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"]
