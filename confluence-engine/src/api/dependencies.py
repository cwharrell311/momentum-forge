"""
Shared application dependencies.

This module holds the singleton instances of the FMP client,
signal processors, confluence engine, and result cache. They're
created once at startup (in main.py lifespan) and shared across
all routes.

Why singletons? Because:
1. The FMP client has a rate limiter — one shared instance
   ensures we never exceed the API limit across endpoints.
2. Creating new HTTP connections for every request is wasteful.
3. The result cache stores scan results so the dashboard loads
   instantly without burning API calls.
"""

from __future__ import annotations

from src.services.cache import ResultCache
from src.services.confluence import ConfluenceEngine
from src.signals.base import SignalProcessor
from src.signals.momentum import MomentumProcessor
from src.signals.vix_regime import VixRegimeProcessor
from src.utils.data_providers import FMPClient

# These get populated by init_app() during startup
_fmp_client: FMPClient | None = None
_processors: list[SignalProcessor] = []
_engine: ConfluenceEngine | None = None
_vix_processor: VixRegimeProcessor | None = None
_cache: ResultCache = ResultCache()


def init_app(fmp_api_key: str) -> None:
    """
    Initialize all shared dependencies.

    Called once during FastAPI lifespan startup. Creates:
    - FMP client (shared HTTP client with rate limiting)
    - Momentum processor (uses FMP)
    - VIX regime processor (uses FMP)
    - Confluence engine (combines all processors)
    """
    global _fmp_client, _processors, _engine, _vix_processor

    _fmp_client = FMPClient(api_key=fmp_api_key)

    # Create signal processors — add more here as they're implemented
    momentum = MomentumProcessor(fmp_client=_fmp_client)
    vix = VixRegimeProcessor(fmp_client=_fmp_client)
    _vix_processor = vix

    _processors = [momentum, vix]

    _engine = ConfluenceEngine(processors=_processors)


async def cleanup_app() -> None:
    """Close all shared resources. Called during shutdown."""
    global _fmp_client
    if _fmp_client:
        await _fmp_client.close()


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
