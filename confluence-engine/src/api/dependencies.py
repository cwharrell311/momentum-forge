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

from src.services.ai_router import AIRouter
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
from src.utils.ai_clients import ClaudeClient, OpenAIClient
from src.utils.data_providers import AlpacaClient, FMPClient, UnusualWhalesClient

# Non-stock tickers to exclude from universe discovery.
# We only want individual stocks — ETFs, ETNs, indexes, and leveraged
# products have distorted flow (hedging, not directional conviction).
_NON_STOCK_TICKERS = {
    # ── Index products ──
    "SPX", "SPXW", "NDX", "VIX", "VIXW", "RUT", "DJX", "OEX", "XSP",
    # ── Broad market ETFs ──
    "SPY", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR", "VTI", "VOO", "RSP",
    "IVV", "VUG", "VTV", "SCHD", "SPLG", "IUSG", "IUSV",
    # ── Sector ETFs ──
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLB", "XLP", "XLU", "XLRE", "XLC",
    "XBI", "XOP", "XHB", "XME", "XRT", "SMH", "ITB", "KRE", "KBE", "HACK",
    "SOXX", "IGV", "IYR", "IYT", "IYF",
    # ── Commodity ETFs ──
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "SLX", "COPX", "IAU",
    "PPLT", "PALL", "DBA", "DBC", "WEAT", "CORN",
    # ── Bond ETFs ──
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "AGG", "BND",
    "BNDX", "VCIT", "VCSH", "MUB", "EMB", "GOVT", "SHV", "BIL",
    # ── Volatility ETFs/ETNs ──
    "VXX", "UVXY", "UVIX", "SVXY", "VIXY", "VIXM", "SVOL",
    # ── Leveraged / Inverse ETFs ──
    "TQQQ", "SQQQ", "SPXL", "SPXS", "QLD", "SSO", "SDS", "SH", "DOG",
    "PSQ", "UPRO", "SPXU", "TNA", "TZA", "LABU", "LABD", "SOXL", "SOXS",
    "FNGU", "FNGD", "JNUG", "JDST", "NUGT", "DUST", "FAS", "FAZ",
    "ERX", "ERY", "TECL", "TECS", "UDOW", "SDOW", "YANG", "YINN",
    # ── Thematic / ARK / International ETFs ──
    "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ", "ARKX",
    "EEM", "EFA", "FXI", "INDA", "EWZ", "EWJ", "IEMG", "VWO", "VEA",
    "KWEB", "MCHI", "ASHR",
}

# These get populated by init_app() during startup
_fmp_client: FMPClient | None = None
_uw_client: UnusualWhalesClient | None = None
_alpaca_client: AlpacaClient | None = None
_processors: list[SignalProcessor] = []
_engine: ConfluenceEngine | None = None
_vix_processor: VixRegimeProcessor | None = None
_cache: ResultCache = ResultCache()
_claude_client: ClaudeClient | None = None
_openai_client: OpenAIClient | None = None
_ai_router: AIRouter | None = None


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

    # ── Alpaca-powered processors (no quota limit) ──
    momentum = MomentumProcessor(alpaca_client=_alpaca_client)

    # ── Alpaca-powered regime (computed from SPY realized vol — no FMP quota) ──
    vix = VixRegimeProcessor(alpaca_client=_alpaca_client)
    insider = InsiderProcessor(fmp_client=_fmp_client, uw_client=_uw_client)
    _vix_processor = vix

    # ── UW-powered processors (activate when UW_API_KEY is set) ──
    options_flow = OptionsFlowProcessor(uw_client=_uw_client)
    gex = GexProcessor(uw_client=_uw_client)
    volatility = VolatilityProcessor(uw_client=_uw_client)
    dark_pool = DarkPoolProcessor(uw_client=_uw_client)
    short_interest = ShortInterestProcessor(uw_client=_uw_client)

    # All 8 processors — UW/Alpaca ones return empty results if not configured
    _processors = [
        momentum,           # Alpaca market data (no quota limit)
        vix,                # FMP free tier (regime filter, not scored — 1 call per 15min)
        insider,            # UW primary, FMP fallback
        options_flow,       # UW Basic ($150/mo) — highest weight
        gex,                # UW Basic
        volatility,         # UW Basic
        dark_pool,          # UW Basic
        short_interest,     # UW Basic
    ]

    _engine = ConfluenceEngine(processors=_processors)


def init_ai(
    anthropic_api_key: str = "",
    openai_api_key: str = "",
    default_provider: str = "auto",
    claude_model: str = "claude-sonnet-4-20250514",
    openai_model: str = "gpt-4o",
) -> None:
    """
    Initialize AI clients and the routing engine.

    Called during FastAPI lifespan startup, after init_app(). Creates:
    - Claude client (if ANTHROPIC_API_KEY is set)
    - OpenAI client (if OPENAI_API_KEY is set)
    - AI Router that picks the best provider per task
    """
    global _claude_client, _openai_client, _ai_router

    _claude_client = ClaudeClient(api_key=anthropic_api_key, model=claude_model)
    _openai_client = OpenAIClient(api_key=openai_api_key, model=openai_model)
    _ai_router = AIRouter(
        claude=_claude_client,
        openai_client=_openai_client,
        default_provider=default_provider,
    )


async def cleanup_app() -> None:
    """Close all shared resources. Called during shutdown."""
    global _fmp_client, _uw_client, _alpaca_client, _claude_client, _openai_client
    if _fmp_client:
        await _fmp_client.close()
    if _uw_client:
        await _uw_client.close()
    if _alpaca_client:
        await _alpaca_client.close()
    if _claude_client:
        await _claude_client.close()
    if _openai_client:
        await _openai_client.close()


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


def get_ai_router() -> AIRouter:
    """Get the shared AI router instance."""
    if _ai_router is None:
        raise RuntimeError("AI router not initialized — call init_ai() first")
    return _ai_router


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


async def discover_active_tickers() -> list[str]:
    """
    Discover individual stocks with unusual options activity.

    Pulls a large batch of flow alerts from UW, filters out all ETFs/ETNs/
    index products, and returns the top stocks ranked by premium volume.

    Controlled by config:
    - UNIVERSE_DISCOVERY=true  → scan market-wide flow (stocks only)
    - UNIVERSE_DISCOVERY=false → watchlist only (original behavior)
    - UNIVERSE_MAX_TICKERS     → cap on how many stocks to scan
    """
    import logging

    from src.config import get_settings

    log = logging.getLogger("confluence.discovery")
    settings = get_settings()

    if not settings.universe_discovery:
        log.debug("Universe discovery disabled — using watchlist only")
        return await get_watchlist_tickers()

    if not _uw_client or not _uw_client.is_configured:
        log.info("UW not configured — using watchlist only")
        return await get_watchlist_tickers()

    try:
        # Pull a large batch of flow alerts (1 API call regardless of limit).
        # We fetch 500 to get past the ETF-dominated top of the list and
        # find individual stocks with unusual activity.
        flow_alerts = await _uw_client.get_market_flow(limit=500)

        if not flow_alerts:
            log.info("No market flow data — falling back to watchlist")
            return await get_watchlist_tickers()

        # Extract tickers from flow data.
        # If universe_stocks_only=True, filter out ETFs/ETNs (legacy).
        # Default (False): include ALL instruments — ETFs, treasuries,
        # commodities, futures proxies, everything tradeable.
        stocks_only = settings.universe_stocks_only
        ticker_premium: dict[str, float] = {}
        skipped_etfs: set[str] = set()
        for alert in flow_alerts:
            ticker = (
                alert.get("ticker_symbol")
                or alert.get("underlying_symbol")
                or alert.get("ticker")
                or ""
            ).upper().strip()
            if not ticker or len(ticker) > 6:
                continue
            # Always skip cash-settled index options (not directly tradeable)
            if ticker.startswith("^") or ticker.startswith("$"):
                continue
            if ticker in ("SPX", "SPXW", "NDX", "VIX", "VIXW", "RUT", "DJX", "OEX", "XSP"):
                skipped_etfs.add(ticker)
                continue
            # Only filter ETFs if explicitly configured to stocks-only
            if stocks_only:
                if ticker in _NON_STOCK_TICKERS:
                    skipped_etfs.add(ticker)
                    continue
                if alert.get("is_etf") is True:
                    skipped_etfs.add(ticker)
                    continue

            premium = 0.0
            for key in ("total_premium", "premium", "cost_basis"):
                val = alert.get(key)
                if val is not None:
                    try:
                        premium = abs(float(val))
                    except (ValueError, TypeError):
                        pass
                    break
            ticker_premium[ticker] = ticker_premium.get(ticker, 0) + premium

        # Sort by total premium (most active first), cap at max tickers
        max_tickers = settings.universe_max_tickers
        discovered = sorted(
            ticker_premium.keys(),
            key=lambda t: ticker_premium[t],
            reverse=True,
        )[:max_tickers]

        filter_mode = "stocks-only" if stocks_only else "all-instruments"
        log.info(
            "Universe [%s]: %d tickers discovered from %d flow alerts "
            "(skipped %d non-tradeable: %s)",
            filter_mode,
            len(discovered),
            len(flow_alerts),
            len(skipped_etfs),
            ", ".join(sorted(skipped_etfs)) if skipped_etfs else "none",
        )
        return discovered

    except Exception as e:
        log.warning("Discovery failed: %s — using watchlist only", e)
        return await get_watchlist_tickers()
