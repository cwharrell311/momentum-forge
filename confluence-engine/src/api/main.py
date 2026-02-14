"""
Confluence Engine API

FastAPI application providing REST endpoints for the confluence
trading platform. This is the entry point that wires everything
together — config, data clients, signal processors, routes, and
the background scheduler.

Run with: uvicorn src.api.main:app --reload --port 8000
Then visit: http://localhost:8000/docs for interactive API docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from src.api.dependencies import cleanup_app, init_ai, init_app
from src.api.routes import broker, command_center, confluence, performance, regime, signals, system, trades, trading_engine, watchlist
from src.config import get_settings
from src.services.scheduler import start_scheduler, stop_scheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("confluence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    This runs ONCE when the server starts, and the cleanup runs
    when the server stops. Everything between 'yield' is the
    server's active lifetime.

    Startup:
    1. Load settings from .env
    2. Initialize FMP client + signal processors
    3. Start background scanner

    Shutdown:
    1. Stop scheduler
    2. Close HTTP clients and DB connections
    """
    settings = get_settings()
    logger.info("Confluence starting up...")

    # Initialize shared dependencies (FMP, UW, Alpaca clients + all 8 processors)
    init_app(
        fmp_api_key=settings.fmp_api_key,
        uw_api_key=settings.uw_api_key,
        alpaca_key=settings.alpaca_api_key,
        alpaca_secret=settings.alpaca_secret_key,
        alpaca_base_url=settings.alpaca_base_url,
    )
    logger.info(f"FMP client initialized (key: ...{settings.fmp_api_key[-4:]})")
    if settings.uw_api_key:
        logger.info("UW client initialized — all 8 signal layers ACTIVE")
    else:
        logger.info("UW not configured — 3/8 layers active (add UW_API_KEY for all 8)")
    if settings.alpaca_api_key:
        logger.info(f"Alpaca client initialized ({'paper' if 'paper' in settings.alpaca_base_url else 'LIVE'} trading)")
    else:
        logger.info("Alpaca not configured — add ALPACA_API_KEY to .env for paper trading")

    # Initialize AI Command Center (Claude + OpenAI routing)
    init_ai(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        default_provider=settings.ai_default_provider,
        claude_model=settings.ai_claude_model,
        openai_model=settings.ai_openai_model,
    )
    ai_providers = []
    if settings.anthropic_api_key:
        ai_providers.append("Claude")
    if settings.openai_api_key:
        ai_providers.append("OpenAI")
    if ai_providers:
        logger.info(f"AI Command Center ready — providers: {', '.join(ai_providers)} (routing: {settings.ai_default_provider})")
    else:
        logger.info("AI Command Center initialized — no providers configured (add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env)")

    # Create database tables (if PostgreSQL is running)
    try:
        from src.utils.db import create_tables

        await create_tables()
        logger.info("Database tables ready")
    except Exception as e:
        logger.warning(
            f"Database not available: {e} — "
            "Screener works fine, but Trade Journal needs PostgreSQL. "
            "Run: docker compose up -d"
        )

    # Start background scanner
    start_scheduler(interval_seconds=settings.scan_interval)

    logger.info("Ready! Visit http://localhost:8000 for the dashboard")

    yield  # Server is running and handling requests here

    # Shutdown
    logger.info("Confluence shutting down...")
    stop_scheduler()
    await cleanup_app()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Confluence",
    description=(
        "Multi-layer signal confluence platform for options and equity trading. "
        "Identifies high-probability setups by detecting when multiple independent "
        "signal layers align on the same ticker."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# CORS — allow local frontend dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount Routes ──
app.include_router(confluence.router, prefix="/api/v1/confluence", tags=["confluence"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
app.include_router(watchlist.router, prefix="/api/v1/watchlist", tags=["watchlist"])
app.include_router(regime.router, prefix="/api/v1/regime", tags=["regime"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(broker.router, prefix="/api/v1/broker", tags=["broker"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])
app.include_router(command_center.router, prefix="/api/v1/ai", tags=["ai-command-center"])
app.include_router(trading_engine.router, prefix="/api/v1/engine", tags=["trading-engine"])


# ── Dashboard ──

_dashboard_path = Path(__file__).parent.parent / "ui" / "dashboard.html"


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the main dashboard. Open http://localhost:8000 in your browser."""
    return Response(
        content=_dashboard_path.read_text(),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── Health Check ──

@app.get("/health", tags=["system"])
async def health():
    """Basic health check — returns OK if the server is running."""
    return {"status": "ok", "version": "0.3.0"}
