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
from fastapi.responses import HTMLResponse

from src.api.dependencies import cleanup_app, init_app
from src.api.routes import confluence, regime, signals, trades, watchlist
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
    logger.info("Confluence Engine starting up...")

    # Initialize shared dependencies (FMP client, processors, engine)
    init_app(fmp_api_key=settings.fmp_api_key)
    logger.info(f"FMP client initialized (key: ...{settings.fmp_api_key[-4:]})")

    # Start background scanner
    start_scheduler(interval_seconds=settings.scan_interval)

    logger.info("Ready! Visit http://localhost:8000 for the dashboard")

    yield  # Server is running and handling requests here

    # Shutdown
    logger.info("Confluence Engine shutting down...")
    stop_scheduler()
    await cleanup_app()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Confluence Engine",
    description=(
        "Multi-layer signal confluence platform for options and equity trading. "
        "Identifies high-probability setups by detecting when multiple independent "
        "signal layers align on the same ticker."
    ),
    version="0.1.0",
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


# ── Dashboard ──

_dashboard_path = Path(__file__).parent.parent / "ui" / "dashboard.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the main dashboard. Open http://localhost:8000 in your browser."""
    return _dashboard_path.read_text()


# ── Health Check ──

@app.get("/health", tags=["system"])
async def health():
    """Basic health check — returns OK if the server is running."""
    return {"status": "ok", "version": "0.1.0"}
