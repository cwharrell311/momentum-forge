"""
Confluence Engine API

FastAPI application providing REST endpoints for the confluence
trading platform. WebSocket support planned for Phase 4.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# TODO: Import routes as they're built
# from src.api.routes import confluence, signals, watchlist, trades, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    On startup:
    - Initialize database connection pool
    - Initialize Redis connection
    - Start signal processor scheduler
    - Load watchlist

    On shutdown:
    - Close all HTTP clients
    - Close DB/Redis connections
    - Stop scheduler
    """
    print("🚀 Confluence Engine starting up...")

    # TODO: Initialize services
    # db = await init_db()
    # redis = await init_redis()
    # scheduler = init_scheduler()

    yield

    print("🛑 Confluence Engine shutting down...")
    # TODO: Cleanup


app = FastAPI(
    title="Confluence Engine",
    description="Multi-layer signal confluence platform for options and equity trading",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check ---

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# --- Routes (uncomment as built) ---

# app.include_router(confluence.router, prefix="/api/v1/confluence", tags=["confluence"])
# app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
# app.include_router(watchlist.router, prefix="/api/v1/watchlist", tags=["watchlist"])
# app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
# app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])


# --- Placeholder route for testing ---

@app.get("/api/v1/confluence")
async def get_confluence():
    """
    Placeholder — returns mock data until signal processors are wired up.
    Replace with real ConfluenceEngine.scan_all() in Phase 1.
    """
    return {
        "regime": "elevated",
        "scores": [
            {
                "ticker": "NVDA",
                "direction": "bullish",
                "conviction": 78,
                "active_layers": 4,
                "total_layers": 8,
                "signals": ["options_flow", "gex", "momentum", "dark_pool"],
            },
            {
                "ticker": "AAPL",
                "direction": "bearish",
                "conviction": 62,
                "active_layers": 3,
                "total_layers": 8,
                "signals": ["options_flow", "short_interest", "momentum"],
            },
        ],
        "message": "⚠️ Mock data — signal processors not yet connected",
    }
