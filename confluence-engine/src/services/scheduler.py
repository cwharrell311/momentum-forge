"""
Background task scheduler.

Runs the confluence scan on a timer so scores are always fresh.
Uses APScheduler which runs inside the same Python process as
the FastAPI server — no extra services needed.

How it works:
1. On startup, the scheduler registers a job: "run scan_watchlist every N seconds"
2. Every N seconds, it loads the watchlist, scans all tickers, and stores results
3. When you hit GET /confluence, you get the latest cached scores
"""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger("confluence.scheduler")

# Module-level scheduler instance
_scheduler: AsyncIOScheduler | None = None


async def run_confluence_scan() -> None:
    """
    Execute a full confluence scan across all watchlist tickers.

    This is the job that runs on the timer. It:
    1. Loads the active watchlist
    2. Runs all signal processors
    3. Computes confluence scores
    4. Logs results (DB persistence comes in a later step)
    """
    from src.api.dependencies import get_engine, get_watchlist_tickers

    try:
        tickers = await get_watchlist_tickers()
        engine = get_engine()

        logger.info(f"Scanning {len(tickers)} tickers...")
        scores = await engine.scan_all(tickers)

        # Log top results
        if scores:
            top = scores[:5]
            for s in top:
                logger.info(
                    f"  {s.ticker}: {s.direction.value} @ {s.conviction_pct}% "
                    f"({s.active_layers} layers)"
                )
        else:
            logger.info("  No signals firing")

        logger.info(
            f"Scan complete: {len(scores)} tickers with signals "
            f"(out of {len(tickers)} scanned)"
        )

    except Exception as e:
        logger.error(f"Confluence scan failed: {e}")


def start_scheduler(interval_seconds: int = 300) -> AsyncIOScheduler:
    """
    Start the background scheduler.

    Args:
        interval_seconds: How often to run the full scan (default 5 min)
    """
    global _scheduler

    _scheduler = AsyncIOScheduler()

    _scheduler.add_job(
        run_confluence_scan,
        trigger="interval",
        seconds=interval_seconds,
        id="confluence_scan",
        name="Full confluence scan",
        next_run_time=datetime.utcnow(),  # Run immediately on startup
    )

    _scheduler.start()
    logger.info(f"Scheduler started — scanning every {interval_seconds}s")
    return _scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
