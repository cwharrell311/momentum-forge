"""
Background task scheduler.

Runs the confluence scan on a timer so scores are always fresh.
Uses APScheduler which runs inside the same Python process as
the FastAPI server — no extra services needed.

How it works:
1. On startup, the scheduler registers a job: "run scan every N seconds"
2. Every N seconds, it loads the watchlist, scans all tickers, and
   stores results in the cache
3. When you open the dashboard, it reads from the cache instantly
   (zero API calls for page loads)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

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
    4. Stores results in the cache for instant dashboard access
    """
    from src.api.dependencies import discover_active_tickers, get_cache, get_engine

    try:
        tickers = await discover_active_tickers()
        engine = get_engine()
        cache = get_cache()

        logger.info(f"Scanning {len(tickers)} tickers...")
        scores = await engine.scan_all(tickers)
        regime = await engine.get_current_regime()

        # Store results in cache — dashboard reads from here
        await cache.update(
            scores=scores,
            regime=regime,
            scanned_tickers=len(tickers),
            dp_regime=engine.dp_divergence,
        )

        # Log top results
        if scores:
            top = scores[:5]
            trade_worthy_count = sum(1 for s in scores if s.trade_worthy)
            logger.info(f"  {trade_worthy_count} of {len(scores)} signals are TRADE WORTHY (flow gate passed)")
            for s in top:
                tw = " [TRADE WORTHY]" if s.trade_worthy else ""
                logger.info(
                    f"  {s.ticker}: {s.direction.value} @ {s.conviction_pct}% "
                    f"({s.active_layers} layers){tw}"
                )
        else:
            logger.info("  No signals firing")

        # Save to database for historical tracking (non-blocking)
        try:
            await _save_scan_history(scores, regime)
        except Exception as e:
            logger.debug(f"History save skipped (DB not available): {e}")

        # Auto-journal: log qualifying signals to trade journal
        try:
            from src.config import get_settings
            from src.services.auto_journal import process_scan_results

            settings = get_settings()
            if settings.auto_trade_enabled:
                regime_str = regime.value if regime else None
                logged = await process_scan_results(
                    scores=scores,
                    regime_value=regime_str,
                    min_conviction=settings.auto_trade_min_conviction,
                    min_layers=settings.auto_trade_min_layers,
                )
                if logged:
                    logger.info(
                        "Auto-journal: %d new entries — %s",
                        len(logged),
                        ", ".join(
                            f"{e['ticker']} ({e['conviction']}%)" for e in logged
                        ),
                    )
        except Exception as e:
            logger.warning(f"Auto-journal failed: {e}")

        # Forward testing: record qualifying signals for grading
        try:
            from src.services.signal_tracker import record_signals

            regime_str = regime.value if regime else None
            recorded = await record_signals(scores, regime_str)
            if recorded:
                logger.info(f"Signal tracker: {recorded} new signal(s) recorded for forward testing")
        except Exception as e:
            logger.debug(f"Signal tracker skipped: {e}")

        logger.info(
            f"Scan complete: {len(scores)} tickers with signals "
            f"(out of {len(tickers)} scanned) — cached."
        )

    except Exception as e:
        logger.error(f"Confluence scan failed: {e}")


def start_scheduler(interval_seconds: int = 300) -> AsyncIOScheduler:
    """
    Start the background scheduler.

    Args:
        interval_seconds: How often to run the full scan
    """
    global _scheduler

    _scheduler = AsyncIOScheduler()

    _scheduler.add_job(
        run_confluence_scan,
        trigger="interval",
        seconds=interval_seconds,
        id="confluence_scan",
        name="Full confluence scan",
        next_run_time=datetime.now(),  # Run immediately on startup
    )

    # Grade past signals every 2 hours (no rush — outcome prices don't change fast)
    _scheduler.add_job(
        _run_signal_grading,
        trigger="interval",
        seconds=7200,  # 2 hours
        id="signal_grading",
        name="Grade past signals",
        next_run_time=datetime.now() + timedelta(seconds=120),  # First run 2 min after startup
    )

    _scheduler.start()
    logger.info(f"Scheduler started — scanning every {interval_seconds}s, grading every 2h")
    return _scheduler


async def _run_signal_grading() -> None:
    """Run the signal grading job — fills in outcome prices for past signals."""
    try:
        from src.services.signal_tracker import grade_signals

        graded = await grade_signals()
        if graded:
            logger.info(f"Signal grading: updated {graded} signal(s)")
    except Exception as e:
        logger.debug(f"Signal grading skipped: {e}")


async def _save_scan_history(scores, regime) -> None:
    """
    Save scan results to the database for historical tracking.

    Each scan creates ConfluenceScoreRecord rows and Signal rows.
    This builds a history so you can later analyze how signals
    evolved over time — useful for backtesting and pattern recognition.
    """
    from src.models.tables import ConfluenceScoreRecord, Signal
    from src.utils.db import get_session

    async with get_session() as session:
        for score in scores:
            # Save individual signals
            signal_ids = []
            for sig in score.signals:
                signal_row = Signal(
                    ticker=score.ticker,
                    layer=sig.layer,
                    direction=sig.direction.value,
                    strength=sig.strength,
                    confidence=sig.confidence,
                    metadata_=sig.metadata,
                    explanation=sig.explanation,
                )
                session.add(signal_row)
                await session.flush()
                signal_ids.append(signal_row.id)

            # Save confluence score
            score_row = ConfluenceScoreRecord(
                ticker=score.ticker,
                direction=score.direction.value,
                conviction=score.conviction_pct,
                active_layers=score.active_layers,
                regime=regime.value if regime else None,
                signal_ids=signal_ids if signal_ids else None,
            )
            session.add(score_row)

        await session.commit()
        logger.debug(f"Saved {len(scores)} scores to history")


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
