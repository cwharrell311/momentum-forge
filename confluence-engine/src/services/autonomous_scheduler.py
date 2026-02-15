"""
Autonomous Scheduler — the cron that runs the trading machine.

Schedules:
- Every 5 min (market hours): Run trading cycle (check signals, execute, monitor)
- Every day (pre-market): Sync portfolio, reset daily counters
- Every week (Sunday): Full reoptimization (re-backtest, re-deploy winners)
- On startup: Load state, resume trading

This is the process that keeps the autonomous trader alive.
Run it as a background service or via systemd/supervisor.

Usage:
    # Start the scheduler (runs forever)
    python -m src.services.autonomous_scheduler

    # Or import and run programmatically
    from src.services.autonomous_scheduler import start_scheduler
    await start_scheduler()
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, time as dt_time, timezone

log = logging.getLogger("forge.scheduler")

# Market hours (Eastern Time)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
PRE_MARKET = dt_time(9, 0)


def _is_market_hours() -> bool:
    """Check if US stock market is open (approximate — no holiday calendar)."""
    try:
        import zoneinfo
        eastern = zoneinfo.ZoneInfo("US/Eastern")
    except ImportError:
        from datetime import timezone as tz, timedelta
        eastern = tz(timedelta(hours=-5))

    now = datetime.now(eastern)
    # Skip weekends
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def _is_pre_market() -> bool:
    """Check if it's pre-market time."""
    try:
        import zoneinfo
        eastern = zoneinfo.ZoneInfo("US/Eastern")
    except ImportError:
        from datetime import timezone as tz, timedelta
        eastern = tz(timedelta(hours=-5))

    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return PRE_MARKET <= current_time < MARKET_OPEN


class AutonomousScheduler:
    """
    Runs the autonomous trader on a schedule.

    - Trading cycles every 5 minutes during market hours
    - Daily pre-market sync
    - Weekly reoptimization on Sundays
    """

    def __init__(
        self,
        cycle_interval_seconds: int = 300,    # 5 minutes
        reoptimize_day: int = 6,              # Sunday
        state_file: str = "autonomous_state.json",
    ):
        self.cycle_interval = cycle_interval_seconds
        self.reoptimize_day = reoptimize_day
        self.state_file = state_file
        self._running = False
        self._trader = None
        self._last_daily_reset: str = ""
        self._last_reoptimize: str = ""

    async def start(self) -> None:
        """Start the scheduler. Runs until interrupted."""
        from src.services.autonomous_trader import AutonomousTrader

        log.info("=" * 60)
        log.info("  MOMENTUM FORGE — AUTONOMOUS SCHEDULER")
        log.info("  Cycle interval: %ds", self.cycle_interval)
        log.info("  State file: %s", self.state_file)
        log.info("=" * 60)

        # Connect to Alpaca if credentials are available
        alpaca_client = None
        try:
            from src.services.alpaca_client import AlpacaClient
            alpaca_client = AlpacaClient.from_env()
            account = await alpaca_client.get_account()
            log.info(
                "Alpaca connected: equity=$%s, cash=$%s (%s)",
                account["equity"], account["cash"],
                "PAPER" if alpaca_client.paper else "LIVE",
            )
        except ValueError:
            log.warning("No Alpaca credentials — running in backtest-only mode")
        except Exception as e:
            log.warning("Alpaca connection failed: %s — running in backtest-only mode", e)

        self._trader = AutonomousTrader(paper_mode=True, alpaca_client=alpaca_client)

        # Load existing state
        if self._trader.load_state(self.state_file):
            log.info("Resumed from saved state")
        else:
            log.info("No saved state — running initial discovery...")
            candidates = await self._trader.discover_strategies()
            if candidates:
                self._trader.deploy_strategies(candidates)
                self._trader.save_state(self.state_file)
                log.info("Initial discovery complete: %d strategies deployed", len(self._trader.portfolio.deployed_strategies))
            else:
                log.warning("No strategies found in initial discovery")

        self._running = True
        log.info("Scheduler started. Press Ctrl+C to stop.")

        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Weekly reoptimization (Sunday)
                today_str = now.strftime("%Y-%m-%d")
                if (now.weekday() == self.reoptimize_day
                        and self._last_reoptimize != today_str):
                    log.info("Running weekly reoptimization...")
                    await self._trader.reoptimize()
                    self._last_reoptimize = today_str
                    self._trader.save_state(self.state_file)

                # Daily reset (pre-market)
                if _is_pre_market() and self._last_daily_reset != today_str:
                    log.info("Pre-market daily reset")
                    self._trader.portfolio.daily_pnl = 0.0
                    if self._trader.portfolio.halted:
                        # Auto-unhalt at start of new day (unless max DD)
                        if self._trader.portfolio.current_drawdown_pct < self._trader.max_drawdown_halt_pct:
                            self._trader.portfolio.halted = False
                            self._trader.portfolio.halt_reason = ""
                            log.info("Trading unhalted for new day")
                    self._last_daily_reset = today_str
                    self._trader.save_state(self.state_file)

                # Trading cycle (market hours only)
                if _is_market_hours():
                    result = await self._trader.run_cycle()
                    log.info(
                        "Cycle: eq=$%.0f dd=%.1f%% pos=%d sig=%d ord=%d",
                        result.get("equity", 0),
                        result.get("drawdown_pct", 0),
                        result.get("open_positions", 0),
                        result.get("signals_generated", 0),
                        result.get("orders_submitted", 0),
                    )
                else:
                    log.debug("Market closed — sleeping")

                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("Scheduler error: %s", e, exc_info=True)
                await asyncio.sleep(30)  # Back off on error

        log.info("Scheduler stopped")
        if self._trader:
            self._trader.save_state(self.state_file)

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False


async def start_scheduler(**kwargs) -> None:
    """Convenience function to start the scheduler."""
    scheduler = AutonomousScheduler(**kwargs)

    # Handle SIGTERM/SIGINT
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig_name, scheduler.stop)

    await scheduler.start()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Momentum Forge Autonomous Scheduler")
    parser.add_argument("--interval", type=int, default=300, help="Cycle interval in seconds")
    parser.add_argument("--state", type=str, default="autonomous_state.json", help="State file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(start_scheduler(
        cycle_interval_seconds=args.interval,
        state_file=args.state,
    ))


if __name__ == "__main__":
    main()
