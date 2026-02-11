"""
In-memory cache for scan results.

Why this exists: The FMP free tier gives you 250 API calls per day.
Without caching, every dashboard refresh triggers a new scan (15+ calls).
With caching, the background scanner runs on a timer and stores results
here. The dashboard reads from the cache instantly — zero API calls.

This is intentionally simple (just Python dicts in memory). When the
server restarts, the cache is empty and refills on the next scan.
Redis caching comes later when we need persistence across restarts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.signals.base import ConfluenceScore, Regime, SignalResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ScanResult:
    """Snapshot of one complete scan cycle."""
    scores: list[ConfluenceScore] = field(default_factory=list)
    regime: Regime = Regime.ELEVATED
    scanned_tickers: int = 0
    timestamp: datetime = field(default_factory=_utcnow)
    dp_regime: str | None = None  # "market_wide" when DP diverges across >60% of signals


class ResultCache:
    """
    Stores the latest scan results in memory.

    Thread-safe: uses asyncio.Lock to prevent concurrent read/write
    corruption between the scheduler (writes) and API routes (reads).

    Usage:
        cache = ResultCache()
        await cache.update(scores, regime, ticker_count)  # Called by scheduler
        result = await cache.latest()                      # Called by API routes
    """

    def __init__(self):
        self._latest: ScanResult | None = None
        self._signal_history: dict[str, list[SignalResult]] = {}
        self._lock = asyncio.Lock()

    async def update(
        self,
        scores: list[ConfluenceScore],
        regime: Regime,
        scanned_tickers: int,
        dp_regime: str | None = None,
    ) -> None:
        """Store new scan results. Called by the scheduler after each scan."""
        async with self._lock:
            self._latest = ScanResult(
                scores=scores,
                regime=regime,
                scanned_tickers=scanned_tickers,
                timestamp=_utcnow(),
                dp_regime=dp_regime,
            )

            # Also store individual signals per ticker for history
            for score in scores:
                self._signal_history[score.ticker] = score.signals

    async def latest(self) -> ScanResult | None:
        """Get the most recent scan results. Returns None if no scan has run yet."""
        async with self._lock:
            return self._latest

    async def get_ticker_signals(self, ticker: str) -> list[SignalResult]:
        """Get the latest signals for a specific ticker."""
        async with self._lock:
            return self._signal_history.get(ticker, [])

    @property
    def has_data(self) -> bool:
        return self._latest is not None

    @property
    def age_seconds(self) -> float:
        """How many seconds since the last scan."""
        if not self._latest:
            return float("inf")
        return (_utcnow() - self._latest.timestamp).total_seconds()
