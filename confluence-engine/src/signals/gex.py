"""
Signal processor stub — implement in the appropriate roadmap phase.
See ARCHITECTURE.md for the SignalProcessor interface.
"""

from __future__ import annotations

from src.signals.base import SignalProcessor, SignalResult


class GexProcessor(SignalProcessor):
    """TODO: Implement this signal processor."""

    @property
    def name(self) -> str:
        return "gex"

    @property
    def refresh_interval_seconds(self) -> int:
        return 300

    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        raise NotImplementedError("This signal processor is not yet implemented.")

    async def scan_single(self, ticker: str) -> SignalResult | None:
        raise NotImplementedError("This signal processor is not yet implemented.")
