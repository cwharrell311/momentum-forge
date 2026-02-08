"""
Base classes for all signal processors.

Every signal layer in the Confluence Engine implements SignalProcessor
and returns SignalResult objects. This ensures uniform scoring and
easy composition in the confluence scoring engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Regime(Enum):
    CALM = "calm"              # VIX < 15, contango
    ELEVATED = "elevated"      # VIX 15-25
    STRESSED = "stressed"      # VIX 25-35, backwardation
    CRISIS = "crisis"          # VIX > 35, deep backwardation


@dataclass
class SignalResult:
    """A single signal from one layer for one ticker."""

    ticker: str
    layer: str                          # e.g., "options_flow", "gex", "momentum"
    direction: Direction
    strength: float                     # 0.0 to 1.0 — how strong is the signal?
    confidence: float                   # 0.0 to 1.0 — how reliable is the data?
    timestamp: datetime
    metadata: dict = field(default_factory=dict)   # Layer-specific details
    explanation: str = ""               # Human-readable summary

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0.0-1.0, got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")


@dataclass
class ConfluenceScore:
    """Combined score across all active signal layers for a single ticker."""

    ticker: str
    direction: Direction
    conviction: float                   # 0.0 to 1.0 — the composite score
    active_layers: int                  # How many layers fired a signal
    total_layers: int                   # How many layers had data available
    signals: list[SignalResult] = field(default_factory=list)
    regime: Regime = Regime.CALM
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def conviction_pct(self) -> int:
        """Conviction as a 0-100 integer for display."""
        return round(self.conviction * 100)


class SignalProcessor(ABC):
    """
    Abstract base class for all signal layers.

    Each signal processor is responsible for:
    1. Fetching raw data from its data source
    2. Normalizing it into a 0.0-1.0 strength score
    3. Determining direction (bullish/bearish/neutral)
    4. Returning SignalResult objects

    Processors must be independent — they don't import from each other.
    """

    @abstractmethod
    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        """
        Scan multiple tickers and return signals for any that fire.

        Returns an empty list for tickers with no meaningful signal.
        Should handle API errors gracefully and skip failed tickers.
        """
        ...

    @abstractmethod
    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Deep scan a single ticker. Returns None if no signal.

        This is used for the ticker deep-dive view and may make
        additional API calls for more detailed data.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this layer, e.g., 'options_flow'."""
        ...

    @property
    @abstractmethod
    def refresh_interval_seconds(self) -> int:
        """
        How often this layer should be re-scanned.

        Fast layers (options flow): 60-300 seconds
        Slow layers (insider buying): 3600-86400 seconds
        """
        ...

    @property
    def weight(self) -> float:
        """
        Default weight in confluence scoring.
        Override in config/signals.yaml for customization.
        """
        return 0.10
