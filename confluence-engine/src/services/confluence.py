"""
Confluence Scoring Engine

The brain of the platform. Combines independent signal layers into
a single conviction score per ticker. This is where the "holy grail"
lives — not in any single signal, but in their convergence.

Architecture: All signal processors run in PARALLEL using asyncio.gather.
Each layer is a fully independent agent — if one is slow or fails, the
others complete unaffected. The engine scores whatever data is available.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

log = logging.getLogger(__name__)

from src.signals.base import (
    ConfluenceScore,
    Direction,
    Regime,
    SignalProcessor,
    SignalResult,
)

# Default weights — override via config/signals.yaml
DEFAULT_WEIGHTS: dict[str, float] = {
    "options_flow": 0.25,
    "gex": 0.18,
    "volatility": 0.12,
    "dark_pool": 0.15,
    "insider": 0.10,
    "short_interest": 0.08,
    "momentum": 0.12,
    # vix_regime is NOT weighted — it modifies other weights
}

# How much to boost conviction when multiple layers agree
CONFLUENCE_MULTIPLIERS: dict[int, float] = {
    1: 0.80,    # Single layer — discount, not enough confirmation
    2: 1.00,    # Two layers — baseline
    3: 1.15,    # Three layers — moderate boost
    4: 1.30,    # Four layers — strong boost
    5: 1.45,    # Five+ layers — high conviction
}

# Regime modifiers: adjust bullish/bearish signal weights based on VIX
REGIME_MODIFIERS: dict[Regime, dict[str, float]] = {
    Regime.CALM: {"bullish_mult": 1.10, "bearish_mult": 0.80},
    Regime.ELEVATED: {"bullish_mult": 1.00, "bearish_mult": 1.00},
    Regime.STRESSED: {"bullish_mult": 0.70, "bearish_mult": 1.30},
    Regime.CRISIS: {"bullish_mult": 0.40, "bearish_mult": 1.50},
}


class ConfluenceEngine:
    """
    Combines signals from all active processors into confluence scores.

    All processors run in PARALLEL — each is an independent agent that
    fetches its own data, scores independently, and returns results.
    The engine then combines everything into a unified conviction score.

    Usage:
        engine = ConfluenceEngine(processors=[flow, gex, vol, momentum, vix])
        scores = await engine.scan_all(tickers=["AAPL", "NVDA", "TSLA"])
        # Returns list of ConfluenceScore sorted by conviction descending
    """

    def __init__(
        self,
        processors: list[SignalProcessor],
        weights: dict[str, float] | None = None,
    ):
        self.processors = processors
        self.weights = weights or DEFAULT_WEIGHTS

        # Separate the VIX regime processor from scoring processors
        self._regime_processor: SignalProcessor | None = None
        self._signal_processors: list[SignalProcessor] = []

        for p in processors:
            if p.name == "vix_regime":
                self._regime_processor = p

        self._signal_processors = [
            p for p in processors if p.name != "vix_regime"
        ]

    async def get_current_regime(self) -> Regime:
        """Determine current market regime from VIX data."""
        if self._regime_processor is None:
            return Regime.ELEVATED  # Default if no regime data

        # VIX regime processor returns a single signal with regime in metadata
        result = await self._regime_processor.scan_single("VIX")
        if result and "regime" in result.metadata:
            return Regime(result.metadata["regime"])
        return Regime.ELEVATED

    async def scan_all(self, tickers: list[str]) -> list[ConfluenceScore]:
        """
        Scan all tickers across all signal layers and compute confluence scores.

        All processors run in PARALLEL via asyncio.gather. Each processor is
        an independent agent — if one fails or is slow, the others complete
        unaffected. The engine scores whatever data is available.

        Returns a list of ConfluenceScore objects sorted by conviction (highest first).
        Only returns tickers that have at least 1 active signal.
        """
        regime = await self.get_current_regime()

        # Gather signals from all processors IN PARALLEL
        all_signals: dict[str, list[SignalResult]] = {t: [] for t in tickers}

        async def _run_processor(processor: SignalProcessor) -> list[SignalResult]:
            """Run a single processor as an independent agent."""
            try:
                return await processor.scan(tickers)
            except Exception as e:
                log.warning("%s agent failed: %s", processor.name, e)
                return []

        # Fire all processors simultaneously — each is an independent agent
        results_per_processor = await asyncio.gather(
            *[_run_processor(p) for p in self._signal_processors]
        )

        # Collect all signals by ticker
        for results in results_per_processor:
            for signal in results:
                if signal.ticker in all_signals:
                    all_signals[signal.ticker].append(signal)

        # Score each ticker
        scores: list[ConfluenceScore] = []
        for ticker, signals in all_signals.items():
            if not signals:
                continue
            score = self._calculate_confluence(ticker, signals, regime)
            if score.conviction > 0:
                scores.append(score)

        # Sort by conviction descending
        scores.sort(key=lambda s: s.conviction, reverse=True)
        return scores

    async def scan_single(self, ticker: str) -> ConfluenceScore | None:
        """
        Deep scan a single ticker with detailed signal data.

        All processors run in parallel for this single ticker.
        """
        regime = await self.get_current_regime()

        async def _run_single(processor: SignalProcessor) -> SignalResult | None:
            """Run a single processor scan for one ticker."""
            try:
                return await processor.scan_single(ticker)
            except Exception as e:
                log.warning("%s agent failed for %s: %s", processor.name, ticker, e)
                return None

        # Fire all processors simultaneously
        results = await asyncio.gather(
            *[_run_single(p) for p in self._signal_processors]
        )

        signals = [r for r in results if r is not None]

        if not signals:
            return None

        return self._calculate_confluence(ticker, signals, regime)

    def _calculate_confluence(
        self,
        ticker: str,
        signals: list[SignalResult],
        regime: Regime,
    ) -> ConfluenceScore:
        """
        Core scoring algorithm:
        1. Determine dominant direction (bullish vs bearish by weighted vote)
        2. Apply regime modifier to each signal's effective weight
        3. Calculate weighted average of aligned signals
        4. Apply confluence multiplier based on number of agreeing layers
        5. Penalize if strong contradicting signals exist
        """
        regime_mod = REGIME_MODIFIERS[regime]

        # Step 1: Tally directional weight
        bullish_weight = 0.0
        bearish_weight = 0.0

        for signal in signals:
            w = self.weights.get(signal.layer, 0.10)
            effective = w * signal.strength * signal.confidence
            if signal.direction == Direction.BULLISH:
                bullish_weight += effective * regime_mod["bullish_mult"]
            elif signal.direction == Direction.BEARISH:
                bearish_weight += effective * regime_mod["bearish_mult"]

        if bullish_weight >= bearish_weight:
            dominant = Direction.BULLISH
            dominant_weight = bullish_weight
            opposing_weight = bearish_weight
        else:
            dominant = Direction.BEARISH
            dominant_weight = bearish_weight
            opposing_weight = bullish_weight

        if dominant_weight == 0:
            return ConfluenceScore(
                ticker=ticker,
                direction=Direction.NEUTRAL,
                conviction=0.0,
                active_layers=0,
                total_layers=len(self._signal_processors),
                signals=signals,
                regime=regime,
            )

        # Step 2: Count layers agreeing with dominant direction
        agreeing = [s for s in signals if s.direction == dominant]
        active_layers = len(agreeing)

        # Step 3: Weighted average strength of agreeing signals
        total_weight = sum(self.weights.get(s.layer, 0.10) for s in agreeing)
        if total_weight == 0:
            weighted_strength = 0.0
        else:
            weighted_strength = sum(
                (self.weights.get(s.layer, 0.10) / total_weight)
                * s.strength
                * s.confidence
                for s in agreeing
            )

        # Step 4: Apply confluence multiplier
        multiplier_key = min(active_layers, 5)
        confluence_mult = CONFLUENCE_MULTIPLIERS.get(multiplier_key, 1.50)
        raw_conviction = weighted_strength * confluence_mult

        # Step 5: Penalize for strong opposition
        if opposing_weight > 0 and dominant_weight > 0:
            opposition_ratio = opposing_weight / dominant_weight
            penalty = min(opposition_ratio * 0.3, 0.5)  # Max 50% penalty
            raw_conviction *= (1.0 - penalty)

        # Clamp to 0.0 - 1.0
        conviction = max(0.0, min(1.0, raw_conviction))

        return ConfluenceScore(
            ticker=ticker,
            direction=dominant,
            conviction=conviction,
            active_layers=active_layers,
            total_layers=len(self._signal_processors),
            signals=signals,
            regime=regime,
            timestamp=datetime.utcnow(),
        )
