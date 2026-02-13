"""
Volatility Regime Filter (via Alpaca SPY bars)

This is NOT a trade signal — it's a context signal that modifies how
the confluence engine weights other layers. When the market is stressed,
bullish signals should be discounted and bearish signals amplified.

Uses 20-day realized volatility of SPY (annualized) as a proxy for VIX.
This is what VIX measures anyway — expected S&P 500 volatility — and it
uses zero FMP quota (Alpaca has no rate limit for bars).

Regime classification (realized vol thresholds):
- CALM:     RV < 12    -> Risk-on, trust bullish flow
- ELEVATED: RV 12-22   -> Balanced weighting
- STRESSED: RV 22-32   -> Discount bullish, boost bearish
- CRISIS:   RV > 32    -> Heavy bearish bias, cash is king
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from src.signals.base import Direction, Regime, SignalProcessor, SignalResult
from src.utils.data_providers import AlpacaClient

log = logging.getLogger(__name__)


class VixRegimeProcessor(SignalProcessor):
    """Volatility regime classification using Alpaca SPY bars."""

    def __init__(self, alpaca_client: AlpacaClient):
        self._alpaca = alpaca_client
        self._cached_regime: Regime | None = None

    @property
    def name(self) -> str:
        return "vix_regime"

    @property
    def refresh_interval_seconds(self) -> int:
        return 900  # 15 minutes — regime doesn't change fast

    @property
    def weight(self) -> float:
        return 0.0  # Not weighted in scoring — it modifies other weights

    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        """Regime is market-wide, so we return a single result regardless of tickers."""
        result = await self.scan_single("VIX")
        return [result] if result else []

    async def scan_single(self, ticker: str = "VIX") -> SignalResult | None:
        """Classify current volatility regime from SPY realized vol."""
        try:
            # Fetch 30 days of SPY bars to compute 20-day realized vol
            bars = await self._alpaca.get_bars("SPY", timeframe="1Day", limit=30)
            if not bars or len(bars) < 5:
                log.warning("Not enough SPY bars for regime calculation")
                return None

            # Extract daily close prices
            closes = [bar["c"] for bar in bars if "c" in bar]
            if len(closes) < 5:
                return None

            # Compute annualized realized volatility (20-day window or available)
            rv = self._realized_volatility(closes)
            regime = self._classify_regime(rv)

            # Determine directional bias from regime
            if regime == Regime.CALM:
                direction = Direction.BULLISH
                strength = 0.7
            elif regime == Regime.ELEVATED:
                direction = Direction.NEUTRAL
                strength = 0.5
            elif regime == Regime.STRESSED:
                direction = Direction.BEARISH
                strength = 0.7
            else:  # CRISIS
                direction = Direction.BEARISH
                strength = 0.9

            self._cached_regime = regime

            return SignalResult(
                ticker="VIX",
                layer=self.name,
                direction=direction,
                strength=strength,
                confidence=0.85,
                timestamp=datetime.utcnow(),
                metadata={
                    "regime": regime.value,
                    "realized_vol": round(rv, 1),
                    "spy_bars_used": len(closes),
                },
                explanation=self._build_explanation(regime, rv),
            )

        except Exception as e:
            log.error("Volatility regime check failed: %s", e)
            return None

    def _realized_volatility(self, closes: list[float]) -> float:
        """
        Compute annualized realized volatility from daily closes.

        Formula: stdev(daily log returns) * sqrt(252)
        Uses all available closes (up to 30 days).
        """
        # Daily log returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))

        if len(returns) < 3:
            return 20.0  # Default to elevated if not enough data

        # Standard deviation of returns
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)

        # Annualize and convert to percentage (same scale as VIX)
        return daily_vol * math.sqrt(252) * 100

    def _classify_regime(self, realized_vol: float) -> Regime:
        """Classify regime based on annualized realized volatility."""
        if realized_vol < 12:
            return Regime.CALM
        elif realized_vol < 22:
            return Regime.ELEVATED
        elif realized_vol < 32:
            return Regime.STRESSED
        else:
            return Regime.CRISIS

    def _build_explanation(self, regime: Regime, realized_vol: float) -> str:
        """Human-readable regime explanation."""
        descriptions = {
            Regime.CALM: "Risk-on environment.",
            Regime.ELEVATED: "Elevated caution warranted.",
            Regime.STRESSED: "Stressed market — discount bullish signals.",
            Regime.CRISIS: "Crisis mode — extreme caution.",
        }
        return (
            f"Market regime: {regime.value.upper()}. "
            f"SPY realized vol {realized_vol:.1f}%. "
            f"{descriptions[regime]}"
        )
