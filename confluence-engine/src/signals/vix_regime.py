"""
VIX Regime Filter

This is NOT a trade signal — it's a context signal that modifies how
the confluence engine weights other layers. When the market is stressed,
bullish signals should be discounted and bearish signals amplified.

Regime classification:
- CALM:     VIX < 15, futures in contango       -> Risk-on, trust bullish flow
- ELEVATED: VIX 15-25, normal                   -> Balanced weighting
- STRESSED: VIX 25-35, futures in backwardation  -> Discount bullish, boost bearish
- CRISIS:   VIX > 35, deep backwardation         -> Heavy bearish bias, cash is king
"""

from __future__ import annotations

from datetime import datetime

from src.signals.base import Direction, Regime, SignalProcessor, SignalResult
from src.utils.data_providers import FMPClient


class VixRegimeProcessor(SignalProcessor):
    """VIX regime classification using FMP API."""

    def __init__(self, fmp_client: FMPClient):
        self._fmp = fmp_client
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
        """Classify current VIX regime."""
        try:
            vix_quote = await self._fmp.get_vix_quote()
            if not vix_quote:
                return None

            vix_level = vix_quote.get("price", 20)
            regime = self._classify_regime(vix_level)

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
                confidence=0.9,  # VIX data is very reliable
                timestamp=datetime.utcnow(),
                metadata={
                    "regime": regime.value,
                    "vix_level": vix_level,
                    "vix_change": vix_quote.get("changesPercentage", 0),
                },
                explanation=self._build_explanation(regime, vix_level),
            )

        except Exception as e:
            print(f"VIX regime check failed: {e}")
            return None

    def _classify_regime(self, vix_level: float) -> Regime:
        """Classify regime based on VIX level."""
        if vix_level < 15:
            return Regime.CALM
        elif vix_level < 25:
            return Regime.ELEVATED
        elif vix_level < 35:
            return Regime.STRESSED
        else:
            return Regime.CRISIS

    def _build_explanation(self, regime: Regime, vix_level: float) -> str:
        """Human-readable regime explanation."""
        descriptions = {
            Regime.CALM: "Risk-on environment.",
            Regime.ELEVATED: "Elevated caution warranted.",
            Regime.STRESSED: "Stressed market — discount bullish signals.",
            Regime.CRISIS: "Crisis mode — extreme caution.",
        }
        return (
            f"Market regime: {regime.value.upper()}. "
            f"VIX at {vix_level:.1f}. "
            f"{descriptions[regime]}"
        )
