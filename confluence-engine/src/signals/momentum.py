"""
Momentum / Technical Signal Processor

Layer 8 in the confluence engine, but the FIRST to implement because
it uses the free FMP API tier. Provides trend confirmation via:
- RSI (relative strength)
- MACD (momentum direction)
- Moving average alignment (20/50/200 SMA)
- Relative volume (current vs average)
- Price position relative to 52-week range

This is a lagging indicator by design — it confirms what's already
happening, not what's about to happen. That's fine. Its job in the
confluence engine is to prevent you from taking flow signals that
fight the trend.
"""

from __future__ import annotations

from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import FMPClient


class MomentumProcessor(SignalProcessor):
    """Technical momentum signal processor using FMP API."""

    def __init__(self, fmp_client: FMPClient):
        self._fmp = fmp_client

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def refresh_interval_seconds(self) -> int:
        return 300  # 5 minutes

    @property
    def weight(self) -> float:
        return 0.12

    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        results: list[SignalResult] = []
        for ticker in tickers:
            try:
                result = await self.scan_single(ticker)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Momentum scan failed for {ticker}: {e}")
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Fetch technical indicators and score momentum.

        Scoring logic:
        - RSI: >60 bullish, <40 bearish, 40-60 neutral
        - MACD: histogram positive = bullish, negative = bearish
        - MA alignment: price > 20 > 50 > 200 = strong bull trend
        - Relative volume: >1.5x avg = confirmation of move
        """
        try:
            # Fetch data through the shared, rate-limited FMP client
            quote = await self._fmp.get_quote(ticker)
            rsi = await self._fmp.get_rsi(ticker)
            macd = await self._fmp.get_macd(ticker)
            sma = await self._fmp.get_sma_bundle(ticker)

            if not quote:
                return None

            # Score individual components
            scores = self._score_components(quote, rsi, macd, sma)

            # Determine overall direction and strength
            bull_score = scores.get("bull_score", 0)
            bear_score = scores.get("bear_score", 0)

            if bull_score > bear_score:
                direction = Direction.BULLISH
                strength = min(bull_score / 5.0, 1.0)
            elif bear_score > bull_score:
                direction = Direction.BEARISH
                strength = min(bear_score / 5.0, 1.0)
            else:
                direction = Direction.NEUTRAL
                strength = 0.2

            return SignalResult(
                ticker=ticker,
                layer=self.name,
                direction=direction,
                strength=strength,
                confidence=0.7,  # Technicals are inherently medium-confidence
                timestamp=datetime.utcnow(),
                metadata={
                    "rsi": scores.get("rsi"),
                    "macd_histogram": scores.get("macd_histogram"),
                    "ma_alignment": scores.get("ma_alignment"),
                    "relative_volume": scores.get("relative_volume"),
                    "price_vs_52w": scores.get("price_vs_52w"),
                    "components": scores,
                },
                explanation=self._build_explanation(ticker, scores, direction),
            )

        except Exception as e:
            print(f"Momentum analysis failed for {ticker}: {e}")
            return None

    def _score_components(
        self,
        quote: dict,
        rsi: dict | None,
        macd: dict | None,
        sma: dict | None,
    ) -> dict:
        """Score each technical component and tally bull/bear points."""
        bull_score = 0
        bear_score = 0
        components: dict = {}

        # RSI scoring
        rsi_value = rsi.get("rsi") if rsi else None
        components["rsi"] = rsi_value
        if rsi_value:
            if rsi_value > 70:
                bull_score += 0.5  # Overbought — strong but less conviction
            elif rsi_value > 60:
                bull_score += 1
            elif rsi_value < 30:
                bear_score += 0.5  # Oversold — could bounce
            elif rsi_value < 40:
                bear_score += 1

        # MACD scoring
        macd_hist = macd.get("histogram") if macd else None
        components["macd_histogram"] = macd_hist
        if macd_hist is not None:
            if macd_hist > 0:
                bull_score += 1
            elif macd_hist < 0:
                bear_score += 1

        # Moving average alignment
        price = quote.get("price", 0)
        sma_20 = sma.get("sma_20") if sma else None
        sma_50 = sma.get("sma_50") if sma else None
        sma_200 = sma.get("sma_200") if sma else None

        if all([price, sma_20, sma_50, sma_200]):
            if price > sma_20 > sma_50 > sma_200:
                components["ma_alignment"] = "strong_bull"
                bull_score += 2
            elif price > sma_20 > sma_50:
                components["ma_alignment"] = "bull"
                bull_score += 1
            elif price < sma_20 < sma_50 < sma_200:
                components["ma_alignment"] = "strong_bear"
                bear_score += 2
            elif price < sma_20 < sma_50:
                components["ma_alignment"] = "bear"
                bear_score += 1
            else:
                components["ma_alignment"] = "mixed"

        # Relative volume
        volume = quote.get("volume", 0)
        avg_volume = quote.get("avgVolume", 1)
        rel_vol = volume / avg_volume if avg_volume > 0 else 1.0
        components["relative_volume"] = round(rel_vol, 2)
        if rel_vol > 1.5:
            if bull_score > bear_score:
                bull_score += 1
            elif bear_score > bull_score:
                bear_score += 1

        # Price position in 52-week range
        year_high = quote.get("yearHigh", 0)
        year_low = quote.get("yearLow", 0)
        if year_high > year_low > 0:
            pct = (price - year_low) / (year_high - year_low)
            components["price_vs_52w"] = round(pct, 2)
            if pct > 0.8:
                bull_score += 0.5
            elif pct < 0.2:
                bear_score += 0.5

        components["bull_score"] = bull_score
        components["bear_score"] = bear_score
        return components

    def _build_explanation(self, ticker: str, scores: dict, direction: Direction) -> str:
        """Build a human-readable explanation of the momentum signal."""
        parts = [f"{ticker} momentum is {direction.value}."]

        rsi = scores.get("rsi")
        if rsi:
            parts.append(f"RSI at {rsi:.0f}.")

        ma = scores.get("ma_alignment")
        if ma:
            parts.append(f"MA alignment: {ma.replace('_', ' ')}.")

        rv = scores.get("relative_volume")
        if rv and rv > 1.5:
            parts.append(f"Volume {rv:.1f}x average.")

        return " ".join(parts)
