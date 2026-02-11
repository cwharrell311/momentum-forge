"""
Momentum / Technical Signal Processor

Uses Alpaca Market Data API (free tier, IEX exchange) to score technical
momentum from historical daily bars. One API call per ticker returns up
to 252 trading days of data — enough to calculate 200-day SMA, RSI,
52-week range, and relative volume.

No daily quota limit. Replaces the earlier FMP implementation that was
capped at 250 calls/day (burned through in a single scan of 83 tickers).

Scoring components (max 6 points bull or bear):
- Moving average alignment (price vs 50-day vs 200-day SMA)     → 2 pts
- Golden/death cross (50 SMA vs 200 SMA relationship)           → 1 pt
- Relative volume (current vs 20-day average)                    → 1 pt
- Price position in 52-week range                                → 1 pt
- Daily price change direction and magnitude                     → 0.5 pt
- Trend strength (distance from key SMAs)                        → 0.5 pt

This is a lagging indicator by design — it confirms what's already
happening, not what's about to happen. Its job is to prevent you
from taking flow signals that fight the trend.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import AlpacaClient

log = logging.getLogger(__name__)


class MomentumProcessor(SignalProcessor):
    """Technical momentum signal processor using Alpaca market data (no quota limit)."""

    def __init__(self, alpaca_client: AlpacaClient):
        self._alpaca = alpaca_client

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
                log.warning("Momentum scan failed for %s: %s", ticker, e)
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Score momentum from Alpaca daily bars.

        Fetches 252 trading days (1 year) and calculates:
        - Current price, daily change %
        - 50-day and 200-day simple moving averages
        - 52-week high and low
        - Volume vs 20-day average volume
        """
        try:
            bars = await self._alpaca.get_bars(ticker, timeframe="1Day", limit=252)
            if not bars or len(bars) < 20:
                return None

            # Convert raw bars into the quote dict the scoring engine expects
            quote = self._bars_to_quote(bars)
            if not quote:
                return None

            scores = self._score_components(quote)

            bull_score = scores.get("bull_score", 0)
            bear_score = scores.get("bear_score", 0)

            if bull_score > bear_score:
                direction = Direction.BULLISH
                strength = min(bull_score / 6.0, 1.0)
            elif bear_score > bull_score:
                direction = Direction.BEARISH
                strength = min(bear_score / 6.0, 1.0)
            else:
                direction = Direction.NEUTRAL
                strength = 0.15

            return SignalResult(
                ticker=ticker,
                layer=self.name,
                direction=direction,
                strength=strength,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "price": quote.get("price"),
                    "change_pct": quote.get("changesPercentage"),
                    "ma_alignment": scores.get("ma_alignment"),
                    "ma_cross": scores.get("ma_cross"),
                    "trend_strength": scores.get("trend_strength"),
                    "relative_volume": scores.get("relative_volume"),
                    "price_vs_52w": scores.get("price_vs_52w"),
                    "sma_50": quote.get("priceAvg50"),
                    "sma_200": quote.get("priceAvg200"),
                    "year_high": quote.get("yearHigh"),
                    "year_low": quote.get("yearLow"),
                    "volume": quote.get("volume"),
                    "avg_volume": quote.get("avgVolume"),
                },
                explanation=self._build_explanation(ticker, quote, scores, direction),
            )

        except Exception as e:
            log.error("Momentum analysis failed for %s: %s", ticker, e)
            return None

    def _bars_to_quote(self, bars: list[dict]) -> dict | None:
        """
        Convert Alpaca daily bars into the quote format the scoring engine expects.

        Calculates SMAs, volume averages, and 52-week range from raw OHLCV data.
        This lets us keep the exact same _score_components() logic — just a
        different data source feeding the same scoring algorithm.
        """
        if not bars:
            return None

        closes = [b["c"] for b in bars]
        highs = [b["h"] for b in bars]
        lows = [b["l"] for b in bars]
        volumes = [b["v"] for b in bars]

        price = closes[-1]

        # Daily change %
        if len(closes) >= 2:
            prev_close = closes[-2]
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        else:
            change_pct = 0

        # Simple moving averages
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
        sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

        # 52-week high/low (all bars we have, up to 252)
        year_high = max(highs)
        year_low = min(lows)

        # Volume: current day vs 20-day average
        volume = volumes[-1] if volumes else 0
        vol_window = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_volume = sum(vol_window) / len(vol_window) if vol_window else 1

        return {
            "price": round(price, 2),
            "changesPercentage": round(change_pct, 2),
            "priceAvg50": round(sma_50, 2) if sma_50 else None,
            "priceAvg200": round(sma_200, 2) if sma_200 else None,
            "yearHigh": round(year_high, 2),
            "yearLow": round(year_low, 2),
            "volume": volume,
            "avgVolume": round(avg_volume),
        }

    def _score_components(self, quote: dict) -> dict:
        """
        Score momentum from quote data. Tally bull/bear points.

        Max bull or bear score = 6 points:
        - MA alignment: up to 2 points
        - Golden/death cross: 1 point
        - Relative volume: 1 point
        - 52-week position: 1 point
        - Daily change: 0.5 points
        - Trend strength: 0.5 points
        """
        bull_score = 0.0
        bear_score = 0.0
        components: dict = {}

        price = quote.get("price", 0)
        sma_50 = quote.get("priceAvg50")
        sma_200 = quote.get("priceAvg200")

        # ── Moving average alignment (most important momentum signal) ──
        # price > 50 SMA > 200 SMA = strong uptrend (2 pts)
        if price and sma_50 and sma_200:
            if price > sma_50 > sma_200:
                components["ma_alignment"] = "strong_bull"
                bull_score += 2
            elif price > sma_50:
                components["ma_alignment"] = "bull"
                bull_score += 1
            elif price < sma_50 < sma_200:
                components["ma_alignment"] = "strong_bear"
                bear_score += 2
            elif price < sma_50:
                components["ma_alignment"] = "bear"
                bear_score += 1
            else:
                components["ma_alignment"] = "mixed"
        elif price and sma_50:
            if price > sma_50:
                components["ma_alignment"] = "bull"
                bull_score += 1
            else:
                components["ma_alignment"] = "bear"
                bear_score += 1

        # ── Golden/death cross detection (50 SMA vs 200 SMA) ──
        if sma_50 and sma_200 and sma_200 > 0:
            sma_spread = (sma_50 - sma_200) / sma_200
            if sma_50 > sma_200:
                components["ma_cross"] = "golden_cross"
                bull_score += 1
            else:
                components["ma_cross"] = "death_cross"
                bear_score += 1
            components["sma_spread_pct"] = round(sma_spread * 100, 2)
        else:
            components["ma_cross"] = "unknown"

        # ── Trend strength (how far price is from 50 SMA) ──
        if price and sma_50 and sma_50 > 0:
            distance_pct = (price - sma_50) / sma_50 * 100
            components["trend_strength"] = round(distance_pct, 2)
            if distance_pct > 5:
                bull_score += 0.5
            elif distance_pct < -5:
                bear_score += 0.5

        # ── Relative volume — confirms conviction behind the move ──
        volume = quote.get("volume", 0)
        avg_volume = quote.get("avgVolume", 1)
        rel_vol = volume / avg_volume if avg_volume and avg_volume > 0 else 1.0
        components["relative_volume"] = round(rel_vol, 2)
        if rel_vol > 1.5:
            if bull_score > bear_score:
                bull_score += 1
            elif bear_score > bull_score:
                bear_score += 1

        # ── 52-week range position ──
        year_high = quote.get("yearHigh", 0)
        year_low = quote.get("yearLow", 0)
        if year_high and year_low and year_high > year_low > 0:
            pct = (price - year_low) / (year_high - year_low)
            components["price_vs_52w"] = round(pct, 2)
            if pct > 0.85:
                bull_score += 1
            elif pct > 0.7:
                bull_score += 0.5
            elif pct < 0.15:
                bear_score += 1
            elif pct < 0.3:
                bear_score += 0.5

        # ── Daily price change — short-term momentum ──
        change_pct = quote.get("changesPercentage", 0) or 0
        if change_pct > 1.5:
            bull_score += 0.5
        elif change_pct < -1.5:
            bear_score += 0.5

        components["bull_score"] = bull_score
        components["bear_score"] = bear_score
        return components

    def _build_explanation(
        self, ticker: str, quote: dict, scores: dict, direction: Direction
    ) -> str:
        """Build a human-readable explanation of the momentum signal."""
        price = quote.get("price", 0)
        change = quote.get("changesPercentage", 0) or 0
        parts = [f"{ticker} at ${price:.2f} ({change:+.1f}%) — momentum is {direction.value}."]

        ma = scores.get("ma_alignment")
        if ma:
            parts.append(f"MA alignment: {ma.replace('_', ' ')}.")

        cross = scores.get("ma_cross")
        if cross and cross != "unknown":
            label = "Golden cross" if cross == "golden_cross" else "Death cross"
            spread = scores.get("sma_spread_pct", 0)
            parts.append(f"{label} (50/200 SMA spread: {spread:+.1f}%).")

        ts = scores.get("trend_strength")
        if ts is not None:
            if abs(ts) > 3:
                parts.append(f"Price is {ts:+.1f}% from 50-day SMA.")

        rv = scores.get("relative_volume")
        if rv and rv > 1.5:
            parts.append(f"Volume {rv:.1f}x average.")

        pos = scores.get("price_vs_52w")
        if pos is not None:
            parts.append(f"52-week range: {pos:.0%}.")

        return " ".join(parts)
