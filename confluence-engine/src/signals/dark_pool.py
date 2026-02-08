"""
Dark Pool Signal Processor

Dark pools are private exchanges where institutional investors trade
large blocks of stock without showing their hand to public markets.
FINRA requires reporting (ATS data), which is what we analyze.

Why this matters:
- Dark pool volume reveals what institutions are ACTUALLY doing
  (as opposed to what they say they're doing on CNBC)
- A stock being accumulated heavily in dark pools while price is flat
  = institutions building a position quietly (very bullish)
- A stock being distributed in dark pools while price is rising
  = institutions selling into retail buying (distribution = bearish)

Key metrics:
- Dark pool volume vs total volume (DP %)
- Dark pool volume vs 30-day average DP volume
- Short volume ratio in dark pools
- Block trade frequency (>10K shares or >$200K)
- Net dollar flow direction (accumulation vs distribution)

Scoring components (max 6 points bull or bear):
- DP volume vs average (unusual activity)           → 1.5 pts
- Short volume ratio in DP                          → 1.5 pts
- Block trade frequency and direction               → 1.5 pts
- DP % of total volume (high = institutional)       → 1 pt
- Multi-day trend consistency                       → 0.5 pts
"""

from __future__ import annotations

from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import UnusualWhalesClient


class DarkPoolProcessor(SignalProcessor):
    """
    Dark pool signal processor using Unusual Whales FINRA ATS data.

    Analyzes institutional dark pool activity to detect accumulation
    or distribution patterns invisible on public exchanges.
    """

    def __init__(self, uw_client: UnusualWhalesClient):
        self._uw = uw_client

    @property
    def name(self) -> str:
        return "dark_pool"

    @property
    def refresh_interval_seconds(self) -> int:
        return 900  # 15 min — DP data updates throughout the day

    @property
    def weight(self) -> float:
        return 0.15

    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        if not self._uw.is_configured:
            return []
        results: list[SignalResult] = []
        for ticker in tickers:
            try:
                result = await self.scan_single(ticker)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Dark pool scan failed for {ticker}: {e}")
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze dark pool activity for a ticker.

        Pulls recent dark pool prints and analyzes:
        - Volume relative to 30-day average
        - Short volume ratio
        - Block trade frequency
        - Accumulation vs distribution pattern
        """
        if not self._uw.is_configured:
            return None

        try:
            dp_data = await self._uw.get_dark_pool_flow(ticker)
            if not dp_data:
                return None

            analysis = self._analyze_dark_pool(dp_data)

            if analysis["total_dp_volume"] == 0:
                return None

            direction = analysis["direction"]
            strength = analysis["strength"]

            if strength < 0.10:
                return None

            return SignalResult(
                ticker=ticker,
                layer=self.name,
                direction=direction,
                strength=strength,
                confidence=analysis["confidence"],
                timestamp=datetime.utcnow(),
                metadata={
                    "total_dp_volume": analysis["total_dp_volume"],
                    "dp_vs_avg": analysis["dp_vs_avg"],
                    "short_volume_ratio": analysis["short_volume_ratio"],
                    "block_trades": analysis["block_trades"],
                    "dp_pct_of_total": analysis["dp_pct_of_total"],
                    "net_flow": analysis["net_flow"],
                    "days_analyzed": analysis["days_analyzed"],
                    "trend": analysis["trend"],
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            print(f"Dark pool analysis failed for {ticker}: {e}")
            return None

    def _analyze_dark_pool(self, prints: list[dict]) -> dict:
        """
        Core dark pool analysis.

        Processes recent dark pool prints to determine accumulation
        or distribution patterns.
        """
        if not prints:
            return self._empty_analysis()

        total_dp_volume = 0
        total_short_volume = 0
        total_volume = 0
        block_trades = 0
        daily_volumes: list[float] = []
        daily_short_ratios: list[float] = []

        for dp_print in prints:
            dp_vol = self._safe_float(
                dp_print.get("volume")
                or dp_print.get("dark_pool_volume")
                or dp_print.get("dp_volume", 0)
            )
            short_vol = self._safe_float(
                dp_print.get("dark_pool_short_volume")
                or dp_print.get("short_volume")
                or dp_print.get("dp_short_volume", 0)
            )
            tot_vol = self._safe_float(
                dp_print.get("total_volume")
                or dp_print.get("market_volume", 0)
            )
            trades = self._safe_float(
                dp_print.get("trade_count")
                or dp_print.get("trades", 0)
            )

            total_dp_volume += dp_vol
            total_short_volume += short_vol
            if tot_vol > 0:
                total_volume += tot_vol

            # Count block trades (>10K shares per print or large notional)
            notional = self._safe_float(dp_print.get("notional_value", 0))
            if dp_vol > 10_000 or notional > 200_000:
                block_trades += 1

            if dp_vol > 0:
                daily_volumes.append(dp_vol)
            if dp_vol > 0 and short_vol > 0:
                daily_short_ratios.append(short_vol / dp_vol)

        days_analyzed = len(daily_volumes)
        if days_analyzed == 0:
            return self._empty_analysis()

        # Calculate averages
        avg_daily_volume = sum(daily_volumes) / days_analyzed if daily_volumes else 0

        # DP volume vs average (compare recent to older data)
        if days_analyzed >= 5:
            recent_avg = sum(daily_volumes[:3]) / min(3, len(daily_volumes[:3]))
            older_avg = sum(daily_volumes[3:]) / max(1, len(daily_volumes[3:]))
            dp_vs_avg = recent_avg / older_avg if older_avg > 0 else 1.0
        else:
            dp_vs_avg = 1.0

        # Short volume ratio
        short_volume_ratio = (
            total_short_volume / total_dp_volume if total_dp_volume > 0 else 0.5
        )

        # DP as % of total volume
        dp_pct = total_dp_volume / total_volume if total_volume > 0 else 0

        # Trend detection — data is ordered most recent first
        # Compare recent sessions vs older sessions
        if days_analyzed >= 3:
            midpoint = days_analyzed // 2
            recent_vols = daily_volumes[:midpoint]     # More recent
            older_vols = daily_volumes[midpoint:]      # Older
            avg_recent = sum(recent_vols) / len(recent_vols) if recent_vols else 0
            avg_older = sum(older_vols) / len(older_vols) if older_vols else 0
            if avg_recent > avg_older * 1.2:
                trend = "increasing"
            elif avg_older > avg_recent * 1.2:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # ── Scoring (max 6 points) ──
        bull_points = 0.0
        bear_points = 0.0

        # DP volume vs average (1.5 pts) — unusual activity is the signal
        if dp_vs_avg > 2.0:
            # 2x+ normal DP volume = major institutional activity
            # Direction depends on short ratio
            if short_volume_ratio < 0.40:
                bull_points += 1.5  # Low short ratio = accumulation
            else:
                bear_points += 1.0  # High DP + high shorts = distribution
        elif dp_vs_avg > 1.5:
            if short_volume_ratio < 0.40:
                bull_points += 1.0
            else:
                bear_points += 0.5

        # Short volume ratio (1.5 pts)
        # Below 40% short = net buying (accumulation)
        # Above 55% short = net selling (distribution)
        if short_volume_ratio < 0.35:
            bull_points += 1.5  # Very low short ratio = strong accumulation
        elif short_volume_ratio < 0.40:
            bull_points += 1.0
        elif short_volume_ratio > 0.60:
            bear_points += 1.5  # Very high short ratio = heavy distribution
        elif short_volume_ratio > 0.55:
            bear_points += 1.0

        # Block trades (1.5 pts) — institutional size
        if block_trades >= 5:
            if short_volume_ratio < 0.45:
                bull_points += 1.5  # Many blocks + low shorts = accumulation
            else:
                bear_points += 1.5  # Many blocks + high shorts = distribution
        elif block_trades >= 2:
            if short_volume_ratio < 0.45:
                bull_points += 0.75
            else:
                bear_points += 0.75

        # DP % of total volume (1 pt)
        if dp_pct > 0.50:
            # More than 50% in dark pools = heavy institutional activity
            if short_volume_ratio < 0.45:
                bull_points += 1.0
            else:
                bear_points += 1.0
        elif dp_pct > 0.35:
            if short_volume_ratio < 0.45:
                bull_points += 0.5
            else:
                bear_points += 0.5

        # Multi-day trend (0.5 pts)
        if trend == "increasing":
            if short_volume_ratio < 0.45:
                bull_points += 0.5  # Rising DP volume + accumulation
            else:
                bear_points += 0.5  # Rising DP volume + distribution

        # Net flow categorization
        if short_volume_ratio < 0.45:
            net_flow = "accumulation"
        elif short_volume_ratio > 0.55:
            net_flow = "distribution"
        else:
            net_flow = "neutral"

        # Determine direction and strength
        max_points = 6.0
        if bull_points > bear_points:
            direction = Direction.BULLISH
            strength = min(bull_points / max_points, 1.0)
        elif bear_points > bull_points:
            direction = Direction.BEARISH
            strength = min(bear_points / max_points, 1.0)
        else:
            direction = Direction.NEUTRAL
            strength = 0.10

        # Confidence
        confidence = 0.5
        if days_analyzed >= 5:
            confidence += 0.15
        if block_trades >= 3:
            confidence += 0.1
        if dp_vs_avg > 1.5:
            confidence += 0.1
        confidence = min(confidence, 0.9)

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "total_dp_volume": total_dp_volume,
            "dp_vs_avg": round(dp_vs_avg, 2),
            "short_volume_ratio": round(short_volume_ratio, 3),
            "block_trades": block_trades,
            "dp_pct_of_total": round(dp_pct, 3),
            "net_flow": net_flow,
            "days_analyzed": days_analyzed,
            "trend": trend,
        }

    def _empty_analysis(self) -> dict:
        """Return empty analysis dict."""
        return {
            "direction": Direction.NEUTRAL,
            "strength": 0.0,
            "confidence": 0.0,
            "total_dp_volume": 0,
            "dp_vs_avg": 0, "short_volume_ratio": 0,
            "block_trades": 0, "dp_pct_of_total": 0,
            "net_flow": "unknown", "days_analyzed": 0,
            "trend": "unknown",
        }

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build human-readable dark pool explanation."""
        d = analysis["direction"]
        flow = analysis["net_flow"]
        parts = [f"{ticker} dark pool: {d.value}. Pattern: {flow}."]

        dp_vs = analysis["dp_vs_avg"]
        if dp_vs > 1.5:
            parts.append(f"DP volume {dp_vs:.1f}x average — unusual institutional activity.")

        sr = analysis["short_volume_ratio"]
        if sr < 0.40:
            parts.append(f"Low short ratio ({sr:.0%}) — net accumulation.")
        elif sr > 0.55:
            parts.append(f"High short ratio ({sr:.0%}) — net distribution.")

        blocks = analysis["block_trades"]
        if blocks >= 3:
            parts.append(f"{blocks} block trades detected.")

        dp_pct = analysis["dp_pct_of_total"]
        if dp_pct > 0.40:
            parts.append(f"{dp_pct:.0%} of volume in dark pools.")

        trend = analysis["trend"]
        if trend == "increasing":
            parts.append("DP activity is increasing over recent sessions.")

        return " ".join(parts)
