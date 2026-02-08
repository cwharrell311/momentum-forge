"""
Short Interest Signal Processor

Short interest measures how many shares are sold short — borrowed and
sold in the expectation the price will drop. Updated bi-weekly by FINRA.

Why this matters:
- High short interest (>10% of float) = crowded bearish trade
- If a heavily shorted stock starts going UP, shorts scramble to cover
  (buy back shares), creating a self-reinforcing upward spiral = squeeze
- Declining short interest = bears are capitulating = potential bottom
- Rising short interest during a rally = smart money disagrees with price

Key metrics:
- Short interest as % of float (SI%)
- Days to cover (SI / average daily volume) — how long to unwind shorts
- Short interest change (is it increasing or decreasing?)
- Short squeeze potential (high SI + declining + bullish flow = squeeze)

Scoring components (max 6 points bull or bear):
- SI% of float level                                     → 2 pts
- Days to cover                                          → 1.5 pts
- SI change direction                                    → 1.5 pts
- Squeeze potential combo                                → 1 pt
"""

from __future__ import annotations

from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import UnusualWhalesClient


class ShortInterestProcessor(SignalProcessor):
    """
    Short interest signal processor using Unusual Whales data.

    Analyzes short interest levels, trends, and squeeze potential
    to identify contrarian opportunities and crowded trades.
    """

    def __init__(self, uw_client: UnusualWhalesClient):
        self._uw = uw_client

    @property
    def name(self) -> str:
        return "short_interest"

    @property
    def refresh_interval_seconds(self) -> int:
        return 3600  # 1 hour — SI data updates bi-weekly, no rush

    @property
    def weight(self) -> float:
        return 0.08  # Lowest weight — slow-moving, bi-weekly data

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
                print(f"Short interest scan failed for {ticker}: {e}")
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze short interest for a ticker.

        Pulls SI data from UW and evaluates:
        - SI as % of float
        - Days to cover
        - Change in SI (increasing/decreasing)
        - Short squeeze potential
        """
        if not self._uw.is_configured:
            return None

        try:
            si_data = await self._uw.get_short_interest(ticker)
            if not si_data:
                return None

            analysis = self._analyze_short_interest(si_data)

            if analysis["si_pct"] is None or analysis["si_pct"] == 0:
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
                    "si_pct": analysis["si_pct"],
                    "days_to_cover": analysis["days_to_cover"],
                    "si_change_pct": analysis["si_change_pct"],
                    "si_direction": analysis["si_direction"],
                    "squeeze_score": analysis["squeeze_score"],
                    "si_shares": analysis["si_shares"],
                    "report_date": analysis["report_date"],
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            print(f"Short interest analysis failed for {ticker}: {e}")
            return None

    def _analyze_short_interest(self, data: dict) -> dict:
        """
        Core short interest analysis.

        Evaluates SI level, trend, and squeeze potential.
        """
        # Extract data — UW may nest it differently
        inner = data.get("data", data)
        if isinstance(inner, list) and inner:
            inner = inner[0]  # Take most recent entry

        si_pct = self._safe_float(
            inner.get("short_interest_pct_float")
            or inner.get("short_percent_of_float")
            or inner.get("si_pct_float", 0)
        )
        # Normalize: if value is 0-1 (decimal), convert to 0-100 (percentage)
        if 0 < si_pct <= 1:
            si_pct = si_pct * 100

        days_to_cover = self._safe_float(
            inner.get("days_to_cover")
            or inner.get("short_ratio", 0)
        )

        si_change_pct = self._safe_float(
            inner.get("short_interest_change_pct")
            or inner.get("si_change_pct")
            or inner.get("change_pct", 0)
        )

        si_shares = self._safe_float(
            inner.get("short_interest")
            or inner.get("short_shares")
            or inner.get("si_shares", 0)
        )

        report_date = (
            inner.get("date")
            or inner.get("report_date")
            or inner.get("settlement_date")
            or "unknown"
        )

        # SI direction
        if si_change_pct > 5:
            si_direction = "increasing"
        elif si_change_pct < -5:
            si_direction = "decreasing"
        else:
            si_direction = "stable"

        # ── Scoring (max 6 points) ──
        bull_points = 0.0
        bear_points = 0.0

        # SI% of float level (2 pts)
        # High SI is a double-edged sword:
        # - It means bears are positioned aggressively (bearish)
        # - But it also means squeeze potential (bullish contrarian)
        # For swing trading, we lean BULLISH on very high SI because
        # squeezes create explosive moves that our other layers will detect
        if si_pct > 25:
            # Extreme SI (>25%) — high squeeze risk
            bull_points += 2.0  # Contrarian: potential squeeze
        elif si_pct > 15:
            bull_points += 1.5  # Heavily shorted
        elif si_pct > 10:
            bull_points += 1.0  # Moderately high
        elif si_pct > 5:
            # Normal range — slight bearish lean (bears have conviction)
            bear_points += 0.5
        # Below 5% = low SI, no meaningful signal either way

        # Days to cover (1.5 pts)
        # More days = harder to unwind = more squeeze pressure
        if days_to_cover > 10:
            bull_points += 1.5  # Extremely crowded, squeeze fuel
        elif days_to_cover > 5:
            bull_points += 1.0
        elif days_to_cover > 3:
            bull_points += 0.5

        # SI change direction (1.5 pts)
        if si_direction == "decreasing" and si_pct > 10:
            # High SI + decreasing = shorts covering = bullish
            bull_points += 1.5
        elif si_direction == "decreasing":
            bull_points += 0.5
        elif si_direction == "increasing" and si_pct > 10:
            # Rising SI on already high base = bears doubling down
            # Could be bearish, but also builds more squeeze fuel
            bear_points += 1.0
        elif si_direction == "increasing":
            bear_points += 0.75

        # Squeeze potential (1 pt)
        # Squeeze = high SI + high days to cover + shorts starting to cover
        squeeze_score = 0.0
        if si_pct > 15 and days_to_cover > 5:
            squeeze_score += 0.5
        if si_pct > 20 and days_to_cover > 8:
            squeeze_score += 0.3
        if si_direction == "decreasing" and si_pct > 15:
            squeeze_score += 0.2  # Cover has begun

        if squeeze_score > 0.5:
            bull_points += 1.0  # High squeeze potential
        elif squeeze_score > 0.2:
            bull_points += 0.5

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

        # Confidence — SI data is reliable but slow-moving
        confidence = 0.6  # Baseline — data is from FINRA, it's accurate
        if si_pct > 10:
            confidence += 0.1  # Higher SI = more meaningful signal
        if days_to_cover > 3:
            confidence += 0.1
        confidence = min(confidence, 0.85)

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "si_pct": round(si_pct, 2) if si_pct else None,
            "days_to_cover": round(days_to_cover, 2) if days_to_cover else None,
            "si_change_pct": round(si_change_pct, 2),
            "si_direction": si_direction,
            "squeeze_score": round(squeeze_score, 2),
            "si_shares": int(si_shares) if si_shares else None,
            "report_date": report_date,
        }

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build human-readable short interest explanation."""
        d = analysis["direction"]
        parts = [f"{ticker} short interest: {d.value}."]

        si = analysis["si_pct"]
        if si:
            parts.append(f"SI: {si:.1f}% of float.")

        dtc = analysis["days_to_cover"]
        if dtc and dtc > 0:
            parts.append(f"Days to cover: {dtc:.1f}.")

        si_dir = analysis["si_direction"]
        change = analysis["si_change_pct"]
        if si_dir == "increasing":
            parts.append(f"SI increasing ({change:+.1f}%) — bears adding positions.")
        elif si_dir == "decreasing":
            parts.append(f"SI decreasing ({change:+.1f}%) — shorts covering.")

        sq = analysis["squeeze_score"]
        if sq > 0.5:
            parts.append("HIGH squeeze potential — crowded short with limited exit.")
        elif sq > 0.2:
            parts.append("Moderate squeeze potential.")

        return " ".join(parts)
