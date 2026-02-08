"""
GEX (Gamma Exposure) / Dealer Positioning Signal Processor

GEX measures how market makers (dealers) are positioned based on the
options they've sold. This determines whether the market is likely to
mean-revert (bounce between levels) or trend (move aggressively).

Core concept:
- When dealers are LONG gamma (positive GEX): they buy dips, sell rips.
  The market is "pinned" and mean-reverts. Ranges tighten.
- When dealers are SHORT gamma (negative GEX): they sell dips, buy rips.
  This AMPLIFIES moves. Trends accelerate, volatility expands.

Formula: GEX = Gamma × OI × 100 × Spot Price
- Calls contribute POSITIVE gamma (dealers hedge by buying stock)
- Puts contribute NEGATIVE gamma (dealers hedge by selling stock)

Why this matters for swing trading:
- Positive GEX zones act as support/resistance (gamma walls)
- Negative GEX = trending environment = ride momentum longer
- Key GEX levels tell you where price is magnetically attracted to
- A flip from positive to negative GEX often precedes big moves

Scoring components (max 6 points bull or bear):
- GEX direction (positive=mean-revert, negative=trending)    → 1.5 pts
- Key level proximity (near gamma wall = high conviction)     → 1.5 pts
- Net GEX magnitude vs historical                            → 1 pt
- Call vs put GEX skew                                       → 1 pt
- GEX regime change (flip detection)                         → 1 pt
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import UnusualWhalesClient

log = logging.getLogger(__name__)


class GexProcessor(SignalProcessor):
    """
    GEX/Dealer positioning signal processor using Unusual Whales data.

    Analyzes gamma exposure to determine market microstructure regime
    and identify key support/resistance levels from dealer hedging.
    """

    def __init__(self, uw_client: UnusualWhalesClient):
        self._uw = uw_client

    @property
    def name(self) -> str:
        return "gex"

    @property
    def refresh_interval_seconds(self) -> int:
        return 300  # 5 min — GEX changes with options trading

    @property
    def weight(self) -> float:
        return 0.18  # Second highest weight

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
                log.warning("GEX scan failed for %s: %s", ticker, e)
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze GEX for a ticker.

        Pulls greek exposure data from UW and determines:
        - Whether dealers are long/short gamma (mean-revert vs trend)
        - Key gamma wall levels (support/resistance)
        - Net GEX direction and magnitude
        - Call/put gamma skew
        """
        if not self._uw.is_configured:
            return None

        try:
            gex_data = await self._uw.get_greek_exposure(ticker)
            if not gex_data:
                return None

            analysis = self._analyze_gex(ticker, gex_data)

            if analysis["net_gex"] == 0 and not analysis["key_levels"]:
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
                    "net_gex": analysis["net_gex"],
                    "gex_regime": analysis["gex_regime"],
                    "call_gex": analysis["call_gex"],
                    "put_gex": analysis["put_gex"],
                    "key_levels": analysis["key_levels"],
                    "gamma_wall": analysis["gamma_wall"],
                    "put_wall": analysis["put_wall"],
                    "gex_skew": analysis["gex_skew"],
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            log.error("GEX analysis failed for %s: %s", ticker, e)
            return None

    def _analyze_gex(self, ticker: str, data: dict) -> dict:
        """
        Core GEX analysis.

        Processes strike-level gamma data to determine:
        1. Net gamma direction (positive = dealers stabilize, negative = amplify)
        2. Key levels where gamma is concentrated (walls)
        3. Directional bias from call/put gamma skew
        """
        # Extract data — UW returns it in various formats
        strikes = data.get("data", data.get("strikes", []))
        if isinstance(strikes, dict):
            strikes = list(strikes.values())

        # Track gamma by strike
        call_gex_total = 0.0
        put_gex_total = 0.0
        strike_gex: dict[float, float] = {}

        for strike_data in strikes if isinstance(strikes, list) else []:
            strike = self._safe_float(
                strike_data.get("strike") or strike_data.get("strike_price", 0)
            )
            if strike <= 0:
                continue

            call_gamma = self._safe_float(
                strike_data.get("call_gex")
                or strike_data.get("call_gamma_exposure")
                or strike_data.get("call_gamma_oi", 0)
            )
            put_gamma = self._safe_float(
                strike_data.get("put_gex")
                or strike_data.get("put_gamma_exposure")
                or strike_data.get("put_gamma_oi", 0)
            )

            call_gex_total += call_gamma
            put_gex_total += abs(put_gamma)  # Put gamma is negative
            net_at_strike = call_gamma - abs(put_gamma)
            strike_gex[strike] = net_at_strike

        # Also check for aggregate fields
        if not strike_gex:
            call_gex_total = self._safe_float(
                data.get("call_gex") or data.get("total_call_gex", 0)
            )
            put_gex_total = self._safe_float(
                data.get("put_gex") or data.get("total_put_gex", 0)
            )

        net_gex = call_gex_total - put_gex_total

        # GEX regime
        if net_gex > 0:
            gex_regime = "positive"  # Mean-reverting
        elif net_gex < 0:
            gex_regime = "negative"  # Trending
        else:
            gex_regime = "neutral"

        # Find key levels (gamma walls)
        key_levels: list[dict] = []
        gamma_wall = 0.0
        put_wall = 0.0

        if strike_gex:
            # Sort by absolute gamma — highest concentration = wall
            sorted_strikes = sorted(
                strike_gex.items(), key=lambda x: abs(x[1]), reverse=True
            )

            for strike, gex_val in sorted_strikes[:5]:
                key_levels.append({
                    "strike": strike,
                    "gex": round(gex_val, 2),
                    "type": "call_wall" if gex_val > 0 else "put_wall",
                })

            # Gamma wall = strike with highest positive GEX (resistance/magnet)
            positive_strikes = [(s, g) for s, g in sorted_strikes if g > 0]
            if positive_strikes:
                gamma_wall = positive_strikes[0][0]

            # Put wall = strike with most negative GEX (support)
            negative_strikes = [(s, g) for s, g in sorted_strikes if g < 0]
            if negative_strikes:
                put_wall = negative_strikes[0][0]

        # Call/put skew
        total_gex = call_gex_total + put_gex_total
        gex_skew = (call_gex_total / total_gex) if total_gex > 0 else 0.5

        # ── Scoring (max 6 points) ──
        bull_points = 0.0
        bear_points = 0.0

        # GEX direction (1.5 pts)
        # Positive GEX in uptrend = bullish (support holds)
        # Negative GEX = trending (momentum amplified — direction from skew)
        if gex_regime == "positive":
            # Positive GEX = mean-reverting, slightly bullish bias
            # (dealers stabilize, support holds)
            bull_points += 1.0
        elif gex_regime == "negative":
            # Negative GEX = trending
            # Direction depends on call/put skew
            if gex_skew > 0.6:
                bull_points += 1.5  # More call gamma = upside amplified
            elif gex_skew < 0.4:
                bear_points += 1.5  # More put gamma = downside amplified
            else:
                # Balanced — slight bearish bias (negative GEX = risk)
                bear_points += 0.5

        # Key level proximity (1.5 pts)
        # Having clear gamma walls provides high-conviction levels
        if gamma_wall > 0 and put_wall > 0:
            bull_points += 0.75  # Clear support from put wall
            bear_points += 0.75  # Clear resistance from gamma wall
            # Net: zero, but increases confidence
        elif gamma_wall > 0:
            bull_points += 0.5
        elif put_wall > 0:
            bear_points += 0.5

        # GEX magnitude (1 pt)
        # Large absolute GEX = strong dealer positioning = more predictive
        if abs(net_gex) > 0:
            magnitude_bonus = min(abs(net_gex) / max(total_gex, 1) * 2, 1.0)
            if net_gex > 0:
                bull_points += magnitude_bonus
            else:
                bear_points += magnitude_bonus

        # Call/put skew (1 pt)
        if gex_skew > 0.65:
            bull_points += 1.0  # Heavily skewed to call gamma
        elif gex_skew > 0.55:
            bull_points += 0.5
        elif gex_skew < 0.35:
            bear_points += 1.0  # Heavily skewed to put gamma
        elif gex_skew < 0.45:
            bear_points += 0.5

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
            strength = 0.15

        # Confidence based on data quality
        confidence = 0.6
        if len(key_levels) >= 3:
            confidence = 0.75
        if len(key_levels) >= 5:
            confidence = 0.85

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "net_gex": round(net_gex, 2),
            "gex_regime": gex_regime,
            "call_gex": round(call_gex_total, 2),
            "put_gex": round(put_gex_total, 2),
            "key_levels": key_levels[:5],
            "gamma_wall": gamma_wall,
            "put_wall": put_wall,
            "gex_skew": round(gex_skew, 3),
        }

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build human-readable GEX explanation."""
        regime = analysis["gex_regime"]
        d = analysis["direction"]
        parts = [f"{ticker} GEX: {d.value}. Dealer regime: {regime}."]

        if regime == "positive":
            parts.append("Positive GEX — mean-reverting, support likely holds.")
        elif regime == "negative":
            parts.append("Negative GEX — trending, moves will be amplified.")

        gw = analysis["gamma_wall"]
        pw = analysis["put_wall"]
        if gw > 0:
            parts.append(f"Gamma wall (resistance/magnet): ${gw:.0f}.")
        if pw > 0:
            parts.append(f"Put wall (support): ${pw:.0f}.")

        skew = analysis["gex_skew"]
        if skew > 0.6:
            parts.append(f"Call-heavy gamma skew ({skew:.0%}).")
        elif skew < 0.4:
            parts.append(f"Put-heavy gamma skew ({skew:.0%}).")

        return " ".join(parts)
