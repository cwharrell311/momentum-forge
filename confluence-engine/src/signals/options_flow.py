"""
Options Flow Signal Processor — THE PRIMARY SIGNAL LAYER

This is the highest-weighted layer (0.25) because institutional options
flow is the single most predictive indicator for swing trading.

How it works:
1. Pull recent flow alerts from Unusual Whales for a ticker
2. Classify each alert by type (Golden Sweep > Sweep > Block > Unusual)
3. Determine TRUE sentiment from bid/ask side (not just calls/puts):
   - Ask-side calls = bullish (buyer paying up for calls)
   - Bid-side calls = bearish (someone selling/closing calls)
   - Ask-side puts = bearish (buyer paying up for puts)
   - Bid-side puts = bullish (someone selling puts = betting won't drop)
4. Weight by premium size ($5K+, $15K+, $30K+ tiers)
5. Score net bullish vs bearish premium
6. Apply conviction multipliers for golden sweeps and repeated hits

Scoring components (max 6 points bull or bear):
- Net premium direction (weighted by alert type)      → 2 pts
- Golden sweep presence                               → 1.5 pts
- Volume/OI ratio signal (new positions vs closing)    → 1 pt
- Premium concentration (many small vs few large)      → 1 pt
- Repeated hits / sweep clusters                       → 0.5 pts
"""

from __future__ import annotations

from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import UnusualWhalesClient


# Alert type weights — golden sweeps are the highest conviction signal
ALERT_TYPE_WEIGHTS: dict[str, float] = {
    "golden_sweep": 3.0,     # $1M+ premium, vol > OI, multi-exchange
    "sweep": 2.0,            # Multi-exchange urgency
    "block": 1.5,            # Large institutional 1:1 trade
    "split": 1.2,            # Split across exchanges
    "floor_trade": 1.5,      # Floor trades — often informed money
    "unusual_volume": 1.0,   # Standard unusual activity
}

# Premium thresholds — larger trades carry more weight
PREMIUM_TIERS: list[tuple[float, float]] = [
    (1_000_000, 3.0),    # $1M+ = massive conviction
    (500_000, 2.5),      # $500K+
    (100_000, 2.0),      # $100K+
    (30_000, 1.5),       # $30K+ = institutional minimum
    (15_000, 1.2),       # $15K+
    (5_000, 1.0),        # $5K+ = baseline meaningful
]


class OptionsFlowProcessor(SignalProcessor):
    """
    Options flow signal processor using Unusual Whales data.

    The highest-weighted layer in the confluence engine. Analyzes
    institutional options flow to determine smart money positioning.
    """

    def __init__(self, uw_client: UnusualWhalesClient):
        self._uw = uw_client

    @property
    def name(self) -> str:
        return "options_flow"

    @property
    def refresh_interval_seconds(self) -> int:
        return 300  # 5 minutes — flow data is time-sensitive

    @property
    def weight(self) -> float:
        return 0.25  # Highest weight — most predictive for swing

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
                print(f"Options flow scan failed for {ticker}: {e}")
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze recent options flow for a ticker.

        Pulls flow alerts from UW and scores based on:
        - Alert types and their conviction levels
        - True bid/ask sentiment (not just call/put direction)
        - Premium size and concentration
        - Volume vs open interest (new positions vs closing)
        """
        if not self._uw.is_configured:
            return None

        try:
            alerts = await self._uw.get_flow_alerts(ticker)
            if not alerts:
                return None

            analysis = self._analyze_flow(alerts)

            if analysis["total_alerts"] == 0:
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
                    "total_alerts": analysis["total_alerts"],
                    "golden_sweeps": analysis["golden_sweeps"],
                    "sweeps": analysis["sweeps"],
                    "blocks": analysis["blocks"],
                    "bullish_premium": analysis["bullish_premium"],
                    "bearish_premium": analysis["bearish_premium"],
                    "net_premium": analysis["net_premium"],
                    "largest_trade": analysis["largest_trade"],
                    "avg_vol_oi_ratio": analysis["avg_vol_oi_ratio"],
                    "dominant_expiry": analysis["dominant_expiry"],
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            print(f"Options flow analysis failed for {ticker}: {e}")
            return None

    def _analyze_flow(self, alerts: list[dict]) -> dict:
        """
        Core flow analysis engine.

        For each alert:
        1. Determine true sentiment from bid/ask side + option type
        2. Weight by alert type (golden sweep highest)
        3. Weight by premium tier
        4. Track volume/OI ratios (>1 = new positions)
        """
        bullish_premium = 0.0
        bearish_premium = 0.0
        bull_weighted = 0.0
        bear_weighted = 0.0
        golden_sweeps = 0
        sweeps = 0
        blocks = 0
        vol_oi_ratios: list[float] = []
        largest_trade = 0.0
        expiry_premium: dict[str, float] = {}

        for alert in alerts:
            premium = self._get_premium(alert)
            if premium < 5_000:
                continue  # Skip tiny trades

            # Determine alert type and its weight
            alert_type = self._get_alert_type(alert)
            type_weight = ALERT_TYPE_WEIGHTS.get(alert_type, 1.0)

            # Count alert types
            if alert_type == "golden_sweep":
                golden_sweeps += 1
            elif alert_type == "sweep":
                sweeps += 1
            elif alert_type == "block":
                blocks += 1

            # Premium tier multiplier
            premium_mult = self._get_premium_multiplier(premium)

            # TRUE sentiment from bid/ask side (not just call/put)
            is_bullish = self._is_bullish(alert)
            weighted_value = premium * type_weight * premium_mult

            if is_bullish:
                bullish_premium += premium
                bull_weighted += weighted_value
            else:
                bearish_premium += premium
                bear_weighted += weighted_value

            # Track vol/OI ratio (>1 means new positions opening)
            vol = self._safe_float(alert.get("volume", 0))
            oi = self._safe_float(alert.get("open_interest", 1))
            if oi > 0:
                vol_oi_ratios.append(vol / oi)

            # Track largest trade
            if premium > largest_trade:
                largest_trade = premium

            # Track premium by expiry
            expiry = alert.get("expiry") or alert.get("expiration_date") or "unknown"
            expiry_premium[expiry] = expiry_premium.get(expiry, 0) + premium

        total_alerts = golden_sweeps + sweeps + blocks + max(0, len(alerts) - golden_sweeps - sweeps - blocks)

        if total_alerts == 0:
            return self._empty_analysis()

        # ── Scoring (max 6 points) ──
        bull_points = 0.0
        bear_points = 0.0

        # Net premium direction (up to 2 pts)
        total_weighted = bull_weighted + bear_weighted
        if total_weighted > 0:
            bull_ratio = bull_weighted / total_weighted
            bear_ratio = bear_weighted / total_weighted
            if bull_ratio > 0.6:
                bull_points += min(bull_ratio * 2.5, 2.0)
            elif bear_ratio > 0.6:
                bear_points += min(bear_ratio * 2.5, 2.0)

        # Golden sweep bonus (up to 1.5 pts)
        if golden_sweeps >= 3:
            # 3+ golden sweeps = maximum conviction
            if bull_weighted > bear_weighted:
                bull_points += 1.5
            else:
                bear_points += 1.5
        elif golden_sweeps >= 1:
            if bull_weighted > bear_weighted:
                bull_points += 1.0
            else:
                bear_points += 1.0

        # Volume/OI ratio (up to 1 pt) — high ratio = new positions
        avg_vol_oi = sum(vol_oi_ratios) / len(vol_oi_ratios) if vol_oi_ratios else 0
        if avg_vol_oi > 2.0:
            # Very high vol/OI = many new positions opening
            if bull_weighted > bear_weighted:
                bull_points += 1.0
            else:
                bear_points += 1.0
        elif avg_vol_oi > 1.0:
            if bull_weighted > bear_weighted:
                bull_points += 0.5
            else:
                bear_points += 0.5

        # Premium concentration (up to 1 pt) — few large trades > many small
        if largest_trade > 500_000:
            if bull_weighted > bear_weighted:
                bull_points += 1.0
            else:
                bear_points += 1.0
        elif largest_trade > 100_000:
            if bull_weighted > bear_weighted:
                bull_points += 0.5
            else:
                bear_points += 0.5

        # Sweep cluster bonus (up to 0.5 pts)
        if sweeps >= 5:
            if bull_weighted > bear_weighted:
                bull_points += 0.5
            else:
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
            strength = 0.1

        # Confidence based on data quality
        confidence = min(0.5 + (total_alerts / 30), 0.95)
        if golden_sweeps >= 2:
            confidence = min(confidence + 0.1, 0.95)

        # Find dominant expiry
        dominant_expiry = max(expiry_premium, key=expiry_premium.get) if expiry_premium else "unknown"

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "total_alerts": total_alerts,
            "golden_sweeps": golden_sweeps,
            "sweeps": sweeps,
            "blocks": blocks,
            "bullish_premium": round(bullish_premium, 2),
            "bearish_premium": round(bearish_premium, 2),
            "net_premium": round(bullish_premium - bearish_premium, 2),
            "largest_trade": round(largest_trade, 2),
            "avg_vol_oi_ratio": round(avg_vol_oi, 2),
            "dominant_expiry": dominant_expiry,
        }

    def _is_bullish(self, alert: dict) -> bool:
        """
        Determine true sentiment from bid/ask side analysis.

        This is the key insight most retail traders miss:
        - Ask-side calls = bullish (buyer paying up)
        - Bid-side calls = bearish (seller closing/selling)
        - Ask-side puts = bearish (buyer paying up for protection)
        - Bid-side puts = bullish (seller selling puts = bullish bet)

        Falls back to UW's sentiment field if bid/ask data isn't available.
        """
        option_type = (alert.get("option_type") or alert.get("put_call") or "").lower()
        side = (alert.get("bid_ask_side") or alert.get("side") or "").lower()

        # If we have bid/ask side data, use true sentiment
        if side and option_type:
            if option_type in ("call", "c"):
                return "ask" in side  # Ask-side calls = bullish
            elif option_type in ("put", "p"):
                return "bid" in side  # Bid-side puts = bullish

        # Fallback: use UW's sentiment classification
        sentiment = (alert.get("sentiment") or "").lower()
        if sentiment:
            return sentiment == "bullish"

        # Last resort: calls = bullish, puts = bearish
        return option_type in ("call", "c")

    def _get_alert_type(self, alert: dict) -> str:
        """Extract and normalize the alert type."""
        alert_rule = (
            alert.get("alert_rule")
            or alert.get("option_activity_type")
            or alert.get("type")
            or ""
        ).lower().replace(" ", "_")

        if "golden" in alert_rule:
            return "golden_sweep"
        if "sweep" in alert_rule:
            return "sweep"
        if "block" in alert_rule:
            return "block"
        if "floor" in alert_rule:
            return "floor_trade"
        if "split" in alert_rule:
            return "split"
        return "unusual_volume"

    def _get_premium(self, alert: dict) -> float:
        """Extract total premium from an alert."""
        for key in ("total_premium", "premium", "cost_basis", "total_size"):
            val = alert.get(key)
            if val is not None:
                return abs(self._safe_float(val))
        # Calculate from price × volume × 100
        price = self._safe_float(alert.get("price") or alert.get("ask") or 0)
        vol = self._safe_float(alert.get("volume") or 0)
        if price > 0 and vol > 0:
            return price * vol * 100
        return 0.0

    def _get_premium_multiplier(self, premium: float) -> float:
        """Get weight multiplier based on premium size tier."""
        for threshold, mult in PREMIUM_TIERS:
            if premium >= threshold:
                return mult
        return 0.5  # Below $5K — de-emphasized

    def _empty_analysis(self) -> dict:
        """Return empty analysis dict."""
        return {
            "direction": Direction.NEUTRAL,
            "strength": 0.0,
            "confidence": 0.0,
            "total_alerts": 0,
            "golden_sweeps": 0, "sweeps": 0, "blocks": 0,
            "bullish_premium": 0, "bearish_premium": 0,
            "net_premium": 0, "largest_trade": 0,
            "avg_vol_oi_ratio": 0, "dominant_expiry": "unknown",
        }

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build human-readable flow explanation."""
        d = analysis["direction"]
        parts = [f"{ticker} options flow: {d.value}."]

        alerts = analysis["total_alerts"]
        parts.append(f"{alerts} significant alerts.")

        gs = analysis["golden_sweeps"]
        if gs > 0:
            parts.append(f"{gs} GOLDEN SWEEP(s) — highest conviction.")

        net = analysis["net_premium"]
        if abs(net) >= 1_000_000:
            parts.append(f"Net premium: ${net / 1_000_000:+.1f}M.")
        elif abs(net) >= 1_000:
            parts.append(f"Net premium: ${net / 1_000:+.0f}K.")

        ratio = analysis["avg_vol_oi_ratio"]
        if ratio > 1.5:
            parts.append(f"Vol/OI ratio {ratio:.1f}x — new positions opening.")

        largest = analysis["largest_trade"]
        if largest >= 1_000_000:
            parts.append(f"Largest single trade: ${largest / 1_000_000:.1f}M.")
        elif largest >= 100_000:
            parts.append(f"Largest trade: ${largest / 1_000:.0f}K.")

        return " ".join(parts)
