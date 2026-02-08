"""
Volatility Surface Signal Processor

Analyzes implied volatility (IV) across the options chain to identify:
1. IV Rank — Is current IV high or low vs its own history?
   - Low IV rank (< 25%) = options are cheap, good for buying
   - High IV rank (> 75%) = options are expensive, consider selling/spreads
2. Put/Call IV Skew — Are puts priced higher than calls?
   - High put skew = market pricing in downside risk (bearish fear)
   - Flat or call-heavy skew = complacency or bullish positioning
3. Term Structure — Near-month IV vs far-month IV
   - Inverted (near > far) = event risk, expect near-term volatility
   - Normal (far > near) = calm, gradual time decay environment

Why this matters for swing trading:
- IV Rank tells you if you're buying expensive or cheap options
- Skew reveals hidden fear/greed not visible in price action
- Term structure warns of incoming volatility events
- Mean-reversion: very high IV tends to come down, very low tends to expand

Scoring components (max 6 points bull or bear):
- IV rank position (extreme high/low)                    → 1.5 pts
- Put/call skew direction                                → 1.5 pts
- Term structure slope                                    → 1 pt
- IV vs historical realized volatility                    → 1 pt
- Overall vol regime classification                       → 1 pt
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import UnusualWhalesClient

log = logging.getLogger(__name__)


class VolatilityProcessor(SignalProcessor):
    """
    Volatility surface signal processor using Unusual Whales data.

    Analyzes implied volatility, skew, and term structure to determine
    whether options are cheap/expensive and what the vol surface implies
    about future direction.
    """

    def __init__(self, uw_client: UnusualWhalesClient):
        self._uw = uw_client

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def refresh_interval_seconds(self) -> int:
        return 600  # 10 min — IV doesn't change as fast as flow

    @property
    def weight(self) -> float:
        return 0.12

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
                log.warning("Volatility scan failed for %s: %s", ticker, e)
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze the volatility surface for a ticker.

        Uses the options chain to calculate:
        - IV rank and percentile
        - Put/call IV skew
        - Near vs far term structure
        """
        if not self._uw.is_configured:
            return None

        try:
            # Get interpolated IV data (IV rank, percentile, current IV)
            overview = await self._uw.get_interpolated_iv(ticker)
            # Get options chain for detailed IV analysis
            chain = await self._uw.get_option_chain(ticker)

            if not overview and not chain:
                return None

            analysis = self._analyze_volatility(overview or {}, chain)

            if analysis["iv_rank"] is None and not chain:
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
                    "iv_rank": analysis["iv_rank"],
                    "iv_percentile": analysis["iv_percentile"],
                    "current_iv": analysis["current_iv"],
                    "put_call_skew": analysis["put_call_skew"],
                    "skew_direction": analysis["skew_direction"],
                    "term_structure": analysis["term_structure"],
                    "vol_regime": analysis["vol_regime"],
                    "options_are": analysis["options_are"],
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            log.error("Volatility analysis failed for %s: %s", ticker, e)
            return None

    def _analyze_volatility(self, overview: dict, chain: list[dict]) -> dict:
        """
        Core volatility analysis.

        Extracts IV metrics from overview data and options chain.
        """
        # Unwrap UW's {"data": [...]} envelope if present
        inner = overview.get("data", overview)
        if isinstance(inner, list) and inner:
            inner = inner[0]
        elif isinstance(inner, list):
            inner = {}

        # Try to get IV data from overview first (less API calls)
        iv_rank = self._safe_float(
            overview.get("iv_rank")
            or overview.get("ivRank")
            or inner.get("iv_rank")
        )
        iv_percentile = self._safe_float(
            overview.get("iv_percentile")
            or inner.get("iv_percentile")
        )
        current_iv = self._safe_float(
            overview.get("implied_volatility")
            or overview.get("iv30")
            or inner.get("implied_volatility")
            or inner.get("iv30")
        )

        # If no IV data from overview, calculate from chain
        put_ivs: list[float] = []
        call_ivs: list[float] = []
        near_month_ivs: list[float] = []
        far_month_ivs: list[float] = []
        all_ivs: list[float] = []

        if chain:
            # Sort chain by expiry to identify near/far month
            expiries: set[str] = set()
            for contract in chain:
                exp = contract.get("expiry") or contract.get("expiration_date") or ""
                if exp:
                    expiries.add(exp)

            sorted_expiries = sorted(expiries)
            near_expiries = set(sorted_expiries[:2]) if len(sorted_expiries) >= 2 else set(sorted_expiries)
            far_expiries = set(sorted_expiries[-2:]) if len(sorted_expiries) >= 4 else set()

            for contract in chain:
                iv = self._safe_float(
                    contract.get("implied_volatility")
                    or contract.get("iv")
                    or contract.get("impliedVolatility")
                )
                if iv <= 0 or iv > 10:  # Sanity check (IV > 1000% is noise)
                    continue

                all_ivs.append(iv)
                opt_type = (contract.get("option_type") or contract.get("put_call") or "").lower()
                exp = contract.get("expiry") or contract.get("expiration_date") or ""

                if opt_type in ("put", "p"):
                    put_ivs.append(iv)
                elif opt_type in ("call", "c"):
                    call_ivs.append(iv)

                if exp in near_expiries:
                    near_month_ivs.append(iv)
                elif exp in far_expiries:
                    far_month_ivs.append(iv)

            if not current_iv and all_ivs:
                current_iv = sum(all_ivs) / len(all_ivs)

        # Convert IV rank to 0-100 if it's in 0-1 format
        if iv_rank and iv_rank <= 1.0:
            iv_rank = iv_rank * 100
        if iv_percentile and iv_percentile <= 1.0:
            iv_percentile = iv_percentile * 100

        # Put/call IV skew
        avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0
        avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0
        if avg_call_iv > 0:
            put_call_skew = avg_put_iv / avg_call_iv
        else:
            put_call_skew = 1.0

        skew_direction = "neutral"
        if put_call_skew > 1.15:
            skew_direction = "put_heavy"  # Fear — puts priced higher
        elif put_call_skew < 0.90:
            skew_direction = "call_heavy"  # Greed — calls priced higher
        else:
            skew_direction = "balanced"

        # Term structure
        avg_near = sum(near_month_ivs) / len(near_month_ivs) if near_month_ivs else 0
        avg_far = sum(far_month_ivs) / len(far_month_ivs) if far_month_ivs else 0
        if avg_near > 0 and avg_far > 0:
            if avg_near > avg_far * 1.05:
                term_structure = "inverted"  # Event risk — near-term vol spike
            elif avg_far > avg_near * 1.05:
                term_structure = "normal"  # Calm — normal contango
            else:
                term_structure = "flat"
        else:
            term_structure = "unknown"

        # Vol regime classification
        if iv_rank and iv_rank > 0:
            if iv_rank > 80:
                vol_regime = "high"
                options_are = "expensive"
            elif iv_rank > 50:
                vol_regime = "elevated"
                options_are = "moderately_priced"
            elif iv_rank > 20:
                vol_regime = "normal"
                options_are = "fairly_priced"
            else:
                vol_regime = "low"
                options_are = "cheap"
        else:
            vol_regime = "unknown"
            options_are = "unknown"

        # ── Scoring (max 6 points) ──
        bull_points = 0.0
        bear_points = 0.0

        # IV rank extremes (1.5 pts)
        # Very low IV = options are cheap = good for directional buyers
        # Very high IV = options are expensive = vol likely to contract
        if iv_rank:
            if iv_rank < 15:
                # Very low IV = vol expansion coming, good for buyers
                # Slightly bullish bias (low fear environment)
                bull_points += 1.5
            elif iv_rank < 25:
                bull_points += 0.75
            elif iv_rank > 85:
                # Very high IV = vol contraction likely, bearish for premium buyers
                # Also indicates elevated fear (bearish)
                bear_points += 1.5
            elif iv_rank > 75:
                bear_points += 0.75

        # Put/call skew (1.5 pts)
        if skew_direction == "put_heavy":
            # Elevated put skew = hidden fear, bearish lean
            bear_points += 1.5 if put_call_skew > 1.25 else 1.0
        elif skew_direction == "call_heavy":
            # Elevated call skew = bullish positioning
            bull_points += 1.5 if put_call_skew < 0.80 else 1.0

        # Term structure (1 pt)
        if term_structure == "inverted":
            # Near-term vol spike expected — typically bearish/uncertain
            bear_points += 1.0
        elif term_structure == "normal":
            # Calm environment — slight bullish bias
            bull_points += 0.5

        # IV vs regime alignment (1 pt)
        if vol_regime == "low" and skew_direction == "balanced":
            bull_points += 1.0  # Low vol + no fear = green light
        elif vol_regime == "high" and skew_direction == "put_heavy":
            bear_points += 1.0  # High vol + put skew = maximum fear

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

        # Confidence
        confidence = 0.5
        if iv_rank and iv_rank > 0:
            confidence += 0.15
        if chain and len(chain) > 20:
            confidence += 0.15
        if put_ivs and call_ivs:
            confidence += 0.1
        confidence = min(confidence, 0.9)

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "iv_rank": round(iv_rank, 1) if iv_rank else None,
            "iv_percentile": round(iv_percentile, 1) if iv_percentile else None,
            "current_iv": round(current_iv, 4) if current_iv else None,
            "put_call_skew": round(put_call_skew, 3),
            "skew_direction": skew_direction,
            "term_structure": term_structure,
            "vol_regime": vol_regime,
            "options_are": options_are,
        }

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build human-readable volatility explanation."""
        d = analysis["direction"]
        parts = [f"{ticker} volatility: {d.value}."]

        iv_rank = analysis["iv_rank"]
        if iv_rank is not None:
            parts.append(f"IV rank: {iv_rank:.0f}%.")
            opts = analysis["options_are"]
            if opts != "unknown":
                parts.append(f"Options are {opts.replace('_', ' ')}.")

        skew = analysis["skew_direction"]
        if skew == "put_heavy":
            parts.append(f"Put-heavy skew ({analysis['put_call_skew']:.2f}x) — hidden fear.")
        elif skew == "call_heavy":
            parts.append(f"Call-heavy skew ({analysis['put_call_skew']:.2f}x) — bullish positioning.")

        ts = analysis["term_structure"]
        if ts == "inverted":
            parts.append("Inverted term structure — near-term event risk.")
        elif ts == "normal":
            parts.append("Normal term structure — calm environment.")

        return " ".join(parts)
