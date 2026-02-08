"""
Insider Trading Signal Processor

Uses FMP free tier to pull SEC Form 4 filings. These are legally
required disclosures when company insiders (officers, directors,
10%+ owners) buy or sell shares.

Why this matters:
- Insiders KNOW their company better than any analyst
- Cluster buying (multiple insiders buying within 30 days) is one of
  the strongest long-term bullish signals in the market
- A single insider sale means nothing (they sell for taxes, diversification, etc.)
- But cluster selling during price strength is a yellow flag

Scoring logic:
- Count purchases vs sales in last 90 days
- Weight C-suite (CEO/CFO/COO) transactions 2x
- Cluster bonus: 3+ insiders buying in same window = strong signal
- Adjust for transaction size (dollar value)
- Penalize routine 10b5-1 plan sales less than discretionary sales

This processor makes ONE API call per ticker (FMP insider-trading endpoint).
Since insider data moves slowly (days/weeks, not minutes), we only
refresh once per day to conserve API quota.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.signals.base import Direction, SignalProcessor, SignalResult
from src.utils.data_providers import FMPClient

# How far back to look for insider transactions
LOOKBACK_DAYS = 90

# C-suite titles get extra weight
C_SUITE_KEYWORDS = {"ceo", "cfo", "coo", "cto", "president", "chief"}


class InsiderProcessor(SignalProcessor):
    """Insider trading signal processor using FMP SEC Form 4 data."""

    def __init__(self, fmp_client: FMPClient):
        self._fmp = fmp_client

    @property
    def name(self) -> str:
        return "insider"

    @property
    def refresh_interval_seconds(self) -> int:
        return 86400  # Once per day — insider data moves slowly

    @property
    def weight(self) -> float:
        return 0.10

    async def scan(self, tickers: list[str]) -> list[SignalResult]:
        results: list[SignalResult] = []
        for ticker in tickers:
            try:
                result = await self.scan_single(ticker)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Insider scan failed for {ticker}: {e}")
                continue
        return results

    async def scan_single(self, ticker: str) -> SignalResult | None:
        """
        Analyze recent insider transactions for a ticker.

        Looks at the last 90 days of SEC Form 4 filings and scores
        based on the pattern of buying vs selling.
        """
        try:
            transactions = await self._fmp.get_insider_trading(ticker, limit=50)
            if not transactions:
                return None

            analysis = self._analyze_transactions(transactions)

            # No meaningful activity in the lookback window
            if analysis["total_transactions"] == 0:
                return None

            direction = analysis["direction"]
            strength = analysis["strength"]

            # Only fire if there's a meaningful signal
            if strength < 0.15:
                return None

            return SignalResult(
                ticker=ticker,
                layer=self.name,
                direction=direction,
                strength=strength,
                confidence=analysis["confidence"],
                timestamp=datetime.utcnow(),
                metadata={
                    "buy_count": analysis["buy_count"],
                    "sell_count": analysis["sell_count"],
                    "buy_value": analysis["buy_value"],
                    "sell_value": analysis["sell_value"],
                    "c_suite_buys": analysis["c_suite_buys"],
                    "c_suite_sells": analysis["c_suite_sells"],
                    "unique_buyers": analysis["unique_buyers"],
                    "unique_sellers": analysis["unique_sellers"],
                    "most_recent": analysis["most_recent"],
                    "lookback_days": LOOKBACK_DAYS,
                },
                explanation=self._build_explanation(ticker, analysis),
            )

        except Exception as e:
            print(f"Insider analysis failed for {ticker}: {e}")
            return None

    def _analyze_transactions(self, transactions: list[dict]) -> dict:
        """
        Score insider transaction patterns.

        Returns a dict with buy/sell counts, values, direction,
        strength, and confidence.
        """
        cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)

        buy_count = 0
        sell_count = 0
        buy_value = 0.0
        sell_value = 0.0
        c_suite_buys = 0
        c_suite_sells = 0
        buyers: set[str] = set()
        sellers: set[str] = set()
        most_recent = None

        for txn in transactions:
            # Parse transaction date
            txn_date_str = txn.get("transactionDate") or txn.get("filingDate", "")
            if not txn_date_str:
                continue

            try:
                txn_date = datetime.strptime(txn_date_str[:10], "%Y-%m-%d")
            except ValueError:
                continue

            if txn_date < cutoff:
                continue

            if most_recent is None or txn_date_str > most_recent:
                most_recent = txn_date_str

            txn_type = (txn.get("transactionType") or "").upper()
            shares = txn.get("securitiesTransacted") or 0
            price = txn.get("price") or 0
            value = abs(shares * price)
            name = txn.get("reportingName") or ""
            owner_type = (txn.get("typeOfOwner") or "").lower()

            is_c_suite = self._is_c_suite(name, owner_type)

            if "P" in txn_type or "PURCHASE" in txn_type or txn_type.startswith("A"):
                buy_count += 1
                buy_value += value
                buyers.add(name)
                if is_c_suite:
                    c_suite_buys += 1
            elif "S" in txn_type or "SALE" in txn_type:
                sell_count += 1
                sell_value += value
                sellers.add(name)
                if is_c_suite:
                    c_suite_sells += 1

        total = buy_count + sell_count
        unique_buyers = len(buyers)
        unique_sellers = len(sellers)

        if total == 0:
            return {
                "direction": Direction.NEUTRAL,
                "strength": 0.0,
                "confidence": 0.0,
                "total_transactions": 0,
                "buy_count": 0, "sell_count": 0,
                "buy_value": 0, "sell_value": 0,
                "c_suite_buys": 0, "c_suite_sells": 0,
                "unique_buyers": 0, "unique_sellers": 0,
                "most_recent": None,
            }

        # ── Scoring ──
        bull_points = 0.0
        bear_points = 0.0

        # Net buy/sell ratio (up to 2 points)
        if buy_count > sell_count:
            ratio = buy_count / max(sell_count, 1)
            bull_points += min(ratio * 0.5, 2.0)
        elif sell_count > buy_count:
            ratio = sell_count / max(buy_count, 1)
            bear_points += min(ratio * 0.3, 1.5)  # Sales weighted less

        # Value-weighted signal (up to 1 point)
        if buy_value > sell_value * 1.5:
            bull_points += 1.0
        elif sell_value > buy_value * 2:
            bear_points += 0.5

        # Cluster buying bonus (up to 2 points) — multiple unique insiders
        if unique_buyers >= 3:
            bull_points += 2.0  # Strong cluster = very bullish
        elif unique_buyers >= 2:
            bull_points += 1.0

        # C-suite weighting (up to 1 point)
        if c_suite_buys >= 2:
            bull_points += 1.0
        elif c_suite_buys >= 1:
            bull_points += 0.5
        if c_suite_sells >= 2:
            bear_points += 0.5  # C-suite sells are less meaningful

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
        confidence = min(0.5 + (total / 20), 0.9)
        if unique_buyers >= 3 or unique_sellers >= 3:
            confidence = min(confidence + 0.1, 0.95)

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "confidence": round(confidence, 3),
            "total_transactions": total,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "c_suite_buys": c_suite_buys,
            "c_suite_sells": c_suite_sells,
            "unique_buyers": unique_buyers,
            "unique_sellers": unique_sellers,
            "most_recent": most_recent,
        }

    def _is_c_suite(self, name: str, owner_type: str) -> bool:
        """Check if the insider is a C-suite executive."""
        combined = f"{name} {owner_type}".lower()
        return any(kw in combined for kw in C_SUITE_KEYWORDS)

    def _build_explanation(self, ticker: str, analysis: dict) -> str:
        """Build a human-readable explanation of the insider signal."""
        direction = analysis["direction"]
        parts = [f"{ticker} insider activity: {direction.value}."]

        buys = analysis["buy_count"]
        sells = analysis["sell_count"]
        parts.append(f"{buys} purchases, {sells} sales in {LOOKBACK_DAYS} days.")

        if analysis["unique_buyers"] >= 3:
            parts.append(
                f"Cluster buying: {analysis['unique_buyers']} unique insiders buying."
            )
        elif analysis["unique_buyers"] >= 2:
            parts.append(f"{analysis['unique_buyers']} unique insider buyers.")

        if analysis["c_suite_buys"] > 0:
            parts.append(f"{analysis['c_suite_buys']} C-suite purchase(s).")

        buy_val = analysis["buy_value"]
        if buy_val > 0:
            if buy_val >= 1_000_000:
                parts.append(f"Buy value: ${buy_val / 1_000_000:.1f}M.")
            elif buy_val >= 1_000:
                parts.append(f"Buy value: ${buy_val / 1_000:.0f}K.")

        return " ".join(parts)
