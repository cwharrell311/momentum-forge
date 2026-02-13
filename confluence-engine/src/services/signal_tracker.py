"""
Signal forward-testing tracker.

Records every qualifying signal at the moment it fires (with the price
at that time), then grades it later by looking up actual prices at
T+1, T+5, T+10, and T+20 trading days.

This answers the critical question: "Are these signals actually predictive?"

Two main jobs:
1. record_signals() — called after each scan, logs qualifying signals
2. grade_signals() — called periodically, fills in outcome prices for
   signals that have aged enough
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import and_, select

from src.signals.base import ConfluenceScore, Direction

log = logging.getLogger("confluence.signal_tracker")

# In-memory set: (ticker, date_str) already recorded today
_recorded_today: set[tuple[str, str]] = set()
_last_clear_date: str = ""


def _today_str() -> str:
    return date.today().isoformat()


def _clear_if_new_day() -> None:
    """Reset the dedup set at the start of each new trading day."""
    global _last_clear_date
    today = _today_str()
    if today != _last_clear_date:
        _recorded_today.clear()
        _last_clear_date = today


async def record_signals(
    scores: list[ConfluenceScore],
    regime_value: str | None,
    min_conviction: int = 40,
    min_layers: int = 2,
) -> int:
    """
    Record qualifying signals from the latest scan into signal_history.

    Lower thresholds than auto-journal (40% conviction, 2 layers) so we
    capture MORE signals for forward testing — we want to know if 40%
    conviction signals actually work, not just 60%+ ones.

    Returns count of newly recorded signals.
    """
    _clear_if_new_day()

    qualifying = [
        s for s in scores
        if s.conviction_pct >= min_conviction
        and s.active_layers >= min_layers
        and s.direction != Direction.NEUTRAL
    ]

    if not qualifying:
        return 0

    today = _today_str()
    new_signals = []

    for score in qualifying:
        key = (score.ticker, today)
        if key in _recorded_today:
            continue

        # Get price from momentum signal metadata
        entry_price = None
        for sig in score.signals:
            if sig.layer == "momentum" and sig.metadata:
                entry_price = sig.metadata.get("price")
                break

        # Build layer details snapshot
        layer_details = {}
        for sig in score.signals:
            if sig.direction == Direction.NEUTRAL:
                continue
            layer_details[sig.layer] = {
                "direction": sig.direction.value,
                "strength": round(sig.strength, 3),
                "confidence": round(sig.confidence, 3),
                "explanation": sig.explanation,
            }

        new_signals.append({
            "ticker": score.ticker,
            "direction": score.direction.value,
            "conviction_pct": score.conviction_pct,
            "active_layers": score.active_layers,
            "trade_worthy": score.trade_worthy,
            "regime": regime_value,
            "entry_price": entry_price,
            "signal_date": datetime.now(timezone.utc),
            "layer_details": layer_details,
        })
        _recorded_today.add(key)

    if not new_signals:
        return 0

    try:
        from src.models.tables import SignalHistory
        from src.utils.db import get_session

        async with get_session() as session:
            for sig_data in new_signals:
                row = SignalHistory(**sig_data)
                session.add(row)
            await session.commit()

        log.info("Signal tracker: recorded %d signals for forward testing", len(new_signals))
        return len(new_signals)

    except Exception as e:
        log.warning("Signal tracker DB write failed: %s", e)
        return 0


async def grade_signals() -> int:
    """
    Grade past signals by looking up current prices for tickers
    that have aged enough.

    For each ungraded signal, checks if enough trading days have
    passed for each horizon (T+1, T+5, T+10, T+20) and fills in
    the outcome price and return.

    Returns count of signals that were graded (at least partially).
    """
    try:
        from src.api.dependencies import get_alpaca_client
        from src.models.tables import SignalHistory
        from src.utils.db import get_session
    except Exception as e:
        log.warning("Grade signals setup failed: %s", e)
        return 0

    now = datetime.now(timezone.utc)
    graded_count = 0

    # Horizons: (column suffix, trading days required, calendar days approx)
    # Trading days ≈ calendar days * 5/7, plus buffer for holidays
    horizons = [
        ("t1", 1, 2),    # T+1: need ~2 calendar days
        ("t5", 5, 8),    # T+5: need ~8 calendar days
        ("t10", 10, 15), # T+10: need ~15 calendar days
        ("t20", 20, 30), # T+20: need ~30 calendar days
    ]

    try:
        async with get_session() as session:
            # Find signals where at least T+1 should be gradeable
            cutoff = now - timedelta(days=2)
            result = await session.execute(
                select(SignalHistory).where(
                    and_(
                        SignalHistory.signal_date <= cutoff,
                        SignalHistory.entry_price.isnot(None),
                        # Not fully graded yet (T+20 not filled)
                        SignalHistory.price_t20.is_(None),
                    )
                ).order_by(SignalHistory.signal_date).limit(50)
            )
            ungraded = result.scalars().all()

            if not ungraded:
                return 0

            # Collect unique tickers we need prices for
            tickers = list({s.ticker for s in ungraded})

            # Fetch current prices via Alpaca
            alpaca = get_alpaca_client()
            if not alpaca or not alpaca.is_configured:
                log.warning("Alpaca not available for grading")
                return 0

            # Get recent bars for each ticker to find closing prices
            ticker_prices = {}
            for ticker in tickers:
                try:
                    bars = await alpaca.get_bars(ticker, timeframe="1Day", limit=30)
                    if bars:
                        # Build a date→close price map
                        price_by_date = {}
                        for b in bars:
                            bar_date = b.get("t", "")[:10]  # YYYY-MM-DD
                            price_by_date[bar_date] = b["c"]
                        ticker_prices[ticker] = {
                            "by_date": price_by_date,
                            "latest": bars[-1]["c"],
                            "dates_sorted": sorted(price_by_date.keys()),
                        }
                except Exception as e:
                    log.debug("Failed to fetch bars for %s: %s", ticker, e)

            # Grade each signal
            for signal in ungraded:
                if signal.ticker not in ticker_prices:
                    continue

                prices = ticker_prices[signal.ticker]
                signal_date_str = signal.signal_date.strftime("%Y-%m-%d")
                dates_after = [
                    d for d in prices["dates_sorted"]
                    if d > signal_date_str
                ]

                updated = False
                is_bullish = signal.direction == "BULLISH"

                for suffix, trading_days, cal_days in horizons:
                    price_col = f"price_{suffix}"
                    return_col = f"return_{suffix}"
                    hit_col = f"hit_{suffix}"

                    # Skip if already graded for this horizon
                    if getattr(signal, price_col) is not None:
                        continue

                    # Check if enough calendar days have passed
                    age_days = (now - signal.signal_date).days
                    if age_days < cal_days:
                        continue

                    # Find the trading day close price
                    if len(dates_after) >= trading_days:
                        target_date = dates_after[trading_days - 1]
                        outcome_price = prices["by_date"][target_date]
                    else:
                        # Not enough trading days in our data yet
                        continue

                    # Calculate return
                    pct_return = round(
                        (outcome_price - signal.entry_price) / signal.entry_price * 100, 2
                    )

                    # Was the direction correct?
                    if is_bullish:
                        hit = outcome_price > signal.entry_price
                    else:
                        hit = outcome_price < signal.entry_price

                    setattr(signal, price_col, outcome_price)
                    setattr(signal, return_col, pct_return if is_bullish else -pct_return)
                    setattr(signal, hit_col, hit)
                    updated = True

                if updated:
                    signal.graded_at = now
                    graded_count += 1

            await session.commit()

        if graded_count:
            log.info("Signal tracker: graded %d signals", graded_count)
        return graded_count

    except Exception as e:
        log.warning("Signal grading failed: %s", e)
        return 0
