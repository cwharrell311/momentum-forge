"""
Auto-journal: automatically log trade journal entries when signals fire.

When the confluence scanner detects a ticker above your conviction
threshold, this service creates a trade journal entry so you have
a timestamped record of the setup — even when you're away from
the dashboard during market hours.

Deduplication: won't create duplicate entries for the same ticker
if there's already an open (un-exited) trade in the journal.

Works with or without Alpaca/PostgreSQL. If the DB is down,
entries are queued and logged to the console instead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.signals.base import ConfluenceScore, Direction

log = logging.getLogger("confluence.auto_journal")

# In-memory set of tickers we've already logged this session
# Prevents duplicate entries across repeated scans
_logged_tickers: set[str] = set()


async def process_scan_results(
    scores: list[ConfluenceScore],
    regime_value: str | None,
    min_conviction: int = 60,
    min_layers: int = 3,
) -> list[dict]:
    """
    Check scan results and auto-log qualifying signals to the trade journal.

    Args:
        scores: Confluence scores from the latest scan
        regime_value: Current VIX regime string (e.g., "calm", "stressed")
        min_conviction: Minimum conviction % to log (default 60)
        min_layers: Minimum agreeing layers to log (default 3)

    Returns:
        List of dicts describing what was logged (for API/dashboard visibility)
    """
    qualifying = [
        s for s in scores
        if s.conviction_pct >= min_conviction
        and s.active_layers >= min_layers
        and s.direction != Direction.NEUTRAL
        and s.trade_worthy  # Flow gate must pass — smart money must agree
    ]

    if not qualifying:
        return []

    # Check which tickers already have open trades in the DB
    already_open = await _get_open_trade_tickers()

    logged = []
    for score in qualifying:
        ticker = score.ticker

        # Skip if already logged this session or already has an open trade
        if ticker in _logged_tickers or ticker in already_open:
            continue

        entry = await _log_signal_to_journal(score, regime_value)
        if entry:
            _logged_tickers.add(ticker)
            logged.append(entry)
            log.info(
                "AUTO-JOURNAL: %s %s @ %d%% conviction, %d layers, regime=%s → journal #%s",
                score.direction.value,
                ticker,
                score.conviction_pct,
                score.active_layers,
                regime_value or "unknown",
                entry.get("trade_id", "?"),
            )

    if logged:
        log.info("Auto-journal: logged %d new signal(s) this scan", len(logged))

    return logged


async def _get_open_trade_tickers() -> set[str]:
    """Get tickers that already have open (un-exited) trades in the journal."""
    try:
        from sqlalchemy import select

        from src.models.tables import Trade
        from src.utils.db import get_session

        async with get_session() as session:
            result = await session.execute(
                select(Trade.ticker).where(Trade.exit_price.is_(None))
            )
            return {row[0] for row in result.all()}
    except Exception:
        # DB not available — rely on in-memory tracking only
        return set()


async def _log_signal_to_journal(
    score: ConfluenceScore,
    regime_value: str | None,
) -> dict | None:
    """Create a trade journal entry for a qualifying signal."""
    # Build detailed notes showing every signal that contributed
    lines = [
        f"[AUTO-LOG] {score.direction.value.upper()} @ {score.conviction_pct}% conviction",
        f"Regime: {regime_value or 'unknown'} | Layers: {score.active_layers}",
        f"Flow Gate: {score.gate_details}",
        "",
    ]

    for sig in score.signals:
        if sig.direction == Direction.NEUTRAL:
            continue
        dir_arrow = "^" if sig.direction == Direction.BULLISH else "v"
        strength_pct = round(sig.strength * 100)
        conf_pct = round(sig.confidence * 100)
        lines.append(
            f"  {dir_arrow} {sig.layer}: {strength_pct}% str / {conf_pct}% conf"
        )
        if sig.explanation:
            lines.append(f"    {sig.explanation}")

        # Include key metadata values for each layer
        if sig.metadata:
            meta_parts = []
            m = sig.metadata
            if sig.layer == "momentum":
                if m.get("ma_cross") and m["ma_cross"] != "unknown":
                    meta_parts.append(f"MA cross: {m['ma_cross']}")
                if m.get("relative_volume"):
                    meta_parts.append(f"RelVol: {m['relative_volume']}x")
            elif sig.layer == "options_flow":
                if m.get("golden_sweeps"):
                    meta_parts.append(f"Golden sweeps: {m['golden_sweeps']}")
                if m.get("net_premium"):
                    meta_parts.append(f"Net premium: ${m['net_premium']:,.0f}")
            elif sig.layer == "gex":
                if m.get("gex_regime"):
                    meta_parts.append(f"GEX: {m['gex_regime']}")
                if m.get("gamma_wall"):
                    meta_parts.append(f"Gamma wall: ${m['gamma_wall']:,.0f}")
            elif sig.layer == "volatility":
                if m.get("iv_rank") is not None:
                    meta_parts.append(f"IV rank: {m['iv_rank']:.0f}%")
                if m.get("options_are"):
                    meta_parts.append(f"Options: {m['options_are']}")
            elif sig.layer == "dark_pool":
                if m.get("net_flow"):
                    meta_parts.append(f"Flow: {m['net_flow']}")
                if m.get("dp_vs_avg"):
                    meta_parts.append(f"DP vs avg: {m['dp_vs_avg']:.1f}x")
            elif sig.layer == "insider":
                if m.get("buy_count"):
                    meta_parts.append(f"Buys: {m['buy_count']}")
                if m.get("c_suite_buys"):
                    meta_parts.append(f"C-suite buys: {m['c_suite_buys']}")
            elif sig.layer == "short_interest":
                if m.get("si_pct") is not None:
                    meta_parts.append(f"SI: {m['si_pct']:.1f}%")
                if m.get("days_to_cover") is not None:
                    meta_parts.append(f"DTC: {m['days_to_cover']:.1f}")

            if meta_parts:
                lines.append(f"    [{', '.join(meta_parts)}]")

    notes = "\n".join(lines)

    # Get current price from the momentum signal metadata
    entry_price = None
    for sig in score.signals:
        if sig.layer == "momentum" and sig.metadata:
            entry_price = sig.metadata.get("price")
            break

    side = "long" if score.direction == Direction.BULLISH else "short"

    try:
        from src.models.tables import Trade
        from src.utils.db import get_session

        trade = Trade(
            ticker=score.ticker,
            side=side,
            instrument="equity",
            entry_price=entry_price,
            quantity=0,  # 0 = signal only, no position taken yet
            confluence_score_id=None,
            entry_at=datetime.now(timezone.utc),
            notes=notes,
        )

        async with get_session() as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)

            return {
                "trade_id": trade.id,
                "ticker": score.ticker,
                "direction": score.direction.value,
                "conviction": score.conviction_pct,
                "active_layers": score.active_layers,
                "entry_price": entry_price,
                "regime": regime_value,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as e:
        log.warning("Auto-journal DB write failed for %s: %s", score.ticker, e)
        # Still return the entry info so the dashboard/logs show it
        return {
            "trade_id": None,
            "ticker": score.ticker,
            "direction": score.direction.value,
            "conviction": score.conviction_pct,
            "active_layers": score.active_layers,
            "entry_price": entry_price,
            "regime": regime_value,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "db_error": str(e),
        }


def clear_session_log() -> None:
    """
    Clear the in-memory logged tickers set.

    Call this at the start of each trading day so tickers can be
    re-logged if they fire again on a new day.
    """
    count = len(_logged_tickers)
    _logged_tickers.clear()
    log.info("Auto-journal session cleared (%d tickers reset)", count)


def get_session_log() -> list[str]:
    """Get list of tickers logged this session."""
    return sorted(_logged_tickers)
