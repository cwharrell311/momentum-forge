#!/usr/bin/env python3
"""
Momentum Signal Backtester

Replays the EXACT momentum scoring algorithm from src/signals/momentum.py
against historical Alpaca daily bars for all watchlist tickers.

For each trading day in history (where we have 252 bars of lookback),
computes the momentum signal and checks whether the direction call
was correct at T+1, T+5, T+10, and T+20 trading days.

Two views:
1. All daily signals — what does the momentum reading predict on any given day?
2. Direction-change signals — what happens specifically when the signal FLIPS?
   These are the actionable entry points.

Usage:
    cd confluence-engine
    python -m scripts.backtest_momentum

Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY in .env
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.momentum import MomentumProcessor
from src.utils.data_providers import AlpacaClient

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backtest")


# ── Config ────────────────────────────────────────────────────────────

# How many daily bars to fetch per ticker (2000 ≈ 8 years)
MAX_BARS = 2000

# Lookback window for signal calculation (matches live system)
LOOKBACK = 252

# Forward horizons to grade
HORIZONS = [1, 5, 10, 20]

# Minimum strength to count as a signal (filters out near-ties)
MIN_STRENGTH = 0.25  # 1.5 of 6 points on one side


def load_tickers() -> list[str]:
    """Load tickers from watchlist.yaml."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


def compute_signal(bars_window: list[dict], processor: MomentumProcessor) -> dict | None:
    """
    Compute momentum signal for a bar window using the EXACT same
    logic as the live MomentumProcessor._bars_to_quote + _score_components.

    Returns signal dict or None if neutral/insufficient data.
    """
    if len(bars_window) < 20:
        return None

    quote = processor._bars_to_quote(bars_window)
    if not quote:
        return None

    scores = processor._score_components(quote)

    bull_score = scores.get("bull_score", 0)
    bear_score = scores.get("bear_score", 0)

    if bull_score > bear_score:
        direction = "BULLISH"
        strength = min(bull_score / 6.0, 1.0)
    elif bear_score > bull_score:
        direction = "BEARISH"
        strength = min(bear_score / 6.0, 1.0)
    else:
        return None  # Neutral — skip

    if strength < MIN_STRENGTH:
        return None

    return {
        "direction": direction,
        "strength": round(strength, 3),
        "bull_score": bull_score,
        "bear_score": bear_score,
        "price": quote["price"],
        "ma_alignment": scores.get("ma_alignment"),
        "ma_cross": scores.get("ma_cross"),
        "trend_strength": scores.get("trend_strength"),
        "relative_volume": scores.get("relative_volume"),
        "price_vs_52w": scores.get("price_vs_52w"),
    }


async def backtest_ticker(
    ticker: str,
    alpaca: AlpacaClient,
    processor: MomentumProcessor,
) -> list[dict]:
    """
    Run momentum backtest on a single ticker.

    Slides a 252-bar window across all available history and records
    every non-neutral signal with its forward returns.
    """
    bars = await alpaca.get_bars(ticker, timeframe="1Day", limit=MAX_BARS)
    if not bars or len(bars) < LOOKBACK + max(HORIZONS):
        print(f"  {ticker}: insufficient data ({len(bars) if bars else 0} bars)")
        return []

    signals = []
    n = len(bars)
    prev_direction = None

    # Slide: position i is "today". Need LOOKBACK bars behind, max(HORIZONS) ahead.
    for i in range(LOOKBACK - 1, n - max(HORIZONS)):
        window_start = max(0, i - LOOKBACK + 1)
        window = bars[window_start : i + 1]

        signal = compute_signal(window, processor)
        if signal is None:
            prev_direction = None
            continue

        # Tag direction changes (the actionable entry points)
        is_direction_change = prev_direction is not None and signal["direction"] != prev_direction
        is_fresh = prev_direction is None  # First signal after a neutral streak
        prev_direction = signal["direction"]

        # Forward returns at each horizon
        entry_price = signal["price"]
        bar_date = bars[i].get("t", "")[:10]

        forward = {}
        for h in HORIZONS:
            future_price = bars[i + h]["c"]
            pct_return = (future_price - entry_price) / entry_price * 100

            if signal["direction"] == "BULLISH":
                hit = future_price > entry_price
            else:
                hit = future_price < entry_price

            forward[f"t{h}_price"] = round(future_price, 2)
            forward[f"t{h}_return"] = round(pct_return, 2)
            forward[f"t{h}_hit"] = hit

        signals.append({
            "ticker": ticker,
            "date": bar_date,
            "direction_change": is_direction_change,
            "fresh_signal": is_fresh,
            **signal,
            **forward,
        })

    print(f"  {ticker}: {len(bars)} bars, {n - LOOKBACK - max(HORIZONS) + 1} days tested → {len(signals)} signals")
    return signals


def print_hit_table(signals: list[dict], label: str) -> None:
    """Print a formatted hit-rate table for a set of signals."""
    if not signals:
        print(f"  No signals in this group.")
        return

    print(f"  {'Horizon':<10} {'Signals':>8} {'Hits':>8} {'Hit Rate':>10} {'Avg Ret':>10} {'Dir Ret':>10}")

    for h in HORIZONS:
        key_hit = f"t{h}_hit"
        key_ret = f"t{h}_return"
        graded = [s for s in signals if key_hit in s]
        if not graded:
            continue
        hits = sum(1 for s in graded if s[key_hit])
        avg_ret = sum(s[key_ret] for s in graded) / len(graded)
        dir_ret = sum(
            s[key_ret] if s["direction"] == "BULLISH" else -s[key_ret]
            for s in graded
        ) / len(graded)
        print(
            f"  T+{h:<7} {len(graded):>8} {hits:>8} {hits / len(graded) * 100:>8.1f}%"
            f" {avg_ret:>+9.2f}% {dir_ret:>+9.2f}%"
        )


def print_report(all_signals: list[dict], tickers: list[str]) -> None:
    """Print the full backtest report."""

    total = len(all_signals)
    bull_ct = sum(1 for s in all_signals if s["direction"] == "BULLISH")
    bear_ct = total - bull_ct
    direction_changes = [s for s in all_signals if s["direction_change"]]
    fresh_signals = [s for s in all_signals if s["fresh_signal"]]

    print(f"\n{'='*70}")
    print(f"  VICUNA MOMENTUM BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Total signals: {total:,}  ({bull_ct:,} bullish, {bear_ct:,} bearish)")
    print(f"  Direction flips: {len(direction_changes):,} (actionable entry points)")
    print(f"  Fresh signals (after neutral): {len(fresh_signals):,}")

    # ── 1. Overall hit rates ──────────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  1. OVERALL HIT RATES — All Daily Signals")
    print(f"     'Hit' = signal direction matched actual price move")
    print(f"     'Dir Ret' = avg return in signal direction (positive = making money)")
    print(f"{'─'*70}")
    print_hit_table(all_signals, "Overall")

    # ── 2. Direction flips only ───────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  2. DIRECTION FLIPS ONLY — When Signal Changes Direction")
    print(f"     These are the actual entry/exit points you'd trade on.")
    print(f"{'─'*70}")
    print_hit_table(direction_changes, "Direction Changes")

    # ── 3. By direction ───────────────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  3. BY DIRECTION")
    print(f"{'─'*70}")
    for direction in ["BULLISH", "BEARISH"]:
        dir_signals = [s for s in all_signals if s["direction"] == direction]
        if not dir_signals:
            continue
        print(f"\n  {direction} ({len(dir_signals):,} signals):")
        print_hit_table(dir_signals, direction)

    # ── 4. By strength bucket ─────────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  4. BY SIGNAL STRENGTH")
    print(f"     Does a stronger momentum reading predict better?")
    print(f"{'─'*70}")

    buckets = [
        ("Weak (25-40%)", 0.25, 0.40),
        ("Medium (40-60%)", 0.40, 0.60),
        ("Strong (60%+)", 0.60, 1.01),
    ]

    for label, lo, hi in buckets:
        bucket_signals = [s for s in all_signals if lo <= s["strength"] < hi]
        if not bucket_signals:
            continue
        print(f"\n  {label} ({len(bucket_signals):,} signals):")
        print_hit_table(bucket_signals, label)

    # ── 5. By MA alignment pattern ────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  5. BY MA ALIGNMENT PATTERN")
    print(f"     Which setups work best?")
    print(f"{'─'*70}")

    for pattern in ["strong_bull", "bull", "strong_bear", "bear"]:
        pattern_signals = [s for s in all_signals if s.get("ma_alignment") == pattern]
        if not pattern_signals:
            continue
        print(f"\n  {pattern.replace('_', ' ').upper()} ({len(pattern_signals):,} signals):")
        print_hit_table(pattern_signals, pattern)

    # ── 6. Per-ticker summary ─────────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  6. PER-TICKER SUMMARY (sorted by T+5 directional return)")
    print(f"{'─'*70}")
    print(f"  {'Ticker':<8} {'Signals':>8} {'Bull':>6} {'Bear':>6} {'T+5 Hit':>8} {'T+5 Dir':>9} {'T+20 Dir':>9}")

    ticker_stats: dict[str, dict] = {}
    for s in all_signals:
        t = s["ticker"]
        if t not in ticker_stats:
            ticker_stats[t] = {"signals": [], "bull": 0, "bear": 0}
        ticker_stats[t]["signals"].append(s)
        if s["direction"] == "BULLISH":
            ticker_stats[t]["bull"] += 1
        else:
            ticker_stats[t]["bear"] += 1

    # Sort by T+5 directional return
    def _t5_dir_ret(ticker: str) -> float:
        sigs = ticker_stats[ticker]["signals"]
        t5 = [s for s in sigs if "t5_hit" in s]
        if not t5:
            return 0.0
        return sum(
            s["t5_return"] if s["direction"] == "BULLISH" else -s["t5_return"]
            for s in t5
        ) / len(t5)

    for ticker in sorted(ticker_stats.keys(), key=_t5_dir_ret, reverse=True):
        stats = ticker_stats[ticker]
        t5 = [s for s in stats["signals"] if "t5_hit" in s]
        t20 = [s for s in stats["signals"] if "t20_hit" in s]
        if not t5:
            continue
        t5_hits = sum(1 for s in t5 if s["t5_hit"])
        t5_dir = sum(
            s["t5_return"] if s["direction"] == "BULLISH" else -s["t5_return"]
            for s in t5
        ) / len(t5)
        t20_dir = 0.0
        if t20:
            t20_dir = sum(
                s["t20_return"] if s["direction"] == "BULLISH" else -s["t20_return"]
                for s in t20
            ) / len(t20)
        print(
            f"  {ticker:<8} {len(stats['signals']):>8} {stats['bull']:>6} {stats['bear']:>6}"
            f" {t5_hits / len(t5) * 100:>6.1f}% {t5_dir:>+8.2f}% {t20_dir:>+8.2f}%"
        )

    # ── 7. By year ────────────────────────────────────────────────────

    print(f"\n{'─'*70}")
    print(f"  7. BY YEAR — Does It Work Across Market Regimes?")
    print(f"{'─'*70}")
    print(f"  {'Year':<8} {'Signals':>8} {'T+5 Hit':>8} {'T+5 Dir':>10} {'T+20 Dir':>10}")

    year_signals: dict[str, list] = defaultdict(list)
    for s in all_signals:
        if s.get("date"):
            year_signals[s["date"][:4]].append(s)

    for year in sorted(year_signals.keys()):
        sigs = year_signals[year]
        t5 = [s for s in sigs if "t5_hit" in s]
        t20 = [s for s in sigs if "t20_hit" in s]
        if not t5:
            continue
        t5_hits = sum(1 for s in t5 if s["t5_hit"])
        t5_dir = sum(
            s["t5_return"] if s["direction"] == "BULLISH" else -s["t5_return"]
            for s in t5
        ) / len(t5)
        t20_dir = 0.0
        if t20:
            t20_dir = sum(
                s["t20_return"] if s["direction"] == "BULLISH" else -s["t20_return"]
                for s in t20
            ) / len(t20)
        print(
            f"  {year:<8} {len(sigs):>8} {t5_hits / len(t5) * 100:>6.1f}% {t5_dir:>+9.2f}% {t20_dir:>+9.2f}%"
        )

    # ── 8. Key takeaway ──────────────────────────────────────────────

    # Calculate overall directional returns for the summary
    t5_all = [s for s in all_signals if "t5_hit" in s]
    t5_hit_rate = 0.0
    t5_dir_ret = 0.0
    if t5_all:
        t5_hit_rate = sum(1 for s in t5_all if s["t5_hit"]) / len(t5_all) * 100
        t5_dir_ret = sum(
            s["t5_return"] if s["direction"] == "BULLISH" else -s["t5_return"]
            for s in t5_all
        ) / len(t5_all)

    # Direction flip stats
    t5_flip = [s for s in direction_changes if "t5_hit" in s]
    flip_hit = 0.0
    flip_dir = 0.0
    if t5_flip:
        flip_hit = sum(1 for s in t5_flip if s["t5_hit"]) / len(t5_flip) * 100
        flip_dir = sum(
            s["t5_return"] if s["direction"] == "BULLISH" else -s["t5_return"]
            for s in t5_flip
        ) / len(t5_flip)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Overall T+5:  {t5_hit_rate:.1f}% hit rate, {t5_dir_ret:+.2f}% avg directional return")
    print(f"  Flip signals: {flip_hit:.1f}% hit rate, {flip_dir:+.2f}% avg directional return")
    print()
    if t5_hit_rate > 55:
        print(f"  Momentum signal shows EDGE (>{55}% hit rate).")
        print(f"  Combined with flow gate + other layers, this is a solid foundation.")
    elif t5_hit_rate > 50:
        print(f"  Momentum signal shows SLIGHT EDGE ({t5_hit_rate:.1f}% > 50%).")
        print(f"  Confirms its role as a trend filter — not a standalone signal.")
    else:
        print(f"  Momentum signal is NEAR COIN-FLIP ({t5_hit_rate:.1f}%).")
        print(f"  This is expected — it's a lagging indicator. Its value is as a")
        print(f"  confirmation layer, not a standalone predictor.")
    print(f"{'='*70}")


async def run_backtest() -> None:
    """Main backtest runner."""
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        sys.exit(1)

    alpaca = AlpacaClient(api_key, secret_key, base_url)
    processor = MomentumProcessor(alpaca)
    tickers = load_tickers()

    print(f"\n{'='*70}")
    print(f"  VICUNA MOMENTUM BACKTESTER")
    print(f"  {len(tickers)} tickers · up to {MAX_BARS} bars each · min strength {MIN_STRENGTH}")
    print(f"  Horizons: T+{', T+'.join(str(h) for h in HORIZONS)} trading days")
    print(f"{'='*70}")
    print()
    print("Fetching historical data from Alpaca...")
    print()

    all_signals: list[dict] = []
    for ticker in tickers:
        try:
            signals = await backtest_ticker(ticker, alpaca, processor)
            all_signals.extend(signals)
        except Exception as e:
            print(f"  {ticker}: ERROR — {e}")

    await alpaca.close()

    if not all_signals:
        print("\nNo signals generated. Check your Alpaca API keys and network.")
        return

    # Print full report
    print_report(all_signals, tickers)

    # Save detailed results to JSON for further analysis
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "backtest_momentum_results.json"

    summary = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_bars": MAX_BARS,
            "lookback": LOOKBACK,
            "min_strength": MIN_STRENGTH,
            "horizons": HORIZONS,
        },
        "tickers_tested": tickers,
        "total_signals": len(all_signals),
        "signals": all_signals,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Raw signal data saved to: data/backtest_momentum_results.json")
    print(f"  ({len(all_signals):,} signals with full detail for further analysis)")
    print()


if __name__ == "__main__":
    asyncio.run(run_backtest())
