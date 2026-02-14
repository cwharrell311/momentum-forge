#!/usr/bin/env python3
"""
TODAY'S SPY TRADING SIGNALS

One command. Plain English. What to trade right now.

Usage:
    PYTHONPATH=. python3 -m backtest.signals
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

from .data import fetch_spy_daily, fetch_vix_daily, add_indicators
from .strategies import (
    GapAndGo, GapFade, OpeningRangeBreakout, BollingerMeanReversion,
    RSIReversal, VWAPBounce, MomentumCrossover, VolatilityBreakout,
    StochasticReversal, RegimeTrend, CompositeOptimal,
)

# ── Strategy configs with backtest-proven parameters ──
STRATEGIES = [
    ("Volatility Breakout", VolatilityBreakout(channel_period=20, volume_filter=1.3, stop_atr_mult=1.5)),
    ("Composite (2-vote)",  CompositeOptimal(min_votes=2, stop_atr_mult=1.5)),
    ("EMA Crossover",       MomentumCrossover(fast_ema=9, slow_ema=21, macd_confirm=True, stop_atr_mult=1.5)),
    ("Composite (3-vote)",  CompositeOptimal(min_votes=3, stop_atr_mult=1.5)),
    ("Gap Fade",            GapFade(min_gap_pct=0.5, rsi_threshold=70, stop_atr_mult=1.5)),
    ("Gap & Go",            GapAndGo(min_gap_pct=0.3, max_gap_pct=2.0, volume_filter=1.2, stop_atr_mult=1.0)),
    ("ORB Breakout",        OpeningRangeBreakout(atr_filter=0.5, volume_filter=1.0, stop_atr_mult=1.0)),
    ("BB Mean Reversion",   BollingerMeanReversion(bb_entry=0.0, rsi_oversold=30, rsi_overbought=70, stop_atr_mult=2.0)),
    ("RSI Reversal",        RSIReversal(rsi_oversold=25, rsi_overbought=75, stop_atr_mult=1.5)),
    ("Stochastic Reversal", StochasticReversal(oversold=20, overbought=80, stop_atr_mult=1.5)),
    ("VWAP Bounce",         VWAPBounce(threshold_pct=0.3, trend_ma=50, stop_atr_mult=1.0)),
    ("Regime Trend",        RegimeTrend(vix_calm_threshold=20, vix_fear_threshold=30, stop_atr_mult=1.5)),
]


def get_market_data():
    """Fetch recent SPY + VIX data with indicators."""
    print("  Fetching SPY data...")
    spy = fetch_spy_daily(years=1, use_cache=True)
    print("  Fetching VIX data...")
    vix = fetch_vix_daily(years=1, use_cache=True)
    spy = spy.join(vix, how="left")
    spy["VIX_Close"] = spy["VIX_Close"].ffill()
    df = add_indicators(spy)
    df.dropna(subset=["SMA_50", "ATR", "RSI", "BB_pct"], inplace=True)
    return df


def scan_signals(df):
    """Run all strategies and collect signals for the latest bar."""
    latest = df.index[-1]
    price = df["Close"].iloc[-1]
    signals = []

    for name, strategy in STRATEGIES:
        sig_df = strategy.generate_signals(df)
        last_row = sig_df.iloc[-1]
        signal = int(last_row.get("signal", 0))
        if signal != 0:
            entry = last_row.get("entry_price", np.nan)
            stop = last_row.get("stop_loss", np.nan)
            direction = "LONG" if signal == 1 else "SHORT"
            risk_pct = abs(entry - stop) / entry * 100 if not np.isnan(stop) and not np.isnan(entry) and entry > 0 else 0
            signals.append({
                "strategy": name,
                "direction": direction,
                "entry": entry,
                "stop": stop,
                "risk_pct": risk_pct,
            })

    return latest, price, signals


def get_indicator_snapshot(df):
    """Get current indicator readings."""
    row = df.iloc[-1]
    prev = df.iloc[-2]
    return {
        "price": row["Close"],
        "rsi": row["RSI"],
        "vix": row.get("VIX_Close", 0),
        "bb_pct": row["BB_pct"],
        "macd_hist": row["MACD_Hist"],
        "stoch_k": row["Stoch_K"],
        "stoch_d": row["Stoch_D"],
        "atr": row["ATR"],
        "atr_pct": row["ATR"] / row["Close"] * 100,
        "vol_ratio": row["Vol_Ratio"],
        "ema9": row["EMA_9"],
        "ema21": row["EMA_21"],
        "sma50": row["SMA_50"],
        "trend": "UP" if row["EMA_9"] > row["EMA_21"] > row["SMA_50"] else
                 "DOWN" if row["EMA_9"] < row["EMA_21"] < row["SMA_50"] else "CHOP",
        "gap_pct": row["Gap_Pct"],
        "prev_close": prev["Close"],
    }


def print_dashboard(date, price, signals, indicators):
    """Print the trading dashboard."""
    n_long = sum(1 for s in signals if s["direction"] == "LONG")
    n_short = sum(1 for s in signals if s["direction"] == "SHORT")
    n_total = len(signals)
    n_strats = len(STRATEGIES)

    print()
    print("=" * 62)
    print("       SPY DAYTRADING SIGNAL SCANNER")
    print("=" * 62)
    print(f"  Date:          {date.strftime('%A, %B %d, %Y')}")
    print(f"  SPY Close:     ${price:.2f}")
    print(f"  Prev Close:    ${indicators['prev_close']:.2f}")
    print(f"  Gap:           {indicators['gap_pct']:+.2f}%")
    print()

    # ── THE VERDICT ──
    print("-" * 62)
    if n_total == 0:
        print("  SIGNAL:   SIT OUT  --  No strategies firing today")
        print("            Cash is a position. Wait for a setup.")
    elif n_long > 0 and n_short == 0:
        confidence = n_long / n_strats * 100
        if n_long >= 3:
            print(f"  SIGNAL:   BUY SPY AT OPEN")
            print(f"            {n_long}/{n_strats} strategies say LONG ({confidence:.0f}% confluence)")
        else:
            print(f"  SIGNAL:   LEAN LONG (weak)")
            print(f"            Only {n_long}/{n_strats} strategies agree ({confidence:.0f}% confluence)")
    elif n_short > 0 and n_long == 0:
        confidence = n_short / n_strats * 100
        if n_short >= 3:
            print(f"  SIGNAL:   SHORT SPY AT OPEN")
            print(f"            {n_short}/{n_strats} strategies say SHORT ({confidence:.0f}% confluence)")
        else:
            print(f"  SIGNAL:   LEAN SHORT (weak)")
            print(f"            Only {n_short}/{n_strats} strategies agree ({confidence:.0f}% confluence)")
    else:
        print(f"  SIGNAL:   MIXED  --  {n_long} LONG vs {n_short} SHORT")
        print(f"            Conflicting signals. Reduce size or sit out.")
    print("-" * 62)

    # ── STRATEGY BREAKDOWN ──
    if signals:
        print()
        print("  STRATEGIES FIRING:")
        print(f"  {'Strategy':<22} {'Dir':<7} {'Entry':>8} {'Stop':>8} {'Risk':>6}")
        print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
        for s in signals:
            entry_str = f"${s['entry']:.2f}" if not np.isnan(s['entry']) else "Open"
            stop_str = f"${s['stop']:.2f}" if not np.isnan(s['stop']) else "  --"
            risk_str = f"{s['risk_pct']:.1f}%" if s['risk_pct'] > 0 else "  --"
            print(f"  {s['strategy']:<22} {s['direction']:<7} {entry_str:>8} {stop_str:>8} {risk_str:>6}")

    # ── KEY LEVELS ──
    print()
    print("  KEY LEVELS:")
    if signals:
        entries = [s["entry"] for s in signals if not np.isnan(s["entry"])]
        stops = [s["stop"] for s in signals if not np.isnan(s["stop"])]
        if entries:
            avg_entry = np.mean(entries)
            print(f"    Avg Entry:    ${avg_entry:.2f}")
        if stops:
            avg_stop = np.mean(stops)
            print(f"    Avg Stop:     ${avg_stop:.2f}")
            if entries:
                avg_risk = abs(avg_entry - avg_stop) / avg_entry * 100
                print(f"    Avg Risk:     {avg_risk:.2f}%")
                # Position sizing: risk 1% of account
                for acct in [10000, 25000, 50000, 100000]:
                    risk_dollars = acct * 0.01
                    risk_per_share = abs(avg_entry - avg_stop)
                    if risk_per_share > 0:
                        shares = int(risk_dollars / risk_per_share)
                        cost = shares * avg_entry
                        print(f"    ${acct:>7,} acct -> {shares:>4} shares (${cost:,.0f}) risking ${risk_dollars:,.0f}")

    # ── MARKET SNAPSHOT ──
    print()
    print("  MARKET SNAPSHOT:")
    trend = indicators["trend"]
    trend_icon = "Bullish" if trend == "UP" else "Bearish" if trend == "DOWN" else "Choppy"
    print(f"    Trend:        {trend_icon} (EMA9/21/SMA50)")
    print(f"    RSI:          {indicators['rsi']:.1f}" +
          (" (Overbought)" if indicators['rsi'] > 70 else
           " (Oversold)" if indicators['rsi'] < 30 else ""))
    print(f"    VIX:          {indicators['vix']:.1f}" +
          (" (Fear!)" if indicators['vix'] > 30 else
           " (Calm)" if indicators['vix'] < 20 else " (Elevated)"))
    print(f"    BB %K:        {indicators['bb_pct']:.2f}" +
          (" (At upper band)" if indicators['bb_pct'] > 0.95 else
           " (At lower band)" if indicators['bb_pct'] < 0.05 else ""))
    print(f"    MACD Hist:    {indicators['macd_hist']:.3f}" +
          (" (Bullish)" if indicators['macd_hist'] > 0 else " (Bearish)"))
    print(f"    Stoch K/D:    {indicators['stoch_k']:.1f}/{indicators['stoch_d']:.1f}")
    print(f"    ATR:          ${indicators['atr']:.2f} ({indicators['atr_pct']:.2f}% of price)")
    print(f"    Volume:       {indicators['vol_ratio']:.2f}x avg" +
          (" (High!)" if indicators['vol_ratio'] > 1.5 else
           " (Low)" if indicators['vol_ratio'] < 0.7 else ""))

    # ── RULES ──
    print()
    print("-" * 62)
    print("  RULES (non-negotiable):")
    print("    1. Max 1% account risk per trade")
    print("    2. Always use the stop loss")
    print("    3. Exit at close (daytrade -- no overnight holds)")
    print("    4. No trading on <3 strategy confluence")
    print("    5. Skip if VIX > 35 (crisis mode)")
    print("=" * 62)
    print()


def main():
    print()
    print("  Loading market data...")
    df = get_market_data()
    date, price, signals = scan_signals(df)
    indicators = get_indicator_snapshot(df)
    print_dashboard(date, price, signals, indicators)


if __name__ == "__main__":
    main()
