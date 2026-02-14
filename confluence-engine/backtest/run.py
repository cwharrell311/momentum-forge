#!/usr/bin/env python3
"""
SPY Daytrading Backtest Runner.

Fetches historical data, runs all strategies, optimizes parameters,
and produces a comprehensive performance report with equity curve charts.

Usage:
    python -m backtest.run
    python -m backtest.run --years 3 --optimize
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .data import fetch_spy_daily, fetch_vix_daily, add_indicators
from .strategies import (
    get_all_strategies, Strategy,
    GapAndGo, GapFade, OpeningRangeBreakout, BollingerMeanReversion,
    RSIReversal, VWAPBounce, MomentumCrossover, VolatilityBreakout,
    StochasticReversal, RegimeTrend, CompositeOptimal,
)
from .engine import BacktestEngine, run_walk_forward
from .metrics import (
    BacktestResult, format_results_table, format_detailed_report,
)

OUTPUT_DIR = Path(__file__).parent / "results"


def run_all_strategies(df: pd.DataFrame, engine: BacktestEngine) -> list[BacktestResult]:
    """Run all strategies with default parameters."""
    strategies = get_all_strategies()
    results = []
    for strat in strategies:
        result = engine.run(df, strat)
        results.append(result)
    return results


def optimize_top_strategies(
    df: pd.DataFrame, engine: BacktestEngine, top_n: int = 5,
    base_results: list[BacktestResult] = None,
) -> list[BacktestResult]:
    """Optimize parameters for the top performing strategies."""
    print("\n--- Parameter Optimization ---")

    param_grids = {
        "Gap & Go": (GapAndGo, {
            "min_gap_pct": [0.2, 0.3, 0.5],
            "max_gap_pct": [1.5, 2.0, 3.0],
            "volume_filter": [1.0, 1.2, 1.5],
            "stop_atr_mult": [0.5, 1.0, 1.5],
        }),
        "Gap Fade": (GapFade, {
            "min_gap_pct": [0.3, 0.5, 0.8],
            "rsi_threshold": [65, 70, 75],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
        "ORB Breakout": (OpeningRangeBreakout, {
            "volume_filter": [0.8, 1.0, 1.3],
            "stop_atr_mult": [0.5, 1.0, 1.5, 2.0],
        }),
        "BB Mean Reversion": (BollingerMeanReversion, {
            "bb_entry": [-0.1, 0.0, 0.1],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
        "RSI Reversal": (RSIReversal, {
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75, 80],
            "require_reversal": [True, False],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
        "VWAP Bounce": (VWAPBounce, {
            "threshold_pct": [0.2, 0.3, 0.5],
            "trend_ma": [20, 50],
            "stop_atr_mult": [0.5, 1.0, 1.5],
        }),
        "EMA Crossover": (MomentumCrossover, {
            "fast_ema": [5, 9, 12],
            "slow_ema": [15, 21, 26],
            "macd_confirm": [True, False],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
        "Volatility Breakout": (VolatilityBreakout, {
            "channel_period": [10, 15, 20, 30],
            "volume_filter": [1.0, 1.3, 1.5],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
        "Stochastic Reversal": (StochasticReversal, {
            "oversold": [15, 20, 25],
            "overbought": [75, 80, 85],
            "stop_atr_mult": [1.0, 1.5, 2.0],
        }),
    }

    # Determine which strategies to optimize
    if base_results:
        ranked = sorted(base_results, key=lambda r: r.sharpe_ratio, reverse=True)
        top_names = [r.strategy_name for r in ranked[:top_n]]
    else:
        top_names = list(param_grids.keys())[:top_n]

    all_optimized = []
    for name in top_names:
        if name.startswith("Composite"):
            continue
        if name not in param_grids:
            continue

        strat_class, grid = param_grids[name]
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  Optimizing {name}: {n_combos} parameter combinations...")

        sweep_results = engine.run_parameter_sweep(df, strat_class, grid)
        if sweep_results:
            best = sweep_results[0]
            best.strategy_name = f"{name} (Optimized)"
            all_optimized.append(best)
            print(f"    Best: Sharpe={best.sharpe_ratio:.2f}, "
                  f"Win%={best.win_rate_pct:.1f}, Params={best.params}")

    return all_optimized


def plot_equity_curves(
    results: list[BacktestResult],
    title: str = "SPY Daytrading Strategy Comparison",
    filename: str = "equity_curves.png",
) -> str:
    """Plot equity curves for all strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax2 = axes[1]

    # Sort by Sharpe for legend ordering
    sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(sorted_results), 10)))

    for idx, result in enumerate(sorted_results[:10]):
        if len(result.equity_curve) < 2:
            continue
        color = colors[idx % len(colors)]
        label = f"{result.strategy_name} (Sharpe={result.sharpe_ratio:.2f})"
        ax1.plot(result.equity_curve.index, result.equity_curve.values,
                 label=label, color=color, linewidth=1.5)

        # Drawdown
        cummax = result.equity_curve.cummax()
        drawdown = (result.equity_curve - cummax) / cummax * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         alpha=0.3, color=color, label=result.strategy_name)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return str(filepath)


def plot_monthly_returns(result: BacktestResult, filename: str = "monthly_returns.png") -> str:
    """Plot monthly returns heatmap for the best strategy."""
    equity = result.equity_curve
    if len(equity) < 30:
        return ""

    monthly = equity.resample("ME").last().pct_change() * 100
    monthly = monthly.dropna()

    # Pivot into year x month
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="sum")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-5, vmax=5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 3 else "white")

    ax.set_title(f"Monthly Returns (%) — {result.strategy_name}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Return %")
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return str(filepath)


def plot_trade_distribution(result: BacktestResult, filename: str = "trade_dist.png") -> str:
    """Plot trade P&L distribution."""
    if not result.trades:
        return ""

    pnls = [t.pnl_pct for t in result.trades]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(pnls, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax1.axvline(x=np.mean(pnls), color="green", linestyle="--", linewidth=1,
                label=f"Mean: {np.mean(pnls):.3f}%")
    ax1.set_title(f"Trade P&L Distribution — {result.strategy_name}")
    ax1.set_xlabel("P&L (%)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Cumulative P&L
    ax2 = axes[1]
    cum_pnl = np.cumsum(pnls)
    ax2.plot(cum_pnl, color="steelblue", linewidth=1.5)
    ax2.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=np.array(cum_pnl) >= 0, color="green", alpha=0.2)
    ax2.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=np.array(cum_pnl) < 0, color="red", alpha=0.2)
    ax2.set_title("Cumulative P&L (%)")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Cumulative P&L (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return str(filepath)


def print_buy_and_hold_benchmark(df: pd.DataFrame) -> BacktestResult:
    """Calculate buy-and-hold SPY benchmark."""
    equity = df["Close"] / df["Close"].iloc[0] * 100_000
    result = BacktestResult(
        strategy_name="SPY Buy & Hold",
        params={"strategy": "passive"},
        trades=[],
        equity_curve=equity,
    )
    # Manual metrics for B&H
    n_days = (equity.index[-1] - equity.index[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 1
    result.total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    result.cagr_pct = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100

    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() > 0:
        result.sharpe_ratio = float(
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        )
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    result.max_drawdown_pct = float(drawdown.min()) * 100
    result.exposure_pct = 100.0
    result.win_rate_pct = 0
    result.profit_factor = 0
    result.total_trades = 0

    if result.max_drawdown_pct != 0:
        result.calmar_ratio = result.cagr_pct / abs(result.max_drawdown_pct)

    return result


def _result_to_dict(r: BacktestResult) -> dict:
    """Convert a BacktestResult to a JSON-serializable dict."""
    return {
        "strategy": r.strategy_name,
        "params": r.params,
        "trades": r.total_trades,
        "win_rate_pct": round(r.win_rate_pct, 2),
        "cagr_pct": round(r.cagr_pct, 2),
        "total_return_pct": round(r.total_return_pct, 2),
        "sharpe": round(r.sharpe_ratio, 3),
        "sortino": round(r.sortino_ratio, 3),
        "calmar": round(r.calmar_ratio, 3),
        "max_drawdown_pct": round(r.max_drawdown_pct, 2),
        "profit_factor": round(r.profit_factor, 3),
        "expectancy_pct": round(r.expectancy_pct, 4),
        "exposure_pct": round(r.exposure_pct, 2),
        "annual_trades": round(r.annual_trades, 1),
        "avg_win_pct": round(r.avg_win_pct, 4),
        "avg_loss_pct": round(r.avg_loss_pct, 4),
        "best_trade_pct": round(r.best_trade_pct, 4),
        "worst_trade_pct": round(r.worst_trade_pct, 4),
        "max_consecutive_wins": r.max_consecutive_wins,
        "max_consecutive_losses": r.max_consecutive_losses,
    }


def save_results_json(
    results: list[BacktestResult],
    bh_result: BacktestResult,
    best: BacktestResult,
    optimized: list[BacktestResult] = None,
) -> str:
    """Save all backtest results to a JSON file for machine-readable analysis."""
    active = [r for r in results if r.total_trades > 0]
    ranked = sorted(active, key=lambda r: r.sharpe_ratio, reverse=True)

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ranking": [_result_to_dict(r) for r in ranked],
        "benchmark": _result_to_dict(bh_result),
        "best_by_category": {
            "sharpe": _result_to_dict(max(active, key=lambda r: r.sharpe_ratio)),
            "win_rate": _result_to_dict(max(active, key=lambda r: r.win_rate_pct)),
            "total_return": _result_to_dict(max(active, key=lambda r: r.total_return_pct)),
            "calmar": _result_to_dict(max(active, key=lambda r: r.calmar_ratio)),
        },
        "recommendation": {
            **_result_to_dict(best),
            "alpha_vs_bh_cagr": round(best.cagr_pct - bh_result.cagr_pct, 2),
        },
    }

    if optimized:
        data["optimized"] = [_result_to_dict(r) for r in optimized if r.total_trades > 0]

    filepath = OUTPUT_DIR / "results.json"
    filepath.write_text(json.dumps(data, indent=2))
    return str(filepath)


def print_results_summary() -> str:
    """Print a concise summary of the last saved backtest results."""
    filepath = OUTPUT_DIR / "results.json"
    if not filepath.exists():
        return "No results found. Run a backtest first: ./go.sh backtest"

    data = json.loads(filepath.read_text())

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  BACKTEST RESULTS SUMMARY")
    lines.append(f"  Generated: {data['generated_at']}")
    lines.append("=" * 60)

    # Ranking table
    lines.append("")
    lines.append("  STRATEGY RANKING (by Sharpe ratio)")
    lines.append(f"  {'#':<3} {'Strategy':<28} {'Sharpe':>7} {'CAGR%':>7} {'Win%':>6} {'MaxDD%':>7} {'PF':>6}")
    lines.append("  " + "-" * 64)
    for i, r in enumerate(data["ranking"], 1):
        lines.append(
            f"  {i:<3} {r['strategy']:<28} {r['sharpe']:>7.2f} {r['cagr_pct']:>7.1f} "
            f"{r['win_rate_pct']:>5.1f} {r['max_drawdown_pct']:>7.1f} {r['profit_factor']:>6.2f}"
        )

    # Benchmark
    bh = data["benchmark"]
    lines.append("")
    lines.append(f"  BENCHMARK: SPY Buy & Hold  Sharpe={bh['sharpe']:.2f}  CAGR={bh['cagr_pct']:.1f}%  MaxDD={bh['max_drawdown_pct']:.1f}%")

    # Recommendation
    rec = data["recommendation"]
    lines.append("")
    lines.append("  RECOMMENDED STRATEGY")
    lines.append(f"    {rec['strategy']}")
    lines.append(f"    Sharpe={rec['sharpe']:.2f}  CAGR={rec['cagr_pct']:.1f}%  Win%={rec['win_rate_pct']:.1f}  Alpha={rec['alpha_vs_bh_cagr']:+.1f}%")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="SPY Daytrading Backtest")
    parser.add_argument("--years", type=int, default=5, help="Years of data")
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data download")
    parser.add_argument("--results", action="store_true", help="Show last backtest results summary")
    args = parser.parse_args()

    if args.results:
        print_results_summary()
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Fetch Data ---
    print(f"Fetching {args.years} years of SPY daily data...")
    spy_df = fetch_spy_daily(years=args.years, use_cache=not args.no_cache)
    print(f"  Got {len(spy_df)} bars from {spy_df.index[0].date()} to {spy_df.index[-1].date()}")

    print("Fetching VIX data for regime filtering...")
    vix_df = fetch_vix_daily(years=args.years, use_cache=not args.no_cache)
    print(f"  Got {len(vix_df)} VIX bars")

    # Merge VIX into SPY data
    spy_df = spy_df.join(vix_df, how="left")
    spy_df["VIX_Close"] = spy_df["VIX_Close"].ffill()

    # Add indicators
    print("Computing technical indicators...")
    df = add_indicators(spy_df)
    df.dropna(subset=["SMA_50", "ATR", "RSI", "BB_pct"], inplace=True)
    print(f"  {len(df)} bars after indicator warmup")

    # --- Run Strategies ---
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_per_trade=1.0,
        slippage_pct=0.02,  # 2 bps slippage
        position_size_pct=100.0,
    )

    print("\n" + "=" * 60)
    print("  RUNNING ALL STRATEGIES")
    print("=" * 60)

    results = run_all_strategies(df, engine)

    # Add buy-and-hold benchmark
    bh_result = print_buy_and_hold_benchmark(df)
    results.append(bh_result)

    # Print comparison table
    print("\n--- Strategy Comparison (sorted by Sharpe) ---")
    print(format_results_table(results))

    # --- Optimize if requested ---
    optimized = []
    if args.optimize:
        optimized = optimize_top_strategies(df, engine, top_n=5, base_results=results)
        if optimized:
            print("\n--- Optimized Strategy Results ---")
            print(format_results_table(optimized))
            results.extend(optimized)

    # --- Find best strategy ---
    active_results = [r for r in results if r.total_trades > 0]
    if not active_results:
        print("\nNo strategies generated trades. Check data or parameters.")
        return

    best = max(active_results, key=lambda r: r.sharpe_ratio)
    print(format_detailed_report(best))

    # Also show the best by other metrics
    best_winrate = max(active_results, key=lambda r: r.win_rate_pct)
    best_return = max(active_results, key=lambda r: r.total_return_pct)
    best_calmar = max(active_results, key=lambda r: r.calmar_ratio)

    print("\n--- Best By Category ---")
    print(f"  Best Sharpe:     {best.strategy_name} ({best.sharpe_ratio:.2f})")
    print(f"  Best Win Rate:   {best_winrate.strategy_name} ({best_winrate.win_rate_pct:.1f}%)")
    print(f"  Best Return:     {best_return.strategy_name} ({best_return.total_return_pct:.1f}%)")
    print(f"  Best Calmar:     {best_calmar.strategy_name} ({best_calmar.calmar_ratio:.2f})")

    # --- Generate Charts ---
    print("\n--- Generating Charts ---")

    chart_path = plot_equity_curves(results)
    print(f"  Equity curves: {chart_path}")

    monthly_path = plot_monthly_returns(best)
    if monthly_path:
        print(f"  Monthly returns: {monthly_path}")

    trade_path = plot_trade_distribution(best)
    if trade_path:
        print(f"  Trade distribution: {trade_path}")

    # If we have optimized results, plot those separately
    if optimized:
        optimized.append(bh_result)
        opt_chart = plot_equity_curves(
            optimized,
            title="Optimized Strategies vs Buy & Hold",
            filename="optimized_equity.png",
        )
        print(f"  Optimized equity: {opt_chart}")

    # --- Walk-Forward Validation on Best ---
    print(f"\n--- Walk-Forward Analysis: {best.strategy_name} ---")
    wf_results = run_walk_forward(df, get_all_strategies()[0], engine, n_splits=5)
    if wf_results:
        print(format_results_table(wf_results))

    # --- Save machine-readable results ---
    json_path = save_results_json(results, bh_result, best, optimized or None)
    print(f"  Results JSON: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  OPTIMAL STRATEGY RECOMMENDATION")
    print("=" * 60)
    print(f"\n  Strategy:       {best.strategy_name}")
    print(f"  Parameters:     {best.params}")
    print(f"  Sharpe Ratio:   {best.sharpe_ratio:.2f}")
    print(f"  CAGR:           {best.cagr_pct:.1f}%")
    print(f"  Max Drawdown:   {best.max_drawdown_pct:.1f}%")
    print(f"  Win Rate:       {best.win_rate_pct:.1f}%")
    print(f"  Profit Factor:  {best.profit_factor:.2f}")
    print(f"  Total Trades:   {best.total_trades}")
    print(f"  Annual Trades:  {best.annual_trades:.0f}")
    print(f"  Expectancy:     {best.expectancy_pct:.3f}% per trade")
    print(f"\n  vs SPY Buy & Hold:")
    print(f"    B&H CAGR:     {bh_result.cagr_pct:.1f}%")
    print(f"    B&H Sharpe:   {bh_result.sharpe_ratio:.2f}")
    print(f"    B&H MaxDD:    {bh_result.max_drawdown_pct:.1f}%")

    alpha = best.cagr_pct - bh_result.cagr_pct
    print(f"\n  Alpha vs B&H:   {alpha:+.1f}% CAGR")
    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
