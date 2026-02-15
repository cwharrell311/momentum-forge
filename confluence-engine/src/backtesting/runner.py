"""
Backtest Runner — CLI entry point for the Momentum Forge backtesting engine.

Usage:
    # Quick backtest on stocks
    python -m src.backtesting.runner --stocks SPY QQQ AAPL --period 2y

    # Crypto backtest
    python -m src.backtesting.runner --crypto BTC/USDT ETH/USDT --days 365

    # Full multi-asset backtest with genetic optimization
    python -m src.backtesting.runner --stocks SPY QQQ --crypto BTC/USDT --optimize --ai-eval

    # Walk-forward analysis only
    python -m src.backtesting.runner --stocks SPY --walk-forward --windows 5

    # Polymarket
    python -m src.backtesting.runner --polymarket-search "bitcoin" --poly-days 60

All results are printed to stdout and optionally saved to JSON.
No database or server required — pure standalone backtesting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data_feeds import (
    AssetClass,
    fetch_crypto_data,
    fetch_economic_calendar,
    fetch_stock_data,
    fetch_stock_intraday,
    discover_polymarket_markets,
    fetch_polymarket_data,
)
from src.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    walk_forward_analysis,
    WalkForwardResult,
)
from src.backtesting.genetic_optimizer import GeneticConfig, optimize
from src.backtesting.ai_analyst import evaluate_strategies, format_report_for_display
from src.backtesting.strategies import (
    get_all_strategies,
    AdaptiveTrend,
    CryptoMeanReversion,
    CryptoMomentum,
    DualMomentum,
    GapFade,
    OpeningRangeBreakout,
    PredictionMomentum,
    PredictionReversion,
    VWAPReversion,
)
from src.backtesting.risk_management import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("forge.runner")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MOMENTUM FORGE — BACKTESTING ENGINE             ║
║         Research-Grade Multi-Asset Strategy Testing           ║
║                                                              ║
║  Walk-Forward Analysis · Kelly Criterion · Genetic Optimizer ║
║  Stocks · Crypto · Prediction Markets · AI Evaluation        ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_stock_backtest(
    symbols: list[str],
    period: str = "2y",
    interval: str = "1d",
    config: BacktestConfig | None = None,
    walk_forward: bool = False,
    wf_windows: int = 5,
) -> list[dict]:
    """Run backtest on stock symbols."""
    if config is None:
        config = BacktestConfig()

    results = []
    strategies = get_all_strategies("stock")

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  Fetching {symbol} ({period}, {interval})...")

        try:
            data = fetch_stock_data(symbol, period=period, interval=interval)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Got {len(data.df)} bars from {data.df.index[0].date()} to {data.df.index[-1].date()}")

        for strategy in strategies:
            meta = strategy.meta()
            print(f"  Testing: {meta.name}...", end=" ", flush=True)

            try:
                if walk_forward:
                    wf = walk_forward_analysis(
                        strategy, data.df, symbol, "stock", config,
                        num_windows=wf_windows,
                    )
                    report = wf.combined_oos_report
                    extra = {"wf_efficiency": f"{wf.wf_efficiency:.3f}"}
                else:
                    bt = run_backtest(strategy, data.df, symbol, "stock", config)
                    report = bt.report
                    extra = {}

                summary = report.summary()
                summary["strategy"] = meta.name
                summary["symbol"] = symbol
                summary["asset_class"] = "stock"
                summary.update(extra)
                results.append(summary)

                if report.is_viable:
                    print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}% DD={report.drawdown.max_drawdown_pct:.1f}%")
                else:
                    print(f"✗ Sharpe={report.sharpe_ratio:.2f} ({report.trades.total_trades} trades)")

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def run_crypto_backtest(
    pairs: list[str],
    days: int = 365,
    timeframe: str = "1d",
    config: BacktestConfig | None = None,
    walk_forward: bool = False,
) -> list[dict]:
    """Run backtest on crypto pairs."""
    if config is None:
        config = BacktestConfig()

    results = []
    strategies = get_all_strategies("crypto")

    for pair in pairs:
        print(f"\n{'─' * 50}")
        print(f"  Fetching {pair} ({days}d, {timeframe})...")

        try:
            data = fetch_crypto_data(pair, timeframe=timeframe, days_back=days)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Got {len(data.df)} bars")

        for strategy in strategies:
            meta = strategy.meta()
            print(f"  Testing: {meta.name}...", end=" ", flush=True)

            try:
                if walk_forward:
                    wf = walk_forward_analysis(
                        strategy, data.df, pair, "crypto", config,
                        num_windows=3,
                    )
                    report = wf.combined_oos_report
                    extra = {"wf_efficiency": f"{wf.wf_efficiency:.3f}"}
                else:
                    bt = run_backtest(strategy, data.df, pair, "crypto", config)
                    report = bt.report
                    extra = {}

                summary = report.summary()
                summary["strategy"] = meta.name
                summary["symbol"] = pair
                summary["asset_class"] = "crypto"
                summary.update(extra)
                results.append(summary)

                if report.is_viable:
                    print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}%")
                else:
                    print(f"✗ Sharpe={report.sharpe_ratio:.2f}")

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def run_polymarket_backtest(
    slugs: list[str] | None = None,
    search: str | None = None,
    days: int = 60,
    config: BacktestConfig | None = None,
) -> list[dict]:
    """Run backtest on Polymarket prediction markets."""
    if config is None:
        config = BacktestConfig()

    results = []
    strategies = get_all_strategies("polymarket")

    # Discover markets if search provided
    if search and not slugs:
        print(f"\n  Searching Polymarket for: {search}")
        markets = discover_polymarket_markets(query=search, limit=5)
        if not markets:
            print("  No markets found.")
            return results
        for m in markets:
            print(f"  Found: {m['question'][:60]}... (slug={m['slug']})")
        slugs = [m["slug"] for m in markets if m["slug"]]

    for slug in (slugs or []):
        print(f"\n{'─' * 50}")
        print(f"  Fetching Polymarket: {slug}...")

        try:
            data = fetch_polymarket_data(slug=slug, days_back=days)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Got {len(data.df)} data points")

        for strategy in strategies:
            meta = strategy.meta()
            print(f"  Testing: {meta.name}...", end=" ", flush=True)

            try:
                bt = run_backtest(strategy, data.df, slug[:16], "polymarket", config)
                report = bt.report
                summary = report.summary()
                summary["strategy"] = meta.name
                summary["symbol"] = slug[:20]
                summary["asset_class"] = "polymarket"
                results.append(summary)

                print(f"Sharpe={report.sharpe_ratio:.2f} trades={report.trades.total_trades}")
            except Exception as e:
                print(f"ERROR: {e}")

    return results


def run_genetic_optimization(
    symbol: str,
    asset_class: str = "stock",
    period: str = "2y",
    days: int = 365,
) -> list[dict]:
    """Run genetic optimization across strategies."""
    results = []

    print(f"\n{'═' * 60}")
    print(f"  GENETIC OPTIMIZATION: {symbol}")
    print(f"{'═' * 60}")

    # Fetch data
    if asset_class == "stock":
        data = fetch_stock_data(symbol, period=period)
    elif asset_class == "crypto":
        data = fetch_crypto_data(symbol, days_back=days)
    else:
        print("Genetic optimization not supported for polymarket yet")
        return results

    # Optimize each strategy class
    strategy_classes = {
        "stock": [VWAPReversion, DualMomentum, GapFade, AdaptiveTrend],
        "crypto": [CryptoMomentum, CryptoMeanReversion, AdaptiveTrend],
    }

    classes = strategy_classes.get(asset_class, [AdaptiveTrend])
    genetic_config = GeneticConfig(
        population_size=30,
        num_generations=20,
        early_stop_generations=6,
    )

    for cls in classes:
        template = cls()
        meta = template.meta()
        print(f"\n  Optimizing: {meta.name} ({meta.param_count} params)...")

        try:
            opt = optimize(
                cls, data.df, symbol, asset_class,
                genetic_config=genetic_config,
                validate_winner=True,
            )

            print(f"  Best params: {opt.best_params}")
            print(f"  Best fitness: {opt.best_individual.fitness:.4f}")
            print(f"  Sharpe: {opt.best_individual.sharpe:.2f}")
            print(f"  CAGR: {opt.best_individual.cagr:.1f}%")
            print(f"  Max DD: {opt.best_individual.max_dd:.1f}%")
            if opt.wf_efficiency is not None:
                print(f"  WF Efficiency: {opt.wf_efficiency:.3f}")
            print(f"  Backtests run: {opt.total_backtests}")

            results.append({
                "strategy": meta.name,
                "symbol": symbol,
                "asset_class": asset_class,
                "best_params": opt.best_params,
                "fitness": opt.best_individual.fitness,
                "sharpe": opt.best_individual.sharpe,
                "cagr": opt.best_individual.cagr,
                "max_dd": opt.best_individual.max_dd,
                "trades": opt.best_individual.trades,
                "wf_efficiency": opt.wf_efficiency,
                "generations": opt.generations_run,
                "total_backtests": opt.total_backtests,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="Momentum Forge Backtesting Engine")

    # Asset selection
    parser.add_argument("--stocks", nargs="+", default=[], help="Stock symbols (e.g., SPY QQQ AAPL)")
    parser.add_argument("--crypto", nargs="+", default=[], help="Crypto pairs (e.g., BTC/USDT ETH/USDT)")
    parser.add_argument("--polymarket-slugs", nargs="+", default=[], help="Polymarket slugs")
    parser.add_argument("--polymarket-search", type=str, default="", help="Search Polymarket")

    # Data params
    parser.add_argument("--period", type=str, default="2y", help="Stock data period (1mo, 3mo, 1y, 2y, 5y)")
    parser.add_argument("--crypto-days", type=int, default=365, help="Days of crypto history")
    parser.add_argument("--poly-days", type=int, default=60, help="Days of Polymarket history")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval (1d, 1h, 5m)")

    # Analysis modes
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward analysis")
    parser.add_argument("--windows", type=int, default=5, help="Walk-forward windows")
    parser.add_argument("--optimize", action="store_true", help="Run genetic optimization")
    parser.add_argument("--ai-eval", action="store_true", help="Use AI to evaluate results")
    parser.add_argument("--ai-provider", type=str, default="claude", choices=["claude", "openai"])

    # Risk params
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--max-dd", type=float, default=15.0, help="Max drawdown % halt")
    parser.add_argument("--kelly", type=float, default=0.5, help="Kelly fraction (0.5 = half-Kelly)")

    # Output
    parser.add_argument("--output", type=str, default="", help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    # Config
    bt_config = BacktestConfig(
        initial_capital=args.capital,
        kelly_fraction=args.kelly,
        use_kelly=True,
    )

    all_results = []
    start_time = time.time()

    # Economic calendar check
    print("  Checking economic calendar...")
    try:
        cal = fetch_economic_calendar(days_ahead=7)
        if len(cal) > 0:
            print(f"  ⚠ {len(cal)} high-impact events in next 7 days:")
            for _, event in cal.iterrows():
                print(f"    {event['date']} {event.get('time', '')} — {event['event']}")
        else:
            print("  No high-impact events in next 7 days")
    except Exception:
        print("  Could not fetch economic calendar")

    # Run backtests
    if args.stocks:
        print(f"\n{'═' * 60}")
        print(f"  STOCKS: {', '.join(args.stocks)}")
        print(f"{'═' * 60}")
        if args.optimize:
            for sym in args.stocks:
                all_results.extend(run_genetic_optimization(sym, "stock", args.period))
        else:
            all_results.extend(run_stock_backtest(
                args.stocks, args.period, args.interval, bt_config,
                args.walk_forward, args.windows,
            ))

    if args.crypto:
        print(f"\n{'═' * 60}")
        print(f"  CRYPTO: {', '.join(args.crypto)}")
        print(f"{'═' * 60}")
        if args.optimize:
            for pair in args.crypto:
                all_results.extend(run_genetic_optimization(pair, "crypto", days=args.crypto_days))
        else:
            all_results.extend(run_crypto_backtest(
                args.crypto, args.crypto_days, args.interval, bt_config,
                args.walk_forward,
            ))

    if args.polymarket_slugs or args.polymarket_search:
        print(f"\n{'═' * 60}")
        print(f"  POLYMARKET")
        print(f"{'═' * 60}")
        all_results.extend(run_polymarket_backtest(
            args.polymarket_slugs or None,
            args.polymarket_search or None,
            args.poly_days,
            bt_config,
        ))

    # Print results
    elapsed = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"  COMPLETED in {elapsed:.1f}s — {len(all_results)} strategy-asset combos tested")
    print(f"{'═' * 60}")

    if all_results:
        print(format_report_for_display(all_results))

        # Sort by Sharpe
        viable = [r for r in all_results if float(str(r.get("sharpe", "0")).replace("+", "")) > 0.5]
        if viable:
            print(f"\n  TOP STRATEGIES (Sharpe > 0.5):")
            for r in sorted(viable, key=lambda x: float(str(x.get("sharpe", "0")).replace("+", "")), reverse=True)[:5]:
                print(f"    {r['strategy']:25s} {r['symbol']:12s} Sharpe={r['sharpe']:>6s} CAGR={r.get('cagr', 'N/A'):>8s}")

    # AI evaluation
    if args.ai_eval and all_results:
        print(f"\n{'═' * 60}")
        print(f"  AI EVALUATION ({args.ai_provider})")
        print(f"{'═' * 60}")
        try:
            analysis = asyncio.run(evaluate_strategies(all_results, args.ai_provider))
            if "error" not in analysis.parsed:
                print(json.dumps(analysis.parsed, indent=2))
            else:
                print(f"  AI error: {analysis.parsed.get('error', analysis.raw_response[:200])}")
        except Exception as e:
            print(f"  AI evaluation failed: {e}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "capital": args.capital,
                    "kelly": args.kelly,
                    "walk_forward": args.walk_forward,
                    "optimize": args.optimize,
                },
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    if not args.stocks and not args.crypto and not args.polymarket_slugs and not args.polymarket_search:
        print("  No assets specified. Examples:")
        print("    python -m src.backtesting.runner --stocks SPY QQQ AAPL")
        print("    python -m src.backtesting.runner --crypto BTC/USDT ETH/USDT")
        print("    python -m src.backtesting.runner --stocks SPY --crypto BTC/USDT --optimize --ai-eval")
        print("    python -m src.backtesting.runner --polymarket-search 'bitcoin'")


if __name__ == "__main__":
    main()
