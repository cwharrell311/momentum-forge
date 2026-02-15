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
    fetch_market_context,
    fetch_stock_data,
    fetch_stock_intraday,
    discover_polymarket_markets,
    fetch_polymarket_data,
)
from src.backtesting.meta_labeling import create_meta_labeler_for_strategy, reset_meta_labeler_registry
from src.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    walk_forward_analysis,
    WalkForwardResult,
    cpcv_analysis,
    CPCVResult,
    compute_sharpe_weights,
    StrategyAllocation,
)
from src.backtesting.genetic_optimizer import GeneticConfig, optimize
from src.backtesting.ai_analyst import evaluate_strategies, format_report_for_display
from src.backtesting.strategies import (
    get_all_strategies,
    AdaptiveTrend,
    CryptoMeanReversion,
    CryptoMomentum,
    DenoisedMomentum,
    DualMomentum,
    EntropyRegimeStrategy,
    GapFade,
    HurstAdaptive,
    PredictionMomentum,
    PredictionReversion,
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
║  CPCV · Walk-Forward · Triple Barrier · Kelly Criterion      ║
║  ATR Barriers · Meta-Labeling · Deflated Sharpe              ║
║  Kalman Denoising · Hurst Adaptive · Entropy Filter          ║
║  HMM Regime · FracDiff · Cross-Asset Context (VIX/DXY)      ║
║  Stocks · Crypto · Prediction Markets · Genetic Optimizer    ║
╚══════════════════════════════════════════════════════════════╝
""")


def _preprocess_data(df: pd.DataFrame, context_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply preprocessing pipeline: context merge + signal processing + fracdiff + regime."""
    from src.backtesting.data_feeds import add_context_to_ohlcv

    result = df.copy()

    # 1. Merge cross-asset context (VIX, DXY, yields) if available
    if context_df is not None:
        try:
            result = add_context_to_ohlcv(result, context_df)
            log.info("  Context merged: VIX, DXY, yields added")
        except Exception as e:
            log.debug("  Context merge failed: %s", e)

    # 2. Add signal processing features (Kalman, Hurst, entropy)
    try:
        from src.backtesting.signal_processing import add_signal_features
        result = add_signal_features(result)
        log.info("  Signal features added: denoised, kalman, hurst, entropy")
    except Exception as e:
        log.debug("  Signal feature extraction skipped: %s", e)

    # 3. Fractional differentiation — stationary features that preserve memory
    try:
        from src.backtesting.fracdiff import add_fracdiff_features
        result = add_fracdiff_features(result)
        d_val = result.attrs.get("fracdiff_d", "?")
        log.info("  FracDiff applied: d=%.2f, close_fracdiff added", d_val)
    except Exception as e:
        log.debug("  FracDiff skipped: %s", e)

    # 4. HMM regime detection — adds regime_trending, regime_confidence columns
    try:
        from src.backtesting.regime_hmm import fit_regime_detector
        detector = fit_regime_detector(result)
        # Vectorize regime states into columns for strategy access
        trending_probs = []
        confidences = []
        for idx in range(len(result)):
            state = detector.get_regime(idx)
            trending_probs.append(state.trending_prob)
            confidences.append(state.confidence)
        result["regime_trending"] = trending_probs
        result["regime_confidence"] = confidences
        log.info("  HMM regime detection applied: trending/confidence columns added")
    except Exception as e:
        log.debug("  HMM regime detection skipped: %s", e)

    return result


def run_stock_backtest(
    symbols: list[str],
    period: str = "2y",
    interval: str = "1d",
    config: BacktestConfig | None = None,
    walk_forward: bool = False,
    wf_windows: int = 5,
    use_cpcv: bool = False,
) -> list[dict]:
    """Run backtest on stock symbols."""
    if config is None:
        config = BacktestConfig()

    results = []
    bt_results = []  # Keep BacktestResult objects for Sharpe weighting
    strategies = get_all_strategies("stock")
    total_combos = len(symbols) * len(strategies)

    # Fetch cross-asset context (VIX, DXY, yields) once for all stocks
    context_df = None
    try:
        context_df = fetch_market_context(period=period)
        print(f"  Market context loaded: VIX, DXY, 10Y/3M yields ({len(context_df)} rows)")
    except Exception as e:
        print(f"  Market context unavailable: {e}")

    # Reset meta-labeler registry for fresh run
    reset_meta_labeler_registry()

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  Fetching {symbol} ({period}, {interval})...")

        try:
            data = fetch_stock_data(symbol, period=period, interval=interval)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Preprocess: add context + signal features
        df = _preprocess_data(data.df, context_df)
        print(f"  Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

        for strategy in strategies:
            meta = strategy.meta()
            print(f"  Testing: {meta.name}...", end=" ", flush=True)

            # Get or create meta-labeler for this strategy
            ml = create_meta_labeler_for_strategy(meta.name)

            try:
                if use_cpcv:
                    cpcv = cpcv_analysis(
                        strategy, df, symbol, "stock", config,
                        num_groups=6, test_groups=2, purge_bars=5,
                    )
                    summary = {
                        "strategy": meta.name,
                        "symbol": symbol,
                        "asset_class": "stock",
                        "sharpe": f"{cpcv.mean_sharpe:.2f}",
                        "cagr": f"{cpcv.mean_cagr:+.2f}%",
                        "max_drawdown": f"{cpcv.mean_max_dd:.2f}%",
                        "prob_overfit": f"{cpcv.prob_overfit:.1%}",
                        "prob_sharpe_pos": f"{cpcv.prob_sharpe_positive:.1%}",
                        "deflated_sharpe": f"{cpcv.deflated_sharpe:.2f}",
                        "cpcv_paths": cpcv.num_paths,
                        "total_trades": "N/A",
                        "win_rate": "N/A",
                    }
                    results.append(summary)
                    print(f"CPCV mean_sharpe={cpcv.mean_sharpe:.2f} P(overfit)={cpcv.prob_overfit:.1%} paths={cpcv.num_paths}")

                elif walk_forward:
                    wf = walk_forward_analysis(
                        strategy, df, symbol, "stock", config,
                        num_windows=wf_windows,
                    )
                    report = wf.combined_oos_report
                    summary = report.summary()
                    summary["strategy"] = meta.name
                    summary["symbol"] = symbol
                    summary["asset_class"] = "stock"
                    summary["wf_efficiency"] = f"{wf.wf_efficiency:.3f}"
                    results.append(summary)

                    if report.is_viable:
                        print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}% DD={report.drawdown.max_drawdown_pct:.1f}%")
                    else:
                        print(f"✗ Sharpe={report.sharpe_ratio:.2f} ({report.trades.total_trades} trades)")

                else:
                    bt = run_backtest(strategy, df, symbol, "stock", config,
                                      num_trials=total_combos, meta_labeler=ml)
                    bt_results.append(bt)
                    report = bt.report
                    summary = report.summary()
                    summary["strategy"] = meta.name
                    summary["symbol"] = symbol
                    summary["asset_class"] = "stock"
                    results.append(summary)

                    if report.is_viable:
                        print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}% DD={report.drawdown.max_drawdown_pct:.1f}%")
                    else:
                        print(f"✗ Sharpe={report.sharpe_ratio:.2f} ({report.trades.total_trades} trades)")

            except Exception as e:
                print(f"ERROR: {e}")

    # Sharpe-weighted allocation across viable strategies
    if bt_results:
        allocations = compute_sharpe_weights(bt_results, config.initial_capital)
        if allocations:
            print(f"\n  SHARPE-WEIGHTED ALLOCATION ({len(allocations)} strategies):")
            for a in allocations[:5]:
                print(f"    {a.strategy_name:25s} {a.symbol:12s} weight={a.weight:.1%} ${a.capital_allocated:,.0f}")

    return results


def run_crypto_backtest(
    pairs: list[str],
    days: int = 365,
    timeframe: str = "1d",
    config: BacktestConfig | None = None,
    walk_forward: bool = False,
    use_cpcv: bool = False,
) -> list[dict]:
    """Run backtest on crypto pairs."""
    if config is None:
        config = BacktestConfig()

    results = []
    bt_results = []
    strategies = get_all_strategies("crypto")

    # Reset meta-labeler registry for fresh crypto run
    reset_meta_labeler_registry()

    for pair in pairs:
        print(f"\n{'─' * 50}")
        print(f"  Fetching {pair} ({days}d, {timeframe})...")

        try:
            data = fetch_crypto_data(pair, timeframe=timeframe, days_back=days)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Preprocess: add signal features (no context for crypto - VIX/DXY less relevant)
        df = _preprocess_data(data.df)
        print(f"  Got {len(df)} bars")

        for strategy in strategies:
            meta = strategy.meta()
            print(f"  Testing: {meta.name}...", end=" ", flush=True)

            # Get or create meta-labeler for this strategy
            ml = create_meta_labeler_for_strategy(meta.name)

            try:
                if use_cpcv:
                    cpcv = cpcv_analysis(
                        strategy, df, pair, "crypto", config,
                        num_groups=6, test_groups=2, purge_bars=5,
                    )
                    summary = {
                        "strategy": meta.name,
                        "symbol": pair,
                        "asset_class": "crypto",
                        "sharpe": f"{cpcv.mean_sharpe:.2f}",
                        "cagr": f"{cpcv.mean_cagr:+.2f}%",
                        "max_drawdown": f"{cpcv.mean_max_dd:.2f}%",
                        "prob_overfit": f"{cpcv.prob_overfit:.1%}",
                        "prob_sharpe_pos": f"{cpcv.prob_sharpe_positive:.1%}",
                        "deflated_sharpe": f"{cpcv.deflated_sharpe:.2f}",
                        "cpcv_paths": cpcv.num_paths,
                        "total_trades": "N/A",
                        "win_rate": "N/A",
                    }
                    results.append(summary)
                    print(f"CPCV mean_sharpe={cpcv.mean_sharpe:.2f} P(overfit)={cpcv.prob_overfit:.1%} paths={cpcv.num_paths}")

                elif walk_forward:
                    wf = walk_forward_analysis(
                        strategy, df, pair, "crypto", config,
                        num_windows=3,
                    )
                    report = wf.combined_oos_report
                    summary = report.summary()
                    summary["strategy"] = meta.name
                    summary["symbol"] = pair
                    summary["asset_class"] = "crypto"
                    summary["wf_efficiency"] = f"{wf.wf_efficiency:.3f}"
                    results.append(summary)

                    if report.is_viable:
                        print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}%")
                    else:
                        print(f"✗ Sharpe={report.sharpe_ratio:.2f}")

                else:
                    total_combos = len(pairs) * len(strategies)
                    bt = run_backtest(strategy, df, pair, "crypto", config,
                                      num_trials=total_combos, meta_labeler=ml)
                    bt_results.append(bt)
                    report = bt.report
                    summary = report.summary()
                    summary["strategy"] = meta.name
                    summary["symbol"] = pair
                    summary["asset_class"] = "crypto"
                    results.append(summary)

                    if report.is_viable:
                        print(f"✓ Sharpe={report.sharpe_ratio:.2f} CAGR={report.cagr_pct:+.1f}%")
                    else:
                        print(f"✗ Sharpe={report.sharpe_ratio:.2f}")

            except Exception as e:
                print(f"ERROR: {e}")

    # Sharpe-weighted allocation
    if bt_results:
        allocations = compute_sharpe_weights(bt_results, config.initial_capital)
        if allocations:
            print(f"\n  SHARPE-WEIGHTED ALLOCATION ({len(allocations)} strategies):")
            for a in allocations[:5]:
                print(f"    {a.strategy_name:25s} {a.symbol:12s} weight={a.weight:.1%} ${a.capital_allocated:,.0f}")

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
        "stock": [DualMomentum, GapFade, AdaptiveTrend],
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
    parser.add_argument("--cpcv", action="store_true", help="Use Combinatorial Purged Cross-Validation (strongest anti-overfit)")
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
                args.walk_forward, args.windows, args.cpcv,
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
                args.walk_forward, args.cpcv,
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
                extra = ""
                if "prob_overfit" in r:
                    extra = f" P(overfit)={r['prob_overfit']}"
                print(f"    {r['strategy']:25s} {r['symbol']:12s} Sharpe={r['sharpe']:>6s} CAGR={r.get('cagr', 'N/A'):>8s}{extra}")

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
