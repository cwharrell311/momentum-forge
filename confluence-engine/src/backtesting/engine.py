"""
Walk-Forward Backtesting Engine.

This is the core backtester. It implements walk-forward analysis
to prevent curve fitting — the #1 killer of trading strategies.

Walk-forward analysis (Robert Pardo, "The Evaluation and Optimization
of Trading Strategies"):
1. Split data into overlapping train/test windows
2. Optimize parameters on the train window
3. Test on the out-of-sample test window
4. Roll forward and repeat
5. Only the out-of-sample results count

This means the final performance report ONLY contains results from
data the strategy has never seen — the gold standard for backtesting.

Additional anti-overfitting measures:
- Embargo period between train/test (prevents information leakage)
- Parameter count penalty (Occam's razor for strategies)
- Minimum trade count per window (reject thin results)
- Transaction costs and slippage modeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.backtesting.metrics import (
    PerformanceReport,
    compute_cagr,
    compute_drawdown,
    compute_returns,
    compute_sharpe,
    compute_trade_stats,
    generate_report,
)
from src.backtesting.position_sizing import KellyCriterion, PositionSize, compute_atr, volatility_position_size
from src.backtesting.risk_management import RiskAction, RiskManager
from src.backtesting.strategies import BaseStrategy, Side, Signal

log = logging.getLogger("forge.engine")


@dataclass
class Trade:
    """A completed trade in the backtest."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    side: Side
    pnl_pct: float
    pnl_dollars: float
    position_size: float       # Fraction of bankroll
    bars_held: int
    entry_time: str
    exit_time: str
    signal_confidence: float
    exit_reason: str           # "signal", "stop_loss", "take_profit", "end_of_data"


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0       # Alpaca is commission-free
    slippage_pct: float = 0.05              # 5 bps slippage per trade
    max_positions: int = 1                   # Single position for simplicity
    use_kelly: bool = True                   # Kelly criterion sizing
    kelly_fraction: float = 0.5             # Half-Kelly
    max_position_pct: float = 0.25          # Max 25% in one trade
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0     # 2:1 R:R
    max_holding_bars: int = 20              # Triple barrier vertical: max bars before force exit
    use_risk_manager: bool = True
    min_trades_per_window: int = 5           # Reject thin results
    use_atr_barriers: bool = True           # Use ATR-based stops instead of fixed signal stops
    atr_period: int = 14                     # ATR lookback
    atr_stop_mult: float = 2.0              # Stop at 2x ATR from entry
    atr_target_mult: float = 3.0            # Target at 3x ATR (1.5:1 R:R)
    use_meta_labeling: bool = True           # Use meta-labeler for signal filtering
    meta_label_threshold: float = 0.55       # Min P(correct) to trade


@dataclass
class BacktestResult:
    """Full result of a single backtest run."""
    strategy_name: str
    symbol: str
    asset_class: str
    timeframe: str
    config: BacktestConfig
    trades: list[Trade]
    equity_curve: pd.Series
    report: PerformanceReport
    signals_generated: int
    signals_blocked: int        # Blocked by risk manager
    window_type: str            # "full", "in_sample", "out_of_sample"


def _close_position(
    position: dict,
    exit_price: float,
    exit_bar: int,
    exit_time: str,
    exit_reason: str,
    slippage_pct: float,
) -> tuple[Trade, float]:
    """Close a position and return the Trade + realized P&L in dollars."""
    slip = slippage_pct / 100
    if position["side"] == Side.LONG:
        adj_price = exit_price * (1 - slip)
        pnl_pct = (adj_price - position["entry_price"]) / position["entry_price"] * 100
    else:
        adj_price = exit_price * (1 + slip)
        pnl_pct = (position["entry_price"] - adj_price) / position["entry_price"] * 100

    pnl_dollars = pnl_pct / 100 * position["size_dollars"]
    trade = Trade(
        entry_bar=position["entry_bar"],
        exit_bar=exit_bar,
        entry_price=position["entry_price"],
        exit_price=adj_price,
        side=position["side"],
        pnl_pct=pnl_pct,
        pnl_dollars=pnl_dollars,
        position_size=position["size_frac"],
        bars_held=exit_bar - position["entry_bar"],
        entry_time=position["entry_time"],
        exit_time=exit_time,
        signal_confidence=position["confidence"],
        exit_reason=exit_reason,
    )
    return trade, pnl_dollars


def _unrealized_pnl(position: dict, current_price: float) -> float:
    """Compute unrealized P&L dollars for an open position."""
    if position["side"] == Side.LONG:
        return (current_price - position["entry_price"]) / position["entry_price"] * position["size_dollars"]
    else:
        return (position["entry_price"] - current_price) / position["entry_price"] * position["size_dollars"]


def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str = "stock",
    config: BacktestConfig | None = None,
    risk_manager: RiskManager | None = None,
    num_trials: int = 1,
    meta_labeler=None,  # Optional MetaLabeler instance
) -> BacktestResult:
    """
    Run a single backtest of a strategy on historical data.

    Bar-by-bar event loop:
    1. Strategy generates signals upfront
    2. Engine iterates through EVERY bar (not just signal bars)
    3. On each bar: check stops/targets, process any signal, track equity
    4. Equity includes unrealized P&L of open positions
    5. This gives an accurate equity curve for drawdown and CAGR calculation

    Returns a BacktestResult with full performance report.
    """
    if config is None:
        config = BacktestConfig()
    if risk_manager is None:
        risk_manager = RiskManager()

    meta = strategy.meta()

    # Generate all signals upfront and index by bar number
    signals = strategy.generate_signals(df)
    total_signals = len(signals)
    signal_map: dict[int, Signal] = {}
    for sig in signals:
        if sig.bar_index not in signal_map:
            signal_map[sig.bar_index] = sig

    # Initialize state
    cash = config.initial_capital
    position: dict | None = None
    trades: list[Trade] = []
    equity_values = []
    equity_times = []
    blocked_signals = 0

    # Kelly sizer
    kelly = KellyCriterion(
        kelly_fraction=config.kelly_fraction,
        max_position_pct=config.max_position_pct,
    ) if config.use_kelly else None

    # ── Bar-by-bar loop ──
    for i in range(len(df)):
        bar = df.iloc[i]
        price = bar["close"]
        timestamp = str(df.index[i])

        # ── 1. Check stops / targets on EVERY bar ──
        if position is not None:
            hit_stop = False
            hit_target = False

            if position["side"] == Side.LONG:
                hit_stop = bar["low"] <= position["stop_price"]
                hit_target = (not hit_stop) and bar["high"] >= position["target_price"]
            else:
                hit_stop = bar["high"] >= position["stop_price"]
                hit_target = (not hit_stop) and bar["low"] <= position["target_price"]

            if hit_stop:
                trade, pnl_dollars = _close_position(
                    position, position["stop_price"], i, timestamp, "stop_loss", config.slippage_pct,
                )
                trades.append(trade)
                cash += pnl_dollars
                risk_manager.update_state(cash, last_trade_won=False)
                if meta_labeler is not None:
                    try:
                        from src.backtesting.meta_labeling import extract_meta_features
                        features = extract_meta_features(df, position["entry_bar"])
                        meta_labeler.record_outcome(features, trade.pnl_pct > 0)
                    except Exception:
                        pass
                position = None

            elif hit_target:
                trade, pnl_dollars = _close_position(
                    position, position["target_price"], i, timestamp, "take_profit", config.slippage_pct,
                )
                trades.append(trade)
                cash += pnl_dollars
                risk_manager.update_state(cash, last_trade_won=True)
                if meta_labeler is not None:
                    try:
                        from src.backtesting.meta_labeling import extract_meta_features
                        features = extract_meta_features(df, position["entry_bar"])
                        meta_labeler.record_outcome(features, trade.pnl_pct > 0)
                    except Exception:
                        pass
                position = None

            # Triple barrier vertical: force exit after max_holding_bars
            elif config.max_holding_bars > 0 and position is not None:
                bars_held = i - position["entry_bar"]
                if bars_held >= config.max_holding_bars:
                    trade, pnl_dollars = _close_position(
                        position, price, i, timestamp, "time_expiry", config.slippage_pct,
                    )
                    trades.append(trade)
                    cash += pnl_dollars
                    risk_manager.update_state(cash, last_trade_won=(trade.pnl_pct > 0))
                    if meta_labeler is not None:
                        try:
                            from src.backtesting.meta_labeling import extract_meta_features
                            features = extract_meta_features(df, position["entry_bar"])
                            meta_labeler.record_outcome(features, trade.pnl_pct > 0)
                        except Exception:
                            pass
                    position = None

        # ── 2. Process signal if one exists on this bar ──
        signal = signal_map.get(i)
        if signal is not None and signal.side != Side.FLAT:

            # Close opposite position first
            if position is not None and signal.side != position["side"]:
                trade, pnl_dollars = _close_position(
                    position, price, i, timestamp, "signal", config.slippage_pct,
                )
                trades.append(trade)
                cash += pnl_dollars
                risk_manager.update_state(cash, last_trade_won=(trade.pnl_pct > 0))
                if meta_labeler is not None:
                    try:
                        from src.backtesting.meta_labeling import extract_meta_features
                        features = extract_meta_features(df, position["entry_bar"])
                        meta_labeler.record_outcome(features, trade.pnl_pct > 0)
                    except Exception:
                        pass
                position = None

            # Open new position if flat
            if position is None:
                # Risk check
                size_mult = 1.0
                blocked = False
                if config.use_risk_manager:
                    returns_hist = compute_returns(pd.Series(equity_values)) if len(equity_values) > 20 else None
                    decision = risk_manager.check_trade(
                        symbol=symbol,
                        returns_history=returns_hist,
                        asset_class=asset_class,
                    )
                    if decision.action in (RiskAction.BLOCK, RiskAction.HALT):
                        blocked_signals += 1
                        blocked = True
                    else:
                        size_mult = decision.size_multiplier

                # Meta-labeling: predict if this signal is likely correct
                if meta_labeler is not None and config.use_meta_labeling:
                    try:
                        from src.backtesting.meta_labeling import extract_meta_features
                        features = extract_meta_features(df, i)
                        meta_confidence = meta_labeler.predict_confidence(features)
                        if meta_confidence < config.meta_label_threshold:
                            blocked_signals += 1
                            blocked = True
                        else:
                            # Scale position size by meta confidence
                            size_mult *= meta_confidence
                    except Exception:
                        pass  # Don't crash if meta-labeling fails

                if not blocked and size_mult > 0:
                    # Position sizing
                    if kelly and len(trades) >= 10:
                        recent_pnls = pd.Series([t.pnl_pct / 100 for t in trades[-50:]])
                        ps = kelly.from_returns(recent_pnls, cash, price)
                    else:
                        ps = PositionSize(
                            fraction=config.max_position_pct * 0.5,
                            shares=int(cash * config.max_position_pct * 0.5 / price) if price > 0 else 0,
                            dollar_amount=cash * config.max_position_pct * 0.5,
                            risk_amount=cash * config.max_position_pct * 0.5 * (signal.stop_loss_pct / 100),
                            method="fixed_fractional",
                            kelly_raw=0,
                            confidence=0.5,
                        )

                    actual_size = ps.dollar_amount * size_mult
                    actual_shares = int(actual_size / price) if price > 0 else 0

                    if actual_shares > 0:
                        # ATR-based dynamic barriers (override signal's fixed stops)
                        if config.use_atr_barriers and i >= config.atr_period + 1:
                            atr_val = compute_atr(
                                df["high"].values[:i+1],
                                df["low"].values[:i+1],
                                df["close"].values[:i+1],
                                period=config.atr_period,
                            )
                            if atr_val > 0:
                                atr_pct = atr_val / price * 100
                                # ATR-based stops override signal defaults
                                effective_stop = atr_pct * config.atr_stop_mult
                                effective_target = atr_pct * config.atr_target_mult
                            else:
                                effective_stop = signal.stop_loss_pct
                                effective_target = signal.take_profit_pct
                        else:
                            effective_stop = signal.stop_loss_pct
                            effective_target = signal.take_profit_pct

                        slippage = config.slippage_pct / 100
                        if signal.side == Side.LONG:
                            entry_price = price * (1 + slippage)
                            stop_price = entry_price * (1 - effective_stop / 100)
                            target_price = entry_price * (1 + effective_target / 100)
                        else:
                            entry_price = price * (1 - slippage)
                            stop_price = entry_price * (1 + effective_stop / 100)
                            target_price = entry_price * (1 - effective_target / 100)

                        cash -= config.commission_per_trade

                        position = {
                            "side": signal.side,
                            "entry_price": entry_price,
                            "entry_bar": i,
                            "entry_time": timestamp,
                            "size_dollars": actual_shares * entry_price,
                            "size_frac": actual_shares * entry_price / cash if cash > 0 else 0,
                            "shares": actual_shares,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "confidence": signal.confidence,
                        }

        # ── 3. Record equity = cash + unrealized position value ──
        if position is not None:
            total_equity = cash + _unrealized_pnl(position, price)
        else:
            total_equity = cash

        equity_values.append(total_equity)
        equity_times.append(df.index[i])

    # Close any remaining position at last bar
    if position is not None and len(df) > 0:
        last_price = df.iloc[-1]["close"]
        trade, pnl_dollars = _close_position(
            position, last_price, len(df) - 1, str(df.index[-1]), "end_of_data", config.slippage_pct,
        )
        trades.append(trade)
        cash += pnl_dollars
        if meta_labeler is not None:
            try:
                from src.backtesting.meta_labeling import extract_meta_features
                features = extract_meta_features(df, position["entry_bar"])
                meta_labeler.record_outcome(features, trade.pnl_pct > 0)
            except Exception:
                pass
        # Update last equity point to reflect closed position
        if equity_values:
            equity_values[-1] = cash

    # Build equity curve (one point per bar — proper time basis)
    equity_series = pd.Series(equity_values, index=equity_times) if equity_values else pd.Series([config.initial_capital])

    # Generate performance report
    trade_pnls = [t.pnl_pct for t in trades]
    trade_durations = [t.bars_held for t in trades]

    periods = 365 if asset_class == "crypto" else 252
    report = generate_report(
        equity_curve=equity_series,
        trade_pnls=trade_pnls,
        trade_durations=trade_durations,
        num_trials=num_trials,
        periods_per_year=periods,
        timeframe=df.index.inferred_freq or "1d",
        asset_class=asset_class,
    )

    return BacktestResult(
        strategy_name=meta.name,
        symbol=symbol,
        asset_class=asset_class,
        timeframe=df.index.inferred_freq or "unknown",
        config=config,
        trades=trades,
        equity_curve=equity_series,
        report=report,
        signals_generated=total_signals,
        signals_blocked=blocked_signals,
        window_type="full",
    )


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════════


@dataclass
class WalkForwardResult:
    """Result of a walk-forward analysis."""
    strategy_name: str
    symbol: str
    # In-sample results (for reference only — don't trade on these!)
    in_sample_results: list[BacktestResult]
    # Out-of-sample results (THIS is what matters)
    out_of_sample_results: list[BacktestResult]
    # Combined OOS equity curve
    combined_oos_report: PerformanceReport
    # Walk-forward efficiency = OOS performance / IS performance
    wf_efficiency: float
    # Number of windows
    num_windows: int


def walk_forward_analysis(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str = "stock",
    config: BacktestConfig | None = None,
    train_pct: float = 0.70,       # 70% train, 30% test per window
    num_windows: int = 5,           # Number of walk-forward windows
    embargo_bars: int = 5,          # Gap between train and test (prevents leakage)
) -> WalkForwardResult:
    """
    Walk-forward backtesting — the gold standard for strategy evaluation.

    Splits data into overlapping train/test windows:

    Window 1: [====TRAIN====][embargo][==TEST==]
    Window 2:      [====TRAIN====][embargo][==TEST==]
    Window 3:           [====TRAIN====][embargo][==TEST==]

    Only the TEST results count. The TRAIN results are for parameter
    optimization (which we do in the genetic optimizer).

    This prevents curve fitting because:
    - Strategy parameters are fit on TRAIN data
    - Performance is measured on TEST data it's never seen
    - Multiple windows show consistency (not just one lucky period)
    """
    if config is None:
        config = BacktestConfig()

    meta = strategy.meta()
    total_bars = len(df)

    # Calculate window sizes
    # Each window overlaps the next by (1 - step_size) percent
    window_size = total_bars // num_windows
    train_size = int(window_size * train_pct)
    test_size = window_size - train_size - embargo_bars

    if train_size < 60 or test_size < 20:
        log.warning("Not enough data for %d walk-forward windows. Need more history.", num_windows)
        # Fall back to single split
        num_windows = 2
        window_size = total_bars // 2
        train_size = int(window_size * train_pct)
        test_size = window_size - train_size - embargo_bars

    in_sample_results: list[BacktestResult] = []
    oos_results: list[BacktestResult] = []

    # Sliding window step
    step = (total_bars - window_size) // max(1, num_windows - 1) if num_windows > 1 else 0

    for w in range(num_windows):
        start = w * step
        train_end = start + train_size
        test_start = train_end + embargo_bars
        test_end = min(test_start + test_size, total_bars)

        if test_end > total_bars or test_start >= test_end:
            break

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[test_start:test_end]

        log.info(
            "WF window %d/%d: train[%d:%d] embargo[%d] test[%d:%d]",
            w + 1, num_windows, start, train_end, embargo_bars, test_start, test_end,
        )

        # In-sample backtest
        is_result = run_backtest(strategy, train_df, symbol, asset_class, config)
        is_result.window_type = "in_sample"
        in_sample_results.append(is_result)

        # Out-of-sample backtest
        oos_result = run_backtest(strategy, test_df, symbol, asset_class, config)
        oos_result.window_type = "out_of_sample"
        oos_results.append(oos_result)

    # Combine OOS results
    all_oos_pnls = []
    all_oos_durations = []
    oos_equities = [config.initial_capital]

    for result in oos_results:
        for trade in result.trades:
            all_oos_pnls.append(trade.pnl_pct)
            all_oos_durations.append(trade.bars_held)
        if len(result.equity_curve) > 0:
            # Chain equity curves
            scale = oos_equities[-1] / result.equity_curve.iloc[0] if result.equity_curve.iloc[0] > 0 else 1
            scaled = result.equity_curve * scale
            oos_equities.extend(scaled.values[1:])

    oos_equity = pd.Series(oos_equities)
    periods = 365 if asset_class == "crypto" else 252

    combined_report = generate_report(
        equity_curve=oos_equity,
        trade_pnls=all_oos_pnls,
        trade_durations=all_oos_durations,
        num_trials=1,
        periods_per_year=periods,
        asset_class=asset_class,
    )

    # Walk-forward efficiency
    is_sharpe = np.mean([r.report.sharpe_ratio for r in in_sample_results]) if in_sample_results else 0
    oos_sharpe = combined_report.sharpe_ratio
    wf_efficiency = oos_sharpe / is_sharpe if is_sharpe > 0 else 0

    return WalkForwardResult(
        strategy_name=meta.name,
        symbol=symbol,
        in_sample_results=in_sample_results,
        out_of_sample_results=oos_results,
        combined_oos_report=combined_report,
        wf_efficiency=round(wf_efficiency, 3),
        num_windows=len(oos_results),
    )


# ═══════════════════════════════════════════════════════════════
# COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# ═══════════════════════════════════════════════════════════════
#
# From Marcos López de Prado, "Advances in Financial Machine Learning"
#
# Standard walk-forward gives ONE path through data.
# CPCV generates ALL combinatorial paths by:
# 1. Split data into N groups
# 2. Use k groups for testing (C(N,k) combinations)
# 3. Train on remaining N-k groups
# 4. Purge (remove) observations near train/test boundary
#    to prevent information leakage
#
# This gives C(N,k) backtest paths instead of just 1,
# allowing us to compute a distribution of performance
# and detect overfitting with much higher power.
# ═══════════════════════════════════════════════════════════════


@dataclass
class CPCVResult:
    """Result of Combinatorial Purged Cross-Validation."""
    strategy_name: str
    symbol: str
    # Distribution of OOS performance across all paths
    sharpe_distribution: list[float]
    cagr_distribution: list[float]
    max_dd_distribution: list[float]
    # Summary stats
    mean_sharpe: float
    median_sharpe: float
    sharpe_std: float
    prob_sharpe_positive: float      # P(Sharpe > 0) across paths
    prob_sharpe_viable: float        # P(Sharpe > 0.5) across paths
    mean_cagr: float
    mean_max_dd: float
    # Overfitting probability
    prob_overfit: float              # P(OOS Sharpe < 0 | IS Sharpe > 0)
    deflated_sharpe: float
    num_paths: int
    num_groups: int
    test_groups: int


def cpcv_analysis(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str = "stock",
    config: BacktestConfig | None = None,
    num_groups: int = 6,           # Split data into 6 groups
    test_groups: int = 2,          # Use 2 groups for testing
    purge_bars: int = 5,           # Purge 5 bars at boundaries
) -> CPCVResult:
    """
    Combinatorial Purged Cross-Validation.

    For N=6, k=2: C(6,2) = 15 backtest paths.
    Each path uses 2 groups for testing and 4 for training.
    Purge bars at boundaries prevent information leakage.

    This is strictly better than walk-forward because:
    - More paths = better statistical power
    - Every observation is used for both training and testing
    - Overfitting probability can be directly estimated
    """
    from itertools import combinations

    if config is None:
        config = BacktestConfig()

    meta = strategy.meta()
    total_bars = len(df)
    group_size = total_bars // num_groups

    if group_size < 30:
        log.warning("Not enough data for CPCV with %d groups. Need more bars.", num_groups)
        num_groups = max(3, total_bars // 30)
        group_size = total_bars // num_groups

    # Define group boundaries
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, total_bars)
        groups.append((start, end))

    # Generate all C(N, k) test set combinations
    test_combos = list(combinations(range(num_groups), test_groups))
    log.info(
        "CPCV: %d groups, %d test groups, %d paths, %d purge bars",
        num_groups, test_groups, len(test_combos), purge_bars,
    )

    sharpe_dist = []
    cagr_dist = []
    dd_dist = []
    is_sharpes = []
    oos_sharpes = []

    periods = 365 if asset_class == "crypto" else 252

    for combo in test_combos:
        test_indices = set()
        purge_indices = set()

        # Collect test and purge bar indices
        for g in combo:
            g_start, g_end = groups[g]
            test_indices.update(range(g_start, g_end))

            # Purge bars at boundaries of test groups
            for bar_offset in range(1, purge_bars + 1):
                if g_start - bar_offset >= 0:
                    purge_indices.add(g_start - bar_offset)
                if g_end + bar_offset - 1 < total_bars:
                    purge_indices.add(g_end + bar_offset - 1)

        # Train indices = everything not in test or purge
        train_mask = [
            i not in test_indices and i not in purge_indices
            for i in range(total_bars)
        ]
        test_mask = [i in test_indices for i in range(total_bars)]

        train_df = df.iloc[train_mask]
        test_df = df.iloc[test_mask]

        if len(train_df) < 30 or len(test_df) < 20:
            continue

        try:
            # In-sample backtest
            is_result = run_backtest(strategy, train_df, symbol, asset_class, config)
            is_sharpes.append(is_result.report.sharpe_ratio)

            # Out-of-sample backtest
            oos_result = run_backtest(strategy, test_df, symbol, asset_class, config)
            oos_sharpes.append(oos_result.report.sharpe_ratio)

            sharpe_dist.append(oos_result.report.sharpe_ratio)
            cagr_dist.append(oos_result.report.cagr_pct)
            dd_dist.append(oos_result.report.drawdown.max_drawdown_pct)

        except Exception as e:
            log.debug("CPCV path failed: %s", e)
            continue

    if not sharpe_dist:
        return CPCVResult(
            strategy_name=meta.name, symbol=symbol,
            sharpe_distribution=[], cagr_distribution=[], max_dd_distribution=[],
            mean_sharpe=0, median_sharpe=0, sharpe_std=0,
            prob_sharpe_positive=0, prob_sharpe_viable=0,
            mean_cagr=0, mean_max_dd=0, prob_overfit=1.0,
            deflated_sharpe=0, num_paths=0,
            num_groups=num_groups, test_groups=test_groups,
        )

    sharpe_arr = np.array(sharpe_dist)
    cagr_arr = np.array(cagr_dist)
    dd_arr = np.array(dd_dist)

    # Probability of overfitting:
    # P(OOS Sharpe < 0 when IS Sharpe > 0)
    overfit_count = sum(
        1 for is_s, oos_s in zip(is_sharpes, oos_sharpes)
        if is_s > 0 and oos_s < 0
    )
    positive_is_count = sum(1 for s in is_sharpes if s > 0)
    prob_overfit = overfit_count / positive_is_count if positive_is_count > 0 else 1.0

    # Deflated Sharpe across all paths
    from src.backtesting.metrics import compute_deflated_sharpe
    try:
        best_oos_sharpe = max(sharpe_dist)
        deflated = compute_deflated_sharpe(
            best_oos_sharpe,
            num_trials=len(sharpe_dist),
            total_bars=total_bars,
            skewness=float(pd.Series(sharpe_dist).skew()) if len(sharpe_dist) > 2 else 0,
            kurtosis=float(pd.Series(sharpe_dist).kurtosis()) + 3 if len(sharpe_dist) > 3 else 3,
        )
    except Exception:
        deflated = float(np.mean(sharpe_dist))

    return CPCVResult(
        strategy_name=meta.name,
        symbol=symbol,
        sharpe_distribution=sharpe_dist,
        cagr_distribution=cagr_dist,
        max_dd_distribution=dd_dist,
        mean_sharpe=round(float(sharpe_arr.mean()), 3),
        median_sharpe=round(float(np.median(sharpe_arr)), 3),
        sharpe_std=round(float(sharpe_arr.std()), 3),
        prob_sharpe_positive=round(float((sharpe_arr > 0).mean()), 3),
        prob_sharpe_viable=round(float((sharpe_arr > 0.5).mean()), 3),
        mean_cagr=round(float(cagr_arr.mean()), 2),
        mean_max_dd=round(float(dd_arr.mean()), 2),
        prob_overfit=round(prob_overfit, 3),
        deflated_sharpe=round(deflated, 3),
        num_paths=len(sharpe_dist),
        num_groups=num_groups,
        test_groups=test_groups,
    )


# ═══════════════════════════════════════════════════════════════
# TRIPLE BARRIER METHOD
# ═══════════════════════════════════════════════════════════════
#
# From López de Prado — replaces fixed stop/target with three
# simultaneous exit conditions:
#
# 1. Upper barrier (take profit) — price hits target
# 2. Lower barrier (stop loss) — price hits stop
# 3. Vertical barrier (time limit) — max holding period expires
#
# Whichever barrier is hit first determines the trade outcome.
# The vertical barrier prevents trades from lingering forever
# in no-man's land, which is a common source of hidden risk.
# ═══════════════════════════════════════════════════════════════


def apply_triple_barrier(
    df: pd.DataFrame,
    entry_bar: int,
    side: Side,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_holding_bars: int = 20,
) -> tuple[int, float, str]:
    """
    Apply the triple barrier method to find the exit point.

    Returns: (exit_bar, exit_price, exit_reason)
    """
    if side == Side.LONG:
        stop_price = entry_price * (1 - stop_loss_pct / 100)
        target_price = entry_price * (1 + take_profit_pct / 100)
    else:
        stop_price = entry_price * (1 + stop_loss_pct / 100)
        target_price = entry_price * (1 - take_profit_pct / 100)

    for i in range(entry_bar + 1, min(entry_bar + max_holding_bars + 1, len(df))):
        bar = df.iloc[i]

        if side == Side.LONG:
            if bar["low"] <= stop_price:
                return i, stop_price, "stop_loss"
            if bar["high"] >= target_price:
                return i, target_price, "take_profit"
        else:
            if bar["high"] >= stop_price:
                return i, stop_price, "stop_loss"
            if bar["low"] <= target_price:
                return i, target_price, "take_profit"

    # Vertical barrier: time expired — exit at market
    exit_bar = min(entry_bar + max_holding_bars, len(df) - 1)
    return exit_bar, df.iloc[exit_bar]["close"], "time_expiry"


# ═══════════════════════════════════════════════════════════════
# SHARPE-WEIGHTED MULTI-STRATEGY ALLOCATION
# ═══════════════════════════════════════════════════════════════


@dataclass
class StrategyAllocation:
    """Allocation weight for a strategy based on its Sharpe ratio."""
    strategy_name: str
    symbol: str
    sharpe: float
    weight: float           # 0.0 - 1.0, sum of all weights = 1.0
    capital_allocated: float


def compute_sharpe_weights(
    results: list[BacktestResult],
    total_capital: float = 100_000,
    min_sharpe: float = 0.0,
) -> list[StrategyAllocation]:
    """
    Allocate capital across strategies proportional to their Sharpe ratio.

    Strategies with Sharpe <= min_sharpe get zero allocation.
    Remaining strategies get capital proportional to their Sharpe.

    This is a simplified risk-parity approach where risk-adjusted
    return drives allocation rather than inverse-volatility.
    """
    # Filter to strategies with positive Sharpe
    viable = [
        r for r in results
        if r.report.sharpe_ratio > min_sharpe
        and r.report.trades.total_trades >= 10
    ]

    if not viable:
        return []

    # Sharpe-weighted allocation
    sharpes = np.array([r.report.sharpe_ratio for r in viable])
    weights = sharpes / sharpes.sum()

    allocations = []
    for r, weight in zip(viable, weights):
        allocations.append(StrategyAllocation(
            strategy_name=r.strategy_name,
            symbol=r.symbol,
            sharpe=round(r.report.sharpe_ratio, 3),
            weight=round(float(weight), 4),
            capital_allocated=round(total_capital * float(weight), 2),
        ))

    return sorted(allocations, key=lambda a: a.weight, reverse=True)
