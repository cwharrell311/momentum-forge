"""
Backtest Engine.

Runs strategies against historical data, simulating trade execution
with realistic assumptions (slippage, commissions, stop losses).
Produces equity curves and trade records for analysis.
"""

import numpy as np
import pandas as pd
from .strategies import Strategy
from .metrics import TradeRecord, BacktestResult, compute_metrics


class BacktestEngine:
    """
    Event-driven backtest engine for daytrading strategies.

    Assumptions:
    - Entries at the strategy-specified entry_price (Open or breakout level)
    - Exits at Close (end of day) unless stop loss is hit intraday
    - Stop losses checked against intraday Low (for longs) or High (for shorts)
    - Commissions and slippage applied per trade
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_per_trade: float = 1.0,  # $1 per trade (modern broker)
        slippage_pct: float = 0.01,  # 0.01% slippage per side
        position_size_pct: float = 100.0,  # % of capital per trade
        max_daily_trades: int = 5,
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_trade
        self.slippage_pct = slippage_pct / 100
        self.position_size_pct = position_size_pct / 100
        self.max_daily_trades = max_daily_trades

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Run backtest for a single strategy."""
        sig_df = strategy.generate_signals(df)
        trades = []
        equity = [self.initial_capital]
        equity_dates = [sig_df.index[0]]
        capital = self.initial_capital

        daily_trade_count = 0
        last_date = None

        for i in range(len(sig_df)):
            row = sig_df.iloc[i]
            current_date = sig_df.index[i]

            # Reset daily trade counter
            if last_date is None or current_date.date() != last_date:
                daily_trade_count = 0
                last_date = current_date.date()

            if row["signal"] == 0 or pd.isna(row["entry_price"]) or pd.isna(row["exit_price"]):
                equity.append(capital)
                equity_dates.append(current_date)
                continue

            if daily_trade_count >= self.max_daily_trades:
                equity.append(capital)
                equity_dates.append(current_date)
                continue

            direction = row["signal"]  # +1 or -1
            entry_price = row["entry_price"]
            exit_price = row["exit_price"]
            stop_loss = row.get("stop_loss", np.nan)

            # Apply slippage to entry
            if direction == 1:
                entry_price *= (1 + self.slippage_pct)
            else:
                entry_price *= (1 - self.slippage_pct)

            # Position sizing
            position_value = capital * self.position_size_pct
            shares = position_value / entry_price

            # Check if stop loss was hit intraday
            stop_hit = False
            if not pd.isna(stop_loss):
                if direction == 1 and row["Low"] <= stop_loss:
                    exit_price = stop_loss
                    stop_hit = True
                elif direction == -1 and row["High"] >= stop_loss:
                    exit_price = stop_loss
                    stop_hit = True

            # Apply slippage to exit
            if direction == 1:
                exit_price *= (1 - self.slippage_pct)
            else:
                exit_price *= (1 + self.slippage_pct)

            # Calculate P&L
            if direction == 1:
                pnl_per_share = exit_price - entry_price
            else:
                pnl_per_share = entry_price - exit_price

            pnl_dollar = pnl_per_share * shares - 2 * self.commission  # round trip
            pnl_pct = (pnl_per_share / entry_price) * 100

            capital += pnl_dollar
            daily_trade_count += 1

            trade = TradeRecord(
                entry_date=current_date,
                exit_date=current_date,
                direction="long" if direction == 1 else "short",
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                pnl_dollar=pnl_dollar,
                hold_bars=1,
                stop_hit=stop_hit,
            )
            trades.append(trade)

            equity.append(capital)
            equity_dates.append(current_date)

        equity_series = pd.Series(equity, index=equity_dates)
        # Remove duplicate indices by keeping last
        equity_series = equity_series[~equity_series.index.duplicated(keep="last")]

        result = BacktestResult(
            strategy_name=strategy.name,
            params=strategy.params_dict,
            trades=trades,
            equity_curve=equity_series,
        )

        return compute_metrics(result)

    def run_parameter_sweep(
        self, df: pd.DataFrame, strategy_class: type,
        param_grid: dict[str, list],
    ) -> list[BacktestResult]:
        """Run backtest across a grid of parameters."""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_class(**params)
            result = self.run(df, strategy)
            results.append(result)

        return sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)


def run_walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    engine: BacktestEngine,
    train_pct: float = 0.7,
    n_splits: int = 5,
) -> list[BacktestResult]:
    """
    Walk-forward analysis: train on in-sample, test on out-of-sample.
    Returns results for each out-of-sample period.
    """
    n = len(df)
    split_size = n // n_splits
    results = []

    for i in range(n_splits):
        start = i * split_size
        end = min(start + split_size, n)
        if end - start < 50:
            continue

        train_end = start + int((end - start) * train_pct)
        test_df = df.iloc[train_end:end]

        if len(test_df) < 10:
            continue

        result = engine.run(test_df, strategy)
        result.strategy_name = f"{strategy.name} [WF {i+1}/{n_splits}]"
        results.append(result)

    return results
