"""
Performance metrics for backtest evaluation.

Calculates standard trading performance statistics:
total return, CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor, etc.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    hold_bars: int
    stop_hit: bool = False
    target_hit: bool = False


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Summary metrics (computed after backtest)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_hold_bars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy_pct: float = 0.0
    calmar_ratio: float = 0.0
    exposure_pct: float = 0.0
    annual_trades: float = 0.0


def compute_metrics(result: BacktestResult, trading_days_per_year: int = 252) -> BacktestResult:
    """Compute all performance metrics from trades and equity curve."""
    trades = result.trades
    equity = result.equity_curve

    if not trades:
        return result

    result.total_trades = len(trades)

    # P&L arrays
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Win rate
    result.win_rate_pct = (len(wins) / len(pnls)) * 100 if len(pnls) > 0 else 0

    # Average win / loss
    result.avg_win_pct = float(wins.mean()) if len(wins) > 0 else 0
    result.avg_loss_pct = float(losses.mean()) if len(losses) > 0 else 0

    # Best / worst
    result.best_trade_pct = float(pnls.max()) if len(pnls) > 0 else 0
    result.worst_trade_pct = float(pnls.min()) if len(pnls) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
    result.profit_factor = float(gross_profit / gross_loss)

    # Average hold
    result.avg_hold_bars = float(np.mean([t.hold_bars for t in trades]))

    # Expectancy
    if result.win_rate_pct > 0:
        wr = result.win_rate_pct / 100
        result.expectancy_pct = wr * result.avg_win_pct + (1 - wr) * result.avg_loss_pct
    else:
        result.expectancy_pct = result.avg_loss_pct

    # Consecutive wins / losses
    result.max_consecutive_wins = _max_consecutive(pnls, positive=True)
    result.max_consecutive_losses = _max_consecutive(pnls, positive=False)

    # Equity curve metrics
    if len(equity) > 1:
        # Total return
        result.total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

        # CAGR
        n_days = (equity.index[-1] - equity.index[0]).days
        if n_days > 0:
            n_years = n_days / 365.25
            result.cagr_pct = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100
        else:
            result.cagr_pct = 0

        # Daily returns
        daily_returns = equity.pct_change().dropna()

        # Sharpe ratio (annualized)
        if daily_returns.std() > 0:
            result.sharpe_ratio = float(
                (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
            )
        else:
            result.sharpe_ratio = 0

        # Sortino ratio (only downside deviation)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = float(
                (daily_returns.mean() / downside.std()) * np.sqrt(trading_days_per_year)
            )
        else:
            result.sortino_ratio = 0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        result.max_drawdown_pct = float(drawdown.min()) * 100

        # Calmar ratio (CAGR / max drawdown)
        if result.max_drawdown_pct != 0:
            result.calmar_ratio = result.cagr_pct / abs(result.max_drawdown_pct)
        else:
            result.calmar_ratio = 0

        # Exposure (% of time in market)
        total_bars = len(equity)
        bars_in_market = sum(t.hold_bars for t in trades)
        result.exposure_pct = min(100.0, (bars_in_market / total_bars) * 100) if total_bars > 0 else 0

        # Annual trades
        if n_days > 0:
            result.annual_trades = len(trades) / (n_days / 365.25)

    return result


def _max_consecutive(arr: np.ndarray, positive: bool = True) -> int:
    """Count max consecutive wins (positive=True) or losses (positive=False)."""
    max_streak = 0
    current = 0
    for val in arr:
        if (positive and val > 0) or (not positive and val <= 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def format_results_table(results: list[BacktestResult]) -> str:
    """Format multiple backtest results as a comparison table."""
    from tabulate import tabulate

    headers = [
        "Strategy", "Trades", "Win%", "CAGR%", "Sharpe", "Sortino",
        "MaxDD%", "PF", "Expect%", "Calmar", "Exposure%"
    ]

    rows = []
    for r in sorted(results, key=lambda x: x.sharpe_ratio, reverse=True):
        rows.append([
            r.strategy_name,
            r.total_trades,
            f"{r.win_rate_pct:.1f}",
            f"{r.cagr_pct:.1f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.sortino_ratio:.2f}",
            f"{r.max_drawdown_pct:.1f}",
            f"{r.profit_factor:.2f}",
            f"{r.expectancy_pct:.3f}",
            f"{r.calmar_ratio:.2f}",
            f"{r.exposure_pct:.1f}",
        ])

    return tabulate(rows, headers=headers, tablefmt="grid")


def format_detailed_report(r: BacktestResult) -> str:
    """Format a detailed report for a single strategy."""
    lines = [
        f"\n{'='*60}",
        f"  {r.strategy_name}",
        f"{'='*60}",
        f"  Parameters: {r.params}",
        f"",
        f"  RETURNS",
        f"    Total Return:     {r.total_return_pct:>8.1f}%",
        f"    CAGR:             {r.cagr_pct:>8.1f}%",
        f"    Sharpe Ratio:     {r.sharpe_ratio:>8.2f}",
        f"    Sortino Ratio:    {r.sortino_ratio:>8.2f}",
        f"    Calmar Ratio:     {r.calmar_ratio:>8.2f}",
        f"",
        f"  RISK",
        f"    Max Drawdown:     {r.max_drawdown_pct:>8.1f}%",
        f"    Exposure:         {r.exposure_pct:>8.1f}%",
        f"",
        f"  TRADES",
        f"    Total Trades:     {r.total_trades:>8d}",
        f"    Annual Trades:    {r.annual_trades:>8.1f}",
        f"    Win Rate:         {r.win_rate_pct:>8.1f}%",
        f"    Profit Factor:    {r.profit_factor:>8.2f}",
        f"    Expectancy:       {r.expectancy_pct:>8.3f}%",
        f"    Avg Win:          {r.avg_win_pct:>8.3f}%",
        f"    Avg Loss:         {r.avg_loss_pct:>8.3f}%",
        f"    Best Trade:       {r.best_trade_pct:>8.3f}%",
        f"    Worst Trade:      {r.worst_trade_pct:>8.3f}%",
        f"    Avg Hold (bars):  {r.avg_hold_bars:>8.1f}",
        f"    Max Consec Wins:  {r.max_consecutive_wins:>8d}",
        f"    Max Consec Loss:  {r.max_consecutive_losses:>8d}",
        f"{'='*60}",
    ]
    return "\n".join(lines)
