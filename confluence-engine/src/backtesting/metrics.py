"""
Research-grade performance metrics for backtesting.

Following Marcos López de Prado (Advances in Financial ML) and
Ernest Chan (Quantitative Trading / Algorithmic Trading).

Metrics computed:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (annualized, risk-free rate adjustable)
- Sortino Ratio (downside deviation only — better for asymmetric returns)
- Calmar Ratio (CAGR / max drawdown — reward-to-risk)
- Max Drawdown (depth, duration, recovery time)
- Profit Factor (gross profits / gross losses)
- Win Rate, Average Win/Loss, Expectancy
- Tail Ratio (95th percentile gain / 5th percentile loss)
- Deflated Sharpe Ratio (accounts for multiple testing / selection bias)
- Return distribution stats (skew, kurtosis)

All metrics assume daily returns unless timeframe is specified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
CRYPTO_DAYS_PER_YEAR = 365  # Crypto trades 24/7


@dataclass
class DrawdownInfo:
    """Detailed drawdown analysis."""
    max_drawdown_pct: float       # Deepest drawdown as percentage
    max_drawdown_duration: int    # Bars in longest drawdown
    avg_drawdown_pct: float       # Average drawdown depth
    avg_drawdown_duration: float  # Average drawdown length
    current_drawdown_pct: float   # Current drawdown from peak
    recovery_time: int            # Bars to recover from max DD (0 if not recovered)
    underwater_pct: float         # % of time spent in drawdown


@dataclass
class TradeStats:
    """Individual trade-level statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float               # Percentage
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_trade_pct: float
    median_trade_pct: float
    expectancy: float             # Expected value per trade
    avg_bars_in_trade: float
    avg_bars_in_winner: float
    avg_bars_in_loser: float
    consecutive_wins_max: int
    consecutive_losses_max: int
    profit_factor: float          # gross_profit / gross_loss


@dataclass
class PerformanceReport:
    """Complete backtest performance report."""
    # Returns
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float            # Probability-weighted ratio of gains vs losses

    # Drawdown
    drawdown: DrawdownInfo

    # Trade stats
    trades: TradeStats

    # Distribution
    skewness: float
    kurtosis: float
    tail_ratio: float             # Right tail / left tail (>1 = positive skew)
    var_95: float                 # Value at Risk (95%)
    cvar_95: float                # Conditional VaR / Expected Shortfall

    # Anti-overfitting
    deflated_sharpe: float        # Adjusted for multiple testing
    num_independent_trials: int   # How many strategy variants were tested

    # Meta
    start_date: str
    end_date: str
    total_bars: int
    timeframe: str
    asset_class: str

    def summary(self) -> dict:
        """Return a clean summary dict for display."""
        return {
            "total_return": f"{self.total_return_pct:+.2f}%",
            "cagr": f"{self.cagr_pct:+.2f}%",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "sortino": f"{self.sortino_ratio:.2f}",
            "calmar": f"{self.calmar_ratio:.2f}",
            "max_drawdown": f"{self.drawdown.max_drawdown_pct:.2f}%",
            "max_dd_duration": f"{self.drawdown.max_drawdown_duration} bars",
            "win_rate": f"{self.trades.win_rate:.1f}%",
            "profit_factor": f"{self.trades.profit_factor:.2f}",
            "total_trades": self.trades.total_trades,
            "expectancy": f"{self.trades.expectancy:+.3f}%",
            "tail_ratio": f"{self.tail_ratio:.2f}",
            "deflated_sharpe": f"{self.deflated_sharpe:.2f}",
            "omega": f"{self.omega_ratio:.2f}",
        }

    @property
    def is_viable(self) -> bool:
        """Quick check: does this strategy have a real edge?"""
        return (
            self.trades.total_trades >= 30
            and self.sharpe_ratio > 0.5
            and self.trades.profit_factor > 1.2
            and self.drawdown.max_drawdown_pct < 35.0
            and self.trades.expectancy > 0
            and self.deflated_sharpe > 0
        )


# ── Core Calculations ──


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Compute percentage returns from an equity curve."""
    return equity_curve.pct_change().dropna()


def compute_log_returns(equity_curve: pd.Series) -> pd.Series:
    """Compute log returns (better for compounding analysis)."""
    return np.log(equity_curve / equity_curve.shift(1)).dropna()


def compute_cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (ending / beginning) ^ (periods_per_year / total_periods) - 1
    """
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_periods = len(equity_curve) - 1
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -100.0
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (total_return ** (1.0 / years) - 1.0) * 100.0


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe Ratio.

    Sharpe = (mean_excess_return / std_return) * sqrt(periods_per_year)

    Using excess returns over risk-free rate (default 5% annual).
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino Ratio — like Sharpe but only penalizes downside volatility.

    Better for strategies with positive skew (large winners, small losers).
    Sortino = excess_return / downside_deviation
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_calmar(
    cagr: float,
    max_drawdown_pct: float,
) -> float:
    """
    Calmar Ratio = CAGR / Max Drawdown.

    Measures return per unit of maximum pain. > 1.0 is good, > 3.0 is excellent.
    """
    if max_drawdown_pct <= 0:
        return 0.0
    return cagr / max_drawdown_pct


def compute_omega(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio — probability-weighted ratio of gains vs losses.

    More complete than Sharpe because it considers the entire return distribution,
    not just mean/variance. Omega > 1.0 means more probability mass above threshold.
    """
    if len(returns) < 2:
        return 0.0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return float("inf") if gains.sum() > 0 else 0.0
    return gains.sum() / losses.sum()


def compute_drawdown(equity_curve: pd.Series) -> DrawdownInfo:
    """
    Full drawdown analysis.

    Computes max drawdown, duration, recovery time, and underwater percentage.
    """
    if len(equity_curve) < 2:
        return DrawdownInfo(0, 0, 0, 0, 0, 0, 0)

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100  # Percentage

    # Max drawdown
    max_dd = abs(drawdown.min())

    # Drawdown durations
    is_underwater = drawdown < 0
    underwater_pct = is_underwater.mean() * 100

    # Find drawdown periods
    dd_starts = []
    dd_ends = []
    in_dd = False
    start_idx = 0

    for i in range(len(is_underwater)):
        if is_underwater.iloc[i] and not in_dd:
            in_dd = True
            start_idx = i
        elif not is_underwater.iloc[i] and in_dd:
            in_dd = False
            dd_starts.append(start_idx)
            dd_ends.append(i)

    if in_dd:
        dd_starts.append(start_idx)
        dd_ends.append(len(is_underwater) - 1)

    durations = [e - s for s, e in zip(dd_starts, dd_ends)]
    max_duration = max(durations) if durations else 0
    avg_duration = np.mean(durations) if durations else 0

    # Average drawdown depth
    dd_depths = []
    for s, e in zip(dd_starts, dd_ends):
        dd_depths.append(abs(drawdown.iloc[s:e + 1].min()))
    avg_depth = np.mean(dd_depths) if dd_depths else 0

    # Recovery time from max DD
    max_dd_idx = drawdown.idxmin()
    if isinstance(max_dd_idx, (pd.Timestamp, int, np.integer)):
        max_dd_pos = equity_curve.index.get_loc(max_dd_idx)
        recovery = 0
        for i in range(max_dd_pos + 1, len(equity_curve)):
            if equity_curve.iloc[i] >= peak.iloc[max_dd_pos]:
                recovery = i - max_dd_pos
                break
    else:
        recovery = 0

    current_dd = abs(drawdown.iloc[-1])

    return DrawdownInfo(
        max_drawdown_pct=round(max_dd, 2),
        max_drawdown_duration=max_duration,
        avg_drawdown_pct=round(avg_depth, 2),
        avg_drawdown_duration=round(avg_duration, 1),
        current_drawdown_pct=round(current_dd, 2),
        recovery_time=recovery,
        underwater_pct=round(underwater_pct, 1),
    )


def compute_trade_stats(trade_pnls: list[float], trade_durations: list[int] | None = None) -> TradeStats:
    """
    Compute detailed trade-level statistics.

    Args:
        trade_pnls: List of P&L percentages for each closed trade.
        trade_durations: Optional list of bars held for each trade.
    """
    if not trade_pnls:
        return TradeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pnls = np.array(trade_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / total * 100) if total > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    # Expectancy (average expected P&L per trade)
    expectancy = pnls.mean()

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for p in pnls:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Duration stats
    durations = trade_durations or [0] * total
    dur_arr = np.array(durations)
    win_mask = pnls > 0
    loss_mask = pnls <= 0

    return TradeStats(
        total_trades=total,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=round(win_rate, 1),
        avg_win_pct=round(float(avg_win), 3),
        avg_loss_pct=round(float(avg_loss), 3),
        largest_win_pct=round(float(pnls.max()), 3),
        largest_loss_pct=round(float(pnls.min()), 3),
        avg_trade_pct=round(float(pnls.mean()), 3),
        median_trade_pct=round(float(np.median(pnls)), 3),
        expectancy=round(float(expectancy), 4),
        avg_bars_in_trade=round(float(dur_arr.mean()), 1) if len(dur_arr) > 0 else 0,
        avg_bars_in_winner=round(float(dur_arr[win_mask].mean()), 1) if win_mask.any() else 0,
        avg_bars_in_loser=round(float(dur_arr[loss_mask].mean()), 1) if loss_mask.any() else 0,
        consecutive_wins_max=max_consec_wins,
        consecutive_losses_max=max_consec_losses,
        profit_factor=round(profit_factor, 2),
    )


def compute_deflated_sharpe(
    sharpe: float,
    num_trials: int,
    total_bars: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts Sharpe for the number of strategies tested (selection bias).
    If you test 100 strategies and pick the best, the Sharpe is inflated.

    DSR accounts for:
    - Number of independent trials
    - Non-normality of returns (skew, kurtosis)
    - Sample length

    Returns a Sharpe that's been "deflated" to account for data snooping.
    A DSR > 0 means the strategy likely has real edge beyond random chance.
    """
    if num_trials <= 1 or total_bars < 10:
        return sharpe

    from scipy import stats as scipy_stats

    # Expected maximum Sharpe under null hypothesis of no skill
    # E[max(SR)] ≈ sqrt(2 * ln(N)) for N independent trials
    euler_mascheroni = 0.5772
    expected_max_sr = np.sqrt(2 * np.log(num_trials)) * (
        1 - euler_mascheroni / (2 * np.log(num_trials))
    ) + euler_mascheroni / (2 * np.sqrt(2 * np.log(num_trials)))

    # Standard error of Sharpe ratio (accounting for non-normality)
    se_sr = np.sqrt(
        (1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis - 3) / 4 * sharpe**2) / (total_bars - 1)
    )

    if se_sr <= 0:
        return 0.0

    # PSR: probability that true Sharpe > 0 given observed Sharpe
    # Deflated: probability that true Sharpe > expected_max_sr
    z = (sharpe - expected_max_sr) / se_sr
    deflated = scipy_stats.norm.cdf(z)

    # Return as an adjusted Sharpe-like metric
    # If deflated > 0.95, the strategy likely has real edge
    return round(sharpe * deflated, 3)


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk — worst expected loss at given confidence."""
    if len(returns) < 2:
        return 0.0
    return abs(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).

    Average of losses beyond VaR — measures tail risk better than VaR alone.
    """
    if len(returns) < 2:
        return 0.0
    var = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var]
    if len(tail_losses) == 0:
        return abs(var)
    return abs(tail_losses.mean())


def compute_tail_ratio(returns: pd.Series) -> float:
    """
    Tail Ratio = 95th percentile / abs(5th percentile).

    > 1.0 means right tail is fatter (more big wins than big losses).
    Good strategies have tail_ratio > 1.0.
    """
    if len(returns) < 20:
        return 1.0
    right = np.percentile(returns, 95)
    left = abs(np.percentile(returns, 5))
    if left == 0:
        return float("inf") if right > 0 else 1.0
    return right / left


# ── Full Report Generator ──


def generate_report(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    trade_durations: list[int] | None = None,
    num_trials: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.05,
    timeframe: str = "1d",
    asset_class: str = "stock",
) -> PerformanceReport:
    """
    Generate a complete performance report from an equity curve and trades.

    Args:
        equity_curve: Series of portfolio values over time (index = timestamps).
        trade_pnls: List of P&L percentages for each closed trade.
        trade_durations: Optional list of bars held for each trade.
        num_trials: Number of strategy variants tested (for deflated Sharpe).
        periods_per_year: 252 for stocks, 365 for crypto.
        risk_free_rate: Annual risk-free rate (default 5%).
        timeframe: Bar timeframe ("1d", "1h", "5m", etc.).
        asset_class: "stock", "crypto", or "polymarket".
    """
    if asset_class == "crypto":
        periods_per_year = CRYPTO_DAYS_PER_YEAR

    returns = compute_returns(equity_curve)
    cagr = compute_cagr(equity_curve, periods_per_year)
    sharpe = compute_sharpe(returns, risk_free_rate, periods_per_year)
    sortino = compute_sortino(returns, risk_free_rate, periods_per_year)
    dd_info = compute_drawdown(equity_curve)
    calmar = compute_calmar(cagr, dd_info.max_drawdown_pct)
    omega = compute_omega(returns)

    trade_stats = compute_trade_stats(trade_pnls, trade_durations)

    # Distribution stats
    skew = float(returns.skew()) if len(returns) > 2 else 0.0
    kurt = float(returns.kurtosis()) + 3.0 if len(returns) > 3 else 3.0  # Convert excess to regular
    tail = compute_tail_ratio(returns)
    var_95 = compute_var(returns)
    cvar_95 = compute_cvar(returns)

    # Anti-overfitting
    try:
        deflated = compute_deflated_sharpe(sharpe, num_trials, len(equity_curve), skew, kurt)
    except ImportError:
        deflated = sharpe  # scipy not available, use raw Sharpe

    ann_vol = float(returns.std() * np.sqrt(periods_per_year) * 100) if len(returns) > 1 else 0.0
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100 if len(equity_curve) > 1 else 0.0

    return PerformanceReport(
        total_return_pct=round(total_ret, 2),
        cagr_pct=round(cagr, 2),
        annualized_volatility_pct=round(ann_vol, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        omega_ratio=round(omega, 3),
        drawdown=dd_info,
        trades=trade_stats,
        skewness=round(skew, 3),
        kurtosis=round(kurt, 3),
        tail_ratio=round(tail, 3),
        var_95=round(var_95, 4),
        cvar_95=round(cvar_95, 4),
        deflated_sharpe=round(deflated, 3),
        num_independent_trials=num_trials,
        start_date=str(equity_curve.index[0])[:10] if len(equity_curve) > 0 else "",
        end_date=str(equity_curve.index[-1])[:10] if len(equity_curve) > 0 else "",
        total_bars=len(equity_curve),
        timeframe=timeframe,
        asset_class=asset_class,
    )


# ── Monte Carlo Robustness Testing ──


def monte_carlo_robustness(
    trade_pnls: list[float],
    n_simulations: int = 5000,
    initial_capital: float = 100_000,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Monte Carlo robustness testing via trade resampling.

    Randomly resamples the trade sequence N times to build
    distributions of key metrics. This tells you whether your
    backtest results are robust or just a lucky ordering of trades.

    Returns:
        Dictionary with confidence intervals for Sharpe, CAGR, max drawdown.
    """
    if len(trade_pnls) < 10:
        return {"error": "Insufficient trades for Monte Carlo", "n_trades": len(trade_pnls)}

    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    sharpes = []
    cagrs = []
    max_dds = []
    final_equities = []

    for _ in range(n_simulations):
        # Resample trade sequence with replacement
        resampled = np.random.choice(pnls, size=n_trades, replace=True)

        # Build equity curve from resampled trades
        equity = [initial_capital]
        for pnl_pct in resampled:
            equity.append(equity[-1] * (1 + pnl_pct / 100))
        eq = np.array(equity)

        # Compute returns
        returns = np.diff(eq) / eq[:-1]
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

        # CAGR
        total_return = eq[-1] / eq[0]
        if total_return > 0:
            years = n_trades / periods_per_year
            if years > 0:
                cagr = (total_return ** (1.0 / years) - 1.0) * 100
            else:
                cagr = 0.0
        else:
            cagr = -100.0
        cagrs.append(cagr)

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        max_dds.append(np.max(dd))

        final_equities.append(eq[-1])

    sharpes = np.array(sharpes)
    cagrs = np.array(cagrs)
    max_dds = np.array(max_dds)

    return {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "sharpe_mean": round(float(np.mean(sharpes)), 3),
        "sharpe_median": round(float(np.median(sharpes)), 3),
        "sharpe_5th": round(float(np.percentile(sharpes, 5)), 3),
        "sharpe_95th": round(float(np.percentile(sharpes, 95)), 3),
        "sharpe_prob_positive": round(float((sharpes > 0).mean()), 3),
        "sharpe_prob_above_05": round(float((sharpes > 0.5).mean()), 3),
        "cagr_mean": round(float(np.mean(cagrs)), 2),
        "cagr_median": round(float(np.median(cagrs)), 2),
        "cagr_5th": round(float(np.percentile(cagrs, 5)), 2),
        "cagr_95th": round(float(np.percentile(cagrs, 95)), 2),
        "max_dd_mean": round(float(np.mean(max_dds)), 2),
        "max_dd_95th": round(float(np.percentile(max_dds, 95)), 2),
        "prob_ruin": round(float((np.array(final_equities) < initial_capital * 0.5).mean()), 4),
    }


# ── GARCH Volatility Forecasting ──


def garch_volatility_forecast(
    returns: pd.Series,
    forecast_horizon: int = 5,
) -> dict:
    """
    GARCH(1,1) volatility forecast.

    Models time-varying volatility: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Uses maximum likelihood estimation. Falls back to EWMA if scipy optimization fails.

    Returns:
        Dictionary with current vol estimate, forecast, and GARCH parameters.
    """
    if len(returns) < 50:
        return {"error": "Insufficient data for GARCH", "n_returns": len(returns)}

    r = returns.values if isinstance(returns, pd.Series) else returns
    r = r[~np.isnan(r)]
    n = len(r)

    # EWMA fallback (always computed as baseline)
    ewma_lambda = 0.94
    ewma_var = np.zeros(n)
    ewma_var[0] = np.var(r[:20]) if n >= 20 else np.var(r)
    for t in range(1, n):
        ewma_var[t] = ewma_lambda * ewma_var[t - 1] + (1 - ewma_lambda) * r[t - 1] ** 2

    ewma_forecast = ewma_var[-1]

    # GARCH(1,1) via maximum likelihood
    try:
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(r)
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(sigma2) + r ** 2 / sigma2)
            return -ll

        # Initial params from variance targeting
        var_r = np.var(r)
        x0 = [var_r * 0.05, 0.08, 0.88]
        bounds = [(1e-8, var_r * 2), (1e-4, 0.5), (0.3, 0.999)]

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        if result.success:
            omega, alpha, beta = result.x
            # Compute conditional variance series
            sigma2 = np.zeros(n)
            sigma2[0] = var_r
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

            # Multi-step forecast
            h_forecast = sigma2[-1]
            long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else var_r
            forecasts = []
            for h in range(1, forecast_horizon + 1):
                h_forecast = omega + (alpha + beta) * h_forecast
                forecasts.append(np.sqrt(h_forecast) * np.sqrt(252) * 100)

            persistence = alpha + beta

            return {
                "model": "GARCH(1,1)",
                "omega": round(float(omega), 8),
                "alpha": round(float(alpha), 4),
                "beta": round(float(beta), 4),
                "persistence": round(float(persistence), 4),
                "current_vol_annual_pct": round(float(np.sqrt(sigma2[-1]) * np.sqrt(252) * 100), 2),
                "long_run_vol_annual_pct": round(float(np.sqrt(long_run_var) * np.sqrt(252) * 100), 2),
                "forecast_vol_pct": [round(f, 2) for f in forecasts],
                "ewma_vol_annual_pct": round(float(np.sqrt(ewma_forecast) * np.sqrt(252) * 100), 2),
                "half_life_days": round(float(-np.log(2) / np.log(persistence)), 1) if persistence > 0 else None,
            }
    except Exception:
        pass

    # Fallback to EWMA
    return {
        "model": "EWMA",
        "lambda": ewma_lambda,
        "current_vol_annual_pct": round(float(np.sqrt(ewma_forecast) * np.sqrt(252) * 100), 2),
        "forecast_vol_pct": [round(float(np.sqrt(ewma_forecast) * np.sqrt(252) * 100), 2)] * forecast_horizon,
    }
