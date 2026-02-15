"""
Fractional Differentiation — preserve memory while achieving stationarity.

From Marcos López de Prado, "Advances in Financial Machine Learning" (Ch. 5).

The problem: standard integer differencing (d=1) makes a series stationary
but destroys all memory/autocorrelation. The signal you trained on is gone.

The solution: use fractional d (0 < d < 1) that removes just enough
non-stationarity while preserving >90% of the series memory.

    (1 - B)^d  where B is the backshift operator, d ∈ (0, 1)

Typical optimal d values:
- S&P 500 daily: d ≈ 0.35-0.45
- Crypto daily: d ≈ 0.20-0.35 (more persistent)
- Commodities: d ≈ 0.40-0.50

We use the Fixed-Width Window (FFD) method for computational efficiency:
- Only keep weights above a threshold (default 1e-5)
- This gives a fixed lookback window instead of expanding

Integration:
- Call add_fracdiff_features(df) before strategy signal generation
- Strategies use 'close_fracdiff' column instead of raw 'close'
- This makes features stationary without losing predictive power
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("forge.fracdiff")


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute Fixed-Width Window fractional differentiation weights.

    The weights follow: w_k = -w_{k-1} * (d - k + 1) / k
    Starting from w_0 = 1.

    We stop when |w_k| < threshold, giving a fixed window width.

    Args:
        d: Fractional differentiation order (0 < d < 1).
        threshold: Minimum absolute weight to include.

    Returns:
        Array of weights (oldest to newest).
    """
    weights = [1.0]
    k = 1
    while True:
        w_next = -weights[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        weights.append(w_next)
        k += 1
        if k > 10000:  # Safety limit
            break
    return np.array(weights[::-1])  # Oldest first


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation using the FFD method.

    (1 - B)^d * X_t = sum_{k=0}^{K} w_k * X_{t-k}

    where K is determined by the weight threshold.

    Args:
        series: Input time series (e.g., log prices).
        d: Fractional differentiation order.
        threshold: Minimum weight threshold for FFD window.

    Returns:
        Fractionally differentiated series (NaN for initial warmup period).
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)

    result = pd.Series(index=series.index, dtype=np.float64)
    result[:] = np.nan

    values = series.values

    for i in range(width - 1, len(values)):
        window = values[i - width + 1:i + 1]
        if not np.any(np.isnan(window)):
            result.iloc[i] = np.dot(weights, window)

    return result


def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.05,
    p_threshold: float = 0.05,
    threshold: float = 1e-3,
) -> float:
    """
    Find the minimum fractional differentiation order d that achieves stationarity.

    Uses the Augmented Dickey-Fuller (ADF) test. We want the smallest d
    where ADF p-value < p_threshold (rejecting the null of unit root).

    This preserves maximum memory while ensuring stationarity.

    Args:
        series: Input time series (e.g., close prices).
        d_range: Range of d values to search.
        d_step: Step size for d search.
        p_threshold: ADF p-value threshold for stationarity.

    Returns:
        Minimum d value that achieves stationarity.
    """
    try:
        from scipy.stats import normaltest
        # Use a simple ADF-like stationarity check without statsmodels
        # We compute autocorrelation and check if it decays
    except ImportError:
        pass

    # Try to use statsmodels ADF if available
    try:
        from statsmodels.tsa.stattools import adfuller
        has_adf = True
    except ImportError:
        has_adf = False

    d_values = np.arange(d_range[0], d_range[1] + d_step, d_step)
    log_series = np.log(series.replace(0, np.nan).dropna())

    for d in d_values:
        if d == 0:
            diff_series = log_series
        else:
            diff_series = frac_diff_ffd(log_series, d, threshold)

        clean = diff_series.dropna()
        if len(clean) < 30:
            continue

        if has_adf:
            try:
                adf_stat, p_value, *_ = adfuller(clean, maxlag=10, autolag="AIC")
                if p_value < p_threshold:
                    log.info("Minimum d for stationarity: %.2f (ADF p=%.4f)", d, p_value)
                    return round(d, 2)
            except Exception:
                continue
        else:
            # Fallback: variance ratio test approximation
            # If the variance of differences is much smaller than the level variance,
            # the series is likely stationary
            if len(clean) > 50:
                level_var = clean.var()
                diff_var = clean.diff().dropna().var()
                if diff_var > 0 and level_var / diff_var < 2.0:
                    log.info("Minimum d for stationarity (variance ratio): %.2f", d)
                    return round(d, 2)

    log.warning("Could not find stationary d, defaulting to 0.40")
    return 0.40


def add_fracdiff_features(
    df: pd.DataFrame,
    d: float | None = None,
    columns: list[str] | None = None,
    threshold: float = 1e-3,
) -> pd.DataFrame:
    """
    Add fractionally differentiated features to a DataFrame.

    If d is None, automatically finds the minimum d for stationarity
    on the 'close' column.

    Adds columns: close_fracdiff, volume_fracdiff (if volume exists).

    Args:
        df: OHLCV DataFrame.
        d: Fractional differentiation order (auto-detected if None).
        columns: Columns to fracdiff (default: close, volume).
        threshold: FFD weight threshold.

    Returns:
        DataFrame with additional fracdiff columns.
    """
    df = df.copy()

    if columns is None:
        columns = ["close"]
        if "volume" in df.columns:
            columns.append("volume")

    # Auto-detect d from close prices
    if d is None:
        close = df["close"].dropna()
        if len(close) > 100:
            d = find_min_d(close, threshold=threshold)
        else:
            d = 0.40  # Default for short series

    log.info("Applying fractional differentiation with d=%.2f", d)

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]
        # Apply on log-transformed values for price columns
        if col in ("close", "open", "high", "low"):
            log_series = np.log(series.replace(0, np.nan))
            fracdiff = frac_diff_ffd(log_series, d, threshold)
        else:
            # Volume: fracdiff raw values
            fracdiff = frac_diff_ffd(series, d, threshold)

        df[f"{col}_fracdiff"] = fracdiff

    # Store d value as attribute for reference
    df.attrs["fracdiff_d"] = d

    return df
