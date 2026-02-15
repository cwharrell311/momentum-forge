"""
Signal Processing Techniques for Trading Signal Quality Improvement.

This module brings well-established techniques from signal processing, control
theory, physics, hydrology, neuroscience, and information theory to improve
trading signal quality. These are techniques used across ALL of science --
seismology, EEG analysis, climate science, aerospace, hydrology -- and are
applied here to financial time series.

Techniques implemented:
    - Wavelet Denoising (signal processing / seismology)
    - Kalman Filter Trend Estimation (control theory / aerospace)
    - Hurst Exponent (hydrology -- R/S analysis)
    - Permutation Entropy (information theory / neuroscience)
    - Empirical Mode Decomposition (NASA Hilbert-Huang Transform)

All external dependencies are soft -- the module falls back gracefully when
optional packages (e.g. pywt) are not installed. Core implementations
(Kalman, Hurst, permutation entropy, EMD) are pure numpy.
"""

from __future__ import annotations

import logging
import warnings
from itertools import permutations
from math import factorial
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import for PyWavelets
# ---------------------------------------------------------------------------
try:
    import pywt

    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    logger.info(
        "pywt (PyWavelets) not installed -- wavelet_denoise will fall back "
        "to Savitzky-Golay smoothing. Install with: pip install PyWavelets"
    )

# ---------------------------------------------------------------------------
# Soft import for scipy (used only in the Savitzky-Golay fallback)
# ---------------------------------------------------------------------------
try:
    from scipy.signal import savgol_filter

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =========================================================================
# 1. Wavelet Denoising
#    Domain: signal processing, seismology, EEG analysis
# =========================================================================

def wavelet_denoise(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",
) -> pd.Series:
    """Denoise a time series using discrete wavelet transform (DWT).

    This is the same technique routinely used to denoise seismic signals
    recorded by geophones, EEG brain-wave recordings, and climate
    proxy data. The idea is simple:

    1. Decompose the signal into wavelet coefficients at multiple
       resolution levels (coarse trend + fine detail).
    2. Apply a threshold to the detail coefficients -- small
       coefficients are noise, large ones carry real information.
    3. Reconstruct the signal from the thresholded coefficients.

    The universal threshold (VisuShrink) of Donoho & Johnstone (1994) is
    used: ``threshold = sigma * sqrt(2 * log(n))`` where *sigma* is
    estimated from the finest-level detail coefficients via the MAD
    estimator.

    Parameters
    ----------
    series : pd.Series
        The raw price or return series to denoise.
    wavelet : str, default ``"db4"``
        Wavelet family to use (e.g. ``"db4"``, ``"sym6"``, ``"coif3"``).
        Daubechies-4 is a solid default for financial data.
    level : int, default ``4``
        Number of decomposition levels. Higher = more aggressive
        smoothing. Capped internally at the maximum level the signal
        length supports.
    threshold_mode : str, default ``"soft"``
        ``"soft"`` (wavelet shrinkage) or ``"hard"`` thresholding.

    Returns
    -------
    pd.Series
        Denoised series with the same index as the input.

    Notes
    -----
    Falls back to a Savitzky-Golay filter (polynomial smoothing) if
    PyWavelets is not installed, and to a simple rolling mean if scipy
    is also unavailable.

    References
    ----------
    Donoho, D.L. & Johnstone, I.M. (1994). "Ideal spatial adaptation
    by wavelet shrinkage." *Biometrika*, 81(3), 425-455.
    """
    values = series.dropna().values.astype(np.float64)
    n = len(values)

    if n < 8:
        logger.warning("Series too short for wavelet denoising; returning as-is.")
        return series.copy()

    # ------------------------------------------------------------------
    # Primary path: PyWavelets available
    # ------------------------------------------------------------------
    if _HAS_PYWT:
        # Cap decomposition level at the maximum the signal supports
        max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)

        coeffs = pywt.wavedec(values, wavelet, level=level)

        # Estimate noise standard deviation from the finest detail coeffs
        # using the median absolute deviation (MAD) estimator
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Universal (VisuShrink) threshold
        threshold = sigma * np.sqrt(2 * np.log(n))

        # Threshold all detail coefficient arrays (skip approximation at index 0)
        denoised_coeffs = [coeffs[0]]  # keep approximation untouched
        for detail in coeffs[1:]:
            denoised_coeffs.append(
                pywt.threshold(detail, value=threshold, mode=threshold_mode)
            )

        reconstructed = pywt.waverec(denoised_coeffs, wavelet)

        # waverec can produce an array one element longer due to padding
        reconstructed = reconstructed[:n]

        result = pd.Series(reconstructed, index=series.dropna().index, name=series.name)
        return result.reindex(series.index)

    # ------------------------------------------------------------------
    # Fallback path: Savitzky-Golay filter (scipy) or rolling mean
    # ------------------------------------------------------------------
    window = min(2 * level * 4 + 1, n if n % 2 == 1 else n - 1)
    window = max(window, 5)  # ensure minimum window
    if window % 2 == 0:
        window += 1

    if _HAS_SCIPY:
        logger.info("Using Savitzky-Golay fallback for wavelet_denoise.")
        polyorder = min(3, window - 1)
        smoothed = savgol_filter(values, window_length=window, polyorder=polyorder)
    else:
        logger.info("Using rolling-mean fallback for wavelet_denoise.")
        smoothed = pd.Series(values).rolling(window, center=True, min_periods=1).mean().values

    result = pd.Series(smoothed, index=series.dropna().index, name=series.name)
    return result.reindex(series.index)


# =========================================================================
# 2. Kalman Filter Trend Estimator
#    Domain: control theory, aerospace (GPS tracking, missile guidance)
# =========================================================================

class KalmanTrendFilter:
    """Linear Kalman filter for extracting trend (position + velocity) from
    a noisy price series.

    This is the *exact same mathematics* used in GPS receivers to track
    position and velocity of a moving object given noisy satellite
    measurements. The state vector is ``[price, velocity]`` where
    *velocity* is the instantaneous slope (trend) of the price.

    The state-space model is:

    .. math::

        \\mathbf{x}_t = F \\, \\mathbf{x}_{t-1} + \\mathbf{w}_t
        \\quad \\text{(transition)}

        z_t = H \\, \\mathbf{x}_t + v_t
        \\quad \\text{(observation)}

    with:

    * State: :math:`\\mathbf{x} = [\\text{price},\\; \\text{slope}]^T`
    * Transition matrix: :math:`F = [[1, 1], [0, 1]]`
      (price updates by adding slope; slope persists)
    * Observation matrix: :math:`H = [1, 0]` (we only observe price)
    * :math:`\\mathbf{w}_t \\sim \\mathcal{N}(0, Q)` -- process noise
    * :math:`v_t \\sim \\mathcal{N}(0, R)` -- observation (measurement) noise

    Parameters
    ----------
    observation_noise : float, default ``1.0``
        Variance *R* of the measurement noise. Larger values mean we
        trust observations less (more smoothing).
    process_noise : float, default ``0.01``
        Scalar multiplier for the process noise covariance *Q*. Larger
        values allow the filter to react faster to true changes.

    Examples
    --------
    >>> kf = KalmanTrendFilter(observation_noise=1.0, process_noise=0.01)
    >>> smoothed = kf.filter(price_series)
    >>> slopes = kf.get_trend_slope()
    """

    def __init__(
        self,
        observation_noise: float = 1.0,
        process_noise: float = 0.01,
    ) -> None:
        self.observation_noise = observation_noise
        self.process_noise = process_noise

        # State-space matrices (set once, used in filter)
        self._F = np.array([[1.0, 1.0],
                            [0.0, 1.0]])  # transition
        self._H = np.array([[1.0, 0.0]])  # observation

        # Process noise covariance
        self._Q = process_noise * np.array([[1.0, 0.0],
                                            [0.0, 1.0]])
        # Measurement noise covariance (scalar wrapped as 1x1)
        self._R = np.array([[observation_noise]])

        # Filtered state history (populated after calling .filter())
        self._states: Optional[np.ndarray] = None
        self._index: Optional[pd.Index] = None

    def filter(self, series: pd.Series) -> pd.Series:
        """Run the Kalman filter forward over the series.

        Parameters
        ----------
        series : pd.Series
            Observed (noisy) price series.

        Returns
        -------
        pd.Series
            Kalman-smoothed price estimate, same index as input.
        """
        values = series.dropna().values.astype(np.float64)
        n = len(values)
        if n == 0:
            return series.copy()

        F, H, Q, R = self._F, self._H, self._Q, self._R

        # Initial state: first observation, zero slope
        x = np.array([values[0], 0.0])
        P = np.eye(2) * 1.0  # initial covariance

        states = np.zeros((n, 2))

        for t in range(n):
            # --- Predict ---
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # --- Update ---
            z = values[t]
            y_residual = z - (H @ x_pred)[0]  # innovation
            S = (H @ P_pred @ H.T + R)  # innovation covariance (1x1)
            K = (P_pred @ H.T) / S[0, 0]  # Kalman gain (2x1)

            x = x_pred + K.flatten() * y_residual
            P = (np.eye(2) - K @ H) @ P_pred

            states[t] = x

        self._states = states
        self._index = series.dropna().index

        smoothed = pd.Series(
            states[:, 0], index=self._index, name="kalman_price"
        )
        return smoothed.reindex(series.index)

    def get_trend_slope(self) -> pd.Series:
        """Return the estimated slope (velocity) at each time step.

        Must call :meth:`filter` first.

        Returns
        -------
        pd.Series
            Trend slope estimate at each point. Positive = uptrend,
            negative = downtrend, magnitude = strength.

        Raises
        ------
        RuntimeError
            If :meth:`filter` has not been called yet.
        """
        if self._states is None or self._index is None:
            raise RuntimeError(
                "Call .filter(series) before .get_trend_slope()."
            )
        return pd.Series(
            self._states[:, 1], index=self._index, name="kalman_slope"
        )


# =========================================================================
# 3. Hurst Exponent (Rescaled Range analysis)
#    Domain: hydrology -- Harold Hurst studying Nile river floods (1951)
# =========================================================================

def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis.

    Originally developed by Harold Edwin Hurst (1880--1978) to study the
    long-term storage capacity of the Nile river. He analysed centuries of
    flood records and discovered that natural phenomena exhibit
    *long-range dependence* -- a finding that was later formalised by
    Mandelbrot as *fractional Brownian motion*.

    Interpretation for trading:

    * **H > 0.5** -- *persistent* (trending): past up-moves predict
      future up-moves. Momentum / trend-following strategies have edge.
    * **H = 0.5** -- *random walk*: no serial dependence. No
      statistical edge.
    * **H < 0.5** -- *anti-persistent* (mean-reverting): past up-moves
      predict future down-moves. Mean-reversion strategies have edge.

    Parameters
    ----------
    series : pd.Series
        Price or return series to analyse.
    max_lag : int, default ``100``
        Maximum sub-series length for R/S calculation. Must be >= 20.

    Returns
    -------
    float
        Estimated Hurst exponent in the range (0, 1).

    References
    ----------
    Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."
    *Transactions of the American Society of Civil Engineers*, 116,
    770-808.
    """
    values = series.dropna().values.astype(np.float64)
    n = len(values)

    if n < 20:
        logger.warning(
            "Series too short (%d points) for reliable Hurst estimation; "
            "returning 0.5 (random walk assumption).",
            n,
        )
        return 0.5

    max_lag = min(max_lag, n // 2)
    # Use lags that are roughly geometrically spaced for efficiency
    lags = np.unique(
        np.logspace(np.log10(10), np.log10(max_lag), num=30).astype(int)
    )
    lags = lags[lags >= 10]

    if len(lags) < 2:
        return 0.5

    rs_values = []
    lag_values = []

    for lag in lags:
        # Number of non-overlapping sub-series of length `lag`
        num_subseries = n // lag
        if num_subseries < 1:
            continue

        rs_list = []
        for i in range(num_subseries):
            subseries = values[i * lag : (i + 1) * lag]
            mean_sub = np.mean(subseries)
            deviations = subseries - mean_sub
            cumulative_dev = np.cumsum(deviations)

            R = np.max(cumulative_dev) - np.min(cumulative_dev)  # range
            S = np.std(subseries, ddof=1)  # standard deviation

            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            lag_values.append(lag)

    if len(lag_values) < 2:
        return 0.5

    # Hurst exponent = slope of log(R/S) vs log(lag)
    log_lags = np.log(np.array(lag_values, dtype=np.float64))
    log_rs = np.log(np.array(rs_values, dtype=np.float64))

    # Ordinary least squares for slope
    slope, _ = np.polyfit(log_lags, log_rs, 1)

    # Clip to valid range
    return float(np.clip(slope, 0.0, 1.0))


# =========================================================================
# 4. Permutation Entropy
#    Domain: information theory, neuroscience (EEG analysis)
# =========================================================================

def permutation_entropy(
    series: pd.Series,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute the permutation entropy of a time series.

    Introduced by Bandt & Pompe (2002), permutation entropy measures the
    complexity of a time series by examining the order patterns (ordinal
    patterns) of consecutive values. It is widely used in neuroscience
    to analyse EEG signals -- for example, to distinguish conscious from
    unconscious brain states, or to detect the onset of epileptic
    seizures.

    In a trading context:

    * **Low entropy** -- the series has predictable, structured patterns.
      A well-tuned strategy may exploit this structure.
    * **High entropy** (close to 1 when normalised) -- the series is
      effectively random. Any apparent pattern is likely noise;
      discretion suggests sitting out.

    Parameters
    ----------
    order : int, default ``3``
        Embedding dimension (length of ordinal patterns). ``3`` gives
        ``3! = 6`` possible patterns; ``5`` gives ``120``.
    delay : int, default ``1``
        Time delay between elements in each pattern.
    normalize : bool, default ``True``
        If True, normalise by ``log2(order!)`` so the result lies in
        ``[0, 1]``.

    Returns
    -------
    float
        Permutation entropy (normalised to [0, 1] if ``normalize=True``).

    References
    ----------
    Bandt, C. & Pompe, B. (2002). "Permutation entropy: a natural
    complexity measure for time series." *Physical Review Letters*,
    88(17), 174102.
    """
    values = series.dropna().values.astype(np.float64)
    n = len(values)

    required_length = (order - 1) * delay + 1
    if n < required_length:
        logger.warning(
            "Series too short (%d) for permutation entropy with order=%d, "
            "delay=%d (need at least %d). Returning NaN.",
            n, order, delay, required_length,
        )
        return float("nan")

    # Build ordinal patterns
    num_patterns = n - (order - 1) * delay
    pattern_counts: dict[tuple[int, ...], int] = {}

    for i in range(num_patterns):
        # Extract the sub-sequence
        indices = [i + k * delay for k in range(order)]
        window = values[indices]
        # The ordinal pattern is the rank order of the values
        pattern = tuple(np.argsort(window).tolist())
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Convert counts to probabilities
    total = sum(pattern_counts.values())
    probabilities = np.array([c / total for c in pattern_counts.values()])

    # Shannon entropy in bits (base 2)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            entropy /= max_entropy

    return float(entropy)


# =========================================================================
# 5. Empirical Mode Decomposition (EMD)
#    Domain: NASA signal processing (Hilbert-Huang Transform)
# =========================================================================

def emd_decompose(
    series: pd.Series,
    max_imfs: int = 5,
) -> List[pd.Series]:
    """Decompose a time series into Intrinsic Mode Functions (IMFs) using
    Empirical Mode Decomposition.

    EMD was developed by Norden Huang and colleagues at NASA in 1998 as
    part of the Hilbert-Huang Transform (HHT). Unlike Fourier analysis
    (which assumes stationarity and linearity), EMD is fully
    *data-driven* and handles **non-stationary, non-linear** signals --
    making it ideal for financial time series which are neither
    stationary nor linear.

    The decomposition produces a set of Intrinsic Mode Functions (IMFs)
    ordered from highest frequency to lowest:

    * **IMF 1** -- fastest oscillation, typically market microstructure
      noise. Usually discard.
    * **IMF 2-3** -- dominant market cycles. These are often the most
      useful for trading signals.
    * **IMF N** (last / residual) -- the slow-moving underlying trend.

    Trading application: build features or generate signals from IMFs
    2-3 (the dominant cycle) rather than from raw price, which is
    contaminated by noise (IMF 1).

    Parameters
    ----------
    series : pd.Series
        The time series to decompose.
    max_imfs : int, default ``5``
        Maximum number of IMFs to extract (including the residual).

    Returns
    -------
    list of pd.Series
        List of IMFs ordered from highest frequency to lowest. The last
        element is the residual (trend). All share the original index.

    References
    ----------
    Huang, N.E. et al. (1998). "The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis." *Proceedings of the Royal Society of London A*, 454,
    903-995.
    """
    values = series.dropna().values.astype(np.float64).copy()
    n = len(values)
    index = series.dropna().index

    if n < 10:
        logger.warning("Series too short for EMD; returning as single IMF.")
        return [pd.Series(values, index=index, name="imf_0")]

    imfs: list[np.ndarray] = []
    residual = values.copy()

    for imf_idx in range(max_imfs - 1):
        # The current candidate IMF starts as the residual
        h = residual.copy()
        converged = False

        # Sifting process
        for _ in range(200):  # max sifting iterations
            # Find local maxima and minima
            maxima_idx = _find_local_extrema(h, mode="max")
            minima_idx = _find_local_extrema(h, mode="min")

            # Need at least 2 of each to form envelopes
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                converged = False
                break

            # Interpolate upper and lower envelopes
            x_axis = np.arange(n)
            upper_env = np.interp(x_axis, maxima_idx, h[maxima_idx])
            lower_env = np.interp(x_axis, minima_idx, h[minima_idx])

            mean_env = (upper_env + lower_env) / 2.0
            prev_h = h.copy()
            h = h - mean_env

            # Cauchy convergence criterion (SD < 0.3 is standard)
            sd = np.sum((prev_h - h) ** 2) / (np.sum(prev_h ** 2) + 1e-12)
            if sd < 0.3:
                converged = True
                break

        if converged and np.std(h) > 1e-10:
            imfs.append(h)
            residual = residual - h
        else:
            # Cannot extract more IMFs
            break

    # The residual is the final component (the trend)
    imfs.append(residual)

    return [
        pd.Series(imf, index=index, name=f"imf_{i}")
        for i, imf in enumerate(imfs)
    ]


def _find_local_extrema(signal: np.ndarray, mode: str = "max") -> np.ndarray:
    """Find indices of local maxima or minima in a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D array.
    mode : str
        ``"max"`` for local maxima, ``"min"`` for local minima.

    Returns
    -------
    np.ndarray
        Integer indices of the detected extrema.
    """
    n = len(signal)
    if n < 3:
        return np.array([], dtype=int)

    # Compute differences
    d = np.diff(signal)

    indices = []
    for i in range(len(d) - 1):
        if mode == "max":
            if d[i] > 0 and d[i + 1] < 0:
                indices.append(i + 1)
            elif d[i] > 0 and d[i + 1] == 0:
                # Plateau -- walk forward to find the end
                j = i + 1
                while j < len(d) and d[j] == 0:
                    j += 1
                if j < len(d) and d[j] < 0:
                    indices.append((i + 1 + j) // 2)
        else:  # min
            if d[i] < 0 and d[i + 1] > 0:
                indices.append(i + 1)
            elif d[i] < 0 and d[i + 1] == 0:
                j = i + 1
                while j < len(d) and d[j] == 0:
                    j += 1
                if j < len(d) and d[j] > 0:
                    indices.append((i + 1 + j) // 2)

    # Include endpoints to prevent envelope edge effects
    result = np.array(indices, dtype=int)

    if len(result) > 0:
        # Prepend index 0 and append index n-1 to anchor the envelopes
        if result[0] != 0:
            result = np.concatenate([[0], result])
        if result[-1] != n - 1:
            result = np.concatenate([result, [n - 1]])
    else:
        result = np.array([0, n - 1], dtype=int)

    return result


# =========================================================================
# 6. Convenience: add all signal features to a DataFrame
# =========================================================================

def add_signal_features(
    df: pd.DataFrame,
    close_col: str = "close",
    hurst_window: int = 100,
    entropy_window: int = 50,
    entropy_order: int = 3,
) -> pd.DataFrame:
    """Add signal-processing features to a DataFrame of OHLCV data.

    This is a convenience function that computes and appends the
    following columns:

    * ``close_denoised`` -- wavelet-denoised close price
    * ``close_kalman`` -- Kalman-filtered close price estimate
    * ``kalman_slope`` -- Kalman trend slope (positive = uptrend)
    * ``hurst`` -- rolling Hurst exponent (H > 0.5 = trending)
    * ``entropy`` -- rolling permutation entropy (low = structured)

    These become features for downstream strategies and meta-labeling
    models.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column named ``close_col`` (default ``"close"``).
    close_col : str, default ``"close"``
        Name of the close price column.
    hurst_window : int, default ``100``
        Rolling window size for the Hurst exponent calculation.
    entropy_window : int, default ``50``
        Rolling window size for the permutation entropy calculation.
    entropy_order : int, default ``3``
        Embedding dimension for permutation entropy.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with the new feature columns added.

    Raises
    ------
    KeyError
        If ``close_col`` is not found in the DataFrame.
    """
    if close_col not in df.columns:
        raise KeyError(
            f"Column '{close_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()
    close = result[close_col]

    # --- Wavelet denoised close ---
    try:
        result["close_denoised"] = wavelet_denoise(close)
    except Exception as exc:
        logger.warning("Wavelet denoising failed: %s. Skipping.", exc)
        result["close_denoised"] = close.copy()

    # --- Kalman filtered close + slope ---
    try:
        kf = KalmanTrendFilter(observation_noise=1.0, process_noise=0.01)
        result["close_kalman"] = kf.filter(close)
        slope = kf.get_trend_slope()
        result["kalman_slope"] = slope.reindex(result.index)
    except Exception as exc:
        logger.warning("Kalman filtering failed: %s. Skipping.", exc)
        result["close_kalman"] = close.copy()
        result["kalman_slope"] = 0.0

    # --- Rolling Hurst exponent ---
    try:
        hurst_values = []
        close_clean = close.dropna()
        for i in range(len(close_clean)):
            if i < hurst_window - 1:
                hurst_values.append(np.nan)
            else:
                window_data = close_clean.iloc[i - hurst_window + 1 : i + 1]
                h = hurst_exponent(window_data, max_lag=hurst_window // 2)
                hurst_values.append(h)
        hurst_series = pd.Series(
            hurst_values, index=close_clean.index, name="hurst"
        )
        result["hurst"] = hurst_series.reindex(result.index)
    except Exception as exc:
        logger.warning("Hurst exponent calculation failed: %s. Skipping.", exc)
        result["hurst"] = np.nan

    # --- Rolling Permutation Entropy ---
    try:
        entropy_values = []
        close_clean = close.dropna()
        for i in range(len(close_clean)):
            if i < entropy_window - 1:
                entropy_values.append(np.nan)
            else:
                window_data = close_clean.iloc[i - entropy_window + 1 : i + 1]
                e = permutation_entropy(
                    window_data, order=entropy_order, normalize=True
                )
                entropy_values.append(e)
        entropy_series = pd.Series(
            entropy_values, index=close_clean.index, name="entropy"
        )
        result["entropy"] = entropy_series.reindex(result.index)
    except Exception as exc:
        logger.warning("Permutation entropy failed: %s. Skipping.", exc)
        result["entropy"] = np.nan

    logger.info(
        "Added signal features: close_denoised, close_kalman, "
        "kalman_slope, hurst, entropy."
    )
    return result
