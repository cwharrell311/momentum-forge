"""
HMM-Based Regime Detection — proper statistical regime classification.

Upgrades from the simple efficiency ratio filter to a Gaussian Hidden
Markov Model that learns regime structure from the data itself.

Two-state model:
- State 0: Low-volatility / trending (favorable for momentum)
- State 1: High-volatility / choppy (unfavorable, reduce exposure)

The HMM learns:
- Mean returns per state
- Volatility per state
- Transition probabilities between states

This is strictly better than the efficiency ratio because:
1. It learns the regime structure from actual return distributions
2. It captures transition dynamics (persistence, switching rates)
3. It provides probability estimates, not binary on/off
4. It adapts to the specific asset being traded

Soft dependency on hmmlearn — falls back to efficiency ratio if not installed.

Reference: "Detecting Regimes in Financial Markets" (Bulla & Bulla, 2006)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger("forge.regime_hmm")


class Regime(Enum):
    """Market regime classification."""
    TRENDING = "trending"     # Low-vol, directional — momentum works
    MEAN_REVERTING = "mean_reverting"  # Range-bound — reversion works
    VOLATILE = "volatile"     # High-vol, unstable — reduce exposure


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: Regime
    confidence: float            # 0-1, how confident in the classification
    trending_prob: float         # P(trending)
    mean_reverting_prob: float   # P(mean-reverting)
    volatile_prob: float         # P(volatile / crisis)
    state_duration: int          # Bars in current state


class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.

    Fits a 2-3 state Gaussian HMM on returns and classifies each bar
    into a market regime.

    Usage:
        detector = HMMRegimeDetector()
        detector.fit(returns)
        state = detector.get_regime(bar_idx)
        if state.regime == Regime.TRENDING:
            # Execute momentum strategy
    """

    def __init__(
        self,
        n_states: int = 2,
        lookback: int = 252,
        min_observations: int = 60,
        retrain_every: int = 50,
    ):
        self.n_states = n_states
        self.lookback = lookback
        self.min_observations = min_observations
        self.retrain_every = retrain_every

        self._model = None
        self._states: np.ndarray | None = None
        self._state_means: np.ndarray | None = None
        self._state_vars: np.ndarray | None = None
        self._returns: np.ndarray | None = None
        self._last_train_idx: int = 0
        self._trending_state: int = 0  # Which HMM state maps to "trending"
        self._has_hmmlearn: bool = False

        # Check if hmmlearn is available
        try:
            from hmmlearn.hmm import GaussianHMM
            self._has_hmmlearn = True
        except ImportError:
            log.info("hmmlearn not installed — falling back to efficiency ratio regime detection")

    def fit(self, returns: pd.Series | np.ndarray) -> None:
        """
        Fit the HMM on historical returns.

        Args:
            returns: Series of daily returns (not prices).
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        self._returns = returns

        if len(returns) < self.min_observations:
            log.warning("Not enough data for HMM (%d < %d)", len(returns), self.min_observations)
            return

        if self._has_hmmlearn:
            self._fit_hmm(returns)
        else:
            self._fit_fallback(returns)

    def _fit_hmm(self, returns: np.ndarray) -> None:
        """Fit using hmmlearn GaussianHMM."""
        from hmmlearn.hmm import GaussianHMM

        # Use last `lookback` observations for fitting
        fit_data = returns[-self.lookback:] if len(returns) > self.lookback else returns
        # Remove NaN/inf
        clean = fit_data[np.isfinite(fit_data)]
        if len(clean) < self.min_observations:
            return

        X = clean.reshape(-1, 1)

        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            model.fit(X)
            self._model = model

            # Predict states for all data
            all_clean = returns[np.isfinite(returns)]
            self._states = model.predict(all_clean.reshape(-1, 1))

            # Identify which state is which by variance
            # Lower variance = trending, higher variance = volatile
            self._state_means = model.means_.flatten()
            self._state_vars = np.array([model.covars_[i][0][0] for i in range(self.n_states)])

            # State with lower variance = trending (favorable)
            self._trending_state = int(np.argmin(self._state_vars))
            self._last_train_idx = len(returns)

            log.info(
                "HMM fitted: %d states, trending_state=%d (mean=%.4f, var=%.6f), "
                "volatile_state mean=%.4f var=%.6f",
                self.n_states, self._trending_state,
                self._state_means[self._trending_state],
                self._state_vars[self._trending_state],
                self._state_means[1 - self._trending_state],
                self._state_vars[1 - self._trending_state],
            )

        except Exception as e:
            log.warning("HMM fitting failed: %s — falling back", e)
            self._fit_fallback(returns)

    def _fit_fallback(self, returns: np.ndarray) -> None:
        """
        Fallback regime detection using rolling statistics.
        Uses efficiency ratio + volatility percentile.
        """
        n = len(returns)
        self._states = np.zeros(n, dtype=int)

        window = min(20, n // 3)
        if window < 5:
            return

        for i in range(window, n):
            segment = returns[i - window:i]
            # Efficiency ratio
            prices_approx = np.cumprod(1 + segment)
            direction = abs(prices_approx[-1] - prices_approx[0])
            volatility = np.sum(np.abs(np.diff(prices_approx)))
            er = direction / volatility if volatility > 0 else 0

            # Realized vol
            rv = np.std(segment) * np.sqrt(252)

            # High ER + low vol = trending (state 0)
            # Low ER or high vol = mean-reverting/volatile (state 1)
            if er > 0.3 and rv < 0.30:
                self._states[i] = 0  # Trending
            else:
                self._states[i] = 1  # Not trending

        self._trending_state = 0
        self._state_means = np.array([np.mean(returns[self._states == 0]) if np.any(self._states == 0) else 0,
                                       np.mean(returns[self._states == 1]) if np.any(self._states == 1) else 0])
        self._state_vars = np.array([np.var(returns[self._states == 0]) if np.any(self._states == 0) else 0.01,
                                      np.var(returns[self._states == 1]) if np.any(self._states == 1) else 0.01])
        self._last_train_idx = n

    def get_regime(self, bar_idx: int) -> RegimeState:
        """
        Get the regime classification for a specific bar.

        Args:
            bar_idx: Index into the returns array.

        Returns:
            RegimeState with regime classification and confidence.
        """
        if self._states is None or bar_idx >= len(self._states):
            return RegimeState(
                regime=Regime.TRENDING,
                confidence=0.3,
                trending_prob=0.5,
                mean_reverting_prob=0.3,
                volatile_prob=0.2,
                state_duration=0,
            )

        state = self._states[bar_idx]
        is_trending = (state == self._trending_state)

        # Compute state duration
        duration = 1
        for j in range(bar_idx - 1, -1, -1):
            if self._states[j] == state:
                duration += 1
            else:
                break

        # Confidence based on how clearly separated the states are
        if self._state_vars is not None and len(self._state_vars) >= 2:
            var_ratio = max(self._state_vars) / (min(self._state_vars) + 1e-10)
            # Higher variance ratio = more clearly separated states = higher confidence
            confidence = min(0.95, 0.5 + 0.1 * np.log(var_ratio + 1))
        else:
            confidence = 0.5

        # Use HMM posterior probabilities if available
        if self._model is not None and self._returns is not None and bar_idx < len(self._returns):
            try:
                clean_returns = self._returns[np.isfinite(self._returns)]
                if bar_idx < len(clean_returns):
                    # Get posterior probability from the model
                    posteriors = self._model.predict_proba(clean_returns[:bar_idx + 1].reshape(-1, 1))
                    if len(posteriors) > 0:
                        last_posterior = posteriors[-1]
                        trending_prob = float(last_posterior[self._trending_state])
                        volatile_prob = float(last_posterior[1 - self._trending_state])
                    else:
                        trending_prob = 0.7 if is_trending else 0.3
                        volatile_prob = 1.0 - trending_prob
                else:
                    trending_prob = 0.7 if is_trending else 0.3
                    volatile_prob = 1.0 - trending_prob
            except Exception:
                trending_prob = 0.7 if is_trending else 0.3
                volatile_prob = 1.0 - trending_prob
        else:
            trending_prob = 0.7 if is_trending else 0.3
            volatile_prob = 1.0 - trending_prob

        # Classify regime
        if trending_prob > 0.6:
            regime = Regime.TRENDING
        elif volatile_prob > 0.7:
            regime = Regime.VOLATILE
        else:
            regime = Regime.MEAN_REVERTING

        return RegimeState(
            regime=regime,
            confidence=round(confidence, 3),
            trending_prob=round(trending_prob, 3),
            mean_reverting_prob=round(max(0, 1 - trending_prob - volatile_prob * 0.3), 3),
            volatile_prob=round(volatile_prob, 3),
            state_duration=duration,
        )

    def is_favorable(self, bar_idx: int, strategy_type: str = "momentum") -> bool:
        """
        Quick check: is the current regime favorable for the strategy type?

        Args:
            bar_idx: Index into the returns array.
            strategy_type: "momentum" or "mean_reversion".

        Returns:
            True if the regime is favorable for the strategy type.
        """
        state = self.get_regime(bar_idx)

        if strategy_type == "momentum":
            # Momentum works in trending regimes
            return state.regime == Regime.TRENDING and state.trending_prob > 0.55
        elif strategy_type == "mean_reversion":
            # Mean reversion works in choppy/range-bound regimes
            return state.regime == Regime.MEAN_REVERTING
        else:
            # Default: avoid volatile regimes
            return state.regime != Regime.VOLATILE


def fit_regime_detector(
    df: pd.DataFrame,
    n_states: int = 2,
    lookback: int = 252,
) -> HMMRegimeDetector:
    """
    Convenience function: fit an HMM regime detector on OHLCV data.

    Args:
        df: OHLCV DataFrame.
        n_states: Number of hidden states.
        lookback: Lookback window for fitting.

    Returns:
        Fitted HMMRegimeDetector.
    """
    returns = df["close"].pct_change().dropna()
    detector = HMMRegimeDetector(n_states=n_states, lookback=lookback)
    detector.fit(returns)
    return detector
