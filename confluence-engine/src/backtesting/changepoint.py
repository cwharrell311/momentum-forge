"""
Bayesian Online Change Point Detection (BOCPD).

Supplements the HMM regime detector with an online algorithm that detects
when the data-generating process changes. While HMM classifies the current
regime (trending vs mean-reverting), BOCPD answers a different question:
"Has a structural change just occurred?"

This is valuable for:
- Resetting strategy parameters after a regime change
- Reducing position size during transition periods
- Triggering meta-labeler retraining

Algorithm: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
- Maintains a run length distribution P(r_t | x_{1:t})
- At each step, either the run continues (r_t = r_{t-1} + 1)
  or a changepoint occurs (r_t = 0)
- Uses a conjugate prior (Normal-Inverse-Gamma) for efficiency
- Hazard function h(r) = 1/lambda controls expected run length

The detector is O(1) per step when using a pruned run-length distribution
(we keep only the top K run lengths).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("forge.changepoint")


@dataclass
class ChangePointState:
    """State of the change point detector at a single time step."""
    is_changepoint: bool        # True if a changepoint was detected
    changepoint_prob: float     # P(r_t = 0) — probability of changepoint
    run_length: int             # Most probable current run length
    run_length_mean: float      # Expected run length
    confidence: float           # 1 - entropy of run length distribution


class BayesianChangepointDetector:
    """
    Online Bayesian changepoint detection (Adams & MacKay 2007).

    Uses Normal-Inverse-Gamma conjugate prior for Gaussian observations.
    Detects changes in mean and/or variance of the return series.

    Args:
        hazard_lambda: Expected run length before a change. Higher = fewer
            changepoints detected. Default 250 (~1 year of trading days).
        threshold: Minimum P(changepoint) to flag. Default 0.3.
        max_run_length: Maximum tracked run lengths (memory limit). Default 500.
        mu0: Prior mean. Default 0.0 (centered returns).
        kappa0: Prior strength on mean. Default 1.0.
        alpha0: Prior shape for variance. Default 1.0.
        beta0: Prior scale for variance. Default 0.01.
    """

    def __init__(
        self,
        hazard_lambda: float = 250.0,
        threshold: float = 0.3,
        max_run_length: int = 500,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 0.01,
    ):
        self.hazard = 1.0 / hazard_lambda
        self.threshold = threshold
        self.max_run_length = max_run_length

        # Prior hyperparameters
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Run length distribution: P(r_t | x_{1:t})
        # Index i represents run length i
        self.run_length_dist = np.array([1.0])  # Start with r_0 = 0, P = 1

        # Sufficient statistics for each run length
        # These are the posterior parameters of the NIG distribution
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

        self._step = 0

    def update(self, x: float) -> ChangePointState:
        """
        Process a new observation and return changepoint state.

        Args:
            x: New observation (typically a log return).

        Returns:
            ChangePointState with detection result.
        """
        self._step += 1

        # ── 1. Evaluate predictive probability P(x_t | r_{t-1}) ──
        # Student-t predictive distribution for each run length
        pred_probs = self._predictive_prob(x)

        # ── 2. Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t}) ──
        growth = self.run_length_dist * pred_probs * (1.0 - self.hazard)

        # ── 3. Changepoint probability: P(r_t = 0, x_{1:t}) ──
        cp = np.sum(self.run_length_dist * pred_probs * self.hazard)

        # ── 4. New joint distribution ──
        new_dist = np.empty(len(growth) + 1)
        new_dist[0] = cp
        new_dist[1:] = growth

        # Normalize
        evidence = new_dist.sum()
        if evidence > 0:
            new_dist /= evidence

        # ── 5. Update sufficient statistics ──
        new_mu = np.empty(len(self.mu) + 1)
        new_kappa = np.empty(len(self.kappa) + 1)
        new_alpha = np.empty(len(self.alpha) + 1)
        new_beta = np.empty(len(self.beta) + 1)

        # Prior for the new run (after changepoint)
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0

        # Update existing runs with new observation
        new_kappa[1:] = self.kappa + 1.0
        new_mu[1:] = (self.kappa * self.mu + x) / new_kappa[1:]
        new_alpha[1:] = self.alpha + 0.5
        new_beta[1:] = self.beta + 0.5 * self.kappa * (x - self.mu) ** 2 / new_kappa[1:]

        # ── 6. Prune to max_run_length ──
        if len(new_dist) > self.max_run_length:
            # Keep top max_run_length entries, redistribute pruned mass
            keep = self.max_run_length
            pruned_mass = new_dist[keep:].sum()
            new_dist = new_dist[:keep]
            if new_dist.sum() > 0:
                new_dist *= (1.0 + pruned_mass / new_dist.sum())
                new_dist /= new_dist.sum()

            new_mu = new_mu[:keep]
            new_kappa = new_kappa[:keep]
            new_alpha = new_alpha[:keep]
            new_beta = new_beta[:keep]

        self.run_length_dist = new_dist
        self.mu = new_mu
        self.kappa = new_kappa
        self.alpha = new_alpha
        self.beta = new_beta

        # ── 7. Extract state ──
        cp_prob = float(new_dist[0])
        most_probable_rl = int(np.argmax(new_dist))
        mean_rl = float(np.sum(np.arange(len(new_dist)) * new_dist))

        # Confidence = 1 - normalized entropy of run length distribution
        entropy = -np.sum(new_dist[new_dist > 0] * np.log(new_dist[new_dist > 0]))
        max_entropy = np.log(len(new_dist)) if len(new_dist) > 1 else 1.0
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return ChangePointState(
            is_changepoint=(cp_prob >= self.threshold),
            changepoint_prob=cp_prob,
            run_length=most_probable_rl,
            run_length_mean=mean_rl,
            confidence=confidence,
        )

    def _predictive_prob(self, x: float) -> np.ndarray:
        """
        Compute Student-t predictive probability for each run length.

        P(x | r) = Student-t(x; mu_r, sigma_r^2, 2*alpha_r)
        where sigma_r^2 = beta_r * (kappa_r + 1) / (alpha_r * kappa_r)
        """
        df = 2.0 * self.alpha  # degrees of freedom
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))

        # Avoid division by zero
        scale = np.maximum(scale, 1e-10)

        # Student-t log probability
        z = (x - self.mu) / scale
        log_prob = (
            _log_gamma(0.5 * (df + 1.0))
            - _log_gamma(0.5 * df)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - 0.5 * (df + 1.0) * np.log(1.0 + z ** 2 / df)
        )

        return np.exp(log_prob)

    def reset(self):
        """Reset the detector to initial state."""
        self.run_length_dist = np.array([1.0])
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])
        self._step = 0


def _log_gamma(x: np.ndarray | float) -> np.ndarray | float:
    """Log-gamma function using scipy if available, else stirling."""
    try:
        from scipy.special import gammaln
        return gammaln(x)
    except ImportError:
        # Stirling approximation for large x
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * np.log(2 * np.pi / x) + x * (np.log(x + 1.0 / (12 * x)) - 1.0)


def detect_changepoints(
    returns: np.ndarray,
    hazard_lambda: float = 250.0,
    threshold: float = 0.3,
) -> list[ChangePointState]:
    """
    Run BOCPD on a full return series (batch mode for backtesting).

    Args:
        returns: Array of log returns.
        hazard_lambda: Expected run length between changes.
        threshold: Min P(changepoint) to flag.

    Returns:
        List of ChangePointState for each time step.
    """
    detector = BayesianChangepointDetector(
        hazard_lambda=hazard_lambda,
        threshold=threshold,
    )
    states = []
    for r in returns:
        state = detector.update(float(r))
        states.append(state)
    return states


def add_changepoint_features(df: 'pd.DataFrame', hazard_lambda: float = 250.0) -> 'pd.DataFrame':
    """
    Add changepoint columns to OHLCV DataFrame for use by strategies.

    Adds:
        - cp_prob: P(changepoint) at each bar
        - cp_detected: boolean flag
        - cp_run_length: most probable run length
        - cp_confidence: detector confidence

    Args:
        df: DataFrame with 'close' column.
        hazard_lambda: Expected run length.

    Returns:
        DataFrame with changepoint columns added.
    """
    import pandas as pd

    result = df.copy()
    closes = result["close"].values
    if len(closes) < 10:
        result["cp_prob"] = 0.0
        result["cp_detected"] = False
        result["cp_run_length"] = 0
        result["cp_confidence"] = 0.0
        return result

    # Compute log returns
    log_returns = np.diff(np.log(closes))
    log_returns = np.concatenate([[0.0], log_returns])  # pad first bar

    states = detect_changepoints(log_returns, hazard_lambda=hazard_lambda)

    result["cp_prob"] = [s.changepoint_prob for s in states]
    result["cp_detected"] = [s.is_changepoint for s in states]
    result["cp_run_length"] = [s.run_length for s in states]
    result["cp_confidence"] = [s.confidence for s in states]

    n_detected = sum(1 for s in states if s.is_changepoint)
    log.info("  BOCPD: %d changepoints detected in %d bars (lambda=%.0f)",
             n_detected, len(states), hazard_lambda)

    return result
