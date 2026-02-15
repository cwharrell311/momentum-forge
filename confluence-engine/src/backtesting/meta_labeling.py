"""
Meta-Labeling — learn which signals to trust and how much to bet.

From Marcos Lopez de Prado, "Advances in Financial Machine Learning" (Ch. 3).

This is a 3-stage architecture that separates signal generation from signal
quality prediction:

    Stage 1: Primary model generates directional signals (our existing strategies).
             These determine SIDE (long/short) — they answer "which direction?"

    Stage 2: Secondary model (the meta-labeler) predicts P(primary signal is correct).
             It answers "should we take this trade?" using features extracted from
             the market state at signal time.

    Stage 3: Position size = f(P(correct)). Higher confidence = larger position.

Why this matters:
- Most strategies generate ~55-60% accurate signals at best
- The OTHER 40-45% are false signals that destroy alpha through losses + costs
- If we can predict WHICH signals will work, we filter out the losers
- Even modest filtering (blocking the worst 20% of signals) dramatically
  improves Sharpe ratio

The meta-labeler trains online: after each completed trade, we record whether
the primary signal was profitable, along with the market features at signal
time. Once we have enough data (min_samples), we start predicting confidence
for new signals and blocking low-confidence ones.

Features are all derived from OHLCV data — no external dependencies.
Uses sklearn LogisticRegression when available, falls back to a pure-numpy
logistic regression with SGD otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("forge.meta_labeling")

# ── Soft dependency on sklearn ──
# Use sklearn LogisticRegression as primary classifier.
# Fall back to a pure-numpy implementation if sklearn is not installed.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    log.info("sklearn not available — using pure-numpy logistic regression fallback")


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════


# Canonical ordering of meta-label features.  Every feature dict produced
# by extract_meta_features uses these exact keys in this order so that
# vectors fed to the classifier are always aligned.
FEATURE_NAMES: list[str] = [
    "volatility_20d",
    "volatility_ratio",
    "volume_ratio",
    "rsi_14",
    "atr_pct",
    "momentum_5d",
    "momentum_20d",
    "efficiency_ratio",
    "bar_return",
    "high_low_range",
    "close_position",
]


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI from the last `period + 1` closing prices.

    Uses the standard exponential-smoothing (Wilder) method:
        RS  = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Returns 50.0 (neutral) if there is not enough data.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def extract_meta_features(
    df: pd.DataFrame,
    bar_idx: int,
    lookback: int = 20,
) -> dict[str, float]:
    """Extract features that predict whether a trade signal will be profitable.

    All features are computed from raw OHLCV data with no external
    dependencies.  They capture the market micro-structure at the moment
    a signal fires: volatility regime, volume context, momentum state,
    and intra-bar dynamics.

    Args:
        df: DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume``.  Index is timestamps.
        bar_idx: Integer position of the signal bar in *df*.
        lookback: Number of bars to use for rolling calculations.
            Minimum is 20 (required for 20-day volatility).

    Returns:
        Dictionary mapping feature name -> value.
        If there is insufficient history (fewer than *lookback* bars before
        *bar_idx*), features that cannot be computed are set to 0.0.
    """
    lookback = max(lookback, 20)  # Need at least 20 bars for vol calculations

    features: dict[str, float] = {}

    # ── Guard: enough history? ──────────────────────────────────
    # We need `lookback` bars of history *before* bar_idx.
    start = max(0, bar_idx - lookback)
    window = df.iloc[start: bar_idx + 1]

    closes = window["close"].values.astype(float)
    highs = window["high"].values.astype(float)
    lows = window["low"].values.astype(float)
    volumes = window["volume"].values.astype(float)

    n = len(closes)
    current_close = closes[-1] if n > 0 else 0.0

    if n < 2 or current_close <= 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # ── Volatility features ─────────────────────────────────────

    # Daily log returns for the window
    log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))

    # 20-day realized volatility (annualized, using available bars)
    vol_20 = float(np.std(log_returns) * math.sqrt(252)) if len(log_returns) > 1 else 0.0
    features["volatility_20d"] = vol_20

    # Volatility ratio: 5d vol / 20d vol
    # > 1 means vol is expanding; < 1 means contracting
    if len(log_returns) >= 5:
        vol_5 = float(np.std(log_returns[-5:]) * math.sqrt(252))
        features["volatility_ratio"] = vol_5 / vol_20 if vol_20 > 0 else 1.0
    else:
        features["volatility_ratio"] = 1.0

    # ── Volume features ─────────────────────────────────────────

    avg_volume = float(np.mean(volumes)) if n > 0 else 1.0
    current_volume = volumes[-1] if n > 0 else 0.0
    features["volume_ratio"] = current_volume / avg_volume if avg_volume > 0 else 1.0

    # ── RSI ──────────────────────────────────────────────────────

    features["rsi_14"] = _compute_rsi(closes, period=14)

    # ── ATR as % of price (normalized volatility) ────────────────

    if n >= 2:
        tr_values = []
        for j in range(1, n):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j - 1]),
                abs(lows[j] - closes[j - 1]),
            )
            tr_values.append(tr)
        atr = float(np.mean(tr_values[-14:])) if tr_values else 0.0
        features["atr_pct"] = atr / current_close * 100.0 if current_close > 0 else 0.0
    else:
        features["atr_pct"] = 0.0

    # ── Momentum features ────────────────────────────────────────

    # 5-day price momentum (return over last 5 bars)
    if n > 5:
        features["momentum_5d"] = (closes[-1] - closes[-6]) / closes[-6]
    else:
        features["momentum_5d"] = 0.0

    # 20-day price momentum
    if n > 20:
        features["momentum_20d"] = (closes[-1] - closes[-21]) / closes[-21]
    elif n > 1:
        features["momentum_20d"] = (closes[-1] - closes[0]) / closes[0]
    else:
        features["momentum_20d"] = 0.0

    # ── Efficiency ratio: direction / path length ────────────────
    # High ER = trending cleanly; low ER = choppy
    direction = abs(closes[-1] - closes[0])
    path_length = float(np.sum(np.abs(np.diff(closes))))
    features["efficiency_ratio"] = direction / path_length if path_length > 0 else 0.0

    # ── Bar-level features ───────────────────────────────────────

    # Current bar's return
    if n >= 2:
        features["bar_return"] = (closes[-1] - closes[-2]) / closes[-2]
    else:
        features["bar_return"] = 0.0

    # High-low range as % of close (intrabar volatility)
    current_high = highs[-1]
    current_low = lows[-1]
    features["high_low_range"] = (
        (current_high - current_low) / current_close if current_close > 0 else 0.0
    )

    # Close position within the high-low range (0 = at low, 1 = at high)
    hl_range = current_high - current_low
    if hl_range > 0:
        features["close_position"] = (current_close - current_low) / hl_range
    else:
        features["close_position"] = 0.5  # No range = midpoint

    return features


def _features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a numpy array in canonical order.

    Always uses ``FEATURE_NAMES`` ordering so that training data and
    prediction inputs are aligned.
    """
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# PURE-NUMPY LOGISTIC REGRESSION FALLBACK
# ═══════════════════════════════════════════════════════════════


class _NumpyLogisticRegression:
    """Minimal logistic regression trained with SGD.

    This is the fallback classifier when sklearn is not installed.
    It implements a simple online logistic regression with stochastic
    gradient descent, L2 regularization, and feature standardization.

    Not meant to compete with sklearn's optimized solver — just needs
    to be "good enough" to separate profitable from unprofitable signals,
    which is a relatively easy binary classification task.
    """

    def __init__(self, learning_rate: float = 0.01, reg_lambda: float = 0.01):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        # Running mean/std for feature standardization
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        """Train the model on the full dataset.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (n_samples,), values in {0, 1}.
            epochs: Number of full passes over the data.
            batch_size: Mini-batch size for SGD.
        """
        n_samples, n_features = X.shape

        # Standardize features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        # Prevent division by zero for constant features
        self._std[self._std < 1e-10] = 1.0
        X_std = (X - self._mean) / self._std

        # Initialize weights (Xavier initialization)
        if self.weights is None or len(self.weights) != n_features:
            self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
            self.bias = 0.0

        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_std[batch_idx]
                y_batch = y[batch_idx]

                # Forward pass
                z = X_batch @ self.weights + self.bias
                predictions = self._sigmoid(z)

                # Gradients (binary cross-entropy + L2)
                error = predictions - y_batch
                grad_w = (X_batch.T @ error) / len(y_batch) + self.reg_lambda * self.weights
                grad_b = float(np.mean(error))

                # SGD update
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(y=1) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with columns [P(y=0), P(y=1)].
        """
        if not self._fitted or self.weights is None or self._mean is None:
            # Not fitted yet — return 0.5 (maximum uncertainty)
            n = X.shape[0] if X.ndim > 1 else 1
            return np.full((n, 2), 0.5)

        X_std = (X - self._mean) / self._std
        z = X_std @ self.weights + self.bias
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])


# ═══════════════════════════════════════════════════════════════
# ONLINE META-LABELER
# ═══════════════════════════════════════════════════════════════


@dataclass
class _TrainingSample:
    """One (features, outcome) pair for meta-labeler training."""
    features: np.ndarray        # Feature vector in canonical order
    was_profitable: bool        # Did the primary signal make money?


class MetaLabeler:
    """Online meta-labeler that learns to filter false signals.

    The meta-labeler sits between the primary strategy (which decides
    direction) and the position sizer (which decides how much to bet).
    It answers: "Given the current market state, what is the probability
    that this signal is correct?"

    Workflow:
        1. Strategy fires a signal (long/short).
        2. Extract features from the bar where the signal fired.
        3. Ask meta-labeler: ``should_trade(features)``
        4. If yes, enter the trade.  ``predict_confidence(features)``
           gives a probability that can scale position size.
        5. When the trade closes, call ``record_outcome(features, was_profitable)``
           to train the meta-labeler.

    After ``min_samples`` outcomes are recorded, the meta-labeler trains
    a classifier and starts making predictions.  It retrains every
    ``retrain_every`` new samples to stay current.

    Args:
        min_samples: Minimum number of recorded outcomes before the
            meta-labeler starts predicting.  Below this threshold,
            ``predict_confidence`` returns 0.5 and ``should_trade``
            returns True (pass all signals through).
        retrain_every: Retrain the classifier after this many new samples
            are added.  Balances staying current vs. computational cost.
    """

    def __init__(self, min_samples: int = 30, retrain_every: int = 10):
        self.min_samples = min_samples
        self.retrain_every = retrain_every

        # Growing dataset of (features, outcome) pairs
        self._samples: list[_TrainingSample] = []
        self._samples_since_train: int = 0

        # Classifier — initialized on first train
        self._model: object | None = None
        self._scaler: object | None = None  # sklearn StandardScaler
        self._is_trained: bool = False

        # Track strategy name for logging
        self._strategy_name: str = "unknown"

        # Performance tracking
        self._total_predictions: int = 0
        self._correct_predictions: int = 0

    @property
    def n_samples(self) -> int:
        """Number of recorded training samples."""
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        """Whether the meta-labeler has enough data to make predictions."""
        return self._is_trained and self.n_samples >= self.min_samples

    @property
    def accuracy(self) -> float:
        """Running accuracy of meta-labeler predictions (0-1).

        Only meaningful after the meta-labeler starts predicting.
        """
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def record_outcome(self, features: dict[str, float], was_profitable: bool) -> None:
        """Record the outcome of a completed trade for future training.

        Call this after every trade closes, passing the features extracted
        at signal time and whether the trade made money.

        Args:
            features: Feature dict from ``extract_meta_features``.
            was_profitable: True if the trade's P&L was positive.
        """
        feature_vec = _features_to_array(features)
        self._samples.append(_TrainingSample(features=feature_vec, was_profitable=was_profitable))
        self._samples_since_train += 1

        log.debug(
            "MetaLabeler[%s] recorded outcome: profitable=%s (total=%d)",
            self._strategy_name,
            was_profitable,
            self.n_samples,
        )

        # Retrain if we have enough samples and it's time
        if self.n_samples >= self.min_samples and self._samples_since_train >= self.retrain_every:
            self._train()

    def predict_confidence(self, features: dict[str, float]) -> float:
        """Predict P(primary signal is correct) given current market features.

        Returns a probability between 0 and 1:
        - Values near 1.0 = high confidence the signal will be profitable
        - Values near 0.5 = uncertain (coin flip)
        - Values near 0.0 = high confidence the signal will LOSE money

        Before the meta-labeler is trained (< min_samples), returns 0.5
        (maximum uncertainty — don't interfere with signals).

        Args:
            features: Feature dict from ``extract_meta_features``.

        Returns:
            P(profitable) in [0, 1].
        """
        if not self._is_trained:
            return 0.5

        feature_vec = _features_to_array(features).reshape(1, -1)

        try:
            if _HAS_SKLEARN and self._scaler is not None:
                feature_vec = self._scaler.transform(feature_vec)
                proba = self._model.predict_proba(feature_vec)  # type: ignore[union-attr]
            else:
                proba = self._model.predict_proba(feature_vec)  # type: ignore[union-attr]

            # proba shape is (1, 2): [P(unprofitable), P(profitable)]
            confidence = float(proba[0, 1])

            # Clip to [0.01, 0.99] to avoid extreme confidence
            confidence = max(0.01, min(0.99, confidence))

            return confidence

        except Exception as e:
            log.warning("MetaLabeler[%s] prediction failed: %s", self._strategy_name, e)
            return 0.5

    def should_trade(self, features: dict[str, float], threshold: float = 0.55) -> bool:
        """Decide whether to take a trade based on meta-label confidence.

        This is the core signal filter.  If the meta-labeler predicts that
        the primary signal has less than ``threshold`` probability of being
        correct, block the trade.

        Before the meta-labeler is trained, always returns True (don't
        block any signals until we have data to learn from).

        Args:
            features: Feature dict from ``extract_meta_features``.
            threshold: Minimum P(correct) to allow the trade.
                Default 0.55 — only slight edge needed, but filters out
                the clearly bad signals.

        Returns:
            True if the trade should be taken, False if it should be blocked.
        """
        if not self._is_trained:
            return True

        confidence = self.predict_confidence(features)
        take_trade = confidence >= threshold

        if not take_trade:
            log.info(
                "MetaLabeler[%s] BLOCKED signal: confidence=%.3f < threshold=%.3f",
                self._strategy_name,
                confidence,
                threshold,
            )

        return take_trade

    def update_prediction_tracking(self, predicted_profitable: bool, actual_profitable: bool) -> None:
        """Track meta-labeler prediction accuracy for monitoring.

        Call this after a trade closes to update the running accuracy.
        This is separate from ``record_outcome`` because you may want
        to track accuracy even for trades that were blocked (by checking
        what would have happened).

        Args:
            predicted_profitable: What the meta-labeler predicted.
            actual_profitable: What actually happened.
        """
        self._total_predictions += 1
        if predicted_profitable == actual_profitable:
            self._correct_predictions += 1

    def get_stats(self) -> dict[str, float | int | bool]:
        """Return diagnostic statistics about the meta-labeler state.

        Useful for dashboards and logging.
        """
        if self.n_samples > 0:
            win_rate_in_data = sum(
                1 for s in self._samples if s.was_profitable
            ) / self.n_samples
        else:
            win_rate_in_data = 0.0

        return {
            "strategy": self._strategy_name,
            "n_samples": self.n_samples,
            "is_ready": self.is_ready,
            "min_samples": self.min_samples,
            "win_rate_in_data": round(win_rate_in_data, 3),
            "meta_accuracy": round(self.accuracy, 3),
            "total_predictions": self._total_predictions,
            "correct_predictions": self._correct_predictions,
            "has_sklearn": _HAS_SKLEARN,
        }

    # ── Private: training ────────────────────────────────────────

    def _train(self) -> None:
        """Train (or retrain) the classifier on all recorded samples.

        Uses sklearn LogisticRegression when available, otherwise falls
        back to the pure-numpy implementation.
        """
        if self.n_samples < self.min_samples:
            return

        # Build training matrices
        X = np.array([s.features for s in self._samples])
        y = np.array([1.0 if s.was_profitable else 0.0 for s in self._samples])

        # Check for degenerate cases: all same label
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            log.warning(
                "MetaLabeler[%s] all %d samples have same outcome (%s) — skipping training",
                self._strategy_name,
                self.n_samples,
                "profitable" if y[0] == 1 else "unprofitable",
            )
            return

        try:
            if _HAS_SKLEARN:
                self._train_sklearn(X, y)
            else:
                self._train_numpy(X, y)

            self._is_trained = True
            self._samples_since_train = 0

            # Log training stats
            train_predictions = self._model.predict_proba(  # type: ignore[union-attr]
                self._scaler.transform(X) if (self._scaler is not None and _HAS_SKLEARN) else X
            )
            train_accuracy = float(np.mean(
                (train_predictions[:, 1] >= 0.5) == (y == 1)
            ))

            log.info(
                "MetaLabeler[%s] trained on %d samples — train accuracy=%.1f%%, "
                "base rate=%.1f%% (sklearn=%s)",
                self._strategy_name,
                self.n_samples,
                train_accuracy * 100,
                float(np.mean(y)) * 100,
                _HAS_SKLEARN,
            )

        except Exception as e:
            log.error("MetaLabeler[%s] training failed: %s", self._strategy_name, e)

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train using sklearn LogisticRegression."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            C=1.0,                      # Regularization strength (inverse)
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",    # Handle class imbalance
            random_state=42,
        )
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler

    def _train_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train using pure-numpy logistic regression fallback."""
        model = _NumpyLogisticRegression(learning_rate=0.01, reg_lambda=0.01)
        model.fit(X, y, epochs=200, batch_size=min(32, len(y)))

        self._model = model
        self._scaler = None  # Numpy model does its own standardization


# ═══════════════════════════════════════════════════════════════
# INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════


# Registry of meta-labelers by strategy name so each strategy
# gets its own independent classifier.
_meta_labeler_registry: dict[str, MetaLabeler] = {}


def create_meta_labeler_for_strategy(
    strategy_name: str,
    min_samples: int = 30,
    retrain_every: int = 10,
) -> MetaLabeler:
    """Get or create a MetaLabeler instance for a specific strategy.

    Each strategy gets its own meta-labeler because the features that
    predict signal quality are strategy-dependent.  A momentum strategy's
    false signals look very different from a mean-reversion strategy's
    false signals.

    Repeated calls with the same ``strategy_name`` return the SAME
    instance, preserving learned state across backtest runs and live
    trading sessions.

    Args:
        strategy_name: Unique identifier for the strategy (e.g.,
            ``"dual_momentum"``, ``"crypto_mean_reversion"``).
        min_samples: Minimum outcomes before the labeler starts predicting.
        retrain_every: Retrain after this many new outcomes.

    Returns:
        A ``MetaLabeler`` instance bound to this strategy.

    Example::

        labeler = create_meta_labeler_for_strategy("opening_range_breakout")

        # On signal:
        features = extract_meta_features(df, bar_idx=42)
        if labeler.should_trade(features, threshold=0.55):
            confidence = labeler.predict_confidence(features)
            position_size = base_size * confidence
            # ... enter trade ...

        # After trade closes:
        labeler.record_outcome(features, was_profitable=True)
    """
    if strategy_name in _meta_labeler_registry:
        return _meta_labeler_registry[strategy_name]

    labeler = MetaLabeler(min_samples=min_samples, retrain_every=retrain_every)
    labeler._strategy_name = strategy_name
    _meta_labeler_registry[strategy_name] = labeler

    log.info(
        "Created MetaLabeler for strategy '%s' (min_samples=%d, retrain_every=%d, sklearn=%s)",
        strategy_name,
        min_samples,
        retrain_every,
        _HAS_SKLEARN,
    )

    return labeler


def reset_meta_labeler_registry() -> None:
    """Clear all registered meta-labelers.

    Useful for testing or when starting a fresh backtest session where
    you don't want to carry over learned state from prior runs.
    """
    _meta_labeler_registry.clear()
    log.info("Meta-labeler registry cleared")
