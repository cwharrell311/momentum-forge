"""
Feature importance ranking via permutation importance + mutual information.

Instead of a Temporal Fusion Transformer (which needs >10k samples to generalize),
this module directly measures which preprocessed features matter for each strategy
by shuffling features and measuring Sharpe degradation.

This is more honest than any neural net at daily-frequency data scales (~1250 bars).

Methods:
1. Permutation importance: shuffle feature → re-run strategy → measure Sharpe drop
2. Mutual information: statistical dependency between feature and forward returns
3. Rolling stability: does feature importance hold across time windows?

Usage:
    from src.backtesting.feature_importance import rank_features
    ranking = rank_features(strategy, df, symbol, asset_class, config)
    for feat in ranking.features[:5]:
        print(f"  {feat.name:25s} importance={feat.importance:.3f} stable={feat.stable}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("forge.features")

# Features added by preprocessing pipeline
PREPROCESSED_FEATURES = [
    # Signal processing
    "close_denoised",
    "close_kalman",
    "hurst_exponent",
    "spectral_entropy",
    # FracDiff
    "close_fracdiff",
    # HMM regime
    "regime_trending",
    "regime_confidence",
    # Multi-timeframe
    "weekly_trend",
    "weekly_momentum",
    # Cross-asset context
    "VIX_close",
    "DXY_close",
    "US10Y_close",
    "US3M_close",
    "yield_curve_spread",
]


@dataclass
class FeatureScore:
    """Importance score for a single feature."""
    name: str
    importance: float  # Mean Sharpe degradation when shuffled (higher = more important)
    std: float  # Std of importance across permutations
    mutual_info: float  # Mutual information with forward returns
    stability: float  # Consistency across rolling windows (0-1, higher = more stable)
    stable: bool  # True if feature is important AND stable

    @property
    def rank_score(self) -> float:
        """Combined score: importance * stability * (1 + MI)."""
        return self.importance * max(0.1, self.stability) * (1 + self.mutual_info)


@dataclass
class FeatureRanking:
    """Complete feature ranking for a strategy-asset pair."""
    strategy_name: str
    symbol: str
    features: list[FeatureScore] = field(default_factory=list)
    top_n: int = 5  # Recommended number of features to use
    baseline_sharpe: float = 0.0

    @property
    def top_features(self) -> list[str]:
        """Return names of top N important + stable features."""
        stable = [f for f in self.features if f.stable]
        if len(stable) >= self.top_n:
            return [f.name for f in stable[:self.top_n]]
        return [f.name for f in self.features[:self.top_n]]


def _compute_mutual_information(feature: np.ndarray, target: np.ndarray, n_bins: int = 20) -> float:
    """Compute mutual information between feature and target using histogram estimator."""
    # Remove NaN pairs
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 50:
        return 0.0

    x = feature[mask]
    y = target[mask]

    # Bin edges
    x_bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    y_bins = np.linspace(np.percentile(y, 1), np.percentile(y, 99), n_bins + 1)

    # Joint histogram
    joint, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    joint = joint / joint.sum()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    return max(0.0, mi)


def _permutation_importance(
    strategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    config,
    feature_name: str,
    n_permutations: int = 5,
    baseline_sharpe: float = 0.0,
) -> tuple[float, float]:
    """Measure Sharpe degradation when a single feature is shuffled.

    Returns (mean_degradation, std_degradation).
    """
    from src.backtesting.engine import run_backtest

    degradations = []

    for _ in range(n_permutations):
        # Create copy with shuffled feature
        df_shuffled = df.copy()
        if feature_name in df_shuffled.columns:
            df_shuffled[feature_name] = np.random.permutation(df_shuffled[feature_name].values)
        else:
            return 0.0, 0.0

        try:
            result = run_backtest(strategy, df_shuffled, symbol, asset_class, config)
            shuffled_sharpe = result.report.sharpe_ratio
            degradation = baseline_sharpe - shuffled_sharpe
            degradations.append(degradation)
        except Exception:
            degradations.append(0.0)

    return float(np.mean(degradations)), float(np.std(degradations))


def _rolling_stability(
    strategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    config,
    feature_name: str,
    n_windows: int = 3,
) -> float:
    """Measure if feature importance is consistent across time windows.

    Returns stability score 0-1 (1 = perfectly consistent importance direction).
    """
    from src.backtesting.engine import run_backtest

    window_size = len(df) // n_windows
    if window_size < 100:
        return 0.5  # Not enough data

    importances = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(df))
        window_df = df.iloc[start:end].copy()

        if len(window_df) < 100:
            continue

        try:
            # Baseline for this window
            baseline = run_backtest(strategy, window_df, symbol, asset_class, config)
            baseline_sharpe = baseline.report.sharpe_ratio

            # Shuffled
            window_shuffled = window_df.copy()
            if feature_name in window_shuffled.columns:
                window_shuffled[feature_name] = np.random.permutation(window_shuffled[feature_name].values)
            else:
                continue

            shuffled = run_backtest(strategy, window_shuffled, symbol, asset_class, config)
            degradation = baseline_sharpe - shuffled.report.sharpe_ratio
            importances.append(degradation)
        except Exception:
            continue

    if len(importances) < 2:
        return 0.5

    # Stability = fraction of windows where importance has same sign as mean
    mean_imp = np.mean(importances)
    if abs(mean_imp) < 1e-6:
        return 0.0
    same_sign = sum(1 for x in importances if np.sign(x) == np.sign(mean_imp))
    return same_sign / len(importances)


def rank_features(
    strategy,
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    config=None,
    n_permutations: int = 5,
    check_stability: bool = True,
) -> FeatureRanking:
    """Rank all preprocessed features by importance for a given strategy.

    Args:
        strategy: Strategy instance to evaluate
        df: DataFrame with preprocessed features already added
        symbol: Asset symbol
        asset_class: "stock" or "crypto"
        config: BacktestConfig (uses default if None)
        n_permutations: Number of shuffle repetitions per feature
        check_stability: Whether to compute rolling stability (slower but recommended)

    Returns:
        FeatureRanking with sorted features
    """
    from src.backtesting.engine import BacktestConfig, run_backtest

    if config is None:
        config = BacktestConfig()

    meta = strategy.meta()
    log.info("Feature importance analysis: %s on %s", meta.name, symbol)

    # Available features in this DataFrame
    available = [f for f in PREPROCESSED_FEATURES if f in df.columns]
    if not available:
        log.warning("No preprocessed features found in DataFrame")
        return FeatureRanking(strategy_name=meta.name, symbol=symbol)

    log.info("  Analyzing %d features: %s", len(available), available)

    # Baseline Sharpe
    try:
        baseline_result = run_backtest(strategy, df, symbol, asset_class, config)
        baseline_sharpe = baseline_result.report.sharpe_ratio
    except Exception as e:
        log.error("  Baseline backtest failed: %s", e)
        return FeatureRanking(strategy_name=meta.name, symbol=symbol)

    log.info("  Baseline Sharpe: %.3f", baseline_sharpe)

    # Forward returns for MI computation
    fwd_returns = df["close"].pct_change().shift(-1).values

    # Score each feature
    scores = []
    for feat in available:
        log.info("  Evaluating: %s...", feat)

        # 1. Permutation importance
        imp_mean, imp_std = _permutation_importance(
            strategy, df, symbol, asset_class, config,
            feat, n_permutations, baseline_sharpe,
        )

        # 2. Mutual information
        feat_values = df[feat].values.astype(float)
        mi = _compute_mutual_information(feat_values, fwd_returns)

        # 3. Rolling stability
        stability = 0.5
        if check_stability and imp_mean > 0:
            stability = _rolling_stability(
                strategy, df, symbol, asset_class, config, feat,
            )

        # Is it both important and stable?
        is_stable = imp_mean > 0.05 and stability >= 0.6

        scores.append(FeatureScore(
            name=feat,
            importance=imp_mean,
            std=imp_std,
            mutual_info=mi,
            stability=stability,
            stable=is_stable,
        ))

    # Sort by combined rank score
    scores.sort(key=lambda x: x.rank_score, reverse=True)

    ranking = FeatureRanking(
        strategy_name=meta.name,
        symbol=symbol,
        features=scores,
        baseline_sharpe=baseline_sharpe,
    )

    log.info("  Top features: %s", ranking.top_features)
    return ranking


def print_feature_ranking(ranking: FeatureRanking) -> str:
    """Format feature ranking for display."""
    lines = [
        f"\n  FEATURE IMPORTANCE: {ranking.strategy_name} on {ranking.symbol}",
        f"  Baseline Sharpe: {ranking.baseline_sharpe:.3f}",
        f"  {'Feature':25s} {'Importance':>10s} {'MI':>8s} {'Stability':>10s} {'Rank':>8s} {'Status':>8s}",
        f"  {'─' * 75}",
    ]

    for i, feat in enumerate(ranking.features):
        status = "KEEP" if feat.stable else ("weak" if feat.importance > 0 else "drop")
        lines.append(
            f"  {feat.name:25s} {feat.importance:>+10.4f} {feat.mutual_info:>8.4f} "
            f"{feat.stability:>10.2f} {feat.rank_score:>8.4f} {status:>8s}"
        )

    lines.append(f"\n  Recommended features ({ranking.top_n}): {', '.join(ranking.top_features)}")
    return "\n".join(lines)
