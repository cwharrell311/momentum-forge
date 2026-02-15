"""
Hierarchical Risk Parity (HRP) portfolio construction.

Replaces simple Sharpe-weighted allocation with the López de Prado (2016)
algorithm that uses hierarchical clustering to build diversified portfolios
without requiring a covariance matrix inversion (which is notoriously unstable).

HRP steps:
1. Compute pairwise distance matrix from return correlations
2. Hierarchical clustering (single-linkage by default)
3. Quasi-diagonalization — reorder assets so similar ones are adjacent
4. Recursive bisection — allocate risk by splitting the dendrogram

Reference: López de Prado, "Building Diversified Portfolios that Outperform
Out-of-Sample" (Journal of Portfolio Management, 2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.backtesting.engine import BacktestResult, StrategyAllocation

log = logging.getLogger("forge.hrp")


@dataclass
class HRPResult:
    """Result of HRP portfolio construction."""
    allocations: list[StrategyAllocation]
    correlation_matrix: pd.DataFrame
    linkage_matrix: np.ndarray
    quasi_diag_order: list[int]


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5 * (1 - corr))."""
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diagonalize(link: np.ndarray, n: int) -> list[int]:
    """Reorder assets using the dendrogram leaf order (quasi-diagonalization)."""
    return list(leaves_list(link).astype(int))


def _recursive_bisection(
    cov: np.ndarray,
    sorted_indices: list[int],
) -> np.ndarray:
    """
    Recursive bisection: allocate weights by splitting the sorted asset list.

    At each split, the two clusters get weight proportional to the inverse
    of their variance (risk parity within the cluster).
    """
    n = len(sorted_indices)
    weights = np.ones(n)

    # Stack-based iterative approach (avoids recursion depth issues)
    clusters = [(sorted_indices, np.arange(n))]

    while clusters:
        next_clusters = []
        for items, w_indices in clusters:
            if len(items) <= 1:
                continue

            # Split in half
            mid = len(items) // 2
            left_items = items[:mid]
            right_items = items[mid:]
            left_w = w_indices[:mid]
            right_w = w_indices[mid:]

            # Compute cluster variances using inverse-variance weighting
            left_var = _cluster_variance(cov, left_items)
            right_var = _cluster_variance(cov, right_items)

            # Allocate inversely proportional to variance
            total_inv_var = 1.0 / left_var + 1.0 / right_var
            alpha = (1.0 / left_var) / total_inv_var  # left gets this fraction

            weights[left_w] *= alpha
            weights[right_w] *= (1.0 - alpha)

            next_clusters.append((left_items, left_w))
            next_clusters.append((right_items, right_w))

        clusters = next_clusters

    return weights


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Compute the variance of a cluster using inverse-variance weights."""
    sub_cov = cov[np.ix_(indices, indices)]
    n = len(indices)

    # Inverse-variance portfolio within the cluster
    ivp = 1.0 / np.diag(sub_cov)
    ivp /= ivp.sum()

    # Portfolio variance = w' * Cov * w
    var = float(ivp @ sub_cov @ ivp)
    return max(var, 1e-10)  # floor to avoid division by zero


def hrp_allocate(
    results: list[BacktestResult],
    total_capital: float = 100_000,
    min_sharpe: float = 0.0,
    min_trades: int = 10,
) -> HRPResult:
    """
    Allocate capital across strategies using Hierarchical Risk Parity.

    Args:
        results: List of BacktestResult objects with equity curves.
        total_capital: Total capital to allocate.
        min_sharpe: Minimum Sharpe ratio to include.
        min_trades: Minimum number of trades to include.

    Returns:
        HRPResult with allocations and diagnostics.
    """
    # Filter viable strategies
    viable = [
        r for r in results
        if r.report.sharpe_ratio > min_sharpe
        and r.report.trades.total_trades >= min_trades
    ]

    if not viable:
        return HRPResult(
            allocations=[], correlation_matrix=pd.DataFrame(),
            linkage_matrix=np.array([]), quasi_diag_order=[],
        )

    if len(viable) == 1:
        r = viable[0]
        alloc = StrategyAllocation(
            strategy_name=r.strategy_name, symbol=r.symbol,
            sharpe=round(r.report.sharpe_ratio, 3),
            weight=1.0, capital_allocated=total_capital,
        )
        return HRPResult(
            allocations=[alloc], correlation_matrix=pd.DataFrame(),
            linkage_matrix=np.array([]), quasi_diag_order=[0],
        )

    # Build return matrix: each column is a strategy's equity curve returns
    returns_dict = {}
    labels = []
    for r in viable:
        label = f"{r.strategy_name}_{r.symbol}"
        returns_dict[label] = r.equity_curve.pct_change().dropna()
        labels.append(label)

    # Align all return series to common index
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how="all").fillna(0)

    if len(returns_df) < 5 or len(returns_df.columns) < 2:
        # Not enough data for HRP, fall back to equal weight
        n = len(viable)
        allocations = []
        for r in viable:
            allocations.append(StrategyAllocation(
                strategy_name=r.strategy_name, symbol=r.symbol,
                sharpe=round(r.report.sharpe_ratio, 3),
                weight=round(1.0 / n, 4),
                capital_allocated=round(total_capital / n, 2),
            ))
        return HRPResult(
            allocations=allocations, correlation_matrix=pd.DataFrame(),
            linkage_matrix=np.array([]), quasi_diag_order=list(range(n)),
        )

    n_assets = len(returns_df.columns)

    # Step 1: Correlation and distance matrices
    corr = returns_df.corr().values
    # Clamp to valid range
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)

    # Convert to condensed form for scipy
    condensed = squareform(dist, checks=False)

    # Step 2: Hierarchical clustering
    link = linkage(condensed, method="single")

    # Step 3: Quasi-diagonalization
    sort_idx = _quasi_diagonalize(link, n_assets)

    # Step 4: Recursive bisection on covariance
    cov = returns_df.cov().values
    weights = _recursive_bisection(cov, sort_idx)

    # Map weights back to original order
    final_weights = np.zeros(n_assets)
    for i, si in enumerate(sort_idx):
        final_weights[si] = weights[i]

    # Normalize
    total_w = final_weights.sum()
    if total_w > 0:
        final_weights /= total_w

    # Build allocations
    allocations = []
    for i, r in enumerate(viable):
        w = float(final_weights[i])
        allocations.append(StrategyAllocation(
            strategy_name=r.strategy_name,
            symbol=r.symbol,
            sharpe=round(r.report.sharpe_ratio, 3),
            weight=round(w, 4),
            capital_allocated=round(total_capital * w, 2),
        ))

    allocations.sort(key=lambda a: a.weight, reverse=True)

    corr_df = pd.DataFrame(corr, index=labels, columns=labels)

    log.info("HRP allocation: %d strategies, top weight=%.1f%%",
             len(allocations), allocations[0].weight * 100 if allocations else 0)

    return HRPResult(
        allocations=allocations,
        correlation_matrix=corr_df,
        linkage_matrix=link,
        quasi_diag_order=sort_idx,
    )
