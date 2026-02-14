"""
Position sizing and bankroll management.

Implements Kelly Criterion and variants for optimal bet sizing.
Following Ed Thorp's work and Ernest Chan's practical adaptations.

Key insight: Full Kelly maximizes long-term growth rate but has
brutal drawdowns (~50% of bankroll). Half-Kelly gives 75% of the
growth rate with far less pain. We default to half-Kelly.

Also implements:
- Fixed fractional sizing
- Volatility-scaled sizing (ATR-based)
- Dynamic Kelly (adjusts based on recent performance)
- Maximum position limits (hard guardrails)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PositionSize:
    """Result of a position sizing calculation."""
    fraction: float          # Fraction of bankroll to risk (0.0 - 1.0)
    shares: int              # Number of shares/contracts to buy
    dollar_amount: float     # Dollar value of the position
    risk_amount: float       # Dollar amount at risk (position * stop_loss)
    method: str              # Which sizing method was used
    kelly_raw: float         # Raw Kelly fraction (before adjustments)
    confidence: float        # 0-1, how confident we are in the sizing


class KellyCriterion:
    """
    Kelly Criterion position sizing.

    f* = (p * b - q) / b

    Where:
        f* = fraction of bankroll to bet
        p = probability of winning
        q = probability of losing (1 - p)
        b = ratio of win size to loss size (odds)

    For continuous distributions (real trading):
        f* = mean_return / variance_of_returns

    We use half-Kelly by default because:
    1. Parameter estimation error means true Kelly is uncertain
    2. Half-Kelly gives ~75% of growth rate with ~50% less drawdown
    3. It's more robust to model misspecification
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,    # Half-Kelly default
        max_position_pct: float = 0.20,  # Never more than 20% in one trade
        min_position_pct: float = 0.01,  # At least 1% or don't bother
        max_portfolio_heat: float = 0.40, # Max 40% of capital at risk across all positions
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_portfolio_heat = max_portfolio_heat

    def from_win_rate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        bankroll: float,
        entry_price: float,
        stop_loss_pct: float = 2.0,
        current_heat: float = 0.0,
    ) -> PositionSize:
        """
        Calculate Kelly sizing from win rate and average win/loss.

        This is the classic Kelly formula for binary outcomes,
        which works well for strategies with clear win/loss trades.

        Args:
            win_rate: Historical win rate (0.0 - 1.0)
            avg_win: Average winning trade P&L (positive, e.g., 2.5 for 2.5%)
            avg_loss: Average losing trade P&L (positive, e.g., 1.0 for 1.0%)
            bankroll: Total account equity
            entry_price: Price per share/unit
            stop_loss_pct: Stop loss percentage for risk calculation
            current_heat: Current portfolio heat (sum of all position risks)
        """
        if avg_loss <= 0:
            avg_loss = 0.001

        p = min(max(win_rate, 0.01), 0.99)
        q = 1.0 - p
        b = avg_win / avg_loss  # Win/loss ratio (odds)

        # Kelly formula: f* = (p*b - q) / b
        kelly_raw = (p * b - q) / b

        return self._apply_constraints(
            kelly_raw=kelly_raw,
            bankroll=bankroll,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            current_heat=current_heat,
            method="kelly_binary",
        )

    def from_returns(
        self,
        returns: pd.Series | np.ndarray,
        bankroll: float,
        entry_price: float,
        stop_loss_pct: float = 2.0,
        current_heat: float = 0.0,
    ) -> PositionSize:
        """
        Kelly sizing from a return distribution (continuous Kelly).

        f* = mu / sigma^2

        More accurate for strategies with variable-sized returns.
        Uses the entire return distribution, not just win rate.
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) < 10:
            return PositionSize(0, 0, 0, 0, "kelly_continuous", 0, 0)

        mu = np.mean(returns)
        sigma2 = np.var(returns)

        if sigma2 <= 0:
            kelly_raw = 0.0
        else:
            kelly_raw = mu / sigma2

        return self._apply_constraints(
            kelly_raw=kelly_raw,
            bankroll=bankroll,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            current_heat=current_heat,
            method="kelly_continuous",
        )

    def dynamic_kelly(
        self,
        recent_returns: pd.Series | np.ndarray,
        lookback_returns: pd.Series | np.ndarray,
        bankroll: float,
        entry_price: float,
        stop_loss_pct: float = 2.0,
        current_heat: float = 0.0,
        decay_factor: float = 0.95,
    ) -> PositionSize:
        """
        Dynamic Kelly — adjusts sizing based on recent vs historical performance.

        If recent returns are worse than historical, we size down.
        If recent returns are better, we size up (but capped).

        This adapts to regime changes — when the strategy starts losing,
        sizing automatically decreases before the drawdown gets deep.

        Uses exponentially weighted moments for faster adaptation.
        """
        if isinstance(recent_returns, pd.Series):
            recent_returns = recent_returns.values
        if isinstance(lookback_returns, pd.Series):
            lookback_returns = lookback_returns.values

        if len(lookback_returns) < 20:
            return PositionSize(0, 0, 0, 0, "dynamic_kelly", 0, 0)

        # Historical Kelly
        hist_mu = np.mean(lookback_returns)
        hist_sigma2 = np.var(lookback_returns)
        hist_kelly = hist_mu / hist_sigma2 if hist_sigma2 > 0 else 0.0

        # Recent Kelly (exponentially weighted)
        if len(recent_returns) >= 5:
            weights = np.array([decay_factor ** i for i in range(len(recent_returns) - 1, -1, -1)])
            weights /= weights.sum()
            recent_mu = np.average(recent_returns, weights=weights)
            recent_sigma2 = np.average((recent_returns - recent_mu) ** 2, weights=weights)
            recent_kelly = recent_mu / recent_sigma2 if recent_sigma2 > 0 else 0.0
        else:
            recent_kelly = hist_kelly

        # Blend: 60% historical, 40% recent (recent pulls us toward current regime)
        kelly_raw = 0.6 * hist_kelly + 0.4 * recent_kelly

        # If recent performance is significantly worse, scale down aggressively
        if len(recent_returns) >= 5:
            recent_sharpe = recent_mu / np.sqrt(recent_sigma2) if recent_sigma2 > 0 else 0
            hist_sharpe = hist_mu / np.sqrt(hist_sigma2) if hist_sigma2 > 0 else 0
            if hist_sharpe > 0 and recent_sharpe < hist_sharpe * 0.5:
                # Strategy degrading — halve the sizing
                kelly_raw *= 0.5

        return self._apply_constraints(
            kelly_raw=kelly_raw,
            bankroll=bankroll,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            current_heat=current_heat,
            method="dynamic_kelly",
        )

    def _apply_constraints(
        self,
        kelly_raw: float,
        bankroll: float,
        entry_price: float,
        stop_loss_pct: float,
        current_heat: float,
        method: str,
    ) -> PositionSize:
        """Apply half-Kelly, position limits, and portfolio heat constraints."""

        # Apply Kelly fraction (half-Kelly by default)
        fraction = kelly_raw * self.kelly_fraction

        # If Kelly says don't trade, don't trade
        if fraction <= 0:
            return PositionSize(
                fraction=0, shares=0, dollar_amount=0, risk_amount=0,
                method=method, kelly_raw=kelly_raw, confidence=0,
            )

        # Cap at max position size
        fraction = min(fraction, self.max_position_pct)

        # Check portfolio heat
        available_heat = max(0, self.max_portfolio_heat - current_heat)
        risk_fraction = fraction * (stop_loss_pct / 100.0)
        if risk_fraction > available_heat:
            fraction = available_heat / (stop_loss_pct / 100.0) if stop_loss_pct > 0 else 0

        # Floor at minimum position
        if fraction < self.min_position_pct:
            return PositionSize(
                fraction=0, shares=0, dollar_amount=0, risk_amount=0,
                method=method, kelly_raw=kelly_raw, confidence=0.1,
            )

        # Convert to shares
        dollar_amount = bankroll * fraction
        shares = int(dollar_amount / entry_price) if entry_price > 0 else 0
        actual_dollar = shares * entry_price
        risk_amount = actual_dollar * (stop_loss_pct / 100.0)

        # Confidence based on how much Kelly we're actually using
        confidence = min(1.0, fraction / self.max_position_pct) if kelly_raw > 0 else 0

        return PositionSize(
            fraction=round(fraction, 4),
            shares=shares,
            dollar_amount=round(actual_dollar, 2),
            risk_amount=round(risk_amount, 2),
            method=method,
            kelly_raw=round(kelly_raw, 4),
            confidence=round(confidence, 3),
        )


# ── Volatility-Scaled Sizing ──


def volatility_position_size(
    bankroll: float,
    entry_price: float,
    atr: float,
    risk_pct: float = 1.0,
    atr_multiplier: float = 2.0,
) -> PositionSize:
    """
    ATR-based position sizing (Van Tharp / Turtle method).

    Position size = (bankroll * risk_pct) / (ATR * multiplier)

    Automatically sizes smaller in volatile markets and larger in calm ones.
    This is how the Turtle Traders sized positions.

    Args:
        bankroll: Total account equity.
        entry_price: Current price.
        atr: Average True Range (e.g., 14-period ATR).
        risk_pct: Max % of bankroll to risk per trade.
        atr_multiplier: Stop distance in ATR units (2.0 = stop at 2x ATR).
    """
    if atr <= 0 or entry_price <= 0:
        return PositionSize(0, 0, 0, 0, "volatility_atr", 0, 0)

    risk_per_share = atr * atr_multiplier
    risk_dollars = bankroll * (risk_pct / 100.0)
    shares = int(risk_dollars / risk_per_share)
    dollar_amount = shares * entry_price
    fraction = dollar_amount / bankroll if bankroll > 0 else 0

    return PositionSize(
        fraction=round(fraction, 4),
        shares=shares,
        dollar_amount=round(dollar_amount, 2),
        risk_amount=round(risk_dollars, 2),
        method="volatility_atr",
        kelly_raw=0,
        confidence=0.7,
    )


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range."""
    if len(highs) < period + 1:
        return 0.0

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(np.mean(tr[-period:]))
