"""
Extended strategy library for multi-asset day trading.

Strategies are organized by asset class and market regime:
- Stock strategies: Gap fade, momentum, denoised momentum
- Crypto strategies: Momentum, mean reversion, denoised momentum
- Polymarket strategies: Probability momentum, mean reversion
- Cross-asset: Regime-adaptive, Hurst-adaptive, entropy-filtered

Each strategy implements the BaseStrategy interface:
- generate_signals(df) -> list of (index, side, confidence, reason)
- parameters are exposed for genetic optimization

Design principle: strategies should be SIMPLE with few parameters.
Complex strategies with 10+ parameters are curve-fitting machines.
We cap at 5 parameters per strategy and penalize parameter count
during optimization.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger("forge.strategies")


class Side(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """A trade signal emitted by a strategy."""
    bar_index: int           # Index into the DataFrame
    side: Side
    confidence: float        # 0.0 - 1.0
    reason: str
    stop_loss_pct: float = 2.0    # Default stop
    take_profit_pct: float = 4.0  # Default target (2:1 R:R)


@dataclass
class StrategyMeta:
    """Metadata about a strategy for the optimizer."""
    name: str
    asset_classes: list[str]       # ["stock", "crypto", "polymarket"]
    param_count: int               # Number of tunable parameters
    param_ranges: dict[str, tuple] # {param_name: (min, max)}
    description: str


class BaseStrategy(ABC):
    """Base class for all backtesting strategies."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        Generate trade signals from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index is timestamps.

        Returns:
            List of Signal objects.
        """
        ...

    @abstractmethod
    def meta(self) -> StrategyMeta:
        """Return strategy metadata for the optimizer."""
        ...


# ═══════════════════════════════════════════════════════════════
# STOCK STRATEGIES
# ═══════════════════════════════════════════════════════════════


class GapFade(BaseStrategy):
    """
    Gap Fade — trade against overnight gaps.

    If the market gaps up > threshold, short it (fade the gap).
    If it gaps down > threshold, go long.

    Gaps tend to fill ~70% of the time for stocks.
    Requires daily bars to detect gaps.

    Parameters: min_gap_pct, max_gap_pct (too big = news, don't fade).
    """

    def __init__(self, min_gap_pct: float = 0.5, max_gap_pct: float = 3.0):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="gap_fade",
            asset_classes=["stock"],
            param_count=2,
            param_ranges={"min_gap_pct": (0.3, 1.5), "max_gap_pct": (2.0, 5.0)},
            description="Fade overnight gaps — gaps tend to fill 70% of the time",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < 2:
            return signals

        for i in range(1, len(df)):
            prev_close = df["close"].iloc[i - 1]
            curr_open = df["open"].iloc[i]

            if prev_close <= 0:
                continue

            gap_pct = (curr_open - prev_close) / prev_close * 100

            # Gap up — fade it (short)
            if self.min_gap_pct <= gap_pct <= self.max_gap_pct:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, gap_pct / self.max_gap_pct),
                    reason=f"Gap fade short: {gap_pct:+.2f}% gap up",
                    stop_loss_pct=gap_pct * 0.5,  # Stop at half the gap extension
                    take_profit_pct=gap_pct * 0.8,  # Target: 80% gap fill
                ))
            # Gap down — fade it (long)
            elif -self.max_gap_pct <= gap_pct <= -self.min_gap_pct:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, abs(gap_pct) / self.max_gap_pct),
                    reason=f"Gap fade long: {gap_pct:+.2f}% gap down",
                    stop_loss_pct=abs(gap_pct) * 0.5,
                    take_profit_pct=abs(gap_pct) * 0.8,
                ))

        return signals


class DualMomentum(BaseStrategy):
    """
    Dual Momentum (Gary Antonacci) — absolute + relative momentum.

    Only go long when:
    1. Absolute momentum is positive (asset > risk-free rate)
    2. Relative momentum ranks it above alternatives

    Enhanced with:
    - Volatility regime filter (skip trades in choppy markets)
    - Wider parameter ranges for longer-term momentum (60/90/120 day)
    - Acceleration confirmation (momentum must be increasing)

    Parameters: fast_period, slow_period, vol_lookback.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 40, vol_lookback: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_lookback = vol_lookback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="dual_momentum",
            asset_classes=["stock", "crypto"],
            param_count=3,
            param_ranges={"fast_period": (5, 30), "slow_period": (30, 120), "vol_lookback": (10, 40)},
            description="Dual momentum with regime filter — long when both timeframes confirm in trending regime",
        )

    def _is_trending_regime(self, closes: np.ndarray, idx: int) -> bool:
        """Check if we're in a trending regime (not choppy) using efficiency ratio."""
        if idx < self.vol_lookback:
            return True  # Default to allowing trades with insufficient data
        window = closes[idx - self.vol_lookback:idx + 1]
        direction = abs(window[-1] - window[0])
        volatility = sum(abs(window[j] - window[j - 1]) for j in range(1, len(window)))
        if volatility == 0:
            return False
        efficiency_ratio = direction / volatility
        # ER > 0.3 = trending, ER < 0.3 = choppy/mean-reverting
        return efficiency_ratio > 0.3

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.slow_period + 2:
            return signals

        closes = df["close"].values

        for i in range(self.slow_period + 1, len(df)):
            # Regime filter: skip choppy markets
            if not self._is_trending_regime(closes, i):
                continue

            fast_mom = (closes[i] - closes[i - self.fast_period]) / closes[i - self.fast_period]
            slow_mom = (closes[i] - closes[i - self.slow_period]) / closes[i - self.slow_period]

            prev_fast = (closes[i-1] - closes[i-1 - self.fast_period]) / closes[i-1 - self.fast_period]

            # Both momentum positive and accelerating = strong long
            if fast_mom > 0 and slow_mom > 0 and fast_mom > prev_fast:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, (fast_mom + slow_mom) * 5),
                    reason=f"Dual momentum long: fast={fast_mom:.3f}, slow={slow_mom:.3f}",
                ))
            # Both negative and accelerating down = short
            elif fast_mom < 0 and slow_mom < 0 and fast_mom < prev_fast:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(fast_mom + slow_mom) * 5),
                    reason=f"Dual momentum short: fast={fast_mom:.3f}, slow={slow_mom:.3f}",
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# CRYPTO STRATEGIES
# ═══════════════════════════════════════════════════════════════


class CryptoMomentum(BaseStrategy):
    """
    Crypto-specific momentum with volume confirmation and regime filter.

    Crypto trends harder than equities — momentum strategies work better.
    Uses ROC + volume surge + volatility regime filter.

    Enhanced:
    - ATR-based dynamic stops (adapt to current volatility)
    - Efficiency ratio regime filter (skip choppy markets)
    - Wider ROC threshold range for optimization

    Parameters: roc_period, volume_multiplier, roc_threshold.
    """

    def __init__(self, roc_period: int = 12, volume_mult: float = 1.5, roc_threshold: float = 3.0):
        self.roc_period = roc_period
        self.volume_mult = volume_mult
        self.roc_threshold = roc_threshold

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="crypto_momentum",
            asset_classes=["crypto"],
            param_count=3,
            param_ranges={"roc_period": (6, 30), "volume_mult": (1.2, 3.0), "roc_threshold": (1.5, 6.0)},
            description="Crypto momentum with volume surge + regime filter + dynamic stops",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        lookback = max(self.roc_period + 1, 20)
        if len(df) < lookback:
            return signals

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values

        for i in range(lookback, len(df)):
            # Regime filter: efficiency ratio
            window = closes[i - 20:i + 1]
            direction = abs(window[-1] - window[0])
            volatility = sum(abs(window[j] - window[j - 1]) for j in range(1, len(window)))
            er = direction / volatility if volatility > 0 else 0
            if er < 0.25:  # Skip choppy regime
                continue

            roc = (closes[i] - closes[i - self.roc_period]) / closes[i - self.roc_period] * 100
            avg_vol = np.mean(volumes[i - 20:i])
            vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0

            # ATR-based dynamic stops
            recent_tr = [max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
                         for j in range(max(1, i - 14), i + 1)]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.03
            atr_pct = atr / closes[i] * 100

            # Strong momentum + volume confirmation
            if roc > self.roc_threshold and vol_ratio > self.volume_mult:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, roc / 10.0 * vol_ratio / 3.0),
                    reason=f"Crypto momentum: ROC={roc:.1f}%, vol={vol_ratio:.1f}x, ER={er:.2f}",
                    stop_loss_pct=max(1.5, atr_pct * 2),
                    take_profit_pct=max(3.0, atr_pct * 4),
                ))
            elif roc < -self.roc_threshold and vol_ratio > self.volume_mult:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(roc) / 10.0 * vol_ratio / 3.0),
                    reason=f"Crypto momentum short: ROC={roc:.1f}%, vol={vol_ratio:.1f}x, ER={er:.2f}",
                    stop_loss_pct=max(1.5, atr_pct * 2),
                    take_profit_pct=max(3.0, atr_pct * 4),
                ))

        return signals


class CryptoMeanReversion(BaseStrategy):
    """
    Crypto Bollinger Band mean reversion.

    Crypto is more volatile than stocks, so we use wider bands.
    Works well in choppy/range-bound crypto markets.

    Parameters: period, num_std.
    """

    def __init__(self, period: int = 20, num_std: float = 2.5):
        self.period = period
        self.num_std = num_std

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="crypto_mean_reversion",
            asset_classes=["crypto"],
            param_count=2,
            param_ranges={"period": (10, 50), "num_std": (2.0, 3.5)},
            description="Bollinger Band reversion with wider bands for crypto volatility",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.period + 1:
            return signals

        closes = df["close"]
        ma = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()

        for i in range(self.period, len(df)):
            if pd.isna(ma.iloc[i]) or pd.isna(std.iloc[i]) or std.iloc[i] <= 0:
                continue

            upper = ma.iloc[i] + self.num_std * std.iloc[i]
            lower = ma.iloc[i] - self.num_std * std.iloc[i]
            z = (closes.iloc[i] - ma.iloc[i]) / std.iloc[i]

            if closes.iloc[i] < lower:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, abs(z) / (self.num_std * 1.5)),
                    reason=f"Crypto BB reversion long: z={z:.2f}, below lower band",
                    stop_loss_pct=abs(z) * std.iloc[i] / closes.iloc[i] * 100,
                ))
            elif closes.iloc[i] > upper:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(z) / (self.num_std * 1.5)),
                    reason=f"Crypto BB reversion short: z={z:.2f}, above upper band",
                    stop_loss_pct=abs(z) * std.iloc[i] / closes.iloc[i] * 100,
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# POLYMARKET STRATEGIES
# ═══════════════════════════════════════════════════════════════


class PredictionMomentum(BaseStrategy):
    """
    Prediction market momentum — ride probability trends.

    When a prediction market's probability starts moving decisively
    in one direction, it often continues (information cascades).

    Buy YES when probability is rising above threshold.
    Buy NO when probability is falling below threshold.

    Parameters: lookback, threshold_change.
    """

    def __init__(self, lookback: int = 12, threshold_change: float = 0.05):
        self.lookback = lookback
        self.threshold_change = threshold_change  # 5% probability shift

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="prediction_momentum",
            asset_classes=["polymarket"],
            param_count=2,
            param_ranges={"lookback": (6, 24), "threshold_change": (0.03, 0.15)},
            description="Ride probability momentum in prediction markets",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.lookback + 1:
            return signals

        prices = df["close"].values  # Price = probability (0 to 1)

        for i in range(self.lookback, len(df)):
            change = prices[i] - prices[i - self.lookback]

            # Probability rising = buy YES
            if change > self.threshold_change and prices[i] < 0.85:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, change / (self.threshold_change * 3)),
                    reason=f"Probability momentum: +{change:.3f} over {self.lookback} bars",
                    stop_loss_pct=5.0,
                    take_profit_pct=10.0,
                ))
            # Probability falling = buy NO (short YES)
            elif change < -self.threshold_change and prices[i] > 0.15:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(change) / (self.threshold_change * 3)),
                    reason=f"Probability momentum short: {change:.3f} over {self.lookback} bars",
                    stop_loss_pct=5.0,
                    take_profit_pct=10.0,
                ))

        return signals


class PredictionReversion(BaseStrategy):
    """
    Prediction market mean reversion — overreaction fade.

    Markets often overreact to news. When probability spikes or
    crashes by a large amount, fade the move.

    Parameters: spike_threshold, lookback.
    """

    def __init__(self, spike_threshold: float = 0.10, lookback: int = 6):
        self.spike_threshold = spike_threshold  # 10% probability spike
        self.lookback = lookback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="prediction_reversion",
            asset_classes=["polymarket"],
            param_count=2,
            param_ranges={"spike_threshold": (0.05, 0.20), "lookback": (3, 12)},
            description="Fade overreactions in prediction market probabilities",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.lookback + 1:
            return signals

        prices = df["close"].values

        for i in range(self.lookback, len(df)):
            change = prices[i] - prices[i - self.lookback]

            # Big spike up — fade it
            if change > self.spike_threshold and prices[i] < 0.90:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, change / (self.spike_threshold * 2)),
                    reason=f"Overreaction fade: +{change:.3f} spike, selling YES",
                    stop_loss_pct=8.0,
                    take_profit_pct=change * 50,  # Target: half the spike reverts
                ))
            # Big spike down — fade it
            elif change < -self.spike_threshold and prices[i] > 0.10:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, abs(change) / (self.spike_threshold * 2)),
                    reason=f"Overreaction fade: {change:.3f} crash, buying YES",
                    stop_loss_pct=8.0,
                    take_profit_pct=abs(change) * 50,
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL STRATEGIES (work across asset classes)
# ═══════════════════════════════════════════════════════════════


class AdaptiveTrend(BaseStrategy):
    """
    Adaptive trend following using Kaufman's Adaptive Moving Average (KAMA).

    KAMA adapts its smoothing based on market noise:
    - Trending market: fast response (follows closely)
    - Choppy market: slow response (filters noise)

    This is anti-curve-fitting by design — the strategy self-adjusts
    to market conditions rather than relying on fixed parameters.

    Parameters: er_period (efficiency ratio lookback), fast/slow constants.
    """

    def __init__(self, er_period: int = 10, fast_period: int = 2, slow_period: int = 30):
        self.er_period = er_period
        self.fast_sc = 2.0 / (fast_period + 1)
        self.slow_sc = 2.0 / (slow_period + 1)

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="adaptive_trend",
            asset_classes=["stock", "crypto", "polymarket"],
            param_count=3,
            param_ranges={"er_period": (5, 20), "fast_period": (2, 5), "slow_period": (20, 50)},
            description="Kaufman Adaptive MA — self-adjusts to trending/choppy conditions",
        )

    def _compute_kama(self, closes: np.ndarray) -> np.ndarray:
        """Compute Kaufman Adaptive Moving Average."""
        kama = np.full(len(closes), np.nan)
        if len(closes) < self.er_period + 1:
            return kama

        kama[self.er_period] = closes[self.er_period]

        for i in range(self.er_period + 1, len(closes)):
            # Efficiency Ratio = direction / volatility
            direction = abs(closes[i] - closes[i - self.er_period])
            volatility = sum(abs(closes[j] - closes[j - 1]) for j in range(i - self.er_period + 1, i + 1))
            er = direction / volatility if volatility > 0 else 0

            # Smoothing constant adapts between fast and slow
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

            kama[i] = kama[i - 1] + sc * (closes[i] - kama[i - 1])

        return kama

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.er_period + 5:
            return signals

        closes = df["close"].values
        kama = self._compute_kama(closes)

        for i in range(self.er_period + 2, len(df)):
            if np.isnan(kama[i]) or np.isnan(kama[i - 1]):
                continue

            # Trend direction from KAMA slope
            kama_slope = (kama[i] - kama[i - 1]) / kama[i - 1] if kama[i - 1] > 0 else 0
            prev_slope = (kama[i - 1] - kama[i - 2]) / kama[i - 2] if kama[i - 2] > 0 else 0

            # Cross above KAMA + KAMA turning up
            if closes[i] > kama[i] and closes[i - 1] <= kama[i - 1] and kama_slope > 0:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, abs(kama_slope) * 100),
                    reason=f"KAMA crossover long: slope={kama_slope:.4f}",
                ))
            # Cross below KAMA + KAMA turning down
            elif closes[i] < kama[i] and closes[i - 1] >= kama[i - 1] and kama_slope < 0:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(kama_slope) * 100),
                    reason=f"KAMA crossover short: slope={kama_slope:.4f}",
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# SIGNAL-PROCESSING STRATEGIES
# ═══════════════════════════════════════════════════════════════


class DenoisedMomentum(BaseStrategy):
    """
    Momentum on denoised price signals.

    The #1 problem with daily momentum: noise drowns the signal.
    Solution from signal processing: denoise FIRST, then compute momentum.

    Uses Kalman filter (from aerospace/control theory) to extract the
    true trend from noisy daily prices, then trades momentum on the
    clean signal. Same math as GPS tracking satellites.

    Parameters: fast_period, slow_period, noise_ratio.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 40, noise_ratio: float = 1.0):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.noise_ratio = noise_ratio  # Higher = more smoothing

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="denoised_momentum",
            asset_classes=["stock", "crypto"],
            param_count=3,
            param_ranges={"fast_period": (5, 20), "slow_period": (20, 80), "noise_ratio": (0.5, 3.0)},
            description="Momentum on Kalman-denoised prices — signal processing meets trading",
        )

    def _kalman_filter(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter on price series.

        State vector: [level, trend]
        Transition: level_t = level_{t-1} + trend_{t-1}
                    trend_t = trend_{t-1}
        Observation: price_t = level_t + noise

        Returns:
            filtered_prices: Kalman-smoothed price levels
            velocities: Kalman-estimated trend (velocity) at each bar
        """
        n = len(prices)
        filtered = np.zeros(n)
        velocity = np.zeros(n)

        # Transition matrix F = [[1, 1], [0, 1]]
        # Observation matrix H = [1, 0]

        # Initial state
        x = np.array([prices[0], 0.0])  # [level, trend]

        # Initial covariance (high uncertainty)
        P = np.array([[1000.0, 0.0],
                       [0.0, 1000.0]])

        # Process noise covariance Q — scaled by noise_ratio
        # Higher noise_ratio = trust the model less = more smoothing
        q_level = 0.01 * self.noise_ratio
        q_trend = 0.001 * self.noise_ratio
        Q = np.array([[q_level, 0.0],
                       [0.0, q_trend]])

        # Measurement noise R — estimate from price variance
        price_std = np.std(np.diff(prices[:min(50, n)])) if n > 2 else 1.0
        R = (price_std * self.noise_ratio) ** 2
        if R < 1e-10:
            R = 1.0  # Guard against zero variance

        filtered[0] = prices[0]
        velocity[0] = 0.0

        for i in range(1, n):
            # === PREDICT ===
            # x_pred = F @ x
            x_pred = np.array([x[0] + x[1], x[1]])

            # P_pred = F @ P @ F.T + Q
            P_pred = np.array([
                [P[0, 0] + P[0, 1] + P[1, 0] + P[1, 1] + Q[0, 0], P[0, 1] + P[1, 1] + Q[0, 1]],
                [P[1, 0] + P[1, 1] + Q[1, 0], P[1, 1] + Q[1, 1]],
            ])

            # === UPDATE ===
            # Innovation: y = z - H @ x_pred (H = [1, 0])
            y = prices[i] - x_pred[0]

            # Innovation covariance: S = H @ P_pred @ H.T + R
            S = P_pred[0, 0] + R

            # Kalman gain: K = P_pred @ H.T / S
            K = np.array([P_pred[0, 0] / S, P_pred[1, 0] / S])

            # Updated state: x = x_pred + K * y
            x = x_pred + K * y

            # Updated covariance: P = (I - K @ H) @ P_pred
            P = np.array([
                [(1 - K[0]) * P_pred[0, 0], (1 - K[0]) * P_pred[0, 1]],
                [-K[1] * P_pred[0, 0] + P_pred[1, 0], -K[1] * P_pred[0, 1] + P_pred[1, 1]],
            ])

            filtered[i] = x[0]
            velocity[i] = x[1]

        return filtered, velocity

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.slow_period + 5:
            return signals

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # Run Kalman filter on close prices
        filtered, velocity = self._kalman_filter(closes)

        # Compute ATR for stops
        for i in range(self.slow_period + 1, len(df)):
            # Fast and slow momentum on FILTERED prices
            if filtered[i - self.fast_period] == 0 or filtered[i - self.slow_period] == 0:
                continue

            fast_mom = (filtered[i] - filtered[i - self.fast_period]) / filtered[i - self.fast_period]
            slow_mom = (filtered[i] - filtered[i - self.slow_period]) / filtered[i - self.slow_period]

            # ATR for position sizing and stops
            recent_tr = [
                max(highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]))
                for j in range(max(1, i - 14), i + 1)
            ]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.02
            atr_pct = atr / closes[i] * 100

            # LONG: both momentums positive AND Kalman velocity confirms uptrend
            if fast_mom > 0 and slow_mom > 0 and velocity[i] > 0:
                conf = min(1.0, (fast_mom + slow_mom) * 5 + abs(velocity[i]) / closes[i] * 100)
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, conf),
                    reason=(
                        f"Denoised momentum long: fast={fast_mom:.4f}, slow={slow_mom:.4f}, "
                        f"velocity={velocity[i]:.3f}"
                    ),
                    stop_loss_pct=max(1.0, atr_pct * 2),     # 2x ATR stop
                    take_profit_pct=max(1.5, atr_pct * 3),    # 3x ATR target
                ))
            # SHORT: both momentums negative AND Kalman velocity confirms downtrend
            elif fast_mom < 0 and slow_mom < 0 and velocity[i] < 0:
                conf = min(1.0, (abs(fast_mom) + abs(slow_mom)) * 5 + abs(velocity[i]) / closes[i] * 100)
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, conf),
                    reason=(
                        f"Denoised momentum short: fast={fast_mom:.4f}, slow={slow_mom:.4f}, "
                        f"velocity={velocity[i]:.3f}"
                    ),
                    stop_loss_pct=max(1.0, atr_pct * 2),
                    take_profit_pct=max(1.5, atr_pct * 3),
                ))

        return signals


class HurstAdaptive(BaseStrategy):
    """
    Adapts strategy type based on Hurst exponent.

    H > 0.55: trending — use momentum (buy breakouts)
    H < 0.45: mean-reverting — use reversion (buy dips)
    H ~ 0.50: random walk — DON'T TRADE

    The Hurst exponent comes from hydrology (Harold Hurst studying
    Nile river flood patterns). It measures persistence vs anti-persistence.

    Parameters: hurst_lookback, momentum_period.
    """

    def __init__(self, hurst_lookback: int = 100, momentum_period: int = 20):
        self.hurst_lookback = hurst_lookback
        self.momentum_period = momentum_period

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="hurst_adaptive",
            asset_classes=["stock", "crypto"],
            param_count=2,
            param_ranges={"hurst_lookback": (50, 200), "momentum_period": (10, 40)},
            description="Adapts between momentum and mean-reversion based on Hurst exponent",
        )

    @staticmethod
    def _compute_hurst(series: np.ndarray) -> float:
        """
        Compute Hurst exponent using rescaled range (R/S) analysis.

        Returns:
            H ~ 0.5: random walk (no memory)
            H > 0.5: persistent (trending)
            H < 0.5: anti-persistent (mean-reverting)
        """
        n = len(series)
        if n < 20:
            return 0.5  # Default to random walk for insufficient data

        # Use multiple sub-series lengths for R/S regression
        max_k = min(n // 2, 128)
        sizes = []
        rs_values = []

        for size in [int(n / d) for d in range(2, min(n // 10 + 1, 20))]:
            if size < 10:
                continue

            rs_list = []
            for start in range(0, n - size + 1, size):
                chunk = series[start:start + size]
                mean = np.mean(chunk)
                deviations = chunk - mean
                cumulative = np.cumsum(deviations)
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                sizes.append(size)
                rs_values.append(np.mean(rs_list))

        if len(sizes) < 2:
            return 0.5

        # log-log regression: log(R/S) = H * log(n) + c
        log_sizes = np.log(np.array(sizes, dtype=float))
        log_rs = np.log(np.array(rs_values, dtype=float))

        # Simple linear regression for slope (= Hurst exponent)
        n_pts = len(log_sizes)
        sum_x = np.sum(log_sizes)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_sizes * log_rs)
        sum_x2 = np.sum(log_sizes ** 2)

        denom = n_pts * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.5

        H = (n_pts * sum_xy - sum_x * sum_y) / denom

        # Clamp to reasonable range
        return max(0.01, min(0.99, H))

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Compute RSI at the last bar of the given closes array."""
        if len(closes) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        min_bars = max(self.hurst_lookback, self.momentum_period * 3) + 1
        if len(df) < min_bars:
            return signals

        closes = df["close"].values

        fast_ma_period = max(5, self.momentum_period // 2)
        slow_ma_period = self.momentum_period

        for i in range(min_bars, len(df)):
            # Rolling Hurst exponent
            window = closes[i - self.hurst_lookback:i + 1]
            H = self._compute_hurst(window)

            if H > 0.55:
                # TRENDING regime — use momentum (fast MA vs slow MA crossover)
                fast_ma = np.mean(closes[i - fast_ma_period + 1:i + 1])
                slow_ma = np.mean(closes[i - slow_ma_period + 1:i + 1])
                prev_fast_ma = np.mean(closes[i - fast_ma_period:i])
                prev_slow_ma = np.mean(closes[i - slow_ma_period:i])

                # Bullish crossover
                if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                    signals.append(Signal(
                        bar_index=i,
                        side=Side.LONG,
                        confidence=min(1.0, (H - 0.5) * 5),
                        reason=f"Hurst trending (H={H:.3f}): MA crossover long",
                        stop_loss_pct=2.0,
                        take_profit_pct=4.0,
                    ))
                # Bearish crossover
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    signals.append(Signal(
                        bar_index=i,
                        side=Side.SHORT,
                        confidence=min(1.0, (H - 0.5) * 5),
                        reason=f"Hurst trending (H={H:.3f}): MA crossover short",
                        stop_loss_pct=2.0,
                        take_profit_pct=4.0,
                    ))

            elif H < 0.45:
                # MEAN-REVERTING regime — use RSI reversion
                rsi = self._compute_rsi(closes[:i + 1], period=14)

                if rsi < 30:
                    signals.append(Signal(
                        bar_index=i,
                        side=Side.LONG,
                        confidence=min(1.0, (0.5 - H) * 5 * (1 - rsi / 50)),
                        reason=f"Hurst mean-revert (H={H:.3f}): RSI={rsi:.1f} oversold",
                        stop_loss_pct=1.5,
                        take_profit_pct=2.5,
                    ))
                elif rsi > 70:
                    signals.append(Signal(
                        bar_index=i,
                        side=Side.SHORT,
                        confidence=min(1.0, (0.5 - H) * 5 * (rsi / 50 - 1)),
                        reason=f"Hurst mean-revert (H={H:.3f}): RSI={rsi:.1f} overbought",
                        stop_loss_pct=1.5,
                        take_profit_pct=2.5,
                    ))

            # H between 0.45 and 0.55: random walk — no edge, skip

        return signals


class EntropyRegimeStrategy(BaseStrategy):
    """
    Only trades when the market is predictable (low entropy).

    Permutation entropy (from neuroscience/EEG analysis) measures
    how random a time series is. We only trade when entropy is low
    (market has structure we can exploit).

    Combines entropy regime filter with simple momentum signal.

    Parameters: entropy_lookback, entropy_threshold, momentum_period.
    """

    def __init__(self, entropy_lookback: int = 50, entropy_threshold: float = 0.7,
                 momentum_period: int = 20):
        self.entropy_lookback = entropy_lookback
        self.entropy_threshold = entropy_threshold  # Below this = predictable
        self.momentum_period = momentum_period

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="entropy_regime",
            asset_classes=["stock", "crypto"],
            param_count=3,
            param_ranges={
                "entropy_lookback": (30, 100),
                "entropy_threshold": (0.5, 0.85),
                "momentum_period": (10, 40),
            },
            description="Only trade when permutation entropy is low — skip random noise, exploit structure",
        )

    @staticmethod
    def _permutation_entropy(series: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """
        Compute normalised permutation entropy of a time series.

        Based on Bandt & Pompe (2002). Maps the series into ordinal
        patterns and computes Shannon entropy of the pattern distribution.

        Returns:
            0.0 = perfectly predictable (single repeating pattern)
            1.0 = maximally random (all patterns equally likely)
        """
        from math import factorial, log

        n = len(series)
        if n < (order - 1) * delay + order:
            return 1.0  # Not enough data, assume random

        # Extract ordinal patterns
        pattern_counts: dict[tuple, int] = {}
        total = 0

        for i in range(n - (order - 1) * delay):
            # Extract the pattern indices
            window = [series[i + j * delay] for j in range(order)]
            # Convert to ordinal pattern (rank order)
            pattern = tuple(sorted(range(order), key=lambda k: window[k]))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total += 1

        if total == 0:
            return 1.0

        # Shannon entropy of pattern distribution
        max_patterns = factorial(order)
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log(p)

        # Normalize by max possible entropy: log(order!)
        max_entropy = log(max_patterns)
        if max_entropy == 0:
            return 1.0

        return entropy / max_entropy

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        min_bars = max(self.entropy_lookback, self.momentum_period * 2) + 1
        if len(df) < min_bars:
            return signals

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        for i in range(min_bars, len(df)):
            # Compute permutation entropy on recent returns
            window = closes[i - self.entropy_lookback:i + 1]
            returns = np.diff(window) / window[:-1]
            pe = self._permutation_entropy(returns, order=3, delay=1)

            # Only trade when entropy is LOW (market has exploitable structure)
            if pe >= self.entropy_threshold:
                continue  # Too random, no edge

            # Simple momentum signal on raw prices
            if closes[i - self.momentum_period] == 0:
                continue
            mom = (closes[i] - closes[i - self.momentum_period]) / closes[i - self.momentum_period]

            # ATR for stops
            recent_tr = [
                max(highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]))
                for j in range(max(1, i - 14), i + 1)
            ]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.02
            atr_pct = atr / closes[i] * 100

            # Confidence scales with how far below the entropy threshold we are
            entropy_edge = (self.entropy_threshold - pe) / self.entropy_threshold

            if mom > 0.01:  # Positive momentum + low entropy
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, entropy_edge * abs(mom) * 20),
                    reason=(
                        f"Entropy regime long: PE={pe:.3f} (thresh={self.entropy_threshold}), "
                        f"mom={mom:.4f}"
                    ),
                    stop_loss_pct=max(1.0, atr_pct * 2),
                    take_profit_pct=max(1.5, atr_pct * 3),
                ))
            elif mom < -0.01:  # Negative momentum + low entropy
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, entropy_edge * abs(mom) * 20),
                    reason=(
                        f"Entropy regime short: PE={pe:.3f} (thresh={self.entropy_threshold}), "
                        f"mom={mom:.4f}"
                    ),
                    stop_loss_pct=max(1.0, atr_pct * 2),
                    take_profit_pct=max(1.5, atr_pct * 3),
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# BREAKOUT & TREND STRATEGIES
# ═══════════════════════════════════════════════════════════════


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout — the Turtle Trading system.

    Buy when price breaks above the highest high of N periods.
    Sell when price breaks below the lowest low of N periods.

    This is the simplest trend-following strategy and one of the most
    robust historically. Richard Dennis made $400M+ with this approach.

    Uses a shorter exit channel than entry channel (asymmetric channels)
    to let winners run while cutting losers quickly.

    Parameters: entry_period, exit_period.
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        self.entry_period = entry_period
        self.exit_period = exit_period

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="donchian_breakout",
            asset_classes=["stock", "crypto"],
            param_count=2,
            param_ranges={"entry_period": (10, 55), "exit_period": (5, 20)},
            description="Turtle-style Donchian channel breakout — the original trend-following system",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.entry_period + 2:
            return signals

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        for i in range(self.entry_period + 1, len(df)):
            upper_channel = np.max(highs[i - self.entry_period:i])
            lower_channel = np.min(lows[i - self.entry_period:i])

            # ATR for confidence scaling
            recent_tr = [max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
                         for j in range(max(1, i - 14), i + 1)]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.02
            atr_pct = atr / closes[i] * 100

            # Breakout above upper channel
            if closes[i] > upper_channel and closes[i - 1] <= upper_channel:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, (closes[i] - upper_channel) / atr) if atr > 0 else 0.5,
                    reason=f"Donchian breakout long: close={closes[i]:.2f} > upper={upper_channel:.2f}",
                    stop_loss_pct=max(1.5, atr_pct * 2),
                    take_profit_pct=max(3.0, atr_pct * 5),  # Wide target for trend
                ))
            # Breakdown below lower channel
            elif closes[i] < lower_channel and closes[i - 1] >= lower_channel:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, (lower_channel - closes[i]) / atr) if atr > 0 else 0.5,
                    reason=f"Donchian breakdown short: close={closes[i]:.2f} < lower={lower_channel:.2f}",
                    stop_loss_pct=max(1.5, atr_pct * 2),
                    take_profit_pct=max(3.0, atr_pct * 5),
                ))

        return signals


class MomentumRankRotation(BaseStrategy):
    """
    Momentum Rank Rotation — cross-sectional relative strength.

    Ranks assets by trailing return and only goes long on those
    in the top momentum tier. This is the Jegadeesh & Titman (1993)
    momentum factor applied at the asset level.

    For single-asset backtesting, this acts as a momentum regime filter:
    only trades when the asset's momentum is in the top percentile
    of its own rolling distribution.

    Parameters: lookback_period, momentum_threshold.
    """

    def __init__(self, lookback_period: int = 60, momentum_threshold: float = 0.7):
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold  # Percentile threshold

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="momentum_rank_rotation",
            asset_classes=["stock", "crypto"],
            param_count=2,
            param_ranges={"lookback_period": (20, 120), "momentum_threshold": (0.5, 0.9)},
            description="Cross-sectional momentum rank — only trade top-percentile momentum",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        min_bars = self.lookback_period + 60  # Need history for percentile calc
        if len(df) < min_bars:
            return signals

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # Pre-compute all trailing returns
        for i in range(min_bars, len(df)):
            # Current momentum
            if closes[i - self.lookback_period] <= 0:
                continue
            current_mom = (closes[i] - closes[i - self.lookback_period]) / closes[i - self.lookback_period]

            # Historical momentum distribution (rolling)
            hist_moms = []
            for j in range(i - 59, i + 1):
                if j >= self.lookback_period and closes[j - self.lookback_period] > 0:
                    m = (closes[j] - closes[j - self.lookback_period]) / closes[j - self.lookback_period]
                    hist_moms.append(m)

            if len(hist_moms) < 20:
                continue

            # Percentile rank of current momentum
            rank = sum(1 for m in hist_moms if m <= current_mom) / len(hist_moms)

            # ATR for stops
            recent_tr = [max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
                         for j in range(max(1, i - 14), i + 1)]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.02
            atr_pct = atr / closes[i] * 100

            # Top percentile momentum = go long
            if rank >= self.momentum_threshold and current_mom > 0:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, rank * abs(current_mom) * 10),
                    reason=f"Momentum rank long: rank={rank:.2f} mom={current_mom:.3f}",
                    stop_loss_pct=max(2.0, atr_pct * 2.5),
                    take_profit_pct=max(4.0, atr_pct * 5),
                ))
            # Bottom percentile = go short
            elif rank <= (1.0 - self.momentum_threshold) and current_mom < 0:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, (1 - rank) * abs(current_mom) * 10),
                    reason=f"Momentum rank short: rank={rank:.2f} mom={current_mom:.3f}",
                    stop_loss_pct=max(2.0, atr_pct * 2.5),
                    take_profit_pct=max(4.0, atr_pct * 5),
                ))

        return signals


class VolatilityBreakoutKeltner(BaseStrategy):
    """
    Keltner Channel Volatility Breakout.

    Keltner Channels = EMA ± ATR * multiplier. Unlike Bollinger Bands
    (which use standard deviation), Keltner uses ATR, making them more
    responsive to directional volatility.

    Entry: price closes outside the channel
    Exit: price returns inside the channel OR trailing stop

    This captures volatility expansion breakouts — the same concept
    as a "squeeze" (Bollinger inside Keltner, then breakout).

    Parameters: ema_period, atr_mult.
    """

    def __init__(self, ema_period: int = 20, atr_mult: float = 2.0):
        self.ema_period = ema_period
        self.atr_mult = atr_mult

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="keltner_breakout",
            asset_classes=["stock", "crypto"],
            param_count=2,
            param_ranges={"ema_period": (10, 40), "atr_mult": (1.5, 3.5)},
            description="Keltner channel breakout — ATR-based volatility expansion plays",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        min_bars = self.ema_period + 15
        if len(df) < min_bars:
            return signals

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # Compute EMA
        ema = np.zeros(len(closes))
        ema[0] = closes[0]
        alpha = 2.0 / (self.ema_period + 1)
        for j in range(1, len(closes)):
            ema[j] = alpha * closes[j] + (1 - alpha) * ema[j - 1]

        for i in range(min_bars, len(df)):
            # ATR
            recent_tr = [max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
                         for j in range(max(1, i - 14), i + 1)]
            atr = np.mean(recent_tr) if recent_tr else closes[i] * 0.02
            atr_pct = atr / closes[i] * 100

            upper = ema[i] + self.atr_mult * atr
            lower = ema[i] - self.atr_mult * atr

            # Breakout above upper Keltner
            if closes[i] > upper and closes[i - 1] <= ema[i - 1] + self.atr_mult * atr:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, (closes[i] - upper) / atr) if atr > 0 else 0.5,
                    reason=f"Keltner breakout long: close={closes[i]:.2f} > upper={upper:.2f}",
                    stop_loss_pct=max(1.0, atr_pct * 1.5),
                    take_profit_pct=max(3.0, atr_pct * 4),
                ))
            # Breakdown below lower Keltner
            elif closes[i] < lower and closes[i - 1] >= ema[i - 1] - self.atr_mult * atr:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, (lower - closes[i]) / atr) if atr > 0 else 0.5,
                    reason=f"Keltner breakdown short: close={closes[i]:.2f} < lower={lower:.2f}",
                    stop_loss_pct=max(1.0, atr_pct * 1.5),
                    take_profit_pct=max(3.0, atr_pct * 4),
                ))

        return signals


# ═══════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════


def get_all_strategies(asset_class: str | None = None) -> list[BaseStrategy]:
    """Get all available strategies, optionally filtered by asset class.

    VWAPReversion removed — negative Sharpe across all assets, unfixable.
    OpeningRangeBreakout removed — fundamentally broken on daily bars (needs intraday data).
    DualMomentum expanded with longer lookback periods (60/90/120 day).
    CryptoMomentum expanded with tuned parameters.
    Added: DenoisedMomentum (Kalman-filtered), HurstAdaptive, EntropyRegimeStrategy.
    Added: DonchianBreakout (Turtle), MomentumRankRotation (cross-sectional), KeltnerBreakout.
    """
    all_strategies = [
        # Stocks — momentum-focused (ORB removed: needs intraday bars)
        GapFade(min_gap_pct=0.5, max_gap_pct=3.0),
        GapFade(min_gap_pct=1.0, max_gap_pct=5.0),
        DualMomentum(fast_period=10, slow_period=40),
        DualMomentum(fast_period=5, slow_period=20),
        DualMomentum(fast_period=15, slow_period=60),    # Medium-term
        DualMomentum(fast_period=20, slow_period=90),    # Longer-term
        DualMomentum(fast_period=30, slow_period=120),   # Macro momentum
        # Crypto — momentum dominates
        CryptoMomentum(roc_period=12, volume_mult=1.5),
        CryptoMomentum(roc_period=8, volume_mult=2.0),
        CryptoMomentum(roc_period=18, volume_mult=1.3, roc_threshold=2.0),  # Longer/looser
        CryptoMomentum(roc_period=6, volume_mult=1.5, roc_threshold=4.0),   # Shorter/stricter
        CryptoMeanReversion(period=20, num_std=2.5),
        CryptoMeanReversion(period=30, num_std=3.0),
        # Polymarket
        PredictionMomentum(lookback=12, threshold_change=0.05),
        PredictionMomentum(lookback=6, threshold_change=0.08),
        PredictionReversion(spike_threshold=0.10, lookback=6),
        PredictionReversion(spike_threshold=0.15, lookback=4),
        # Universal — adaptive
        AdaptiveTrend(er_period=10),
        AdaptiveTrend(er_period=15),
        # Signal-processing strategies — denoised momentum
        DenoisedMomentum(fast_period=10, slow_period=40, noise_ratio=1.0),   # Default
        DenoisedMomentum(fast_period=5, slow_period=20, noise_ratio=0.5),    # Fast/responsive
        DenoisedMomentum(fast_period=15, slow_period=60, noise_ratio=2.0),   # Slow/smooth
        # Hurst-adaptive — auto-switches momentum vs reversion
        HurstAdaptive(hurst_lookback=100, momentum_period=20),
        HurstAdaptive(hurst_lookback=150, momentum_period=30),    # Longer lookback
        # Entropy regime — only trade when market is predictable
        EntropyRegimeStrategy(entropy_lookback=50, entropy_threshold=0.7, momentum_period=20),
        EntropyRegimeStrategy(entropy_lookback=80, entropy_threshold=0.65, momentum_period=30),
        # Trend-following — Turtle system (Donchian breakout)
        DonchianBreakout(entry_period=20, exit_period=10),         # Classic Turtle
        DonchianBreakout(entry_period=55, exit_period=20),         # Long-term Turtle
        DonchianBreakout(entry_period=10, exit_period=5),          # Short-term breakout
        # Cross-sectional momentum — rank-based
        MomentumRankRotation(lookback_period=60, momentum_threshold=0.7),   # 3-month momentum
        MomentumRankRotation(lookback_period=120, momentum_threshold=0.75), # 6-month momentum
        MomentumRankRotation(lookback_period=20, momentum_threshold=0.8),   # 1-month fast rotation
        # Keltner channel — ATR-based volatility breakout
        VolatilityBreakoutKeltner(ema_period=20, atr_mult=2.0),    # Standard
        VolatilityBreakoutKeltner(ema_period=10, atr_mult=1.5),    # Tight/fast
        VolatilityBreakoutKeltner(ema_period=30, atr_mult=2.5),    # Wide/slow
    ]

    if asset_class:
        return [s for s in all_strategies if asset_class in s.meta().asset_classes]
    return all_strategies
