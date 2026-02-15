"""
Extended strategy library for multi-asset day trading.

Strategies are organized by asset class and market regime:
- Stock strategies: VWAP reversion, ORB, gap fade, momentum
- Crypto strategies: Funding rate, liquidation levels, momentum
- Polymarket strategies: Probability momentum, mean reversion
- Cross-asset: Regime-adaptive, volatility harvesting

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


class VWAPReversion(BaseStrategy):
    """
    VWAP Mean Reversion — bread-and-butter day trading strategy.

    Buy when price drops N standard deviations below VWAP.
    Sell when price rises N standard deviations above VWAP.
    Exit at VWAP (mean reversion target).

    Works best in range-bound, liquid markets.
    Parameters: lookback (for rolling VWAP), entry_std, exit_std.
    """

    def __init__(self, lookback: int = 20, entry_std: float = 2.0, exit_std: float = 0.5):
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="vwap_reversion",
            asset_classes=["stock", "crypto"],
            param_count=3,
            param_ranges={"lookback": (10, 50), "entry_std": (1.5, 3.0), "exit_std": (0.2, 1.0)},
            description="Buy below VWAP bands, sell above — mean reversion to fair value",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.lookback + 1:
            return signals

        # Compute rolling VWAP (approximation using typical price * volume)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).rolling(self.lookback).sum()
        cum_vol = df["volume"].rolling(self.lookback).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Standard deviation of price around VWAP
        deviation = (df["close"] - vwap).rolling(self.lookback).std()

        for i in range(self.lookback, len(df)):
            if pd.isna(vwap.iloc[i]) or pd.isna(deviation.iloc[i]) or deviation.iloc[i] <= 0:
                continue

            z_score = (df["close"].iloc[i] - vwap.iloc[i]) / deviation.iloc[i]

            # Long: price well below VWAP
            if z_score < -self.entry_std:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, abs(z_score) / (self.entry_std * 2)),
                    reason=f"VWAP reversion: z={z_score:.2f}, price={df['close'].iloc[i]:.2f}, vwap={vwap.iloc[i]:.2f}",
                    stop_loss_pct=abs(z_score) * deviation.iloc[i] / df["close"].iloc[i] * 100 * 1.5,
                ))
            # Short: price well above VWAP
            elif z_score > self.entry_std:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(z_score) / (self.entry_std * 2)),
                    reason=f"VWAP reversion short: z={z_score:.2f}",
                    stop_loss_pct=abs(z_score) * deviation.iloc[i] / df["close"].iloc[i] * 100 * 1.5,
                ))

        return signals


class OpeningRangeBreakout(BaseStrategy):
    """
    Opening Range Breakout (ORB) — classic day trading strategy.

    Define the opening range as the high/low of the first N bars.
    Go long on breakout above the range, short on breakdown below.
    Stop loss at the opposite side of the range.

    Best on 5m-15m bars. Uses the first 30-60 minutes as the opening range.
    Parameters: range_bars (how many bars define the range), atr_filter.
    """

    def __init__(self, range_bars: int = 6, atr_filter: float = 1.0):
        self.range_bars = range_bars  # e.g., 6 x 5m bars = 30 min opening range
        self.atr_filter = atr_filter

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="opening_range_breakout",
            asset_classes=["stock"],
            param_count=2,
            param_ranges={"range_bars": (3, 12), "atr_filter": (0.5, 2.0)},
            description="Breakout above/below opening range — trend continuation play",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []

        # Group bars by trading day
        df = df.copy()
        df["date"] = df.index.date

        for date, day_df in df.groupby("date"):
            if len(day_df) < self.range_bars + 2:
                continue

            # Opening range = first N bars
            opening = day_df.iloc[:self.range_bars]
            range_high = opening["high"].max()
            range_low = opening["low"].min()
            range_size = range_high - range_low

            if range_size <= 0:
                continue

            # ATR filter: skip if opening range is too narrow (no volatility)
            if self.atr_filter > 0:
                atr_period = min(14, len(day_df) - 1)
                if atr_period < 2:
                    continue

            # Scan remaining bars for breakout
            triggered = False
            for i in range(self.range_bars, len(day_df)):
                if triggered:
                    break

                bar = day_df.iloc[i]
                global_idx = df.index.get_loc(day_df.index[i])

                # Breakout above range
                if bar["close"] > range_high:
                    signals.append(Signal(
                        bar_index=global_idx,
                        side=Side.LONG,
                        confidence=min(1.0, (bar["close"] - range_high) / range_size),
                        reason=f"ORB long: close={bar['close']:.2f} > range_high={range_high:.2f}",
                        stop_loss_pct=(range_high - range_low) / bar["close"] * 100,
                        take_profit_pct=(range_high - range_low) / bar["close"] * 100 * 2,
                    ))
                    triggered = True
                # Breakdown below range
                elif bar["close"] < range_low:
                    signals.append(Signal(
                        bar_index=global_idx,
                        side=Side.SHORT,
                        confidence=min(1.0, (range_low - bar["close"]) / range_size),
                        reason=f"ORB short: close={bar['close']:.2f} < range_low={range_low:.2f}",
                        stop_loss_pct=(range_high - range_low) / bar["close"] * 100,
                        take_profit_pct=(range_high - range_low) / bar["close"] * 100 * 2,
                    ))
                    triggered = True

        return signals


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

    For backtesting, we simplify: go long when both fast and slow
    momentum are positive and accelerating.

    Parameters: fast_period, slow_period.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 40):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="dual_momentum",
            asset_classes=["stock", "crypto"],
            param_count=2,
            param_ranges={"fast_period": (5, 20), "slow_period": (20, 60)},
            description="Dual momentum — long only when both timeframes confirm",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        if len(df) < self.slow_period + 2:
            return signals

        closes = df["close"].values

        for i in range(self.slow_period + 1, len(df)):
            fast_mom = (closes[i] - closes[i - self.fast_period]) / closes[i - self.fast_period]
            slow_mom = (closes[i] - closes[i - self.slow_period]) / closes[i - self.slow_period]

            prev_fast = (closes[i-1] - closes[i-1 - self.fast_period]) / closes[i-1 - self.fast_period]
            prev_slow = (closes[i-1] - closes[i-1 - self.slow_period]) / closes[i-1 - self.slow_period]

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
    Crypto-specific momentum with volume confirmation.

    Crypto trends harder than equities — momentum strategies work better.
    Uses ROC + volume surge as confirmation.

    Parameters: roc_period, volume_multiplier.
    """

    def __init__(self, roc_period: int = 12, volume_mult: float = 1.5):
        self.roc_period = roc_period
        self.volume_mult = volume_mult

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="crypto_momentum",
            asset_classes=["crypto"],
            param_count=2,
            param_ranges={"roc_period": (6, 24), "volume_mult": (1.2, 3.0)},
            description="Crypto momentum with volume surge confirmation",
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        signals = []
        lookback = max(self.roc_period + 1, 20)
        if len(df) < lookback:
            return signals

        closes = df["close"].values
        volumes = df["volume"].values

        for i in range(lookback, len(df)):
            roc = (closes[i] - closes[i - self.roc_period]) / closes[i - self.roc_period] * 100
            avg_vol = np.mean(volumes[i - 20:i])
            vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0

            # Strong momentum + volume confirmation
            if roc > 3.0 and vol_ratio > self.volume_mult:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.LONG,
                    confidence=min(1.0, roc / 10.0 * vol_ratio / 3.0),
                    reason=f"Crypto momentum: ROC={roc:.1f}%, vol={vol_ratio:.1f}x",
                    stop_loss_pct=3.0,
                    take_profit_pct=6.0,
                ))
            elif roc < -3.0 and vol_ratio > self.volume_mult:
                signals.append(Signal(
                    bar_index=i,
                    side=Side.SHORT,
                    confidence=min(1.0, abs(roc) / 10.0 * vol_ratio / 3.0),
                    reason=f"Crypto momentum short: ROC={roc:.1f}%, vol={vol_ratio:.1f}x",
                    stop_loss_pct=3.0,
                    take_profit_pct=6.0,
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
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════


def get_all_strategies(asset_class: str | None = None) -> list[BaseStrategy]:
    """Get all available strategies, optionally filtered by asset class."""
    all_strategies = [
        # Stocks
        VWAPReversion(lookback=20, entry_std=2.0),
        VWAPReversion(lookback=30, entry_std=2.5),
        OpeningRangeBreakout(range_bars=6),
        OpeningRangeBreakout(range_bars=12),
        GapFade(min_gap_pct=0.5, max_gap_pct=3.0),
        GapFade(min_gap_pct=1.0, max_gap_pct=5.0),
        DualMomentum(fast_period=10, slow_period=40),
        DualMomentum(fast_period=5, slow_period=20),
        # Crypto
        CryptoMomentum(roc_period=12, volume_mult=1.5),
        CryptoMomentum(roc_period=8, volume_mult=2.0),
        CryptoMeanReversion(period=20, num_std=2.5),
        CryptoMeanReversion(period=30, num_std=3.0),
        # Polymarket
        PredictionMomentum(lookback=12, threshold_change=0.05),
        PredictionMomentum(lookback=6, threshold_change=0.08),
        PredictionReversion(spike_threshold=0.10, lookback=6),
        PredictionReversion(spike_threshold=0.15, lookback=4),
        # Universal
        AdaptiveTrend(er_period=10),
        AdaptiveTrend(er_period=15),
    ]

    if asset_class:
        return [s for s in all_strategies if asset_class in s.meta().asset_classes]
    return all_strategies
