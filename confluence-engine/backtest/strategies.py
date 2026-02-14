"""
SPY Daytrading Strategies for Backtesting.

Each strategy returns a signal Series: +1 (long), -1 (short), 0 (flat).
Strategies are designed for daily-bar simulation of intraday patterns.

All entries are assumed at Open, exits at Close of the same day (daytrading)
unless otherwise specified. Some strategies hold overnight (swing).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Strategy(ABC):
    """Base class for all strategies."""

    name: str = "Base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Must add columns to df:
          - 'signal': +1 (long), -1 (short), 0 (flat)
          - 'entry_price': price at which we enter
          - 'exit_price': price at which we exit
          - 'stop_loss': stop loss price (optional)
          - 'take_profit': take profit price (optional)

        Returns the modified dataframe.
        """
        pass

    @property
    def params_dict(self) -> dict:
        """Return strategy parameters as a dict."""
        return {}


# ---------------------------------------------------------------------------
# 1. GAP AND GO — Buy gap-up opens, ride momentum to close
# ---------------------------------------------------------------------------
class GapAndGo(Strategy):
    name = "Gap & Go"

    def __init__(self, min_gap_pct: float = 0.3, max_gap_pct: float = 2.0,
                 volume_filter: float = 1.2, stop_atr_mult: float = 1.0):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.volume_filter = volume_filter
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"min_gap": self.min_gap_pct, "max_gap": self.max_gap_pct,
                "vol_filter": self.volume_filter, "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Gap up within range, volume above average
        gap_up = (df["Gap_Pct"] >= self.min_gap_pct) & (df["Gap_Pct"] <= self.max_gap_pct)
        vol_ok = df["Vol_Ratio"] >= self.volume_filter

        # Trend filter: above 20 SMA
        trend_ok = df["Close"].shift(1) > df["SMA_20"].shift(1)

        long_signal = gap_up & vol_ok & trend_ok
        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 2. GAP FADE — Fade large gaps expecting mean reversion / gap fill
# ---------------------------------------------------------------------------
class GapFade(Strategy):
    name = "Gap Fade"

    def __init__(self, min_gap_pct: float = 0.5, max_gap_pct: float = 3.0,
                 rsi_threshold: float = 70, stop_atr_mult: float = 1.5):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.rsi_threshold = rsi_threshold
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"min_gap": self.min_gap_pct, "max_gap": self.max_gap_pct,
                "rsi_thresh": self.rsi_threshold, "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Fade gap ups: short when gap up is large + RSI overbought
        gap_up_fade = (
            (df["Gap_Pct"] >= self.min_gap_pct) &
            (df["Gap_Pct"] <= self.max_gap_pct) &
            (df["RSI"].shift(1) >= self.rsi_threshold)
        )
        df.loc[gap_up_fade, "signal"] = -1
        df.loc[gap_up_fade, "entry_price"] = df.loc[gap_up_fade, "Open"]
        df.loc[gap_up_fade, "exit_price"] = df.loc[gap_up_fade, "Close"]
        df.loc[gap_up_fade, "stop_loss"] = (
            df.loc[gap_up_fade, "Open"] + self.stop_atr_mult * df.loc[gap_up_fade, "ATR"]
        )

        # Fade gap downs: buy when gap down is large + RSI oversold
        gap_dn_fade = (
            (df["Gap_Pct"] <= -self.min_gap_pct) &
            (df["Gap_Pct"] >= -self.max_gap_pct) &
            (df["RSI"].shift(1) <= (100 - self.rsi_threshold))
        )
        df.loc[gap_dn_fade, "signal"] = 1
        df.loc[gap_dn_fade, "entry_price"] = df.loc[gap_dn_fade, "Open"]
        df.loc[gap_dn_fade, "exit_price"] = df.loc[gap_dn_fade, "Close"]
        df.loc[gap_dn_fade, "stop_loss"] = (
            df.loc[gap_dn_fade, "Open"] - self.stop_atr_mult * df.loc[gap_dn_fade, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 3. OPENING RANGE BREAKOUT (ORB) — Simulated on daily bars
#    Uses previous day's range; if open breaks above prev high -> long
# ---------------------------------------------------------------------------
class OpeningRangeBreakout(Strategy):
    name = "ORB Breakout"

    def __init__(self, atr_filter: float = 0.5, volume_filter: float = 1.0,
                 stop_atr_mult: float = 1.0):
        self.atr_filter = atr_filter
        self.volume_filter = volume_filter
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"atr_filter": self.atr_filter, "vol_filter": self.volume_filter,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Breakout above previous day's high
        breakout_long = (
            (df["High"] > df["Prev_High"]) &
            (df["Open"] <= df["Prev_High"]) &  # didn't gap above
            (df["Vol_Ratio"] >= self.volume_filter) &
            (df["ATR"] >= self.atr_filter)
        )
        # Simulated entry at previous high (breakout level)
        df.loc[breakout_long, "signal"] = 1
        df.loc[breakout_long, "entry_price"] = df.loc[breakout_long, "Prev_High"]
        df.loc[breakout_long, "exit_price"] = df.loc[breakout_long, "Close"]
        df.loc[breakout_long, "stop_loss"] = (
            df.loc[breakout_long, "Prev_High"] - self.stop_atr_mult * df.loc[breakout_long, "ATR"]
        )

        # Breakdown below previous day's low
        breakout_short = (
            (df["Low"] < df["Prev_Low"]) &
            (df["Open"] >= df["Prev_Low"]) &  # didn't gap below
            (df["Vol_Ratio"] >= self.volume_filter) &
            (df["ATR"] >= self.atr_filter) &
            (df["signal"] == 0)  # don't conflict with long
        )
        df.loc[breakout_short, "signal"] = -1
        df.loc[breakout_short, "entry_price"] = df.loc[breakout_short, "Prev_Low"]
        df.loc[breakout_short, "exit_price"] = df.loc[breakout_short, "Close"]
        df.loc[breakout_short, "stop_loss"] = (
            df.loc[breakout_short, "Prev_Low"] + self.stop_atr_mult * df.loc[breakout_short, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 4. MEAN REVERSION (Bollinger Band) — Buy at lower band, sell at upper
# ---------------------------------------------------------------------------
class BollingerMeanReversion(Strategy):
    name = "BB Mean Reversion"

    def __init__(self, bb_entry: float = 0.0, bb_exit: float = 0.5,
                 rsi_oversold: float = 30, rsi_overbought: float = 70,
                 stop_atr_mult: float = 2.0):
        self.bb_entry = bb_entry
        self.bb_exit = bb_exit
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"bb_entry": self.bb_entry, "bb_exit": self.bb_exit,
                "rsi_os": self.rsi_oversold, "rsi_ob": self.rsi_overbought,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Long: price touches lower band + RSI oversold
        long_signal = (
            (df["BB_pct"].shift(1) <= self.bb_entry) &
            (df["RSI"].shift(1) <= self.rsi_oversold)
        )
        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        # Short: price touches upper band + RSI overbought
        short_signal = (
            (df["BB_pct"].shift(1) >= (1 - self.bb_entry)) &
            (df["RSI"].shift(1) >= self.rsi_overbought) &
            (df["signal"] == 0)
        )
        df.loc[short_signal, "signal"] = -1
        df.loc[short_signal, "entry_price"] = df.loc[short_signal, "Open"]
        df.loc[short_signal, "exit_price"] = df.loc[short_signal, "Close"]
        df.loc[short_signal, "stop_loss"] = (
            df.loc[short_signal, "Open"] + self.stop_atr_mult * df.loc[short_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 5. RSI REVERSAL — Trade RSI extremes with confirmation
# ---------------------------------------------------------------------------
class RSIReversal(Strategy):
    name = "RSI Reversal"

    def __init__(self, rsi_oversold: float = 25, rsi_overbought: float = 75,
                 require_reversal: bool = True, stop_atr_mult: float = 1.5):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.require_reversal = require_reversal
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"rsi_os": self.rsi_oversold, "rsi_ob": self.rsi_overbought,
                "reversal": self.require_reversal, "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        prev_rsi = df["RSI"].shift(1)
        prev_prev_rsi = df["RSI"].shift(2)

        if self.require_reversal:
            # RSI was oversold and is now turning up
            long_signal = (
                (prev_prev_rsi <= self.rsi_oversold) &
                (prev_rsi > prev_prev_rsi) &
                (prev_rsi <= self.rsi_oversold + 10)
            )
            # RSI was overbought and is now turning down
            short_signal = (
                (prev_prev_rsi >= self.rsi_overbought) &
                (prev_rsi < prev_prev_rsi) &
                (prev_rsi >= self.rsi_overbought - 10)
            )
        else:
            long_signal = prev_rsi <= self.rsi_oversold
            short_signal = prev_rsi >= self.rsi_overbought

        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        short_signal = short_signal & (df["signal"] == 0)
        df.loc[short_signal, "signal"] = -1
        df.loc[short_signal, "entry_price"] = df.loc[short_signal, "Open"]
        df.loc[short_signal, "exit_price"] = df.loc[short_signal, "Close"]
        df.loc[short_signal, "stop_loss"] = (
            df.loc[short_signal, "Open"] + self.stop_atr_mult * df.loc[short_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 6. VWAP BOUNCE — Trade bounces off rolling VWAP proxy
# ---------------------------------------------------------------------------
class VWAPBounce(Strategy):
    name = "VWAP Bounce"

    def __init__(self, threshold_pct: float = 0.3, trend_ma: int = 50,
                 stop_atr_mult: float = 1.0):
        self.threshold_pct = threshold_pct
        self.trend_ma = trend_ma
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"threshold": self.threshold_pct, "trend_ma": self.trend_ma,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        vwap_dist = (df["Close"].shift(1) - df["VWAP_20"].shift(1)) / df["VWAP_20"].shift(1) * 100
        trend_sma = df["Close"].rolling(self.trend_ma).mean()

        # Long: price pulled back to near VWAP in uptrend
        long_signal = (
            (vwap_dist <= -self.threshold_pct) &
            (vwap_dist >= -self.threshold_pct * 3) &  # not too far below
            (df["Close"].shift(1) > trend_sma.shift(1))  # uptrend
        )
        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        # Short: price extended above VWAP in downtrend
        short_signal = (
            (vwap_dist >= self.threshold_pct) &
            (vwap_dist <= self.threshold_pct * 3) &
            (df["Close"].shift(1) < trend_sma.shift(1)) &
            (df["signal"] == 0)
        )
        df.loc[short_signal, "signal"] = -1
        df.loc[short_signal, "entry_price"] = df.loc[short_signal, "Open"]
        df.loc[short_signal, "exit_price"] = df.loc[short_signal, "Close"]
        df.loc[short_signal, "stop_loss"] = (
            df.loc[short_signal, "Open"] + self.stop_atr_mult * df.loc[short_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 7. MOMENTUM CROSSOVER — EMA 9/21 crossover with confirmation
# ---------------------------------------------------------------------------
class MomentumCrossover(Strategy):
    name = "EMA Crossover"

    def __init__(self, fast_ema: int = 9, slow_ema: int = 21,
                 macd_confirm: bool = True, stop_atr_mult: float = 1.5):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.macd_confirm = macd_confirm
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"fast": self.fast_ema, "slow": self.slow_ema,
                "macd": self.macd_confirm, "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        fast = df["Close"].ewm(span=self.fast_ema, adjust=False).mean()
        slow = df["Close"].ewm(span=self.slow_ema, adjust=False).mean()

        # Crossover: fast crosses above slow
        cross_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
        # Crossunder: fast crosses below slow
        cross_dn = (fast.shift(1) >= slow.shift(1)) & (fast < slow)

        if self.macd_confirm:
            cross_up = cross_up & (df["MACD_Hist"] > 0)
            cross_dn = cross_dn & (df["MACD_Hist"] < 0)

        df.loc[cross_up, "signal"] = 1
        df.loc[cross_up, "entry_price"] = df.loc[cross_up, "Open"]
        df.loc[cross_up, "exit_price"] = df.loc[cross_up, "Close"]
        df.loc[cross_up, "stop_loss"] = (
            df.loc[cross_up, "Open"] - self.stop_atr_mult * df.loc[cross_up, "ATR"]
        )

        cross_dn = cross_dn & (df["signal"] == 0)
        df.loc[cross_dn, "signal"] = -1
        df.loc[cross_dn, "entry_price"] = df.loc[cross_dn, "Open"]
        df.loc[cross_dn, "exit_price"] = df.loc[cross_dn, "Close"]
        df.loc[cross_dn, "stop_loss"] = (
            df.loc[cross_dn, "Open"] + self.stop_atr_mult * df.loc[cross_dn, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 8. VOLATILITY BREAKOUT (Donchian) — Trade channel breakouts
# ---------------------------------------------------------------------------
class VolatilityBreakout(Strategy):
    name = "Volatility Breakout"

    def __init__(self, channel_period: int = 20, volume_filter: float = 1.3,
                 stop_atr_mult: float = 1.5):
        self.channel_period = channel_period
        self.volume_filter = volume_filter
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"period": self.channel_period, "vol_filter": self.volume_filter,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        dc_upper = df["High"].rolling(self.channel_period).max().shift(1)
        dc_lower = df["Low"].rolling(self.channel_period).min().shift(1)

        # Breakout above channel
        long_signal = (
            (df["Close"] > dc_upper) &
            (df["Vol_Ratio"] >= self.volume_filter)
        )
        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        # Breakdown below channel
        short_signal = (
            (df["Close"] < dc_lower) &
            (df["Vol_Ratio"] >= self.volume_filter) &
            (df["signal"] == 0)
        )
        df.loc[short_signal, "signal"] = -1
        df.loc[short_signal, "entry_price"] = df.loc[short_signal, "Open"]
        df.loc[short_signal, "exit_price"] = df.loc[short_signal, "Close"]
        df.loc[short_signal, "stop_loss"] = (
            df.loc[short_signal, "Open"] + self.stop_atr_mult * df.loc[short_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 9. STOCHASTIC REVERSAL — Trade stochastic K/D crosses at extremes
# ---------------------------------------------------------------------------
class StochasticReversal(Strategy):
    name = "Stochastic Reversal"

    def __init__(self, oversold: float = 20, overbought: float = 80,
                 stop_atr_mult: float = 1.5):
        self.oversold = oversold
        self.overbought = overbought
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"os": self.oversold, "ob": self.overbought,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Long: K crosses above D in oversold zone
        long_signal = (
            (df["Stoch_K"].shift(2) < df["Stoch_D"].shift(2)) &
            (df["Stoch_K"].shift(1) >= df["Stoch_D"].shift(1)) &
            (df["Stoch_K"].shift(1) <= self.oversold + 10)
        )
        df.loc[long_signal, "signal"] = 1
        df.loc[long_signal, "entry_price"] = df.loc[long_signal, "Open"]
        df.loc[long_signal, "exit_price"] = df.loc[long_signal, "Close"]
        df.loc[long_signal, "stop_loss"] = (
            df.loc[long_signal, "Open"] - self.stop_atr_mult * df.loc[long_signal, "ATR"]
        )

        # Short: K crosses below D in overbought zone
        short_signal = (
            (df["Stoch_K"].shift(2) > df["Stoch_D"].shift(2)) &
            (df["Stoch_K"].shift(1) <= df["Stoch_D"].shift(1)) &
            (df["Stoch_K"].shift(1) >= self.overbought - 10) &
            (df["signal"] == 0)
        )
        df.loc[short_signal, "signal"] = -1
        df.loc[short_signal, "entry_price"] = df.loc[short_signal, "Open"]
        df.loc[short_signal, "exit_price"] = df.loc[short_signal, "Close"]
        df.loc[short_signal, "stop_loss"] = (
            df.loc[short_signal, "Open"] + self.stop_atr_mult * df.loc[short_signal, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 10. REGIME-FILTERED TREND — Only trade with the trend, filtered by VIX
# ---------------------------------------------------------------------------
class RegimeTrend(Strategy):
    name = "Regime Trend"

    def __init__(self, vix_calm_threshold: float = 20, vix_fear_threshold: float = 30,
                 stop_atr_mult: float = 1.5):
        self.vix_calm_threshold = vix_calm_threshold
        self.vix_fear_threshold = vix_fear_threshold
        self.stop_atr_mult = stop_atr_mult

    @property
    def params_dict(self) -> dict:
        return {"vix_calm": self.vix_calm_threshold,
                "vix_fear": self.vix_fear_threshold,
                "stop_atr": self.stop_atr_mult}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        has_vix = "VIX_Close" in df.columns

        # Trend: EMA 9 > EMA 21 > SMA 50
        uptrend = (
            (df["EMA_9"].shift(1) > df["EMA_21"].shift(1)) &
            (df["EMA_21"].shift(1) > df["SMA_50"].shift(1))
        )
        downtrend = (
            (df["EMA_9"].shift(1) < df["EMA_21"].shift(1)) &
            (df["EMA_21"].shift(1) < df["SMA_50"].shift(1))
        )

        # Pullback to EMA 21
        pullback_long = (
            uptrend &
            (df["Low"] <= df["EMA_21"] * 1.005) &
            (df["Close"].shift(1) > df["EMA_21"].shift(1))
        )
        pullback_short = (
            downtrend &
            (df["High"] >= df["EMA_21"] * 0.995) &
            (df["Close"].shift(1) < df["EMA_21"].shift(1))
        )

        if has_vix:
            # In calm markets, favor longs; in fear, favor shorts
            calm = df["VIX_Close"].shift(1) < self.vix_calm_threshold
            fear = df["VIX_Close"].shift(1) > self.vix_fear_threshold
            pullback_long = pullback_long & calm
            pullback_short = pullback_short & fear

        df.loc[pullback_long, "signal"] = 1
        df.loc[pullback_long, "entry_price"] = df.loc[pullback_long, "Open"]
        df.loc[pullback_long, "exit_price"] = df.loc[pullback_long, "Close"]
        df.loc[pullback_long, "stop_loss"] = (
            df.loc[pullback_long, "Open"] - self.stop_atr_mult * df.loc[pullback_long, "ATR"]
        )

        pullback_short = pullback_short & (df["signal"] == 0)
        df.loc[pullback_short, "signal"] = -1
        df.loc[pullback_short, "entry_price"] = df.loc[pullback_short, "Open"]
        df.loc[pullback_short, "exit_price"] = df.loc[pullback_short, "Close"]
        df.loc[pullback_short, "stop_loss"] = (
            df.loc[pullback_short, "Open"] + self.stop_atr_mult * df.loc[pullback_short, "ATR"]
        )

        return df


# ---------------------------------------------------------------------------
# 11. COMPOSITE OPTIMAL — Combine best signals with voting
# ---------------------------------------------------------------------------
class CompositeOptimal(Strategy):
    name = "Composite Optimal"

    def __init__(self, min_votes: int = 3, stop_atr_mult: float = 1.5):
        self.min_votes = min_votes
        self.stop_atr_mult = stop_atr_mult
        self.sub_strategies = [
            GapAndGo(),
            OpeningRangeBreakout(),
            BollingerMeanReversion(),
            RSIReversal(),
            VWAPBounce(),
            MomentumCrossover(),
            VolatilityBreakout(),
            StochasticReversal(),
        ]

    @property
    def params_dict(self) -> dict:
        return {"min_votes": self.min_votes, "stop_atr": self.stop_atr_mult,
                "n_strategies": len(self.sub_strategies)}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Collect signals from all sub-strategies
        all_signals = []
        for strat in self.sub_strategies:
            sig_df = strat.generate_signals(df)
            all_signals.append(sig_df["signal"].values)

        votes = np.array(all_signals)  # shape: (n_strategies, n_bars)
        long_votes = (votes == 1).sum(axis=0)
        short_votes = (votes == -1).sum(axis=0)

        df["signal"] = 0
        df["entry_price"] = np.nan
        df["exit_price"] = np.nan
        df["stop_loss"] = np.nan

        # Long when enough strategies agree
        long_mask = long_votes >= self.min_votes
        df.loc[long_mask, "signal"] = 1
        df.loc[long_mask, "entry_price"] = df.loc[long_mask, "Open"]
        df.loc[long_mask, "exit_price"] = df.loc[long_mask, "Close"]
        df.loc[long_mask, "stop_loss"] = (
            df.loc[long_mask, "Open"] - self.stop_atr_mult * df.loc[long_mask, "ATR"]
        )

        # Short when enough strategies agree
        short_mask = (short_votes >= self.min_votes) & (df["signal"] == 0)
        df.loc[short_mask, "signal"] = -1
        df.loc[short_mask, "entry_price"] = df.loc[short_mask, "Open"]
        df.loc[short_mask, "exit_price"] = df.loc[short_mask, "Close"]
        df.loc[short_mask, "stop_loss"] = (
            df.loc[short_mask, "Open"] + self.stop_atr_mult * df.loc[short_mask, "ATR"]
        )

        return df


def get_all_strategies() -> list[Strategy]:
    """Return all strategies with default parameters."""
    return [
        GapAndGo(),
        GapFade(),
        OpeningRangeBreakout(),
        BollingerMeanReversion(),
        RSIReversal(),
        VWAPBounce(),
        MomentumCrossover(),
        VolatilityBreakout(),
        StochasticReversal(),
        RegimeTrend(),
        CompositeOptimal(min_votes=2),
        CompositeOptimal(min_votes=3),
    ]
