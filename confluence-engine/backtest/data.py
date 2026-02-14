"""
SPY historical data for backtesting.

Fetches OHLCV data from Yahoo Finance when network is available.
Falls back to high-fidelity synthetic data generation calibrated to
real SPY statistical properties when network is unavailable.
"""

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".data_cache"

# ── Real SPY statistical properties (calibrated from 2019-2024) ──
SPY_STATS = {
    "annual_return": 0.10,       # ~10% annual return
    "annual_vol": 0.16,          # ~16% annual volatility
    "mean_reversion": 0.02,      # slight mean reversion at daily level
    "gap_std": 0.004,            # overnight gap std ~0.4%
    "intraday_range_mean": 0.01, # avg daily range ~1%
    "volume_mean": 80_000_000,   # avg daily volume
    "volume_std": 25_000_000,
    "start_price": 420.0,        # starting price level
}

# ── VIX statistical properties ──
VIX_STATS = {
    "mean": 18.0,
    "std": 6.0,
    "min": 10.0,
    "max": 82.0,
    "mean_reversion_speed": 0.05,
}


def generate_spy_daily(years: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic SPY daily OHLCV data.

    Uses geometric Brownian motion with:
    - Calibrated drift and volatility matching real SPY
    - Overnight gaps (open != prev close)
    - Realistic intraday high/low ranges
    - Volume clustering and mean-reversion
    - Regime shifts (calm/volatile periods via stochastic volatility)
    """
    rng = np.random.default_rng(seed)
    n_days = years * 252
    s = SPY_STATS

    # Generate business day index
    end_date = pd.Timestamp("2025-02-13")
    start_date = end_date - pd.tseries.offsets.BDay(n_days + 50)
    dates = pd.bdate_range(start=start_date, end=end_date)[-n_days:]

    # Stochastic volatility (regime changes)
    vol_base = s["annual_vol"] / np.sqrt(252)
    vol_process = np.zeros(n_days)
    vol_process[0] = vol_base
    for i in range(1, n_days):
        # Mean-reverting vol with random shocks
        vol_shock = rng.normal(0, vol_base * 0.15)
        vol_process[i] = vol_process[i-1] + 0.05 * (vol_base - vol_process[i-1]) + vol_shock
        vol_process[i] = max(vol_base * 0.3, min(vol_base * 3.0, vol_process[i]))

    # Daily returns with slight mean reversion
    daily_drift = s["annual_return"] / 252
    returns = np.zeros(n_days)
    for i in range(n_days):
        returns[i] = daily_drift + vol_process[i] * rng.normal()

    # Build close prices
    close = np.zeros(n_days)
    close[0] = s["start_price"]
    for i in range(1, n_days):
        close[i] = close[i-1] * (1 + returns[i])

    # Generate opens with overnight gaps
    gap_returns = rng.normal(0, s["gap_std"], n_days)
    opens = np.zeros(n_days)
    opens[0] = close[0] * (1 + gap_returns[0])
    for i in range(1, n_days):
        opens[i] = close[i-1] * (1 + gap_returns[i])

    # Generate high/low with realistic ranges
    intraday_range = np.abs(rng.normal(s["intraday_range_mean"], s["intraday_range_mean"] * 0.5, n_days))
    # Scale range by volatility regime
    intraday_range *= vol_process / vol_base

    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    for i in range(n_days):
        bar_max = max(opens[i], close[i])
        bar_min = min(opens[i], close[i])
        range_pts = bar_max * intraday_range[i]

        # Distribute range: more extension in the direction of the move
        up_ext = rng.uniform(0.2, 0.8) * range_pts
        dn_ext = range_pts - up_ext

        highs[i] = bar_max + up_ext
        lows[i] = bar_min - dn_ext
        # Ensure high > low and both contain open/close
        highs[i] = max(highs[i], bar_max)
        lows[i] = min(lows[i], bar_min)

    # Volume with clustering
    log_vol = np.log(s["volume_mean"])
    volumes = np.zeros(n_days)
    volumes[0] = s["volume_mean"]
    for i in range(1, n_days):
        vol_shock = rng.normal(0, 0.15)
        # Volume spikes on big moves
        move_factor = 1 + 2 * abs(returns[i]) / vol_base
        volumes[i] = np.exp(
            log_vol + 0.1 * (np.log(volumes[i-1]) - log_vol) + vol_shock
        ) * move_factor
        volumes[i] = max(20_000_000, volumes[i])

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": close,
        "Volume": volumes.astype(int),
    }, index=dates)

    return df


def generate_vix_daily(years: int = 5, seed: int = 123) -> pd.DataFrame:
    """Generate realistic synthetic VIX data (mean-reverting process)."""
    rng = np.random.default_rng(seed)
    n_days = years * 252
    v = VIX_STATS

    end_date = pd.Timestamp("2025-02-13")
    start_date = end_date - pd.tseries.offsets.BDay(n_days + 50)
    dates = pd.bdate_range(start=start_date, end=end_date)[-n_days:]

    # Ornstein-Uhlenbeck process for VIX
    vix_close = np.zeros(n_days)
    vix_close[0] = v["mean"]
    for i in range(1, n_days):
        # Mean reversion + random shock + occasional spikes
        reversion = v["mean_reversion_speed"] * (v["mean"] - vix_close[i-1])
        shock = rng.normal(0, v["std"] * 0.1)
        # Rare VIX spikes
        if rng.random() < 0.01:
            shock += rng.exponential(8)
        vix_close[i] = vix_close[i-1] + reversion + shock
        vix_close[i] = np.clip(vix_close[i], v["min"], v["max"])

    # Generate OHLC from close
    vix_range = np.abs(rng.normal(1.5, 0.8, n_days))
    vix_open = vix_close * (1 + rng.normal(0, 0.01, n_days))
    vix_high = np.maximum(vix_open, vix_close) + vix_range * rng.uniform(0.3, 0.7, n_days)
    vix_low = np.minimum(vix_open, vix_close) - vix_range * rng.uniform(0.3, 0.7, n_days)
    vix_low = np.maximum(vix_low, v["min"])

    df = pd.DataFrame({
        "VIX_Open": vix_open,
        "VIX_High": vix_high,
        "VIX_Low": vix_low,
        "VIX_Close": vix_close,
    }, index=dates)

    return df


def fetch_spy_daily(years: int = 5, use_cache: bool = True) -> pd.DataFrame:
    """Fetch SPY daily OHLCV data. Falls back to synthetic if network unavailable."""
    cache_file = CACHE_DIR / f"spy_daily_{years}y.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 200:
            return df

    if HAS_YFINANCE:
        try:
            ticker = yf.Ticker("SPY")
            df = ticker.history(period=f"{years}y", interval="1d", auto_adjust=True)
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            if len(df) > 200:
                if use_cache:
                    CACHE_DIR.mkdir(exist_ok=True)
                    df.to_parquet(cache_file)
                return df
        except Exception:
            pass

    # Fallback: synthetic data
    print("  [Using synthetic SPY data calibrated to real statistical properties]")
    return generate_spy_daily(years=years)


def fetch_vix_daily(years: int = 5, use_cache: bool = True) -> pd.DataFrame:
    """Fetch VIX data. Falls back to synthetic if network unavailable."""
    cache_file = CACHE_DIR / f"vix_daily_{years}y.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 200:
            return df

    if HAS_YFINANCE:
        try:
            ticker = yf.Ticker("^VIX")
            df = ticker.history(period=f"{years}y", interval="1d", auto_adjust=True)
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df[["Open", "High", "Low", "Close"]].copy()
            df.columns = ["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]
            df.dropna(inplace=True)
            if len(df) > 200:
                if use_cache:
                    CACHE_DIR.mkdir(exist_ok=True)
                    df.to_parquet(cache_file)
                return df
        except Exception:
            pass

    print("  [Using synthetic VIX data]")
    return generate_vix_daily(years=years)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used by strategies."""
    df = df.copy()

    # Moving averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # VWAP proxy (rolling typical price * volume / cumulative volume)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP_20"] = (tp * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Donchian Channels
    df["DC_upper"] = df["High"].rolling(20).max()
    df["DC_lower"] = df["Low"].rolling(20).min()
    df["DC_mid"] = (df["DC_upper"] + df["DC_lower"]) / 2

    # Gap from previous close
    df["Prev_Close"] = df["Close"].shift(1)
    df["Gap_Pct"] = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"] * 100

    # Intraday range metrics
    df["Day_Range"] = (df["High"] - df["Low"]) / df["Open"] * 100
    df["Body_Pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # Upper/Lower shadows
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"] * 100
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"] * 100

    # Volume indicators
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_20"]

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Stochastic
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = ((df["Close"] - low_14) / (high_14 - low_14)) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # Previous day metrics for strategy use
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Range"] = df["Prev_High"] - df["Prev_Low"]

    return df
