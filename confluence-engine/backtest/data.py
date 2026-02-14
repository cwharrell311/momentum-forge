"""
SPY historical data for backtesting.

Data source priority:
  1. Parquet cache (fastest — previously downloaded data)
  2. CSV import (manual download from Yahoo Finance, etc.)
  3. Yahoo Finance via yfinance (auto-download)
  4. Synthetic fallback (calibrated to real SPY stats, for offline use)

To use real data:
  ./go.sh data download        # Auto-fetch via yfinance
  ./go.sh data import file.csv # Import manually-downloaded CSV
  ./go.sh data status          # Show what's cached
  ./go.sh data clear           # Wipe cache, re-download next run
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


def _normalize_csv(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Normalize a CSV DataFrame to standard column names and datetime index."""
    # Handle common column name variations from Yahoo Finance downloads
    # Prefer "Adj Close" over "Close" if both exist (Yahoo Finance CSVs)
    col_lower = [c.strip().lower().replace(" ", "_") for c in df.columns]
    has_adj_close = any(c in ("adj_close",) for c in col_lower)

    col_map = {}
    for col in df.columns:
        lower = col.strip().lower().replace(" ", "_")
        if lower in ("date", "datetime", "timestamp"):
            col_map[col] = "__date__"
        elif lower in ("open", "adj_open"):
            col_map[col] = "Open"
        elif lower in ("high", "adj_high"):
            col_map[col] = "High"
        elif lower in ("low", "adj_low"):
            col_map[col] = "Low"
        elif lower == "adj_close":
            col_map[col] = "Close"  # Adj Close preferred
        elif lower == "close":
            if not has_adj_close:
                col_map[col] = "Close"
            # else: skip raw Close when Adj Close exists
        elif lower == "volume":
            col_map[col] = "Volume"
        elif lower in ("vix_open",):
            col_map[col] = "VIX_Open"
        elif lower in ("vix_high",):
            col_map[col] = "VIX_High"
        elif lower in ("vix_low",):
            col_map[col] = "VIX_Low"
        elif lower in ("vix_close",):
            col_map[col] = "VIX_Close"

    # Drop columns that weren't mapped (e.g. raw "Close" when Adj Close is preferred)
    cols_to_drop = [c for c in df.columns if c not in col_map]
    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns=col_map)

    # Set date index if present as column
    if "__date__" in df.columns:
        df.index = pd.to_datetime(df["__date__"])
        df.drop(columns=["__date__"], inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Strip timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only expected columns
    present = [c for c in expected_cols if c in df.columns]
    df = df[present].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def import_spy_csv(csv_path: str, years: int = 5) -> pd.DataFrame:
    """
    Import SPY data from a CSV file and cache it.

    Works with CSVs downloaded from Yahoo Finance, Google Finance,
    or any source with Date/Open/High/Low/Close/Volume columns.

    Usage:
      1. Go to finance.yahoo.com/quote/SPY/history
      2. Set time period to 5Y, click Download
      3. Run: ./go.sh data import ~/Downloads/SPY.csv
    """
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_csv(df, ["Open", "High", "Low", "Close", "Volume"])

    if len(df) < 50:
        raise ValueError(f"CSV has only {len(df)} rows — need at least 50 for backtesting")

    # Cache as parquet for fast reloads
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"spy_daily_{years}y.parquet"
    df.to_parquet(cache_file)

    # Also save a metadata file so we know where the data came from
    meta_file = CACHE_DIR / f"spy_daily_{years}y.meta"
    meta_file.write_text(
        f"source: csv_import\n"
        f"file: {path}\n"
        f"rows: {len(df)}\n"
        f"range: {df.index[0].date()} to {df.index[-1].date()}\n"
        f"imported: {pd.Timestamp.now().isoformat(timespec='seconds')}\n"
    )

    print(f"  Imported {len(df)} bars from {path.name}")
    print(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Cached to: {cache_file}")
    return df


def import_vix_csv(csv_path: str, years: int = 5) -> pd.DataFrame:
    """Import VIX data from a CSV file and cache it."""
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # VIX CSVs from Yahoo have standard OHLC columns — rename to VIX_ prefix
    df = _normalize_csv(df, ["Open", "High", "Low", "Close"])
    if "Open" in df.columns and "VIX_Open" not in df.columns:
        df = df.rename(columns={
            "Open": "VIX_Open", "High": "VIX_High",
            "Low": "VIX_Low", "Close": "VIX_Close",
        })

    if len(df) < 50:
        raise ValueError(f"CSV has only {len(df)} rows — need at least 50")

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"vix_daily_{years}y.parquet"
    df.to_parquet(cache_file)

    meta_file = CACHE_DIR / f"vix_daily_{years}y.meta"
    meta_file.write_text(
        f"source: csv_import\n"
        f"file: {path}\n"
        f"rows: {len(df)}\n"
        f"range: {df.index[0].date()} to {df.index[-1].date()}\n"
        f"imported: {pd.Timestamp.now().isoformat(timespec='seconds')}\n"
    )

    print(f"  Imported {len(df)} VIX bars from {path.name}")
    return df


def _download_yfinance(symbol: str, years: int, columns: list[str]) -> pd.DataFrame | None:
    """Download data via yfinance. Returns None on failure."""
    if not HAS_YFINANCE:
        print(f"  yfinance not installed. Run: pip install yfinance")
        return None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{years}y", interval="1d", auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        present = [c for c in columns if c in df.columns]
        df = df[present].copy()
        df.dropna(inplace=True)
        if len(df) > 200:
            return df
        print(f"  yfinance returned only {len(df)} bars for {symbol} — too few")
        return None
    except Exception as e:
        print(f"  yfinance download failed for {symbol}: {e}")
        return None


def _get_data_source(cache_file: Path) -> str:
    """Read the .meta file to determine data source, if it exists."""
    meta_file = cache_file.with_suffix(".meta")
    if meta_file.exists():
        for line in meta_file.read_text().splitlines():
            if line.startswith("source:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def fetch_spy_daily(
    years: int = 5,
    use_cache: bool = True,
    force_source: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch SPY daily OHLCV data.

    Returns (dataframe, source_label) where source_label is one of:
      "cache (yahoo_finance)", "cache (csv_import)", "yahoo_finance", "synthetic"

    Args:
        years: How many years of data to fetch.
        use_cache: If True, use cached parquet if available.
        force_source: Force a specific source: "yfinance", "synthetic", or None (auto).
    """
    cache_file = CACHE_DIR / f"spy_daily_{years}y.parquet"

    # Step 1: Cache (unless forcing a specific source)
    if use_cache and force_source is None and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 200:
            source = _get_data_source(cache_file)
            return df, f"cache ({source})"

    # Step 2: yfinance (unless forcing synthetic)
    if force_source != "synthetic":
        df = _download_yfinance("SPY", years, ["Open", "High", "Low", "Close", "Volume"])
        if df is not None:
            if use_cache:
                CACHE_DIR.mkdir(exist_ok=True)
                df.to_parquet(cache_file)
                meta_file = cache_file.with_suffix(".meta")
                meta_file.write_text(
                    f"source: yahoo_finance\n"
                    f"rows: {len(df)}\n"
                    f"range: {df.index[0].date()} to {df.index[-1].date()}\n"
                    f"downloaded: {pd.Timestamp.now().isoformat(timespec='seconds')}\n"
                )
            return df, "yahoo_finance"

    # Step 3: Synthetic fallback
    print("  [Using synthetic SPY data calibrated to real statistical properties]")
    return generate_spy_daily(years=years), "synthetic"


def fetch_vix_daily(
    years: int = 5,
    use_cache: bool = True,
    force_source: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch VIX data. Returns (dataframe, source_label)."""
    cache_file = CACHE_DIR / f"vix_daily_{years}y.parquet"

    if use_cache and force_source is None and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 200:
            source = _get_data_source(cache_file)
            return df, f"cache ({source})"

    if force_source != "synthetic":
        df = _download_yfinance("^VIX", years, ["Open", "High", "Low", "Close"])
        if df is not None:
            df.columns = ["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]
            if use_cache:
                CACHE_DIR.mkdir(exist_ok=True)
                df.to_parquet(cache_file)
                meta_file = cache_file.with_suffix(".meta")
                meta_file.write_text(
                    f"source: yahoo_finance\n"
                    f"rows: {len(df)}\n"
                    f"range: {df.index[0].date()} to {df.index[-1].date()}\n"
                    f"downloaded: {pd.Timestamp.now().isoformat(timespec='seconds')}\n"
                )
            return df, "yahoo_finance"

    print("  [Using synthetic VIX data]")
    return generate_vix_daily(years=years), "synthetic"


def download_all(years: int = 5) -> None:
    """Download SPY + VIX data and cache locally. Run this from your Mac."""
    print(f"Downloading {years} years of real market data...\n")

    print("--- SPY ---")
    spy_df, spy_src = fetch_spy_daily(years=years, use_cache=False)
    print(f"  Source: {spy_src}")
    print(f"  Bars: {len(spy_df)}")
    if len(spy_df) > 0:
        print(f"  Range: {spy_df.index[0].date()} to {spy_df.index[-1].date()}")
        print(f"  Latest close: ${spy_df['Close'].iloc[-1]:.2f}")

    print("\n--- VIX ---")
    vix_df, vix_src = fetch_vix_daily(years=years, use_cache=False)
    print(f"  Source: {vix_src}")
    print(f"  Bars: {len(vix_df)}")
    if len(vix_df) > 0:
        vix_col = "VIX_Close" if "VIX_Close" in vix_df.columns else vix_df.columns[-1]
        print(f"  Range: {vix_df.index[0].date()} to {vix_df.index[-1].date()}")
        print(f"  Latest VIX: {vix_df[vix_col].iloc[-1]:.2f}")

    print(f"\nData cached in: {CACHE_DIR}/")
    print("Run './go.sh backtest' to use this data.")


def cache_status() -> None:
    """Print what data is currently cached."""
    print("\n--- Data Cache Status ---")
    print(f"  Cache dir: {CACHE_DIR}/\n")

    if not CACHE_DIR.exists():
        print("  No cached data. Run: ./go.sh data download")
        return

    parquets = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquets:
        print("  No cached data. Run: ./go.sh data download")
        return

    for pf in parquets:
        size_mb = pf.stat().st_size / (1024 * 1024)
        meta_file = pf.with_suffix(".meta")

        print(f"  {pf.name} ({size_mb:.1f} MB)")
        if meta_file.exists():
            for line in meta_file.read_text().splitlines():
                print(f"    {line}")
        else:
            # Read the parquet to show basic info
            df = pd.read_parquet(pf)
            print(f"    rows: {len(df)}")
            print(f"    range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"    source: unknown (no metadata)")
        print()


def clear_cache() -> None:
    """Delete all cached data files."""
    if not CACHE_DIR.exists():
        print("  No cache directory to clear.")
        return

    count = 0
    for f in CACHE_DIR.iterdir():
        f.unlink()
        count += 1

    print(f"  Cleared {count} cached files from {CACHE_DIR}/")
    print("  Next backtest will re-download or use synthetic data.")


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
