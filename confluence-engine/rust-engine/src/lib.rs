//! Rust hot path for Momentum Forge backtesting engine.
//!
//! Ports the bar-by-bar event loop from engine.py to compiled Rust code.
//! The inner loop iterates every bar and manages trailing stops, partial exits,
//! multi-position tracking, and ATR-based barriers — all in tight native code.
//!
//! Python still handles strategy signal generation, meta-labeling, and Kelly
//! sizing. This module receives the pre-computed signals + OHLCV arrays and
//! returns completed trades + equity curve.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

const SIDE_LONG: i8 = 1;
const SIDE_SHORT: i8 = -1;

// ─────────────────────────────────────────────────────────────
// Position state (internal)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Position {
    side: i8,            // SIDE_LONG or SIDE_SHORT
    entry_price: f64,
    entry_bar: usize,
    size_dollars: f64,
    size_frac: f64,
    shares: i64,
    stop_price: f64,
    target_price: f64,
    confidence: f64,
    trailing_active: bool,
    trailing_distance: f64,
    best_price: f64,
    atr_pct: f64,
    partial_taken: bool,
}

// ─────────────────────────────────────────────────────────────
// Completed trade (returned to Python)
// ─────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
struct RustTrade {
    #[pyo3(get)]
    entry_bar: usize,
    #[pyo3(get)]
    exit_bar: usize,
    #[pyo3(get)]
    entry_price: f64,
    #[pyo3(get)]
    exit_price: f64,
    #[pyo3(get)]
    side: i8,
    #[pyo3(get)]
    pnl_pct: f64,
    #[pyo3(get)]
    pnl_dollars: f64,
    #[pyo3(get)]
    position_size: f64,
    #[pyo3(get)]
    bars_held: usize,
    #[pyo3(get)]
    signal_confidence: f64,
    #[pyo3(get)]
    exit_reason: String,
}

// ─────────────────────────────────────────────────────────────
// Backtest configuration
// ─────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
struct RustBacktestConfig {
    #[pyo3(get, set)]
    initial_capital: f64,
    #[pyo3(get, set)]
    slippage_pct: f64,
    #[pyo3(get, set)]
    commission_per_trade: f64,
    #[pyo3(get, set)]
    max_holding_bars: usize,
    #[pyo3(get, set)]
    use_trailing_stop: bool,
    #[pyo3(get, set)]
    trailing_stop_atr_mult: f64,
    #[pyo3(get, set)]
    max_positions: usize,
    #[pyo3(get, set)]
    partial_exit_enabled: bool,
    #[pyo3(get, set)]
    partial_exit_pct: f64,
    #[pyo3(get, set)]
    partial_exit_atr_mult: f64,
    #[pyo3(get, set)]
    use_atr_barriers: bool,
    #[pyo3(get, set)]
    atr_period: usize,
    #[pyo3(get, set)]
    atr_stop_mult: f64,
    #[pyo3(get, set)]
    atr_target_mult: f64,
}

#[pymethods]
impl RustBacktestConfig {
    #[new]
    #[pyo3(signature = (
        initial_capital = 100_000.0,
        slippage_pct = 0.05,
        commission_per_trade = 0.0,
        max_holding_bars = 40,
        use_trailing_stop = true,
        trailing_stop_atr_mult = 2.5,
        max_positions = 3,
        partial_exit_enabled = true,
        partial_exit_pct = 50.0,
        partial_exit_atr_mult = 2.0,
        use_atr_barriers = true,
        atr_period = 14,
        atr_stop_mult = 2.0,
        atr_target_mult = 3.0,
    ))]
    fn new(
        initial_capital: f64,
        slippage_pct: f64,
        commission_per_trade: f64,
        max_holding_bars: usize,
        use_trailing_stop: bool,
        trailing_stop_atr_mult: f64,
        max_positions: usize,
        partial_exit_enabled: bool,
        partial_exit_pct: f64,
        partial_exit_atr_mult: f64,
        use_atr_barriers: bool,
        atr_period: usize,
        atr_stop_mult: f64,
        atr_target_mult: f64,
    ) -> Self {
        Self {
            initial_capital,
            slippage_pct,
            commission_per_trade,
            max_holding_bars,
            use_trailing_stop,
            trailing_stop_atr_mult,
            max_positions,
            partial_exit_enabled,
            partial_exit_pct,
            partial_exit_atr_mult,
            use_atr_barriers,
            atr_period,
            atr_stop_mult,
            atr_target_mult,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Helper: compute ATR at bar i
// ─────────────────────────────────────────────────────────────

fn compute_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize, i: usize) -> f64 {
    if i < period {
        return 0.0;
    }
    let start = i + 1 - period;
    let mut sum = 0.0;
    for j in start..=i {
        let tr = if j == 0 {
            highs[j] - lows[j]
        } else {
            let hl = highs[j] - lows[j];
            let hc = (highs[j] - closes[j - 1]).abs();
            let lc = (lows[j] - closes[j - 1]).abs();
            hl.max(hc).max(lc)
        };
        sum += tr;
    }
    sum / period as f64
}

// ─────────────────────────────────────────────────────────────
// Helper: close position
// ─────────────────────────────────────────────────────────────

fn close_position(pos: &Position, exit_price: f64, exit_bar: usize, exit_reason: &str, slippage_pct: f64) -> RustTrade {
    let slip = slippage_pct / 100.0;
    let adj_price = if pos.side == SIDE_LONG {
        exit_price * (1.0 - slip)
    } else {
        exit_price * (1.0 + slip)
    };

    let pnl_pct = if pos.side == SIDE_LONG {
        (adj_price - pos.entry_price) / pos.entry_price * 100.0
    } else {
        (pos.entry_price - adj_price) / pos.entry_price * 100.0
    };

    let pnl_dollars = pnl_pct / 100.0 * pos.size_dollars;

    RustTrade {
        entry_bar: pos.entry_bar,
        exit_bar,
        entry_price: pos.entry_price,
        exit_price: adj_price,
        side: pos.side,
        pnl_pct,
        pnl_dollars,
        position_size: pos.size_frac,
        bars_held: exit_bar - pos.entry_bar,
        signal_confidence: pos.confidence,
        exit_reason: exit_reason.to_string(),
    }
}

// ─────────────────────────────────────────────────────────────
// Core backtest loop — the hot path
// ─────────────────────────────────────────────────────────────

/// Run the bar-by-bar backtest loop in Rust.
///
/// Args:
///     open_arr: numpy array of open prices
///     high_arr: numpy array of high prices
///     low_arr: numpy array of low prices
///     close_arr: numpy array of close prices
///     signal_bars: numpy array of bar indices that have signals
///     signal_sides: numpy array of signal sides (1=long, -1=short)
///     signal_confidences: numpy array of signal confidence values
///     signal_stop_pcts: numpy array of signal stop loss percentages
///     signal_target_pcts: numpy array of signal take profit percentages
///     position_sizes: numpy array of dollar amounts per signal (pre-computed by Python)
///     position_shares: numpy array of share counts per signal
///     blocked: numpy array of bools — True if signal is blocked by risk/meta
///     config: RustBacktestConfig
///
/// Returns:
///     (trades, equity_curve) — list of RustTrade, numpy array of equity values
#[pyfunction]
fn run_backtest_loop<'py>(
    py: Python<'py>,
    _open_arr: PyReadonlyArray1<'py, f64>,
    high_arr: PyReadonlyArray1<'py, f64>,
    low_arr: PyReadonlyArray1<'py, f64>,
    close_arr: PyReadonlyArray1<'py, f64>,
    signal_bars: PyReadonlyArray1<'py, i64>,
    signal_sides: PyReadonlyArray1<'py, i8>,
    signal_confidences: PyReadonlyArray1<'py, f64>,
    signal_stop_pcts: PyReadonlyArray1<'py, f64>,
    signal_target_pcts: PyReadonlyArray1<'py, f64>,
    position_sizes: PyReadonlyArray1<'py, f64>,
    position_shares: PyReadonlyArray1<'py, i64>,
    blocked: PyReadonlyArray1<'py, bool>,
    config: &RustBacktestConfig,
) -> PyResult<(Py<PyList>, Bound<'py, PyArray1<f64>>)> {
    let highs = high_arr.as_slice()?;
    let lows = low_arr.as_slice()?;
    let closes = close_arr.as_slice()?;
    let sig_bars = signal_bars.as_slice()?;
    let sig_sides = signal_sides.as_slice()?;
    let sig_confs = signal_confidences.as_slice()?;
    let sig_stops = signal_stop_pcts.as_slice()?;
    let sig_targets = signal_target_pcts.as_slice()?;
    let pos_sizes = position_sizes.as_slice()?;
    let pos_shares = position_shares.as_slice()?;
    let sig_blocked = blocked.as_slice()?;

    let num_bars = closes.len();
    let num_signals = sig_bars.len();

    // Build signal lookup: bar_index -> signal array index
    let mut signal_map: Vec<Option<usize>> = vec![None; num_bars];
    for (si, &bar) in sig_bars.iter().enumerate() {
        if bar >= 0 && (bar as usize) < num_bars {
            signal_map[bar as usize] = Some(si);
        }
    }

    let mut cash = config.initial_capital;
    let mut positions: Vec<Position> = Vec::with_capacity(config.max_positions);
    let mut trades: Vec<RustTrade> = Vec::with_capacity(num_signals);
    let mut equity: Array1<f64> = Array1::zeros(num_bars);

    // ── Bar-by-bar loop ──
    for i in 0..num_bars {
        let price = closes[i];
        let high = highs[i];
        let low = lows[i];

        // ── 1. Update trailing stops + check exits for all positions ──
        let mut to_close: Vec<usize> = Vec::new();
        for (pidx, pos) in positions.iter_mut().enumerate() {
            // Trailing stop ratchet
            if config.use_trailing_stop && pos.trailing_active {
                if pos.side == SIDE_LONG {
                    if high > pos.best_price {
                        pos.best_price = high;
                    }
                    let new_stop = pos.best_price - pos.trailing_distance;
                    if new_stop > pos.stop_price {
                        pos.stop_price = new_stop;
                    }
                } else {
                    if low < pos.best_price {
                        pos.best_price = low;
                    }
                    let new_stop = pos.best_price + pos.trailing_distance;
                    if new_stop < pos.stop_price {
                        pos.stop_price = new_stop;
                    }
                }
            }

            // Activate trailing stop after 1 ATR move in favor
            if config.use_trailing_stop && !pos.trailing_active {
                let unrealized_pct = if pos.side == SIDE_LONG {
                    (high - pos.entry_price) / pos.entry_price
                } else {
                    (pos.entry_price - low) / pos.entry_price
                };
                if unrealized_pct > pos.atr_pct {
                    pos.trailing_active = true;
                    pos.best_price = if pos.side == SIDE_LONG { high } else { low };
                }
            }

            // Check stop/target
            let hit_stop = if pos.side == SIDE_LONG {
                low <= pos.stop_price
            } else {
                high >= pos.stop_price
            };

            let hit_target = if !hit_stop {
                if pos.side == SIDE_LONG {
                    high >= pos.target_price
                } else {
                    low <= pos.target_price
                }
            } else {
                false
            };

            if hit_stop {
                let reason = if pos.trailing_active { "trailing_stop" } else { "stop_loss" };
                let trade = close_position(pos, pos.stop_price, i, reason, config.slippage_pct);
                cash += trade.pnl_dollars;
                trades.push(trade);
                to_close.push(pidx);
            } else if hit_target {
                // Partial exit logic
                if config.partial_exit_enabled && !pos.partial_taken && pos.shares > 1 {
                    let partial_shares = (pos.shares as f64 * config.partial_exit_pct / 100.0).max(1.0) as i64;
                    let remaining = pos.shares - partial_shares;

                    // Close partial
                    let mut partial_pos = pos.clone();
                    partial_pos.size_dollars = partial_shares as f64 * pos.entry_price;
                    partial_pos.shares = partial_shares;
                    let trade = close_position(&partial_pos, pos.target_price, i, "partial_take_profit", config.slippage_pct);
                    cash += trade.pnl_dollars;
                    trades.push(trade);

                    // Update remainder
                    pos.shares = remaining;
                    pos.size_dollars = remaining as f64 * pos.entry_price;
                    pos.stop_price = pos.entry_price; // breakeven
                    pos.target_price *= 1.5;          // widen
                    pos.partial_taken = true;
                    pos.trailing_active = true;
                    pos.best_price = if pos.side == SIDE_LONG { high } else { low };
                } else {
                    let trade = close_position(pos, pos.target_price, i, "take_profit", config.slippage_pct);
                    cash += trade.pnl_dollars;
                    trades.push(trade);
                    to_close.push(pidx);
                }
            } else if config.max_holding_bars > 0 && (i - pos.entry_bar) >= config.max_holding_bars {
                // Triple barrier vertical
                let trade = close_position(pos, price, i, "time_expiry", config.slippage_pct);
                cash += trade.pnl_dollars;
                trades.push(trade);
                to_close.push(pidx);
            }
        }

        // Remove closed positions (reverse order)
        to_close.sort_unstable();
        for &pidx in to_close.iter().rev() {
            positions.remove(pidx);
        }

        // ── 2. Process signal if one exists ──
        if let Some(si) = signal_map[i] {
            let sig_side = sig_sides[si];
            if sig_side != 0 && !sig_blocked[si] {
                // Close opposite-side positions
                let mut opp_indices: Vec<usize> = Vec::new();
                for (pidx, pos) in positions.iter().enumerate() {
                    if pos.side != sig_side {
                        let trade = close_position(pos, price, i, "signal", config.slippage_pct);
                        cash += trade.pnl_dollars;
                        trades.push(trade);
                        opp_indices.push(pidx);
                    }
                }
                for &pidx in opp_indices.iter().rev() {
                    positions.remove(pidx);
                }

                // Open new position if under limit
                if positions.len() < config.max_positions {
                    let shares = pos_shares[si];
                    if shares > 0 {
                        // ATR-based barriers
                        let (effective_stop, effective_target, atr_pct_val) =
                            if config.use_atr_barriers && i >= config.atr_period + 1 {
                                let atr_val = compute_atr(highs, lows, closes, config.atr_period, i);
                                if atr_val > 0.0 {
                                    let atr_pct = atr_val / price * 100.0;
                                    (
                                        atr_pct * config.atr_stop_mult,
                                        atr_pct * config.atr_target_mult,
                                        atr_val / price,
                                    )
                                } else {
                                    (sig_stops[si], sig_targets[si], 0.02)
                                }
                            } else {
                                (sig_stops[si], sig_targets[si], 0.02)
                            };

                        let slippage = config.slippage_pct / 100.0;
                        let (entry_price, stop_price, target_price, trailing_dist) = if sig_side == SIDE_LONG {
                            let ep = price * (1.0 + slippage);
                            let sp = ep * (1.0 - effective_stop / 100.0);
                            let tp = ep * (1.0 + effective_target / 100.0);
                            let td = atr_pct_val * config.trailing_stop_atr_mult * ep;
                            (ep, sp, tp, td)
                        } else {
                            let ep = price * (1.0 - slippage);
                            let sp = ep * (1.0 + effective_stop / 100.0);
                            let tp = ep * (1.0 - effective_target / 100.0);
                            let td = atr_pct_val * config.trailing_stop_atr_mult * ep;
                            (ep, sp, tp, td)
                        };

                        cash -= config.commission_per_trade;

                        positions.push(Position {
                            side: sig_side,
                            entry_price,
                            entry_bar: i,
                            size_dollars: shares as f64 * entry_price,
                            size_frac: if cash > 0.0 { shares as f64 * entry_price / cash } else { 0.0 },
                            shares,
                            stop_price,
                            target_price,
                            confidence: sig_confs[si],
                            trailing_active: false,
                            trailing_distance: trailing_dist,
                            best_price: entry_price,
                            atr_pct: atr_pct_val,
                            partial_taken: false,
                        });
                    }
                }
            }
        }

        // ── 3. Record equity = cash + unrealized ──
        let mut total_equity = cash;
        for pos in &positions {
            let unrealized = if pos.side == SIDE_LONG {
                (price - pos.entry_price) / pos.entry_price * pos.size_dollars
            } else {
                (pos.entry_price - price) / pos.entry_price * pos.size_dollars
            };
            total_equity += unrealized;
        }
        equity[i] = total_equity;
    }

    // Close remaining positions at last bar
    if !positions.is_empty() && num_bars > 0 {
        let last_price = closes[num_bars - 1];
        for pos in &positions {
            let trade = close_position(pos, last_price, num_bars - 1, "end_of_data", config.slippage_pct);
            cash += trade.pnl_dollars;
            trades.push(trade);
        }
        equity[num_bars - 1] = cash;
    }

    // Convert trades to Python list
    let trade_list = PyList::empty(py);
    for t in &trades {
        let py_trade = Py::new(py, t.clone())?;
        trade_list.append(py_trade)?;
    }

    let equity_arr = PyArray1::from_array(py, &equity);

    Ok((trade_list.into(), equity_arr))
}

/// Python module definition
#[pymodule]
fn forge_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustTrade>()?;
    m.add_class::<RustBacktestConfig>()?;
    m.add_function(wrap_pyfunction!(run_backtest_loop, m)?)?;
    Ok(())
}
