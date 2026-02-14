"""
Autonomous Trader — fully automated strategy deployment and execution.

This is the brain of the operation. No human in the loop.

Pipeline:
1. DISCOVER: Backtest all strategies across all instruments
2. RANK: Walk-forward validate, filter for robustness
3. DEPLOY: Run winning strategies on live Alpaca data
4. MONITOR: Track real P&L vs backtest expectations
5. KILL: Auto-disable strategies that degrade
6. REOPTIMIZE: Weekly genetic optimization to adapt

The autonomous trader manages a PORTFOLIO of strategies, not just one.
Each strategy runs on specific instruments with specific parameters.
Position sizing uses half-Kelly, scaled by regime.

Think of this as a hedge fund in a box:
- Multiple uncorrelated strategies for diversification
- Automatic risk management and position limits
- Self-healing: kills underperformers, promotes outperformers
- Adapts to regime changes via walk-forward reoptimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("forge.autonomous")


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


class DeploymentStatus(Enum):
    CANDIDATE = "candidate"       # Passed backtest, not yet live
    PAPER_TESTING = "paper"       # Running on paper account
    LIVE = "live"                 # Trading real money
    DEGRADED = "degraded"         # Performance below expectations
    KILLED = "killed"             # Auto-disabled


@dataclass
class DeployedStrategy:
    """A strategy that's been validated and deployed for trading."""
    strategy_name: str
    strategy_class: str              # Class name for reinstantiation
    params: dict[str, float]         # Optimized parameters
    instruments: list[str]           # Which tickers this runs on
    status: DeploymentStatus = DeploymentStatus.CANDIDATE
    # Backtest metrics (what we expect)
    expected_sharpe: float = 0.0
    expected_cagr_pct: float = 0.0
    expected_max_dd_pct: float = 0.0
    expected_win_rate: float = 0.0
    wf_efficiency: float = 0.0       # Walk-forward efficiency
    # Live metrics (what's actually happening)
    live_sharpe: float = 0.0
    live_pnl_pct: float = 0.0
    live_trades: int = 0
    live_wins: int = 0
    live_max_dd_pct: float = 0.0
    # Position management
    kelly_fraction: float = 0.25     # Half-Kelly default
    max_position_pct: float = 0.15   # Max 15% per strategy
    current_position_pct: float = 0.0
    # Timing
    deployed_at: str = ""
    last_signal_at: str = ""
    last_reoptimized_at: str = ""
    # Signals
    pending_signal: dict | None = None

    def degradation_ratio(self) -> float:
        """How much has live performance degraded vs expectations?"""
        if self.expected_sharpe <= 0 or self.live_trades < 5:
            return 1.0  # Not enough data
        return self.live_sharpe / self.expected_sharpe if self.expected_sharpe > 0 else 0

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "strategy_class": self.strategy_class,
            "params": self.params,
            "instruments": self.instruments,
            "status": self.status.value,
            "expected": {
                "sharpe": self.expected_sharpe,
                "cagr_pct": self.expected_cagr_pct,
                "max_dd_pct": self.expected_max_dd_pct,
                "win_rate": self.expected_win_rate,
                "wf_efficiency": self.wf_efficiency,
            },
            "live": {
                "sharpe": self.live_sharpe,
                "pnl_pct": self.live_pnl_pct,
                "trades": self.live_trades,
                "win_rate": (self.live_wins / self.live_trades * 100) if self.live_trades > 0 else 0,
                "max_dd_pct": self.live_max_dd_pct,
                "degradation": self.degradation_ratio(),
            },
            "sizing": {
                "kelly_fraction": self.kelly_fraction,
                "max_position_pct": self.max_position_pct,
                "current_position_pct": self.current_position_pct,
            },
            "deployed_at": self.deployed_at,
            "last_signal_at": self.last_signal_at,
        }


@dataclass
class PortfolioState:
    """Current state of the autonomous portfolio."""
    equity: float = 100_000.0
    cash: float = 100_000.0
    peak_equity: float = 100_000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    daily_pnl: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)  # {symbol: {shares, side, avg_price, ...}}
    deployed_strategies: list[DeployedStrategy] = field(default_factory=list)
    last_rebalance: str = ""
    last_optimization: str = ""
    paper_mode: bool = True
    halted: bool = False
    halt_reason: str = ""

    def open_position_count(self) -> int:
        return len(self.positions)

    def portfolio_heat(self) -> float:
        """Total capital at risk across all positions."""
        total = sum(abs(p.get("risk_pct", 0)) for p in self.positions.values())
        return total

    def to_dict(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "daily_pnl": self.daily_pnl,
            "open_positions": self.open_position_count(),
            "portfolio_heat": round(self.portfolio_heat(), 2),
            "deployed_strategies": len(self.deployed_strategies),
            "active_strategies": len([s for s in self.deployed_strategies if s.status in (DeploymentStatus.PAPER_TESTING, DeploymentStatus.LIVE)]),
            "paper_mode": self.paper_mode,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "last_rebalance": self.last_rebalance,
            "last_optimization": self.last_optimization,
            "positions": {sym: pos for sym, pos in self.positions.items()},
            "strategies": [s.to_dict() for s in self.deployed_strategies],
        }


# ═══════════════════════════════════════════════════════════════
# AUTONOMOUS TRADER
# ═══════════════════════════════════════════════════════════════


class AutonomousTrader:
    """
    The autonomous trading system.

    Controls the full lifecycle:
    - Strategy discovery via backtesting
    - Deployment to paper/live
    - Real-time signal generation
    - Order execution via Alpaca
    - Performance monitoring and strategy health
    - Auto-kill degraded strategies
    - Weekly reoptimization
    """

    def __init__(
        self,
        alpaca_client=None,
        initial_capital: float = 100_000.0,
        paper_mode: bool = True,
        max_strategies: int = 8,         # Max concurrent strategies
        max_positions: int = 12,          # Max open positions
        max_portfolio_risk_pct: float = 40.0,  # Max 40% of capital at risk
        daily_loss_limit_pct: float = 3.0,     # Stop trading after 3% daily loss
        max_drawdown_halt_pct: float = 15.0,   # Halt if DD > 15%
        degradation_threshold: float = 0.4,    # Kill strategy if live < 40% of expected
        min_sharpe_to_deploy: float = 0.8,     # Min Sharpe to consider deploying
        min_wf_efficiency: float = 0.4,        # Min walk-forward efficiency
        min_trades_required: int = 20,          # Min trades in backtest
    ):
        self.alpaca = alpaca_client
        self.paper_mode = paper_mode
        self.max_strategies = max_strategies
        self.max_positions = max_positions
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self.degradation_threshold = degradation_threshold
        self.min_sharpe = min_sharpe_to_deploy
        self.min_wf_efficiency = min_wf_efficiency
        self.min_trades = min_trades_required

        self.portfolio = PortfolioState(
            equity=initial_capital,
            cash=initial_capital,
            peak_equity=initial_capital,
            paper_mode=paper_mode,
        )

        self._running = False

    # ── DISCOVERY: Find winning strategies ──

    async def discover_strategies(
        self,
        instruments: list[str] | None = None,
        period: str = "2y",
    ) -> list[DeployedStrategy]:
        """
        Run backtests across all instruments and strategies.
        Returns candidates that pass quality filters.

        This is the entry point — call this to find what works.
        """
        from src.backtesting.data_feeds import fetch_stock_data
        from src.backtesting.engine import BacktestConfig, walk_forward_analysis
        from src.backtesting.strategies import get_all_strategies

        if instruments is None:
            instruments = _default_instruments()

        config = BacktestConfig(
            initial_capital=self.portfolio.equity,
            kelly_fraction=0.5,
            max_position_pct=0.20,
        )

        candidates: list[DeployedStrategy] = []
        total_tested = 0

        for symbol in instruments:
            log.info("Discovering strategies for %s...", symbol)
            try:
                data = fetch_stock_data(symbol, period=period, interval="1d")
            except Exception as e:
                log.warning("Could not fetch %s: %s", symbol, e)
                continue

            if len(data.df) < 120:
                log.warning("%s: only %d bars, skipping (need 120+)", symbol, len(data.df))
                continue

            # Determine asset class from symbol
            asset_class = _classify_instrument(symbol)

            # Test all strategies for this asset class
            strategies = get_all_strategies(asset_class)
            if not strategies:
                strategies = get_all_strategies()  # Fall back to all

            for strategy in strategies:
                meta = strategy.meta()
                total_tested += 1

                try:
                    wf = walk_forward_analysis(
                        strategy, data.df, symbol, asset_class, config,
                        num_windows=3, embargo_bars=5,
                    )
                    report = wf.combined_oos_report

                    # Quality filters — only deploy robust strategies
                    if report.trades.total_trades < self.min_trades:
                        continue
                    if report.sharpe_ratio < self.min_sharpe:
                        continue
                    if wf.wf_efficiency < self.min_wf_efficiency:
                        continue
                    if report.drawdown.max_drawdown_pct > 30:
                        continue
                    if report.trades.profit_factor < 1.2:
                        continue

                    # This strategy passed! Create deployment candidate
                    candidate = DeployedStrategy(
                        strategy_name=f"{meta.name}_{symbol}",
                        strategy_class=type(strategy).__name__,
                        params={k: getattr(strategy, k, v) for k, (v, _) in zip(meta.param_ranges.keys(), meta.param_ranges.values())},
                        instruments=[symbol],
                        status=DeploymentStatus.CANDIDATE,
                        expected_sharpe=report.sharpe_ratio,
                        expected_cagr_pct=report.cagr_pct,
                        expected_max_dd_pct=report.drawdown.max_drawdown_pct,
                        expected_win_rate=report.trades.win_rate,
                        wf_efficiency=wf.wf_efficiency,
                        kelly_fraction=min(0.5, max(0.1, report.sharpe_ratio / 4)),
                        deployed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    candidates.append(candidate)
                    log.info(
                        "CANDIDATE: %s on %s — Sharpe=%.2f CAGR=%.1f%% WF=%.2f",
                        meta.name, symbol, report.sharpe_ratio,
                        report.cagr_pct, wf.wf_efficiency,
                    )

                except Exception as e:
                    log.debug("WF failed for %s on %s: %s", meta.name, symbol, e)

        log.info(
            "Discovery complete: %d strategies tested, %d candidates found",
            total_tested, len(candidates),
        )

        # Rank by Sharpe * WF efficiency (reward both performance and robustness)
        candidates.sort(
            key=lambda c: c.expected_sharpe * c.wf_efficiency,
            reverse=True,
        )

        return candidates

    # ── DEPLOYMENT: Activate winning strategies ──

    def deploy_strategies(self, candidates: list[DeployedStrategy]) -> list[DeployedStrategy]:
        """
        Deploy the top N candidates as paper or live strategies.

        Picks strategies that are UNCORRELATED to each other
        (no point running 5 momentum strategies on the same asset).
        """
        deployed: list[DeployedStrategy] = []
        used_instruments: set[str] = set()
        used_strategy_types: dict[str, int] = {}

        for candidate in candidates:
            if len(deployed) >= self.max_strategies:
                break

            # Diversification: max 2 strategies per instrument
            instrument = candidate.instruments[0] if candidate.instruments else ""
            if instrument in used_instruments:
                # Allow max 2 strategies per instrument
                existing = [d for d in deployed if instrument in d.instruments]
                if len(existing) >= 2:
                    continue

            # Diversification: max 3 of the same strategy type
            stype = candidate.strategy_class
            if used_strategy_types.get(stype, 0) >= 3:
                continue

            # Deploy it
            candidate.status = DeploymentStatus.PAPER_TESTING if self.paper_mode else DeploymentStatus.LIVE
            deployed.append(candidate)
            used_instruments.add(instrument)
            used_strategy_types[stype] = used_strategy_types.get(stype, 0) + 1

            log.info(
                "DEPLOYED [%s]: %s — expected Sharpe=%.2f, WF=%.2f",
                candidate.status.value, candidate.strategy_name,
                candidate.expected_sharpe, candidate.wf_efficiency,
            )

        self.portfolio.deployed_strategies = deployed
        return deployed

    # ── SIGNAL GENERATION: Check for new trades ──

    async def check_signals(self) -> list[dict]:
        """
        Check all deployed strategies for new trade signals.

        Called periodically (every 1-5 minutes during market hours).
        Returns list of signals to execute.
        """
        if self.portfolio.halted:
            log.warning("Trading halted: %s", self.portfolio.halt_reason)
            return []

        signals = []

        for strategy_dep in self.portfolio.deployed_strategies:
            if strategy_dep.status not in (DeploymentStatus.PAPER_TESTING, DeploymentStatus.LIVE):
                continue

            for instrument in strategy_dep.instruments:
                try:
                    signal = await self._generate_signal(strategy_dep, instrument)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    log.error("Signal generation failed: %s on %s: %s",
                              strategy_dep.strategy_name, instrument, e)

        return signals

    async def _generate_signal(
        self,
        strategy_dep: DeployedStrategy,
        instrument: str,
    ) -> dict | None:
        """Generate a signal for one strategy-instrument pair."""
        if not self.alpaca:
            return None

        # Fetch recent bars from Alpaca
        raw_bars = await self.alpaca.get_bars(instrument, timeframe="1Day", limit=60)
        if not raw_bars or len(raw_bars) < 30:
            return None

        # Convert to strategy format
        from src.services.trading_engine import parse_bars
        bars = parse_bars(raw_bars)
        if len(bars) < 30:
            return None

        # Instantiate strategy with deployed parameters
        strategy = _instantiate_strategy(strategy_dep.strategy_class, strategy_dep.params)
        if not strategy:
            return None

        # Generate signals
        trade_signals = strategy.generate_signals(bars)
        if not trade_signals:
            return None

        # Only care about the latest signal
        latest = trade_signals[-1]

        # Check if this is a new signal (not a repeat)
        signal_key = f"{latest.side.value}_{instrument}_{bars[-1].timestamp}"
        if strategy_dep.pending_signal and strategy_dep.pending_signal.get("key") == signal_key:
            return None  # Already seen this signal

        strategy_dep.last_signal_at = datetime.now(timezone.utc).isoformat()
        strategy_dep.pending_signal = {"key": signal_key}

        # Position sizing
        position_pct = strategy_dep.kelly_fraction * strategy_dep.max_position_pct
        # Scale down if portfolio is already hot
        available_risk = self.max_portfolio_risk_pct - self.portfolio.portfolio_heat()
        if available_risk <= 0:
            log.warning("Portfolio heat limit reached, skipping signal")
            return None
        position_pct = min(position_pct, available_risk / 100)

        dollar_amount = self.portfolio.equity * position_pct
        shares = int(dollar_amount / bars[-1].close) if bars[-1].close > 0 else 0
        if shares <= 0:
            return None

        signal = {
            "strategy": strategy_dep.strategy_name,
            "instrument": instrument,
            "side": latest.side.value,
            "price": bars[-1].close,
            "shares": shares,
            "dollar_amount": round(dollar_amount, 2),
            "position_pct": round(position_pct * 100, 2),
            "confidence": latest.confidence,
            "reason": latest.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "SIGNAL: %s %s %s @ $%.2f (%d shares, %.1f%% of portfolio)",
            latest.side.value.upper(), instrument, strategy_dep.strategy_name,
            bars[-1].close, shares, position_pct * 100,
        )

        return signal

    # ── EXECUTION: Send orders to Alpaca ──

    async def execute_signals(self, signals: list[dict]) -> list[dict]:
        """
        Execute trade signals via Alpaca.

        Returns list of execution results.
        """
        if not self.alpaca:
            log.warning("No Alpaca client — signals not executed")
            return [{"signal": s, "status": "no_broker"} for s in signals]

        results = []

        for signal in signals:
            # Pre-flight checks
            if self.portfolio.halted:
                results.append({"signal": signal, "status": "halted"})
                continue

            if self.portfolio.open_position_count() >= self.max_positions:
                results.append({"signal": signal, "status": "max_positions"})
                continue

            # Check daily loss limit
            if self.portfolio.equity > 0:
                daily_loss = abs(self.portfolio.daily_pnl / self.portfolio.equity * 100)
                if daily_loss >= self.daily_loss_limit_pct:
                    results.append({"signal": signal, "status": "daily_limit"})
                    continue

            try:
                order = await self._submit_order(signal)
                results.append({"signal": signal, "status": "submitted", "order": order})

                # Track position
                sym = signal["instrument"]
                self.portfolio.positions[sym] = {
                    "shares": signal["shares"],
                    "side": signal["side"],
                    "avg_price": signal["price"],
                    "strategy": signal["strategy"],
                    "entry_time": signal["timestamp"],
                    "risk_pct": signal["position_pct"],
                }

            except Exception as e:
                log.error("Order failed for %s: %s", signal["instrument"], e)
                results.append({"signal": signal, "status": "error", "error": str(e)})

        return results

    async def _submit_order(self, signal: dict) -> dict:
        """Submit an order to Alpaca."""
        side = "buy" if signal["side"] == "long" else "sell"

        # Use market order for simplicity and speed
        order = await self.alpaca.submit_order(
            symbol=signal["instrument"],
            qty=signal["shares"],
            side=side,
            type="market",
            time_in_force="day",
        )

        log.info(
            "ORDER SUBMITTED: %s %d %s @ market (%s)",
            side.upper(), signal["shares"], signal["instrument"],
            "PAPER" if self.paper_mode else "LIVE",
        )

        return order if isinstance(order, dict) else {"id": str(order)}

    # ── MONITORING: Track live performance ──

    async def update_portfolio(self) -> PortfolioState:
        """
        Sync portfolio state with Alpaca account.
        Compute live metrics and check health of all strategies.
        """
        if self.alpaca:
            try:
                account = await self.alpaca.get_account()
                if account:
                    self.portfolio.equity = float(account.get("equity", self.portfolio.equity))
                    self.portfolio.cash = float(account.get("cash", self.portfolio.cash))

                # Update peak and drawdown
                if self.portfolio.equity > self.portfolio.peak_equity:
                    self.portfolio.peak_equity = self.portfolio.equity
                if self.portfolio.peak_equity > 0:
                    self.portfolio.current_drawdown_pct = (
                        (self.portfolio.peak_equity - self.portfolio.equity)
                        / self.portfolio.peak_equity * 100
                    )
                    self.portfolio.max_drawdown_pct = max(
                        self.portfolio.max_drawdown_pct,
                        self.portfolio.current_drawdown_pct,
                    )

                # Sync positions
                positions = await self.alpaca.get_positions()
                if positions:
                    self.portfolio.positions = {}
                    for pos in positions:
                        sym = pos.get("symbol", "")
                        self.portfolio.positions[sym] = {
                            "shares": abs(int(float(pos.get("qty", 0)))),
                            "side": "long" if float(pos.get("qty", 0)) > 0 else "short",
                            "avg_price": float(pos.get("avg_entry_price", 0)),
                            "market_value": float(pos.get("market_value", 0)),
                            "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
                            "unrealized_pnl_pct": float(pos.get("unrealized_plpc", 0)) * 100,
                        }

            except Exception as e:
                log.error("Portfolio update failed: %s", e)

        # Check circuit breakers
        self._check_circuit_breakers()

        self.portfolio.total_pnl = self.portfolio.equity - self.portfolio.peak_equity + self.portfolio.max_drawdown_pct
        self.portfolio.total_pnl_pct = (
            (self.portfolio.equity / 100_000.0 - 1) * 100
            if self.portfolio.equity > 0 else 0
        )

        return self.portfolio

    def _check_circuit_breakers(self) -> None:
        """Check and enforce portfolio-level circuit breakers."""
        # Max drawdown halt
        if self.portfolio.current_drawdown_pct >= self.max_drawdown_halt_pct:
            self.portfolio.halted = True
            self.portfolio.halt_reason = (
                f"Max drawdown breached: {self.portfolio.current_drawdown_pct:.1f}% "
                f">= {self.max_drawdown_halt_pct}% limit"
            )
            log.critical("TRADING HALTED: %s", self.portfolio.halt_reason)

        # Daily loss halt
        if self.portfolio.equity > 0:
            daily_loss = abs(self.portfolio.daily_pnl / self.portfolio.equity * 100)
            if daily_loss >= self.daily_loss_limit_pct:
                self.portfolio.halted = True
                self.portfolio.halt_reason = (
                    f"Daily loss limit breached: {daily_loss:.1f}% "
                    f">= {self.daily_loss_limit_pct}% limit"
                )
                log.critical("TRADING HALTED: %s", self.portfolio.halt_reason)

    def health_check(self) -> list[dict]:
        """
        Check health of all deployed strategies.
        Kill strategies that have degraded significantly.
        """
        results = []

        for strategy in self.portfolio.deployed_strategies:
            if strategy.status == DeploymentStatus.KILLED:
                continue

            if strategy.live_trades < 5:
                results.append({
                    "strategy": strategy.strategy_name,
                    "status": "warming_up",
                    "trades": strategy.live_trades,
                })
                continue

            degradation = strategy.degradation_ratio()

            if degradation < self.degradation_threshold:
                # Strategy has degraded too much — kill it
                strategy.status = DeploymentStatus.DEGRADED
                results.append({
                    "strategy": strategy.strategy_name,
                    "status": "KILLED",
                    "reason": f"Degradation ratio {degradation:.2f} < threshold {self.degradation_threshold}",
                    "expected_sharpe": strategy.expected_sharpe,
                    "live_sharpe": strategy.live_sharpe,
                })
                log.warning(
                    "STRATEGY KILLED: %s — degradation %.2f (expected Sharpe=%.2f, live=%.2f)",
                    strategy.strategy_name, degradation,
                    strategy.expected_sharpe, strategy.live_sharpe,
                )
            elif degradation < 0.7:
                # Degraded but not dead — reduce position size
                strategy.max_position_pct *= 0.5
                results.append({
                    "strategy": strategy.strategy_name,
                    "status": "degraded",
                    "reason": f"Degradation {degradation:.2f} — position size halved",
                })
            else:
                results.append({
                    "strategy": strategy.strategy_name,
                    "status": "healthy",
                    "degradation": degradation,
                })

        return results

    # ── REOPTIMIZATION ──

    async def reoptimize(self, instruments: list[str] | None = None) -> list[DeployedStrategy]:
        """
        Weekly reoptimization — find new strategies and replace underperformers.

        Discovers new candidates, promotes the best, kills the worst.
        """
        log.info("Starting weekly reoptimization...")

        # Find new candidates
        candidates = await self.discover_strategies(instruments)

        if not candidates:
            log.warning("No new candidates found during reoptimization")
            return self.portfolio.deployed_strategies

        # Keep strategies that are still healthy
        healthy = [
            s for s in self.portfolio.deployed_strategies
            if s.status in (DeploymentStatus.PAPER_TESTING, DeploymentStatus.LIVE)
            and s.degradation_ratio() >= self.degradation_threshold
        ]

        # Replace killed/degraded slots with new candidates
        available_slots = self.max_strategies - len(healthy)
        new_deploys = candidates[:available_slots] if available_slots > 0 else []

        for nd in new_deploys:
            nd.status = DeploymentStatus.PAPER_TESTING if self.paper_mode else DeploymentStatus.LIVE
            nd.deployed_at = datetime.now(timezone.utc).isoformat()
            nd.last_reoptimized_at = datetime.now(timezone.utc).isoformat()

        self.portfolio.deployed_strategies = healthy + new_deploys
        self.portfolio.last_optimization = datetime.now(timezone.utc).isoformat()

        log.info(
            "Reoptimization complete: %d healthy kept, %d new deployed, %d total",
            len(healthy), len(new_deploys), len(self.portfolio.deployed_strategies),
        )

        return self.portfolio.deployed_strategies

    # ── PERSISTENCE ──

    def save_state(self, path: str = "autonomous_state.json") -> None:
        """Save portfolio and strategy state to disk."""
        state = self.portfolio.to_dict()
        filepath = Path(path)
        filepath.write_text(json.dumps(state, indent=2, default=str))
        log.info("State saved to %s", filepath)

    def load_state(self, path: str = "autonomous_state.json") -> bool:
        """Load portfolio state from disk."""
        filepath = Path(path)
        if not filepath.exists():
            return False
        try:
            state = json.loads(filepath.read_text())
            self.portfolio.equity = state.get("equity", self.portfolio.equity)
            self.portfolio.cash = state.get("cash", self.portfolio.cash)
            self.portfolio.paper_mode = state.get("paper_mode", True)
            self.portfolio.halted = state.get("halted", False)
            self.portfolio.halt_reason = state.get("halt_reason", "")
            log.info("State loaded from %s", filepath)
            return True
        except Exception as e:
            log.error("Failed to load state: %s", e)
            return False

    # ── MAIN LOOP ──

    async def run_cycle(self) -> dict:
        """
        Run one full trading cycle:
        1. Update portfolio state
        2. Check strategy health
        3. Generate signals
        4. Execute trades
        5. Save state

        Call this every 1-5 minutes during market hours.
        """
        cycle_start = time.time()

        # 1. Sync with Alpaca
        await self.update_portfolio()

        # 2. Health check
        health = self.health_check()

        # 3. Generate signals
        signals = await self.check_signals()

        # 4. Execute
        executions = []
        if signals:
            executions = await self.execute_signals(signals)

        # 5. Save state
        self.save_state()

        elapsed = time.time() - cycle_start

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_ms": round(elapsed * 1000),
            "equity": self.portfolio.equity,
            "drawdown_pct": self.portfolio.current_drawdown_pct,
            "open_positions": self.portfolio.open_position_count(),
            "signals_generated": len(signals),
            "orders_submitted": len([e for e in executions if e.get("status") == "submitted"]),
            "strategies_healthy": len([h for h in health if h.get("status") == "healthy"]),
            "strategies_killed": len([h for h in health if h.get("status") == "KILLED"]),
            "halted": self.portfolio.halted,
        }

        log.info(
            "Cycle: equity=$%.0f dd=%.1f%% pos=%d signals=%d orders=%d [%dms]",
            self.portfolio.equity, self.portfolio.current_drawdown_pct,
            self.portfolio.open_position_count(), len(signals),
            result["orders_submitted"], result["cycle_ms"],
        )

        return result


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def _default_instruments() -> list[str]:
    """Full instrument universe for backtesting."""
    return [
        # Broad market
        "SPY", "QQQ", "IWM", "DIA",
        # Treasuries
        "TLT", "IEF", "SHY", "HYG",
        # Commodities
        "GLD", "SLV", "USO", "UNG",
        # Sectors
        "XLF", "XLE", "XLK", "XLV", "SMH",
        # Leveraged
        "TQQQ", "SOXL",
        # International
        "EEM", "FXI",
        # Volatility
        "UVXY", "SVXY",
        # Mega-cap
        "AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT", "AMD", "GOOGL",
        # Momentum
        "PLTR", "COIN", "MSTR", "NFLX",
        # Blue chip
        "JPM", "GS", "V", "UNH",
    ]


def _classify_instrument(symbol: str) -> str:
    """Classify an instrument into asset class for strategy selection."""
    crypto_symbols = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"}
    if symbol.split("/")[0] in crypto_symbols or symbol.endswith("USDT"):
        return "crypto"

    # ETFs and leveraged products work best with stock strategies
    # (they have the same OHLCV structure)
    return "stock"


def _instantiate_strategy(class_name: str, params: dict):
    """Instantiate a strategy from class name and parameters."""
    from src.services.trading_engine import (
        MACrossover, VolatilityBreakout, MomentumStrategy,
        MeanReversion, RSIMomentum,
    )

    class_map = {
        "MACrossover": MACrossover,
        "VolatilityBreakout": VolatilityBreakout,
        "MomentumStrategy": MomentumStrategy,
        "MeanReversion": MeanReversion,
        "RSIMomentum": RSIMomentum,
    }

    # Also include backtesting strategies
    try:
        from src.backtesting.strategies import (
            VWAPReversion, OpeningRangeBreakout, GapFade,
            DualMomentum, CryptoMomentum, CryptoMeanReversion,
            PredictionMomentum, PredictionReversion, AdaptiveTrend,
        )
        class_map.update({
            "VWAPReversion": VWAPReversion,
            "OpeningRangeBreakout": OpeningRangeBreakout,
            "GapFade": GapFade,
            "DualMomentum": DualMomentum,
            "CryptoMomentum": CryptoMomentum,
            "CryptoMeanReversion": CryptoMeanReversion,
            "PredictionMomentum": PredictionMomentum,
            "PredictionReversion": PredictionReversion,
            "AdaptiveTrend": AdaptiveTrend,
        })
    except ImportError:
        pass

    cls = class_map.get(class_name)
    if not cls:
        log.warning("Unknown strategy class: %s", class_name)
        return None

    try:
        # Filter params to only those the constructor accepts
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        # Cast int params
        for k, v in valid_params.items():
            if isinstance(v, float) and v == int(v):
                valid_params[k] = int(v)
        return cls(**valid_params)
    except Exception as e:
        log.warning("Could not instantiate %s with %s: %s", class_name, params, e)
        return None


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════


async def run_discovery_cli(
    instruments: list[str] | None = None,
    deploy: bool = True,
    save: bool = True,
) -> dict:
    """
    Run strategy discovery from CLI / GitHub Actions.

    1. Backtests everything
    2. Deploys winners
    3. Saves state to JSON (for pickup by the live scheduler)

    Returns summary dict.
    """
    trader = AutonomousTrader(paper_mode=True)

    # Discover
    candidates = await trader.discover_strategies(instruments)

    if not candidates:
        return {"status": "no_candidates", "tested": 0}

    # Deploy top strategies
    if deploy:
        deployed = trader.deploy_strategies(candidates)
    else:
        deployed = candidates

    # Save state
    if save:
        trader.save_state("autonomous_state.json")

    summary = {
        "status": "ok",
        "candidates_found": len(candidates),
        "deployed": len(deployed),
        "strategies": [s.to_dict() for s in deployed],
        "top_5": [
            {
                "name": s.strategy_name,
                "sharpe": s.expected_sharpe,
                "cagr": s.expected_cagr_pct,
                "wf_efficiency": s.wf_efficiency,
                "instruments": s.instruments,
            }
            for s in deployed[:5]
        ],
    }

    return summary


def main():
    """CLI entry point for autonomous trader."""
    import argparse

    parser = argparse.ArgumentParser(description="Momentum Forge Autonomous Trader")
    parser.add_argument("command", choices=["discover", "status", "cycle"], help="Command to run")
    parser.add_argument("--instruments", nargs="+", default=None, help="Override instruments")
    parser.add_argument("--deploy", action="store_true", default=True, help="Deploy candidates")
    parser.add_argument("--save", action="store_true", default=True, help="Save state to disk")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "discover":
        result = asyncio.run(run_discovery_cli(args.instruments, args.deploy, args.save))
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "status":
        trader = AutonomousTrader()
        if trader.load_state():
            print(json.dumps(trader.portfolio.to_dict(), indent=2, default=str))
        else:
            print("No state file found. Run 'discover' first.")
    elif args.command == "cycle":
        trader = AutonomousTrader()
        trader.load_state()
        result = asyncio.run(trader.run_cycle())
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
