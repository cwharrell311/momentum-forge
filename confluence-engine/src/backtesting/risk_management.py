"""
Risk management guardrails for the backtesting engine.

These guardrails run BEFORE and DURING every trade to prevent
catastrophic losses. Think of them as circuit breakers.

Guardrails implemented:
1. Daily loss limit — stop trading after X% daily loss
2. Drawdown halt — pause strategy if drawdown exceeds threshold
3. Economic calendar filter — avoid trading around FOMC/CPI/NFP
4. Correlation filter — don't stack correlated positions
5. Regime filter — adjust behavior based on VIX/volatility regime
6. Max position count — limit concurrent positions
7. Time-of-day filter — only trade during liquid hours
8. Consecutive loss circuit breaker — pause after N straight losses

These are hard guardrails — they override strategy signals.
No strategy is allowed to bypass them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger("forge.risk")


class RiskAction(Enum):
    """What the risk manager decided."""
    ALLOW = "allow"           # Trade is permitted
    REDUCE = "reduce"         # Trade allowed but with reduced size
    BLOCK = "block"           # Trade is blocked
    HALT = "halt"             # All trading halted


@dataclass
class RiskDecision:
    """Result of a risk check."""
    action: RiskAction
    reason: str
    size_multiplier: float = 1.0   # 1.0 = full size, 0.5 = half, 0 = blocked
    guardrails_triggered: list[str] = field(default_factory=list)


class MarketRegime(Enum):
    """Volatility regime classification."""
    LOW_VOL = "low_vol"         # VIX < 15 or realized vol < 10%
    NORMAL = "normal"           # VIX 15-20
    ELEVATED = "elevated"       # VIX 20-30
    HIGH_VOL = "high_vol"       # VIX 30-40
    CRISIS = "crisis"           # VIX > 40


def classify_regime(
    returns: pd.Series,
    lookback: int = 20,
    vix_level: float | None = None,
) -> MarketRegime:
    """
    Classify current market regime from recent returns or VIX.

    If VIX is available, use it directly. Otherwise estimate from
    realized volatility of recent returns.
    """
    if vix_level is not None:
        if vix_level < 15:
            return MarketRegime.LOW_VOL
        elif vix_level < 20:
            return MarketRegime.NORMAL
        elif vix_level < 30:
            return MarketRegime.ELEVATED
        elif vix_level < 40:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.CRISIS

    # Estimate from realized volatility (annualized)
    if len(returns) < lookback:
        return MarketRegime.NORMAL

    recent = returns.iloc[-lookback:]
    realized_vol = recent.std() * np.sqrt(252) * 100  # Annualized %

    if realized_vol < 10:
        return MarketRegime.LOW_VOL
    elif realized_vol < 15:
        return MarketRegime.NORMAL
    elif realized_vol < 25:
        return MarketRegime.ELEVATED
    elif realized_vol < 40:
        return MarketRegime.HIGH_VOL
    else:
        return MarketRegime.CRISIS


# Regime-based position size scaling
REGIME_SCALE = {
    MarketRegime.LOW_VOL: 1.0,     # Full size in calm markets
    MarketRegime.NORMAL: 1.0,      # Full size
    MarketRegime.ELEVATED: 0.75,   # Reduce by 25%
    MarketRegime.HIGH_VOL: 0.50,   # Half size
    MarketRegime.CRISIS: 0.25,     # Quarter size — survival mode
}


class RiskManager:
    """
    Central risk management system.

    All trade signals pass through this before execution.
    The risk manager can reduce size, block trades, or halt trading entirely.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,      # Stop trading after 3% daily loss
        max_drawdown_pct: float = 25.0,        # Halt strategy if DD > 25%
        max_open_positions: int = 6,            # Max concurrent positions
        max_correlation: float = 0.70,          # Don't stack correlated > 0.7
        consecutive_loss_limit: int = 5,        # Pause after 5 straight losses
        economic_event_buffer_hours: int = 2,   # No trades 2h before/after high-impact events
        trading_hours_only: bool = True,         # Stocks: only trade 9:30-16:00 ET
        regime_scaling: bool = True,             # Scale positions by vol regime
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.max_correlation = max_correlation
        self.consecutive_loss_limit = consecutive_loss_limit
        self.economic_event_buffer_hours = economic_event_buffer_hours
        self.trading_hours_only = trading_hours_only
        self.regime_scaling = regime_scaling

        # State tracking
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._open_positions: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""

    def update_state(
        self,
        equity: float,
        daily_pnl: float = 0.0,
        open_positions: int = 0,
        last_trade_won: bool | None = None,
    ) -> None:
        """Update risk manager state with current portfolio info."""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        self._daily_pnl = daily_pnl
        self._open_positions = open_positions

        if last_trade_won is not None:
            if last_trade_won:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self._daily_pnl = 0.0

    def check_trade(
        self,
        symbol: str,
        returns_history: pd.Series | None = None,
        timestamp: datetime | None = None,
        economic_events: pd.DataFrame | None = None,
        portfolio_returns: dict[str, pd.Series] | None = None,
        asset_class: str = "stock",
    ) -> RiskDecision:
        """
        Run all guardrails on a proposed trade.

        Returns a RiskDecision telling the engine whether to proceed.
        """
        triggered = []
        multiplier = 1.0

        # 1. Check if trading is halted
        if self._halted:
            return RiskDecision(
                action=RiskAction.HALT,
                reason=f"Trading halted: {self._halt_reason}",
                size_multiplier=0,
                guardrails_triggered=["halt"],
            )

        # 2. Daily loss limit
        if self._current_equity > 0:
            daily_loss_pct = abs(self._daily_pnl / self._current_equity * 100)
            if daily_loss_pct >= self.max_daily_loss_pct:
                return RiskDecision(
                    action=RiskAction.BLOCK,
                    reason=f"Daily loss limit hit: {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%",
                    size_multiplier=0,
                    guardrails_triggered=["daily_loss_limit"],
                )

        # 3. Drawdown check
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if current_dd >= self.max_drawdown_pct:
                self._halted = True
                self._halt_reason = f"Max drawdown hit: {current_dd:.1f}%"
                return RiskDecision(
                    action=RiskAction.HALT,
                    reason=self._halt_reason,
                    size_multiplier=0,
                    guardrails_triggered=["max_drawdown"],
                )
            # Scale down as drawdown approaches limit
            if current_dd >= self.max_drawdown_pct * 0.6:
                dd_scale = 1.0 - (current_dd / self.max_drawdown_pct)
                multiplier *= max(0.25, dd_scale)
                triggered.append("drawdown_scaling")

        # 4. Position count limit
        if self._open_positions >= self.max_open_positions:
            return RiskDecision(
                action=RiskAction.BLOCK,
                reason=f"Max positions reached: {self._open_positions}/{self.max_open_positions}",
                size_multiplier=0,
                guardrails_triggered=["max_positions"],
            )

        # 5. Consecutive loss circuit breaker
        if self._consecutive_losses >= self.consecutive_loss_limit:
            return RiskDecision(
                action=RiskAction.BLOCK,
                reason=f"Consecutive loss limit: {self._consecutive_losses} straight losses",
                size_multiplier=0,
                guardrails_triggered=["consecutive_losses"],
            )
        if self._consecutive_losses >= self.consecutive_loss_limit - 2:
            multiplier *= 0.5
            triggered.append("consecutive_loss_warning")

        # 6. Economic calendar filter
        if economic_events is not None and timestamp is not None and len(economic_events) > 0:
            buffer_h = self.economic_event_buffer_hours
            high_impact = economic_events[economic_events["impact"] == "HIGH"]
            for _, event in high_impact.iterrows():
                event_time = pd.to_datetime(f"{event['date']} {event.get('time', '12:00')}")
                if event_time.tzinfo is None:
                    event_time = event_time.tz_localize("US/Eastern")
                event_time_utc = event_time.tz_convert("UTC")
                ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
                hours_diff = abs((ts - event_time_utc).total_seconds() / 3600)
                if hours_diff <= buffer_h:
                    return RiskDecision(
                        action=RiskAction.BLOCK,
                        reason=f"Economic event: {event['event']} in {hours_diff:.1f}h",
                        size_multiplier=0,
                        guardrails_triggered=["economic_calendar"],
                    )

        # 7. Trading hours filter (stocks only)
        if self.trading_hours_only and asset_class == "stock" and timestamp:
            ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            # Convert to Eastern Time for market hours check
            try:
                import zoneinfo
                eastern = zoneinfo.ZoneInfo("US/Eastern")
            except ImportError:
                from datetime import timezone as tz
                eastern = tz(offset=-5 * 3600)  # Approximate EST
            et_time = ts.astimezone(eastern).time()
            market_open = time(9, 30)
            market_close = time(16, 0)
            if et_time < market_open or et_time > market_close:
                return RiskDecision(
                    action=RiskAction.BLOCK,
                    reason=f"Outside trading hours: {et_time}",
                    size_multiplier=0,
                    guardrails_triggered=["trading_hours"],
                )

        # 8. Regime-based scaling
        if self.regime_scaling and returns_history is not None and len(returns_history) >= 20:
            regime = classify_regime(returns_history)
            regime_mult = REGIME_SCALE.get(regime, 1.0)
            if regime_mult < 1.0:
                multiplier *= regime_mult
                triggered.append(f"regime_{regime.value}")

        # 9. Correlation check
        if portfolio_returns and symbol in portfolio_returns:
            for other_sym, other_returns in portfolio_returns.items():
                if other_sym == symbol:
                    continue
                sym_returns = portfolio_returns.get(symbol)
                if sym_returns is not None and len(sym_returns) >= 20 and len(other_returns) >= 20:
                    # Align and compute correlation
                    aligned = pd.concat([sym_returns, other_returns], axis=1).dropna()
                    if len(aligned) >= 20:
                        corr = aligned.corr().iloc[0, 1]
                        if abs(corr) > self.max_correlation:
                            multiplier *= 0.5
                            triggered.append(f"correlation_{other_sym}_{corr:.2f}")
                            break

        # Final decision
        if multiplier <= 0.1:
            return RiskDecision(
                action=RiskAction.BLOCK,
                reason="Too many risk factors triggered",
                size_multiplier=0,
                guardrails_triggered=triggered,
            )
        elif multiplier < 1.0:
            return RiskDecision(
                action=RiskAction.REDUCE,
                reason=f"Size reduced to {multiplier:.0%}",
                size_multiplier=round(multiplier, 2),
                guardrails_triggered=triggered,
            )
        else:
            return RiskDecision(
                action=RiskAction.ALLOW,
                reason="All guardrails passed",
                size_multiplier=1.0,
                guardrails_triggered=[],
            )

    def unhalt(self) -> None:
        """Manually unhalt trading (requires explicit action)."""
        self._halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        log.info("Trading unhalted manually")
