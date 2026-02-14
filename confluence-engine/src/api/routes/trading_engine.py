"""
Trading Engine API routes.

Controls the autonomous trading engine: run backtests, get AI
recommendations, activate/deactivate live signal generation,
and view current engine state.

POST /api/v1/engine/backtest          → Run backtests on tickers
POST /api/v1/engine/optimize          → Run full AI optimization cycle
GET  /api/v1/engine/status            → Current engine state
GET  /api/v1/engine/strategies        → List available strategies
GET  /api/v1/engine/results           → Full backtest results
POST /api/v1/engine/activate          → Activate the engine with selected strategy
POST /api/v1/engine/deactivate        → Stop the engine
POST /api/v1/engine/signal            → Check for current trade signal
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
log = logging.getLogger("vicuna.engine.api")


# ── Request/Response Schemas ──


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(default=["SPY", "QQQ"], description="Tickers to backtest")
    lookback_days: int = Field(default=252, ge=60, le=504, description="Days of history (60-504)")


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(default=["SPY", "QQQ"], description="Tickers to optimize")


class ActivateRequest(BaseModel):
    strategy: str | None = Field(default=None, description="Override AI's pick (or None for AI-selected)")
    ticker: str | None = Field(default=None, description="Override ticker (or None for AI-selected)")


# ── Helpers ──


def _get_alpaca():
    from src.api.dependencies import get_alpaca_client
    client = get_alpaca_client()
    if not client or not client.is_configured:
        raise HTTPException(status_code=503, detail="Alpaca not configured — add ALPACA_API_KEY to .env")
    return client


def _get_ai_router():
    from src.api.dependencies import get_ai_router
    return get_ai_router()


# ── Endpoints ──


@router.get("/strategies")
async def list_strategies():
    """List all available trading strategies with their parameters."""
    from src.services.trading_engine import get_all_strategies

    strategies = get_all_strategies()
    return {
        "count": len(strategies),
        "strategies": [
            {
                "name": s.name,
                "full_name": f"{s.name}({s.params})",
                "description": s.description,
                "params": s.params,
            }
            for s in strategies
        ],
    }


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """
    Run all strategies against specified tickers.

    Returns ranked results sorted by Sharpe ratio. Does NOT
    activate trading — just shows what would have worked.
    """
    from src.services.trading_engine import get_all_strategies, get_engine_state, run_full_backtest

    client = _get_alpaca()

    log.info("Running backtest: %s, %d days", req.tickers, req.lookback_days)

    results = await run_full_backtest(
        alpaca_client=client,
        tickers=req.tickers,
        strategies=get_all_strategies(),
        lookback_days=req.lookback_days,
    )

    # Update engine state
    state = get_engine_state()
    state.backtest_results = results
    state.last_backtest_at = datetime.now(timezone.utc).isoformat()

    return {
        "total_combinations": len(results),
        "tickers": req.tickers,
        "lookback_days": req.lookback_days,
        "results": [r.to_summary() for r in results[:20]],
        "best": results[0].to_summary() if results else None,
    }


@router.post("/optimize")
async def run_optimization(req: OptimizeRequest):
    """
    Full AI optimization: backtest everything, then let the AI pick
    the best strategy+ticker combination.

    This is the main entry point — it runs backtests and gets an
    AI recommendation in one call.
    """
    from src.services.strategy_optimizer import run_optimization_cycle

    client = _get_alpaca()
    ai = _get_ai_router()

    recommendation = await run_optimization_cycle(
        alpaca_client=client,
        ai_router=ai,
        tickers=req.tickers,
    )

    return {
        "recommendation": recommendation,
        "message": (
            f"AI recommends: {recommendation.get('recommendation', 'none')} "
            f"on {recommendation.get('ticker', 'none')} "
            f"(confidence: {recommendation.get('confidence', 0) * 100:.0f}%)"
        ),
    }


@router.get("/status")
async def engine_status():
    """Get current trading engine state."""
    from src.services.trading_engine import get_engine_state
    return get_engine_state().to_dict()


@router.get("/results")
async def get_results(limit: int = 50):
    """Get full backtest results from the last run."""
    from src.services.trading_engine import get_engine_state

    state = get_engine_state()
    results = state.backtest_results[:limit]
    return {
        "count": len(results),
        "total": len(state.backtest_results),
        "last_backtest_at": state.last_backtest_at,
        "results": [r.to_summary() for r in results],
    }


@router.post("/activate")
async def activate_engine(req: ActivateRequest):
    """
    Activate the trading engine with the selected (or AI-recommended) strategy.

    Once active, the engine generates real-time signals and can execute
    trades through Alpaca (paper by default).
    """
    from src.services.trading_engine import get_engine_state

    state = get_engine_state()

    strategy = req.strategy or state.selected_strategy
    ticker = req.ticker or state.selected_ticker

    if not strategy:
        raise HTTPException(
            status_code=400,
            detail="No strategy selected. Run /optimize first or specify a strategy.",
        )
    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="No ticker selected. Run /optimize first or specify a ticker.",
        )

    state.active = True
    state.selected_strategy = strategy
    state.selected_ticker = ticker

    log.info("Engine ACTIVATED: %s on %s", strategy, ticker)

    return {
        "status": "active",
        "strategy": strategy,
        "ticker": ticker,
        "message": f"Engine active — running {strategy} on {ticker} (paper trading)",
    }


@router.post("/deactivate")
async def deactivate_engine():
    """Stop the trading engine. No new signals or trades will be generated."""
    from src.services.trading_engine import get_engine_state

    state = get_engine_state()
    state.active = False

    log.info("Engine DEACTIVATED")

    return {"status": "inactive", "message": "Trading engine stopped"}


@router.post("/signal")
async def check_signal():
    """
    Check if the active strategy has a current trade signal.

    Fetches latest bars from Alpaca and runs the selected strategy
    to see if it's generating a signal right now.
    """
    from src.services.trading_engine import (
        get_all_strategies,
        get_engine_state,
        parse_bars,
    )

    state = get_engine_state()

    if not state.active:
        raise HTTPException(status_code=400, detail="Engine is not active. Call /activate first.")

    if not state.selected_strategy or not state.selected_ticker:
        raise HTTPException(status_code=400, detail="No strategy/ticker selected.")

    client = _get_alpaca()

    # Find the matching strategy
    strategies = get_all_strategies()
    full_names = {f"{s.name}({s.params})": s for s in strategies}
    strategy = full_names.get(state.selected_strategy)

    if not strategy:
        # Try matching by base name
        for s in strategies:
            if s.name in state.selected_strategy:
                strategy = s
                break

    if not strategy:
        raise HTTPException(status_code=400, detail=f"Strategy '{state.selected_strategy}' not found")

    # Get latest bars
    raw_bars = await client.get_bars(state.selected_ticker, timeframe="1Day", limit=252)
    if not raw_bars:
        raise HTTPException(status_code=502, detail=f"No data for {state.selected_ticker}")

    bars = parse_bars(raw_bars)
    signals = strategy.generate_signals(bars)

    # Set ticker on signals
    for sig in signals:
        sig.ticker = state.selected_ticker

    if signals:
        latest = signals[-1]
        state.last_signal = latest
        return {
            "has_signal": True,
            "signal": {
                "ticker": latest.ticker,
                "side": latest.side.value,
                "price": latest.price,
                "timestamp": latest.timestamp,
                "strategy": latest.strategy,
                "confidence": latest.confidence,
                "reason": latest.reason,
            },
        }

    return {"has_signal": False, "message": "No signal at this time"}
