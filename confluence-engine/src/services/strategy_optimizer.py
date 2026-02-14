"""
AI Strategy Optimizer — uses Claude/OpenAI to evaluate backtest
results and recommend the best strategy + parameters.

The AI sees raw performance data and decides what to trade.
No human bias — pure data-driven selection.
"""

from __future__ import annotations

import json
import logging

from src.services.trading_engine import BacktestResult, EngineState, get_engine_state

log = logging.getLogger("vicuna.optimizer")

OPTIMIZER_PROMPT = """\
You are an algorithmic trading strategy optimizer. You are given backtest results
for multiple strategies run across one or more tickers using historical daily data.

Your job:
1. Analyze all results and identify the BEST strategy+ticker combination
2. Consider: Sharpe ratio, win rate, profit factor, drawdown, number of trades
3. Reject strategies with fewer than 5 trades (not enough data)
4. Reject strategies with max drawdown > 25% (too risky)
5. Prefer strategies with Sharpe > 1.0 and profit factor > 1.5
6. If no strategy meets these criteria, say "no viable strategy found"

Return your answer as JSON:
{
  "recommendation": "strategy_name",
  "ticker": "TICKER",
  "reasoning": "2-3 sentences on why this is the best choice",
  "risk_notes": "key risks to watch for",
  "confidence": 0.0-1.0,
  "rejected_count": number of strategies you rejected and why
}

Be ruthless — only recommend strategies with real edge. A mediocre strategy
is worse than no strategy.
"""


def format_results_for_ai(results: list[BacktestResult], max_results: int = 30) -> str:
    """Format backtest results into a prompt for the AI."""
    # Take top results by Sharpe
    top = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)[:max_results]

    lines = ["BACKTEST RESULTS (sorted by Sharpe ratio):", "=" * 60]
    for i, r in enumerate(top, 1):
        lines.append(f"\n#{i}: {r.strategy_name} on {r.ticker}")
        lines.append(f"  Total Return: {r.total_return_pct:+.2f}%")
        lines.append(f"  Win Rate: {r.win_rate:.1f}% ({r.total_trades} trades)")
        lines.append(f"  Sharpe Ratio: {r.sharpe_ratio:.2f}")
        lines.append(f"  Profit Factor: {r.profit_factor:.2f}")
        lines.append(f"  Max Drawdown: {r.max_drawdown_pct:.2f}%")
        lines.append(f"  Avg Trade P&L: {r.avg_trade_pnl_pct:+.2f}%")
        lines.append(f"  Bars Tested: {r.bars_tested}")

    return "\n".join(lines)


async def optimize_with_ai(
    results: list[BacktestResult],
    ai_router,
) -> dict:
    """
    Send backtest results to the AI and get a strategy recommendation.

    Returns the AI's recommendation as a parsed dict, plus the raw response.
    """
    if not results:
        return {
            "recommendation": None,
            "reasoning": "No backtest results to evaluate.",
            "confidence": 0.0,
        }

    formatted = format_results_for_ai(results)
    full_prompt = f"{OPTIMIZER_PROMPT}\n\n{formatted}"

    try:
        response, decision = await ai_router.route(
            prompt=full_prompt,
            force_provider="claude",  # Claude is better at analysis
        )

        raw_content = response.content if response else ""

        # Try to extract JSON from the response
        recommendation = _parse_ai_response(raw_content)
        recommendation["raw_response"] = raw_content
        recommendation["provider"] = response.provider if response else "none"

        log.info(
            "AI recommendation: %s on %s (confidence: %.0f%%)",
            recommendation.get("recommendation", "none"),
            recommendation.get("ticker", "none"),
            recommendation.get("confidence", 0) * 100,
        )

        return recommendation

    except Exception as e:
        log.error("AI optimization failed: %s", e)
        # Fall back to pure metrics-based selection
        return _fallback_selection(results)


def _parse_ai_response(content: str) -> dict:
    """Try to extract structured recommendation from AI response."""
    # Try to find JSON in the response
    try:
        # Look for JSON block
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    # If JSON parsing fails, return the raw text as reasoning
    return {
        "recommendation": None,
        "reasoning": content[:500],
        "confidence": 0.0,
    }


def _fallback_selection(results: list[BacktestResult]) -> dict:
    """
    Pure metrics-based strategy selection when AI is unavailable.

    Scores each strategy on a composite of Sharpe, win rate, and profit factor,
    filtered by minimum quality thresholds.
    """
    viable = [
        r for r in results
        if r.total_trades >= 5
        and r.max_drawdown_pct <= 25.0
        and r.sharpe_ratio > 0
    ]

    if not viable:
        return {
            "recommendation": None,
            "ticker": None,
            "reasoning": "No strategy passed minimum quality filters (5+ trades, <25% drawdown, positive Sharpe).",
            "confidence": 0.0,
            "method": "fallback_metrics",
        }

    # Composite score: 40% Sharpe + 30% profit factor + 30% win rate
    def score(r: BacktestResult) -> float:
        sharpe_norm = min(r.sharpe_ratio / 3.0, 1.0)  # Cap at 3.0
        pf_norm = min(r.profit_factor / 3.0, 1.0)
        wr_norm = r.win_rate / 100.0
        return 0.4 * sharpe_norm + 0.3 * pf_norm + 0.3 * wr_norm

    best = max(viable, key=score)
    composite = score(best)

    return {
        "recommendation": best.strategy_name,
        "ticker": best.ticker,
        "reasoning": (
            f"Selected by metrics: Sharpe={best.sharpe_ratio:.2f}, "
            f"Win Rate={best.win_rate:.1f}%, PF={best.profit_factor:.2f}, "
            f"Drawdown={best.max_drawdown_pct:.1f}%"
        ),
        "confidence": round(composite, 2),
        "method": "fallback_metrics",
    }


async def run_optimization_cycle(
    alpaca_client,
    ai_router,
    tickers: list[str],
) -> dict:
    """
    Full optimization cycle:
    1. Backtest all strategies on given tickers
    2. Ask AI to evaluate and pick the best
    3. Update engine state with recommendation

    Returns the recommendation dict.
    """
    from src.services.trading_engine import get_all_strategies, run_full_backtest

    log.info("Starting optimization cycle for %d tickers...", len(tickers))

    # Run backtests
    results = await run_full_backtest(
        alpaca_client=alpaca_client,
        tickers=tickers,
        strategies=get_all_strategies(),
        lookback_days=252,
    )

    log.info("Backtest complete: %d strategy-ticker combinations tested", len(results))

    # Get AI recommendation (or fallback)
    try:
        recommendation = await optimize_with_ai(results, ai_router)
    except Exception as e:
        log.warning("AI unavailable, using fallback: %s", e)
        recommendation = _fallback_selection(results)

    # Update engine state
    state = get_engine_state()
    state.backtest_results = results
    state.ai_recommendation = recommendation.get("reasoning", "")
    state.selected_strategy = recommendation.get("recommendation")
    state.selected_ticker = recommendation.get("ticker")
    from datetime import datetime, timezone
    state.last_backtest_at = datetime.now(timezone.utc).isoformat()

    return recommendation
