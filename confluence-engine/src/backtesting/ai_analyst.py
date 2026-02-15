"""
AI Strategy Analyst — Claude + OpenAI for strategy evaluation.

Uses AI to:
1. Evaluate backtest results and pick the best strategy
2. Analyze market regime and suggest adjustments
3. Review genetic optimization results for overfitting signals
4. Generate natural language reports on portfolio performance
5. Cross-reference strategies with academic research

The AI sees raw data and makes decisions. No human bias baked in.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("forge.ai_analyst")


STRATEGY_EVAL_PROMPT = """\
You are a quantitative analyst evaluating algorithmic trading strategies
across ALL tradeable instruments: stocks, ETFs, index funds, treasuries
(TLT, IEF, SHY), commodities (GLD, SLV, USO, UNG), sector ETFs (XLF,
XLE, XLK, SMH), leveraged products (TQQQ, SOXL, TMF), international
(EEM, FXI), volatility products (UVXY, SVXY), and crypto.

You have a PhD in financial engineering and 15 years of experience building
systematic trading systems at a top quant fund.

Your job: analyze these backtest results and give a BRUTALLY HONEST assessment.
Consider cross-asset correlations and portfolio construction across asset classes.

Criteria (in order of importance):
1. **Deflated Sharpe > 0**: If the deflated Sharpe is negative, the strategy
   likely has no real edge — it's just data mining. REJECT IT.
2. **Walk-forward efficiency > 0.5**: Out-of-sample performance should be at
   least 50% of in-sample. Below that = curve fitting.
3. **Minimum 30 trades**: Statistical significance requires sufficient samples.
4. **Profit factor > 1.3**: Gross profit must meaningfully exceed gross loss.
5. **Max drawdown < 25%**: Capital preservation is paramount.
6. **Tail ratio > 0.8**: We want positive skew (big wins, small losses).
7. **Sortino > 1.0**: Downside risk must be well-compensated.

For each strategy, assess:
- Is the edge REAL or is it curve-fitted?
- Would this survive in live trading with slippage and execution delays?
- What market conditions could kill this strategy?
- What's the recommended Kelly fraction?

Return your analysis as JSON:
{
  "ranking": [
    {
      "strategy": "strategy_name",
      "symbol": "SYMBOL",
      "verdict": "DEPLOY" | "PAPER_TEST" | "REJECT",
      "confidence": 0.0-1.0,
      "reasoning": "2-3 sentences",
      "risk_factors": ["factor1", "factor2"],
      "recommended_kelly_fraction": 0.0-0.5,
      "estimated_live_sharpe": 0.0-3.0
    }
  ],
  "portfolio_notes": "Overall portfolio construction advice",
  "regime_warning": "Any current market conditions to watch"
}

Be RUTHLESS. A mediocre strategy that loses money slowly is worse than cash.
Only recommend DEPLOY for strategies with clear, robust edge.
"""

REGIME_ANALYSIS_PROMPT = """\
You are a macro strategist analyzing current market conditions for a
systematic trading operation that trades stocks, crypto, and prediction markets.

Given the following market data, classify the current regime and recommend
position sizing adjustments:

Market Data:
{market_data}

Return your analysis as JSON:
{
  "regime": "risk_on" | "risk_off" | "transitioning" | "crisis",
  "vix_assessment": "description of volatility environment",
  "crypto_assessment": "crypto market conditions",
  "recommended_exposure": {
    "stocks": 0.0-1.0,
    "crypto": 0.0-1.0,
    "polymarket": 0.0-1.0
  },
  "key_risks": ["risk1", "risk2"],
  "upcoming_catalysts": ["catalyst1", "catalyst2"]
}
"""


@dataclass
class AIAnalysis:
    """Result from AI analysis."""
    raw_response: str
    parsed: dict
    provider: str       # "claude" or "openai"
    model: str
    tokens_used: int


async def evaluate_strategies(
    results: list[dict],
    provider: str = "claude",
) -> AIAnalysis:
    """
    Send backtest results to AI for evaluation.

    Args:
        results: List of strategy result summaries.
        provider: "claude" or "openai".
    """
    # Format results for the prompt
    formatted = "BACKTEST RESULTS FOR EVALUATION:\n" + "=" * 60 + "\n"
    for i, r in enumerate(results, 1):
        formatted += f"\n#{i}: {r.get('strategy', 'unknown')} on {r.get('symbol', '?')}\n"
        for key, val in r.items():
            if key not in ("strategy", "symbol"):
                formatted += f"  {key}: {val}\n"

    full_prompt = STRATEGY_EVAL_PROMPT + "\n\n" + formatted

    if provider == "claude":
        return await _call_claude(full_prompt)
    else:
        return await _call_openai(full_prompt)


async def analyze_regime(market_data: dict, provider: str = "claude") -> AIAnalysis:
    """Get AI regime analysis for position sizing."""
    prompt = REGIME_ANALYSIS_PROMPT.format(market_data=json.dumps(market_data, indent=2))
    if provider == "claude":
        return await _call_claude(prompt)
    else:
        return await _call_openai(prompt)


async def _call_claude(prompt: str) -> AIAnalysis:
    """Call Claude API for analysis."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return AIAnalysis(
            raw_response="No ANTHROPIC_API_KEY set",
            parsed={"error": "No API key"},
            provider="claude",
            model="none",
            tokens_used=0,
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        parsed = _parse_json_response(content)

        return AIAnalysis(
            raw_response=content,
            parsed=parsed,
            provider="claude",
            model="claude-sonnet-4-5-20250929",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )
    except Exception as e:
        log.error("Claude API error: %s", e)
        return AIAnalysis(
            raw_response=str(e),
            parsed={"error": str(e)},
            provider="claude",
            model="error",
            tokens_used=0,
        )


async def _call_openai(prompt: str) -> AIAnalysis:
    """Call OpenAI API for analysis."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return AIAnalysis(
            raw_response="No OPENAI_API_KEY set",
            parsed={"error": "No API key"},
            provider="openai",
            model="none",
            tokens_used=0,
        )

    client = openai.AsyncOpenAI(api_key=api_key)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": "You are a quantitative trading analyst. Return JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        parsed = _parse_json_response(content)

        return AIAnalysis(
            raw_response=content,
            parsed=parsed,
            provider="openai",
            model="gpt-4o",
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )
    except Exception as e:
        log.error("OpenAI API error: %s", e)
        return AIAnalysis(
            raw_response=str(e),
            parsed={"error": str(e)},
            provider="openai",
            model="error",
            tokens_used=0,
        )


def _parse_json_response(content: str) -> dict:
    """Extract JSON from AI response."""
    # Try to find JSON block
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON in code blocks
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    return {"raw_text": content[:1000]}


def format_report_for_display(results: list[dict]) -> str:
    """Format backtest results as a readable text report."""
    lines = [
        "=" * 70,
        "  MOMENTUM FORGE — BACKTEST REPORT",
        "=" * 70,
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"  #{i}  {r.get('strategy', '?')} on {r.get('symbol', '?')}")
        lines.append(f"  {'─' * 40}")
        lines.append(f"  CAGR:           {r.get('cagr', 'N/A')}")
        lines.append(f"  Sharpe:         {r.get('sharpe', 'N/A')}")
        lines.append(f"  Sortino:        {r.get('sortino', 'N/A')}")
        lines.append(f"  Max Drawdown:   {r.get('max_drawdown', 'N/A')}")
        lines.append(f"  Win Rate:       {r.get('win_rate', 'N/A')}")
        lines.append(f"  Profit Factor:  {r.get('profit_factor', 'N/A')}")
        lines.append(f"  Total Trades:   {r.get('total_trades', 'N/A')}")
        lines.append(f"  Expectancy:     {r.get('expectancy', 'N/A')}")
        lines.append(f"  Deflated Sharpe:{r.get('deflated_sharpe', 'N/A')}")
        lines.append(f"  Tail Ratio:     {r.get('tail_ratio', 'N/A')}")
        if r.get('wf_efficiency'):
            lines.append(f"  WF Efficiency:  {r.get('wf_efficiency')}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
