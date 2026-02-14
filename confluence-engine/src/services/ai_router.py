"""
AI Command Center — intelligent routing between Claude and OpenAI.

The router classifies each incoming request by task type and routes it
to whichever provider handles that category best. If the preferred
provider isn't configured, it falls back to the other one automatically.

Routing philosophy:
  Claude  → deep reasoning, narrative analysis, risk assessment, explanations
  OpenAI  → code generation, structured extraction, quantitative math, quick lookups

Both providers receive the same trading-platform system prompt so they
have full context about the Confluence Engine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from src.utils.ai_clients import AIResponse, ClaudeClient, OpenAIClient

log = logging.getLogger("confluence.ai_router")


# ── System prompt injected into every AI call ──

SYSTEM_PROMPT = """\
You are a trading analyst assistant integrated into the Confluence Engine,
an algorithmic trading platform that detects high-probability setups by
finding confluence across 8 independent signal layers:

1. Options Flow (0.25 weight) — institutional sweeps, blocks, unusual OI
2. GEX / Dealer Positioning (0.18) — gamma exposure, dealer hedging
3. Dark Pool Activity (0.15) — FINRA ATS institutional accumulation
4. Volatility Surface (0.12) — IV rank, skew, term structure
5. Momentum / Technical (0.12) — MA alignment, golden cross, volume
6. Insider Cluster Buying (0.10) — SEC Form 4 C-suite purchases
7. Short Interest (0.08) — SI%, days to cover, squeeze setups
8. VIX Regime Filter (modifier) — calm/elevated/stressed/crisis

The system uses a "flow gate": a setup is only trade-worthy when options
flow agrees with the direction AND at least one of {GEX, volatility}
confirms. Conviction scores range from 0-100%.

Be concise, specific, and quantitative. Reference actual signal layers
and data points when making analysis. No disclaimers about not being
financial advice — the user is an experienced trader.
"""


class TaskCategory(Enum):
    """Categories that map to provider strengths."""

    # Claude-preferred tasks
    ANALYSIS = "analysis"             # "analyze this setup", "what does this mean"
    EXPLANATION = "explanation"       # "explain the confluence score", "why is X bearish"
    RISK_ASSESSMENT = "risk"          # "what's the risk", "position sizing"
    TRADE_THESIS = "thesis"           # "build a trade thesis", "bull/bear case"
    SUMMARY = "summary"              # "summarize today's signals", "recap"

    # OpenAI-preferred tasks
    CODE_GENERATION = "code"          # "write a signal processor", "add an endpoint"
    DATA_EXTRACTION = "extraction"    # "extract key levels", "parse this data"
    CALCULATION = "calculation"       # "calculate risk/reward", "what's the expected value"
    QUICK_LOOKUP = "lookup"           # "what's the current price", "IV rank for X"

    # Neutral — either works, route by availability/config
    GENERAL = "general"


# Keywords that signal which category a task falls into
_CATEGORY_PATTERNS: list[tuple[TaskCategory, list[str]]] = [
    (TaskCategory.ANALYSIS, [
        r"\banalyz[es]?\b", r"\banalysis\b", r"\bassess\b", r"\bevaluat[es]?\b",
        r"\binterpret\b", r"\bwhat.+(?:think|see|read)\b", r"\bbreakdown\b",
        r"\bdiagnos[es]?\b", r"\bsetup\b.*\b(?:look|good|bad)\b",
    ]),
    (TaskCategory.EXPLANATION, [
        r"\bexplain\b", r"\bwhy\b", r"\bhow come\b", r"\bwhat does.+mean\b",
        r"\btell me about\b", r"\bwhat.+happening\b", r"\bdescribe\b",
    ]),
    (TaskCategory.RISK_ASSESSMENT, [
        r"\brisk\b", r"\bposition siz\b", r"\bstop.?loss\b", r"\bdownside\b",
        r"\bhedg[ei]\b", r"\bprotect\b", r"\bmax.?loss\b", r"\brisk.?reward\b",
    ]),
    (TaskCategory.TRADE_THESIS, [
        r"\bthesis\b", r"\bbull(?:ish)?.+case\b", r"\bbear(?:ish)?.+case\b",
        r"\btrade plan\b", r"\btrade idea\b", r"\bstrategy\b", r"\bplay\b",
        r"\bentry\b.*\bexit\b",
    ]),
    (TaskCategory.SUMMARY, [
        r"\bsummar", r"\brecap\b", r"\boverview\b", r"\bhighlight\b",
        r"\bdigest\b", r"\bwrap.?up\b", r"\btldr\b", r"\btl;dr\b",
    ]),
    (TaskCategory.CODE_GENERATION, [
        r"\bwrite.+code\b", r"\bgenerate.+code\b", r"\bcreate.+(?:function|class|endpoint)\b",
        r"\bimplement\b", r"\brefactor\b", r"\badd.+(?:feature|endpoint|route)\b",
        r"\bpython\b.*\bcode\b", r"\bscript\b",
    ]),
    (TaskCategory.DATA_EXTRACTION, [
        r"\bextract\b", r"\bparse\b", r"\bpull out\b", r"\bstructur\b",
        r"\bjson\b", r"\bformat\b.*\bdata\b", r"\bconvert\b",
    ]),
    (TaskCategory.CALCULATION, [
        r"\bcalculat\b", r"\bcompute\b", r"\bexpected value\b", r"\bprobability\b",
        r"\bmath\b", r"\bformula\b", r"\bnumber[s]?\b.*\b(?:crunch|run)\b",
    ]),
    (TaskCategory.QUICK_LOOKUP, [
        r"\bwhat.+(?:price|value|level)\b", r"\bcurrent\b.*\b(?:price|iv|volume)\b",
        r"\blookup\b", r"\bcheck\b.*\bquick\b",
    ]),
]

# Which provider is preferred for each category
_PROVIDER_PREFERENCE: dict[TaskCategory, str] = {
    TaskCategory.ANALYSIS: "claude",
    TaskCategory.EXPLANATION: "claude",
    TaskCategory.RISK_ASSESSMENT: "claude",
    TaskCategory.TRADE_THESIS: "claude",
    TaskCategory.SUMMARY: "claude",
    TaskCategory.CODE_GENERATION: "openai",
    TaskCategory.DATA_EXTRACTION: "openai",
    TaskCategory.CALCULATION: "openai",
    TaskCategory.QUICK_LOOKUP: "openai",
    TaskCategory.GENERAL: "claude",  # Default preference
}


@dataclass
class RoutingDecision:
    """Explains why the router chose a specific provider."""

    category: TaskCategory
    provider: str       # "claude" or "openai"
    reason: str         # Human-readable explanation
    was_fallback: bool  # True if we fell back because preferred wasn't configured


def classify_task(prompt: str) -> TaskCategory:
    """
    Classify a user prompt into a task category using keyword matching.

    Scans the prompt against pattern lists for each category and returns
    the first match. Falls back to GENERAL if nothing matches.
    """
    lower = prompt.lower()
    for category, patterns in _CATEGORY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lower):
                return category
    return TaskCategory.GENERAL


class AIRouter:
    """
    Routes AI requests to the best provider based on task classification.

    Usage:
        router = AIRouter(claude_client, openai_client)
        response = await router.route("Analyze the NVDA confluence setup")
        # → routes to Claude (analysis task)

        response = await router.route("Calculate risk/reward for 100 shares at $850")
        # → routes to OpenAI (calculation task)
    """

    def __init__(
        self,
        claude: ClaudeClient,
        openai_client: OpenAIClient,
        default_provider: str = "auto",
    ):
        self.claude = claude
        self.openai = openai_client
        self.default_provider = default_provider
        self._routing_log: list[RoutingDecision] = []

    def _pick_provider(
        self,
        category: TaskCategory,
        force_provider: str | None = None,
    ) -> RoutingDecision:
        """Decide which provider to use for a given task category."""

        # Explicit override from request
        if force_provider and force_provider in ("claude", "openai"):
            configured = (
                self.claude.is_configured if force_provider == "claude"
                else self.openai.is_configured
            )
            if configured:
                return RoutingDecision(
                    category=category,
                    provider=force_provider,
                    reason=f"Forced to {force_provider} by request",
                    was_fallback=False,
                )
            # If forced but not configured, fall through to auto

        # Global override (not "auto")
        if self.default_provider in ("claude", "openai"):
            configured = (
                self.claude.is_configured if self.default_provider == "claude"
                else self.openai.is_configured
            )
            if configured:
                return RoutingDecision(
                    category=category,
                    provider=self.default_provider,
                    reason=f"Global default set to {self.default_provider}",
                    was_fallback=False,
                )

        # Auto routing — pick based on task category
        preferred = _PROVIDER_PREFERENCE.get(category, "claude")

        if preferred == "claude" and self.claude.is_configured:
            return RoutingDecision(
                category=category,
                provider="claude",
                reason=f"Claude preferred for {category.value} tasks",
                was_fallback=False,
            )
        if preferred == "openai" and self.openai.is_configured:
            return RoutingDecision(
                category=category,
                provider="openai",
                reason=f"OpenAI preferred for {category.value} tasks",
                was_fallback=False,
            )

        # Fallback: use whichever is available
        if self.claude.is_configured:
            return RoutingDecision(
                category=category,
                provider="claude",
                reason=f"Fallback to Claude ({preferred} preferred but OpenAI not configured)",
                was_fallback=True,
            )
        if self.openai.is_configured:
            return RoutingDecision(
                category=category,
                provider="openai",
                reason=f"Fallback to OpenAI ({preferred} preferred but Claude not configured)",
                was_fallback=True,
            )

        return RoutingDecision(
            category=category,
            provider="none",
            reason="No AI providers configured — add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env",
            was_fallback=True,
        )

    async def route(
        self,
        prompt: str,
        force_provider: str | None = None,
        context: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> tuple[AIResponse, RoutingDecision]:
        """
        Classify the task, pick the best provider, and execute.

        Args:
            prompt: The user's question or task.
            force_provider: Override auto-routing ("claude" or "openai").
            context: Extra context to prepend (e.g., current confluence data).
            max_tokens: Max response length.
            temperature: Creativity (0.0 = deterministic, 1.0 = creative).

        Returns:
            Tuple of (AI response, routing decision explaining the choice).
        """
        category = classify_task(prompt)
        decision = self._pick_provider(category, force_provider)
        self._routing_log.append(decision)

        log.info(
            "Routing to %s: category=%s reason=%s",
            decision.provider, category.value, decision.reason,
        )

        if decision.provider == "none":
            return (
                AIResponse(
                    provider="none", model="none", content="",
                    error=decision.reason,
                ),
                decision,
            )

        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        system = SYSTEM_PROMPT

        if decision.provider == "claude":
            response = await self.claude.complete(
                prompt=full_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            response = await self.openai.complete(
                prompt=full_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return response, decision

    async def dual_route(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 4096,
    ) -> dict:
        """
        Send the same prompt to BOTH providers and return both responses.

        Useful for comparing answers or getting a second opinion on a
        trade thesis. Only calls providers that are configured.
        """
        import asyncio

        results: dict = {}
        tasks = []

        if self.claude.is_configured:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            tasks.append(("claude", self.claude.complete(
                prompt=full_prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
            )))
        if self.openai.is_configured:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            tasks.append(("openai", self.openai.complete(
                prompt=full_prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
            )))

        if not tasks:
            return {"error": "No AI providers configured"}

        coros = [t[1] for t in tasks]
        responses = await asyncio.gather(*coros, return_exceptions=True)

        for (name, _), resp in zip(tasks, responses):
            if isinstance(resp, Exception):
                results[name] = AIResponse(
                    provider=name, model="error", content="",
                    error=str(resp),
                )
            else:
                results[name] = resp

        return results

    @property
    def status(self) -> dict:
        """Get router status including provider availability and usage."""
        return {
            "default_provider": self.default_provider,
            "claude": self.claude.usage_stats,
            "openai": self.openai.usage_stats,
            "routing_log_size": len(self._routing_log),
            "category_descriptions": {
                "claude_preferred": [
                    "analysis", "explanation", "risk_assessment",
                    "trade_thesis", "summary",
                ],
                "openai_preferred": [
                    "code_generation", "data_extraction",
                    "calculation", "quick_lookup",
                ],
            },
        }

    @property
    def recent_decisions(self) -> list[dict]:
        """Last 20 routing decisions for debugging."""
        return [
            {
                "category": d.category.value,
                "provider": d.provider,
                "reason": d.reason,
                "was_fallback": d.was_fallback,
            }
            for d in self._routing_log[-20:]
        ]
