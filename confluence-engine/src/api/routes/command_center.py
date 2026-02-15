"""
AI Command Center API routes.

Provides endpoints for the intelligent AI routing system that sends
tasks to Claude or OpenAI based on which provider handles the task
type best.

POST /api/v1/ai/ask        → Auto-routed question (picks best provider)
POST /api/v1/ai/dual       → Send to BOTH providers, compare answers
POST /api/v1/ai/analyze    → Analyze a ticker with live confluence context
GET  /api/v1/ai/status     → Provider availability and usage stats
GET  /api/v1/ai/categories → Show routing rules (which tasks go where)
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request / Response Models ──


class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    provider: str | None = Field(
        None,
        description='Force a provider: "claude" or "openai". Omit for auto-routing.',
    )
    max_tokens: int = Field(4096, ge=100, le=16000)
    temperature: float = Field(0.3, ge=0.0, le=1.0)


class AskResponse(BaseModel):
    content: str
    provider: str
    model: str
    category: str
    routing_reason: str
    was_fallback: bool
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None = None


class DualRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(4096, ge=100, le=16000)


class DualResponse(BaseModel):
    claude: AskResponse | None = None
    openai: AskResponse | None = None


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=6)
    question: str = Field(
        "Analyze this setup and provide a trade thesis with entry, target, and stop.",
        max_length=4000,
    )
    provider: str | None = None
    max_tokens: int = Field(4096, ge=100, le=16000)


class ProviderStatus(BaseModel):
    provider: str
    model: str
    configured: bool
    calls: int
    errors: int
    total_input_tokens: int
    total_output_tokens: int


class RouterStatus(BaseModel):
    default_provider: str
    claude: ProviderStatus
    openai: ProviderStatus
    recent_decisions: list[dict]


class CategoryInfo(BaseModel):
    category: str
    preferred_provider: str
    description: str
    example_prompts: list[str]


# ── Endpoints ──


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question — the router picks the best AI provider automatically.

    The task is classified by keyword analysis (analysis, code generation,
    risk assessment, etc.) and routed to whichever provider handles that
    category best. Override with the `provider` field to force one.
    """
    from src.api.dependencies import get_ai_router

    ai_router = get_ai_router()
    response, decision = await ai_router.route(
        prompt=req.prompt,
        force_provider=req.provider,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    return AskResponse(
        content=response.content,
        provider=response.provider,
        model=response.model,
        category=decision.category.value,
        routing_reason=decision.reason,
        was_fallback=decision.was_fallback,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        error=response.error,
    )


@router.post("/dual", response_model=DualResponse)
async def dual(req: DualRequest):
    """
    Send the same prompt to BOTH Claude and OpenAI in parallel.

    Returns both responses so you can compare perspectives — useful for
    getting a second opinion on a trade thesis or risk analysis.
    Only calls providers that are configured (skips unconfigured ones).
    """
    from src.api.dependencies import get_ai_router

    ai_router = get_ai_router()
    results = await ai_router.dual_route(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
    )

    if "error" in results:
        return DualResponse()

    def _to_ask_response(resp, name: str) -> AskResponse:
        return AskResponse(
            content=resp.content,
            provider=resp.provider,
            model=resp.model,
            category="general",
            routing_reason=f"Dual mode — sent to {name}",
            was_fallback=False,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            error=resp.error,
        )

    return DualResponse(
        claude=_to_ask_response(results["claude"], "claude") if "claude" in results else None,
        openai=_to_ask_response(results["openai"], "openai") if "openai" in results else None,
    )


@router.post("/analyze", response_model=AskResponse)
async def analyze_ticker(req: AnalyzeRequest):
    """
    Analyze a ticker with live confluence data injected as context.

    Pulls the latest confluence score and all signal layer results for
    the ticker, formats them as context, then sends the question to the
    AI with that data. This way the AI sees actual signal strengths,
    directions, and metadata — not just the ticker symbol.
    """
    from src.api.dependencies import get_ai_router, get_cache, get_engine

    ticker = req.ticker.upper()

    # Build context from live data
    context_parts = [f"=== Live Confluence Data for {ticker} ==="]

    # Check cache first for recent scan results
    cache = get_cache()
    cached = cache.get_scores() if cache.has_data else []
    ticker_score = None
    for score in cached:
        if score.ticker == ticker:
            ticker_score = score
            break

    if ticker_score:
        context_parts.append(f"\nConfluence Score: {ticker_score.conviction_pct}% {ticker_score.direction.value}")
        context_parts.append(f"Active Layers: {ticker_score.active_layers}/{ticker_score.total_layers}")
        context_parts.append(f"Trade Worthy: {ticker_score.trade_worthy} ({ticker_score.gate_details})")
        context_parts.append(f"Regime: {ticker_score.regime.value}")

        for sig in ticker_score.signals:
            context_parts.append(
                f"\n  [{sig.layer}] {sig.direction.value} "
                f"strength={sig.strength:.2f} confidence={sig.confidence:.2f}"
            )
            if sig.explanation:
                context_parts.append(f"    {sig.explanation}")
            if sig.metadata:
                # Include key metadata fields
                meta_str = ", ".join(f"{k}={v}" for k, v in sig.metadata.items())
                context_parts.append(f"    metadata: {meta_str}")
    else:
        # No cached data — run a fresh single-ticker scan
        try:
            engine = get_engine()
            scores = await engine.scan([ticker])
            if scores:
                score = scores[0]
                context_parts.append(f"\nConfluence Score: {score.conviction_pct}% {score.direction.value}")
                context_parts.append(f"Active Layers: {score.active_layers}/{score.total_layers}")
                context_parts.append(f"Trade Worthy: {score.trade_worthy}")

                for sig in score.signals:
                    context_parts.append(
                        f"\n  [{sig.layer}] {sig.direction.value} "
                        f"strength={sig.strength:.2f} confidence={sig.confidence:.2f}"
                    )
                    if sig.explanation:
                        context_parts.append(f"    {sig.explanation}")
        except Exception as e:
            context_parts.append(f"\n(Could not fetch live data: {e})")

    context = "\n".join(context_parts)

    ai_router = get_ai_router()
    response, decision = await ai_router.route(
        prompt=req.question,
        force_provider=req.provider,
        context=context,
        max_tokens=req.max_tokens,
    )

    return AskResponse(
        content=response.content,
        provider=response.provider,
        model=response.model,
        category=decision.category.value,
        routing_reason=decision.reason,
        was_fallback=decision.was_fallback,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        error=response.error,
    )


@router.get("/status", response_model=RouterStatus)
async def ai_status():
    """
    Get AI provider status — which providers are configured, usage stats,
    and recent routing decisions.
    """
    from src.api.dependencies import get_ai_router

    ai_router = get_ai_router()
    status = ai_router.status

    return RouterStatus(
        default_provider=status["default_provider"],
        claude=ProviderStatus(**status["claude"]),
        openai=ProviderStatus(**status["openai"]),
        recent_decisions=ai_router.recent_decisions,
    )


@router.get("/categories", response_model=list[CategoryInfo])
async def ai_categories():
    """
    Show all task categories and their routing rules.

    Explains which types of tasks go to Claude vs. OpenAI, with example
    prompts for each category. Useful for understanding how to phrase
    questions to get routed to a specific provider.
    """
    return [
        CategoryInfo(
            category="analysis",
            preferred_provider="claude",
            description="Deep analysis of trade setups, signal interpretation, pattern recognition",
            example_prompts=[
                "Analyze the NVDA confluence setup",
                "What do you see in this options flow data?",
                "Evaluate the TSLA short squeeze potential",
            ],
        ),
        CategoryInfo(
            category="explanation",
            preferred_provider="claude",
            description="Explaining why signals are firing, what data means, educational",
            example_prompts=[
                "Explain why GEX is bearish for AAPL",
                "What does negative gamma exposure mean here?",
                "Why is the flow gate blocking this trade?",
            ],
        ),
        CategoryInfo(
            category="risk_assessment",
            preferred_provider="claude",
            description="Risk analysis, position sizing, stop loss placement, hedging",
            example_prompts=[
                "What's the risk/reward for going long NVDA here?",
                "How should I size this position given the VIX regime?",
                "Where should I place my stop loss?",
            ],
        ),
        CategoryInfo(
            category="trade_thesis",
            preferred_provider="claude",
            description="Building complete trade plans with entry, target, stop, and rationale",
            example_prompts=[
                "Build a bull case for AAPL based on current signals",
                "What's the trade plan for this TSLA setup?",
                "Give me a bear thesis with entry and exit levels",
            ],
        ),
        CategoryInfo(
            category="summary",
            preferred_provider="claude",
            description="Summarizing scan results, daily digests, signal recaps",
            example_prompts=[
                "Summarize today's top confluence setups",
                "Give me a recap of all bullish signals",
                "TLDR on the current market regime",
            ],
        ),
        CategoryInfo(
            category="code_generation",
            preferred_provider="openai",
            description="Writing new signal processors, endpoints, scripts, or tools",
            example_prompts=[
                "Write a signal processor for earnings momentum",
                "Create a Python script to backtest this strategy",
                "Add an endpoint that exports trades to CSV",
            ],
        ),
        CategoryInfo(
            category="data_extraction",
            preferred_provider="openai",
            description="Parsing and structuring data, format conversions, JSON extraction",
            example_prompts=[
                "Extract the key support/resistance levels from this data",
                "Convert this signal data to a structured JSON format",
                "Parse these option chain strikes into a table",
            ],
        ),
        CategoryInfo(
            category="calculation",
            preferred_provider="openai",
            description="Quantitative calculations, expected value, probability, Greeks",
            example_prompts=[
                "Calculate the expected value of this trade",
                "What's the probability of profit for this setup?",
                "Compute the Kelly criterion bet size",
            ],
        ),
        CategoryInfo(
            category="quick_lookup",
            preferred_provider="openai",
            description="Quick factual questions, current values, simple lookups",
            example_prompts=[
                "What's the current IV rank for TSLA?",
                "Check the price level quickly",
            ],
        ),
        CategoryInfo(
            category="general",
            preferred_provider="claude",
            description="General questions that don't fit a specific category",
            example_prompts=[
                "What should I be watching today?",
                "Any thoughts on the current market?",
            ],
        ),
    ]
