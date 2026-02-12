"""
Signal API routes.

View individual signal layer results. Useful for debugging
and understanding what each layer is seeing independently.

GET /api/v1/signals/{ticker}       → All signals for a ticker
GET /api/v1/signals/layer/{layer}  → All signals from one layer
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()


class SignalDetailResponse(BaseModel):
    ticker: str
    layer: str
    direction: str
    strength: float
    confidence: float
    explanation: str
    metadata: dict = {}


@router.get("/{ticker}", response_model=list[SignalDetailResponse])
async def get_signals_for_ticker(ticker: str):
    """
    Get all active signal layer results for a single ticker.

    Runs each implemented signal processor against this ticker
    and returns individual results. Unlike /confluence, this
    does NOT combine them — you see each layer independently.
    """
    from src.api.dependencies import get_processors

    ticker = ticker.upper()
    results: list[SignalDetailResponse] = []

    for processor in get_processors():
        try:
            signal = await processor.scan_single(ticker)
            if signal:
                results.append(
                    SignalDetailResponse(
                        ticker=signal.ticker,
                        layer=signal.layer,
                        direction=signal.direction.value,
                        strength=round(signal.strength, 3),
                        confidence=round(signal.confidence, 3),
                        explanation=signal.explanation,
                        metadata=signal.metadata,
                    )
                )
        except NotImplementedError:
            # Skip stub processors that aren't built yet
            continue
        except Exception as e:
            log.warning("Signal scan failed (%s) for %s: %s", processor.name, ticker, e)
            continue

    return results
