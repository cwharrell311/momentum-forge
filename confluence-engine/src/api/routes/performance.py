"""
Signal forward-testing performance API.

Provides endpoints for the signal scorecard — the answer to
"Are these signals actually making money?"

GET  /api/v1/performance/scorecard   → Overall hit rates & returns by horizon
GET  /api/v1/performance/by-layer    → Per-layer accuracy breakdown
GET  /api/v1/performance/signals     → Browse recorded signals with outcomes
POST /api/v1/performance/grade       → Manually trigger grading job
GET  /api/v1/performance/stats       → Quick summary stats
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import and_, case, desc, func, select

from src.models.tables import SignalHistory
from src.utils.db import get_session

router = APIRouter()


class ScorecardResponse(BaseModel):
    total_signals: int
    graded_signals: int
    ungraded_signals: int
    # Hit rates by horizon
    hit_rate_t1: float | None
    hit_rate_t5: float | None
    hit_rate_t10: float | None
    hit_rate_t20: float | None
    # Average returns by horizon
    avg_return_t1: float | None
    avg_return_t5: float | None
    avg_return_t10: float | None
    avg_return_t20: float | None
    # Breakdown by conviction bucket
    conviction_buckets: list[dict]
    # Trade-worthy vs not
    trade_worthy_hit_rate: float | None
    non_trade_worthy_hit_rate: float | None


class LayerAccuracy(BaseModel):
    layer: str
    signal_count: int
    avg_strength: float
    # How often signals from this layer appeared in winning signals
    presence_in_hits_t5: float | None
    presence_in_misses_t5: float | None


class SignalRecord(BaseModel):
    id: int
    ticker: str
    direction: str
    conviction_pct: int
    active_layers: int
    trade_worthy: bool
    regime: str | None
    entry_price: float | None
    signal_date: str
    return_t1: float | None
    return_t5: float | None
    return_t10: float | None
    return_t20: float | None
    hit_t5: bool | None
    graded: bool
    layer_details: dict | None


class QuickStats(BaseModel):
    total_signals: int
    graded_signals: int
    oldest_signal: str | None
    newest_signal: str | None
    best_signal: dict | None
    worst_signal: dict | None


@router.get("/scorecard", response_model=ScorecardResponse)
async def get_scorecard():
    """
    Overall signal scorecard — the big picture.

    Shows hit rates and average returns across all four time horizons
    (T+1, T+5, T+10, T+20), plus breakdowns by conviction level
    and trade-worthiness.
    """
    async with get_session() as session:
        # Total counts
        total = (await session.execute(
            select(func.count(SignalHistory.id))
        )).scalar() or 0

        graded = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                SignalHistory.hit_t1.isnot(None)
            )
        )).scalar() or 0

        # Hit rates and avg returns per horizon
        hit_rates = {}
        avg_returns = {}
        for suffix in ["t1", "t5", "t10", "t20"]:
            hit_col = getattr(SignalHistory, f"hit_{suffix}")
            return_col = getattr(SignalHistory, f"return_{suffix}")

            # Hit rate
            hits = (await session.execute(
                select(func.count(SignalHistory.id)).where(hit_col.is_(True))
            )).scalar() or 0
            total_graded_h = (await session.execute(
                select(func.count(SignalHistory.id)).where(hit_col.isnot(None))
            )).scalar() or 0
            hit_rates[suffix] = round(hits / total_graded_h, 3) if total_graded_h else None

            # Avg return
            avg_ret = (await session.execute(
                select(func.avg(return_col)).where(return_col.isnot(None))
            )).scalar()
            avg_returns[suffix] = round(float(avg_ret), 2) if avg_ret is not None else None

        # Conviction buckets: 40-50%, 50-60%, 60-70%, 70-80%, 80%+
        buckets = []
        bucket_ranges = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        for low, high in bucket_ranges:
            bucket_count = (await session.execute(
                select(func.count(SignalHistory.id)).where(
                    and_(
                        SignalHistory.conviction_pct >= low,
                        SignalHistory.conviction_pct < high,
                    )
                )
            )).scalar() or 0

            bucket_hits = (await session.execute(
                select(func.count(SignalHistory.id)).where(
                    and_(
                        SignalHistory.conviction_pct >= low,
                        SignalHistory.conviction_pct < high,
                        SignalHistory.hit_t5.is_(True),
                    )
                )
            )).scalar() or 0

            bucket_graded = (await session.execute(
                select(func.count(SignalHistory.id)).where(
                    and_(
                        SignalHistory.conviction_pct >= low,
                        SignalHistory.conviction_pct < high,
                        SignalHistory.hit_t5.isnot(None),
                    )
                )
            )).scalar() or 0

            avg_ret = (await session.execute(
                select(func.avg(SignalHistory.return_t5)).where(
                    and_(
                        SignalHistory.conviction_pct >= low,
                        SignalHistory.conviction_pct < high,
                        SignalHistory.return_t5.isnot(None),
                    )
                )
            )).scalar()

            buckets.append({
                "range": f"{low}-{high}%",
                "count": bucket_count,
                "graded": bucket_graded,
                "hit_rate_t5": round(bucket_hits / bucket_graded, 3) if bucket_graded else None,
                "avg_return_t5": round(float(avg_ret), 2) if avg_ret is not None else None,
            })

        # Trade-worthy vs not
        tw_hits = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                and_(SignalHistory.trade_worthy.is_(True), SignalHistory.hit_t5.is_(True))
            )
        )).scalar() or 0
        tw_total = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                and_(SignalHistory.trade_worthy.is_(True), SignalHistory.hit_t5.isnot(None))
            )
        )).scalar() or 0
        ntw_hits = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                and_(SignalHistory.trade_worthy.is_(False), SignalHistory.hit_t5.is_(True))
            )
        )).scalar() or 0
        ntw_total = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                and_(SignalHistory.trade_worthy.is_(False), SignalHistory.hit_t5.isnot(None))
            )
        )).scalar() or 0

        return ScorecardResponse(
            total_signals=total,
            graded_signals=graded,
            ungraded_signals=total - graded,
            hit_rate_t1=hit_rates["t1"],
            hit_rate_t5=hit_rates["t5"],
            hit_rate_t10=hit_rates["t10"],
            hit_rate_t20=hit_rates["t20"],
            avg_return_t1=avg_returns["t1"],
            avg_return_t5=avg_returns["t5"],
            avg_return_t10=avg_returns["t10"],
            avg_return_t20=avg_returns["t20"],
            conviction_buckets=buckets,
            trade_worthy_hit_rate=round(tw_hits / tw_total, 3) if tw_total else None,
            non_trade_worthy_hit_rate=round(ntw_hits / ntw_total, 3) if ntw_total else None,
        )


@router.get("/by-layer", response_model=list[LayerAccuracy])
async def get_layer_accuracy():
    """
    Per-layer accuracy breakdown.

    Shows which signal layers are most predictive by analyzing
    their presence in winning vs losing signals at T+5.
    """
    async with get_session() as session:
        # Get all graded signals with layer details
        result = await session.execute(
            select(SignalHistory).where(
                and_(
                    SignalHistory.hit_t5.isnot(None),
                    SignalHistory.layer_details.isnot(None),
                )
            )
        )
        graded_signals = result.scalars().all()

        if not graded_signals:
            return []

        # Count layer presence in hits vs misses
        layer_stats: dict[str, dict] = {}
        for sig in graded_signals:
            if not sig.layer_details:
                continue
            for layer_name, details in sig.layer_details.items():
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        "hits": 0, "misses": 0, "total": 0,
                        "strengths": [],
                    }
                stats = layer_stats[layer_name]
                stats["total"] += 1
                stats["strengths"].append(details.get("strength", 0))
                if sig.hit_t5:
                    stats["hits"] += 1
                else:
                    stats["misses"] += 1

        layers = []
        for name, stats in sorted(layer_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            total = stats["total"]
            avg_str = sum(stats["strengths"]) / len(stats["strengths"]) if stats["strengths"] else 0
            layers.append(LayerAccuracy(
                layer=name,
                signal_count=total,
                avg_strength=round(avg_str, 3),
                presence_in_hits_t5=round(stats["hits"] / total, 3) if total else None,
                presence_in_misses_t5=round(stats["misses"] / total, 3) if total else None,
            ))

        return layers


@router.get("/signals", response_model=list[SignalRecord])
async def list_signals(
    ticker: str | None = Query(None, description="Filter by ticker"),
    graded_only: bool = Query(False, description="Only show graded signals"),
    trade_worthy_only: bool = Query(False, description="Only show trade-worthy signals"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Browse recorded signals with their outcomes.

    Shows every signal that was recorded for forward testing,
    along with the graded returns at each time horizon.
    """
    async with get_session() as session:
        query = select(SignalHistory).order_by(desc(SignalHistory.signal_date)).limit(limit)

        if ticker:
            query = query.where(SignalHistory.ticker == ticker.upper())
        if graded_only:
            query = query.where(SignalHistory.hit_t1.isnot(None))
        if trade_worthy_only:
            query = query.where(SignalHistory.trade_worthy.is_(True))

        result = await session.execute(query)
        signals = result.scalars().all()

        return [
            SignalRecord(
                id=s.id,
                ticker=s.ticker,
                direction=s.direction,
                conviction_pct=s.conviction_pct,
                active_layers=s.active_layers,
                trade_worthy=s.trade_worthy,
                regime=s.regime,
                entry_price=s.entry_price,
                signal_date=s.signal_date.isoformat() if s.signal_date else "",
                return_t1=s.return_t1,
                return_t5=s.return_t5,
                return_t10=s.return_t10,
                return_t20=s.return_t20,
                hit_t5=s.hit_t5,
                graded=s.graded_at is not None,
                layer_details=s.layer_details,
            )
            for s in signals
        ]


@router.post("/grade")
async def trigger_grading():
    """
    Manually trigger the signal grading job.

    Useful when you don't want to wait for the scheduled 2-hour interval.
    Grades all ungraded signals that have aged enough.
    """
    from src.services.signal_tracker import grade_signals

    graded = await grade_signals()
    return {"status": "ok", "graded": graded}


@router.get("/stats", response_model=QuickStats)
async def quick_stats():
    """
    Quick summary stats for the signal tracker.

    Shows total signals, how many are graded, date range,
    and the best/worst performing signals.
    """
    async with get_session() as session:
        total = (await session.execute(
            select(func.count(SignalHistory.id))
        )).scalar() or 0

        graded = (await session.execute(
            select(func.count(SignalHistory.id)).where(
                SignalHistory.hit_t5.isnot(None)
            )
        )).scalar() or 0

        oldest = (await session.execute(
            select(func.min(SignalHistory.signal_date))
        )).scalar()

        newest = (await session.execute(
            select(func.max(SignalHistory.signal_date))
        )).scalar()

        # Best signal by T+5 return
        best_result = await session.execute(
            select(SignalHistory).where(
                SignalHistory.return_t5.isnot(None)
            ).order_by(desc(SignalHistory.return_t5)).limit(1)
        )
        best = best_result.scalars().first()
        best_dict = None
        if best:
            best_dict = {
                "ticker": best.ticker,
                "direction": best.direction,
                "conviction_pct": best.conviction_pct,
                "return_t5": best.return_t5,
                "signal_date": best.signal_date.isoformat(),
            }

        # Worst signal
        worst_result = await session.execute(
            select(SignalHistory).where(
                SignalHistory.return_t5.isnot(None)
            ).order_by(SignalHistory.return_t5).limit(1)
        )
        worst = worst_result.scalars().first()
        worst_dict = None
        if worst:
            worst_dict = {
                "ticker": worst.ticker,
                "direction": worst.direction,
                "conviction_pct": worst.conviction_pct,
                "return_t5": worst.return_t5,
                "signal_date": worst.signal_date.isoformat(),
            }

        return QuickStats(
            total_signals=total,
            graded_signals=graded,
            oldest_signal=oldest.isoformat() if oldest else None,
            newest_signal=newest.isoformat() if newest else None,
            best_signal=best_dict,
            worst_signal=worst_dict,
        )
