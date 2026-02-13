"""
Database table definitions.

Each class here maps to one table in PostgreSQL. The columns,
types, indexes, and relationships are all defined in Python.
SQLAlchemy translates these into CREATE TABLE statements.

Table overview:
- Watchlist: tickers you're actively tracking
- Signal: individual signal events from each layer
- ConfluenceScoreRecord: combined scores (what you trade from)
- Trade: your trade journal with P&L tracking
- Alert: notifications for high-conviction setups
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Watchlist(Base):
    """
    Tickers being actively scanned by the confluence engine.

    This is your universe of stocks. The engine only scans tickers
    that are in this table with active=True. Start with ~35 liquid,
    optionable names and expand as needed.
    """

    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    avg_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    market_cap: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Signal(Base):
    """
    Individual signal events from each layer.

    Every time a signal processor fires on a ticker, a row is created
    here. This gives you a historical record of every signal the
    engine has ever generated — useful for backtesting later.

    The 'metadata' JSONB column holds layer-specific details
    (e.g., RSI value for momentum, sweep size for options flow).
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    layer: Mapped[str] = mapped_column(String(30), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    strength: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    __table_args__ = (
        Index("idx_signals_ticker_time", "ticker", "created_at"),
        Index("idx_signals_layer", "layer", "created_at"),
        Index("idx_signals_layer_ticker", "layer", "ticker"),
    )


class ConfluenceScoreRecord(Base):
    """
    Combined confluence scores computed by the engine.

    This is the main output of the system — a ranked list of tickers
    sorted by conviction. Each row captures the score, which layers
    were active, and the market regime at the time.

    The signal_ids array links back to which individual signals
    contributed to this score.
    """

    __tablename__ = "confluence_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    conviction: Mapped[float] = mapped_column(Float, nullable=False)
    active_layers: Mapped[int] = mapped_column(Integer, nullable=False)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    signal_ids: Mapped[list[int] | None] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    __table_args__ = (
        Index("idx_confluence_conviction", "conviction", "created_at"),
        Index("idx_confluence_ticker", "ticker", "created_at"),
    )


class Trade(Base):
    """
    Trade journal — every trade you take, with confluence context.

    Links each trade back to the confluence score that triggered it.
    Tracks entry/exit prices, P&L, and your notes. This is how you
    measure whether the engine is actually making you money.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # 'long' or 'short'
    instrument: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'equity', 'call', 'put', 'spread'
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confluence_score_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    alpaca_order_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # Links to Alpaca order for reconciliation
    entry_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    __table_args__ = (
        Index("idx_trade_ticker_entry", "ticker", "entry_at"),
        Index("idx_trade_confluence_id", "confluence_score_id"),
    )


class Alert(Base):
    """
    Alerts generated by the engine.

    When a ticker crosses a conviction threshold (default 70%),
    an alert row is created. The dashboard shows unacknowledged
    alerts so you don't miss high-conviction setups.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(10), nullable=True)
    alert_type: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # 'confluence_high', 'flow_spike', etc.
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    conviction: Mapped[float | None] = mapped_column(Float, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )


class SignalHistory(Base):
    """
    Forward testing: records every qualifying signal with its price
    at the time it fired, then grades it later against actual outcomes.

    Each row = one signal snapshot. The grading job fills in outcome
    prices at T+1, T+5, T+10, T+20 trading days, then marks whether
    the predicted direction was correct.

    This is how you answer: "Are these signals actually predictive?"
    """

    __tablename__ = "signal_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # BULLISH / BEARISH
    conviction_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    active_layers: Mapped[int] = mapped_column(Integer, nullable=False)
    trade_worthy: Mapped[bool] = mapped_column(Boolean, nullable=False)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Layer-level detail stored as JSON
    layer_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Outcome prices — filled in by the grading job
    price_t1: Mapped[float | None] = mapped_column(Float, nullable=True)   # T+1 trading day
    price_t5: Mapped[float | None] = mapped_column(Float, nullable=True)   # T+5 trading days
    price_t10: Mapped[float | None] = mapped_column(Float, nullable=True)  # T+10 trading days
    price_t20: Mapped[float | None] = mapped_column(Float, nullable=True)  # T+20 trading days

    # Graded results — filled in by the grading job
    return_t1: Mapped[float | None] = mapped_column(Float, nullable=True)   # % return at T+1
    return_t5: Mapped[float | None] = mapped_column(Float, nullable=True)   # % return at T+5
    return_t10: Mapped[float | None] = mapped_column(Float, nullable=True)  # % return at T+10
    return_t20: Mapped[float | None] = mapped_column(Float, nullable=True)  # % return at T+20
    hit_t1: Mapped[bool | None] = mapped_column(Boolean, nullable=True)     # Direction correct at T+1?
    hit_t5: Mapped[bool | None] = mapped_column(Boolean, nullable=True)     # Direction correct at T+5?
    hit_t10: Mapped[bool | None] = mapped_column(Boolean, nullable=True)    # Direction correct at T+10?
    hit_t20: Mapped[bool | None] = mapped_column(Boolean, nullable=True)    # Direction correct at T+20?

    graded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_sighist_ticker_date", "ticker", "signal_date"),
        Index("idx_sighist_date", "signal_date"),
        Index("idx_sighist_graded", "graded_at"),
        Index("idx_sighist_conviction", "conviction_pct", "signal_date"),
    )
