"""
Database connection management.

Creates an async SQLAlchemy engine and session factory.
The engine maintains a connection pool — it keeps a few
database connections open and ready so each API request
doesn't have to wait for a new connection.

Usage:
    from src.utils.db import get_session

    async with get_session() as session:
        result = await session.execute(select(Watchlist))
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings

# The engine is created once and reused for the lifetime of the app.
# pool_size=5 means up to 5 simultaneous DB connections.
# echo=False means SQL queries aren't printed (set True for debugging).
_engine = create_async_engine(
    get_settings().database_url,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

# A session factory — each call to _session_factory() creates a new
# session (like a "conversation" with the database).
_session_factory = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with automatic cleanup.

    Usage:
        async with get_session() as session:
            # do database work
            await session.commit()
        # session is automatically closed here
    """
    session = _session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_tables() -> None:
    """
    Create all database tables if they don't exist, then add any
    missing columns to existing tables.

    SQLAlchemy's create_all() only creates NEW tables — it won't add
    columns to tables that already exist. The _COLUMN_MIGRATIONS list
    handles that by running ALTER TABLE for any missing columns.

    Safe to call multiple times.
    """
    from sqlalchemy import text

    from src.models.base import Base

    # Import tables so they're registered with Base.metadata
    import src.models.tables  # noqa: F401

    async with _engine.begin() as conn:
        # Step 1: Create any brand-new tables
        await conn.run_sync(Base.metadata.create_all)

        # Step 2: Add missing columns to existing tables
        # (table, column, sql_type) — validated against allowlist
        ALLOWED_TYPES = {"VARCHAR(64)", "VARCHAR(10)", "INTEGER", "FLOAT", "TEXT", "JSONB"}
        column_migrations = [
            ("trades", "alpaca_order_id", "VARCHAR(64)"),
        ]
        for table, column, sql_type in column_migrations:
            if sql_type not in ALLOWED_TYPES:
                raise ValueError(f"Invalid SQL type in migration: {sql_type}")
            result = await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = :column"
                ),
                {"table": table, "column": column},
            )
            if not result.fetchone():
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
                )

        # Step 3: Add missing indexes to existing tables
        index_migrations = [
            ("idx_signals_layer_ticker", "signals", "layer, ticker"),
            ("idx_confluence_ticker", "confluence_scores", "ticker, created_at"),
            ("idx_trade_ticker_entry", "trades", "ticker, entry_at"),
            ("idx_trade_confluence_id", "trades", "confluence_score_id"),
        ]
        for idx_name, table, columns in index_migrations:
            result = await conn.execute(
                text(
                    "SELECT 1 FROM pg_indexes "
                    "WHERE indexname = :idx_name"
                ),
                {"idx_name": idx_name},
            )
            if not result.fetchone():
                await conn.execute(
                    text(f"CREATE INDEX {idx_name} ON {table} ({columns})")
                )


async def close_engine() -> None:
    """Shut down the connection pool. Called on app shutdown."""
    await _engine.dispose()
