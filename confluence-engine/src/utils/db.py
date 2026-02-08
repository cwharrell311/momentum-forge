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
    Create all database tables if they don't exist.

    Uses the SQLAlchemy models defined in src/models/tables.py.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS.
    """
    from src.models.base import Base

    # Import tables so they're registered with Base.metadata
    import src.models.tables  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_engine() -> None:
    """Shut down the connection pool. Called on app shutdown."""
    await _engine.dispose()
