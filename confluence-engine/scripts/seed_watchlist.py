"""
Seed the watchlist table from config/watchlist.yaml.

Run this after starting the database for the first time:
    python -m scripts.seed_watchlist

It reads the tickers from config/watchlist.yaml and inserts them
into the PostgreSQL watchlist table. Safe to run multiple times —
it skips tickers that already exist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from sqlalchemy import select

from src.models.tables import Watchlist
from src.utils.db import get_session


def load_watchlist() -> list[str]:
    config_path = Path(__file__).parent.parent / "config" / "watchlist.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


async def seed() -> None:
    # Run migrations (creates tables + adds any missing columns)
    from scripts.migrate import run_migrations

    await run_migrations()
    print()  # blank line after migration output

    tickers = load_watchlist()
    print(f"Loaded {len(tickers)} tickers from config")

    async with get_session() as session:
        for ticker in tickers:
            existing = await session.get(Watchlist, ticker)
            if existing:
                print(f"  {ticker} — already exists, skipping")
                continue

            session.add(Watchlist(ticker=ticker))
            print(f"  {ticker} — added")

        await session.commit()

    print(f"\nWatchlist seeded successfully.")


def main():
    asyncio.run(seed())


if __name__ == "__main__":
    main()
