"""
Database migrations — adds missing columns to existing tables.

SQLAlchemy's create_all() creates new tables but does NOT add columns
to tables that already exist. This script bridges that gap without
needing a full Alembic setup.

Run after pulling new code that adds columns:
    python -m scripts.migrate

Safe to run multiple times — checks if columns exist before adding.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from src.utils.db import _engine, create_tables


# Each migration is (table, column, SQL type, description)
MIGRATIONS: list[tuple[str, str, str, str]] = [
    (
        "trades",
        "alpaca_order_id",
        "VARCHAR(64)",
        "Links trade journal entries to Alpaca orders",
    ),
]


async def column_exists(conn, table: str, column: str) -> bool:
    """Check if a column already exists in a table."""
    result = await conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    )
    return result.fetchone() is not None


async def run_migrations() -> None:
    # First, create any brand-new tables
    print("Ensuring all tables exist...")
    await create_tables()
    print("Tables ready.\n")

    # Then add any missing columns to existing tables
    print("Checking for missing columns...")
    applied = 0

    async with _engine.begin() as conn:
        for table, column, sql_type, description in MIGRATIONS:
            if await column_exists(conn, table, column):
                print(f"  {table}.{column} — already exists, skipping")
                continue

            sql = f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
            await conn.execute(text(sql))
            print(f"  {table}.{column} — ADDED ({description})")
            applied += 1

    if applied:
        print(f"\nApplied {applied} migration(s) successfully.")
    else:
        print("\nDatabase is up to date — no migrations needed.")


def main():
    asyncio.run(run_migrations())


if __name__ == "__main__":
    main()
