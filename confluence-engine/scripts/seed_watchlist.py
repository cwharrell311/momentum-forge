"""
Seed the watchlist table from config/watchlist.yaml.
Run: python scripts/seed_watchlist.py
"""

from __future__ import annotations

import yaml
from pathlib import Path


def load_watchlist() -> list[str]:
    config_path = Path(__file__).parent.parent / "config" / "watchlist.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


def main():
    tickers = load_watchlist()
    print(f"Loaded {len(tickers)} tickers from watchlist config:")
    for t in tickers:
        print(f"  {t}")

    # TODO: Insert into PostgreSQL watchlist table
    # For now, just validates the config loads correctly
    print(f"\n✅ Watchlist ready. Connect to DB to seed.")


if __name__ == "__main__":
    main()
