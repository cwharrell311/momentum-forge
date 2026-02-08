"""
Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-check all config values
at startup. If a required value is missing, the app fails fast with
a clear error instead of crashing later with a cryptic KeyError.

Usage:
    from src.config import get_settings
    settings = get_settings()
    print(settings.fmp_api_key)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Data Providers ──
    fmp_api_key: str = ""
    uw_api_key: str = ""

    # ── Database ──
    database_url: str = "postgresql+asyncpg://confluence:localdev123@localhost:5432/confluence"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── Broker ──
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # ── Application ──
    live_trading_enabled: bool = False
    log_level: str = "INFO"
    scan_interval: int = 300  # seconds between full scans

    @property
    def sync_database_url(self) -> str:
        """Sync DB URL for Alembic migrations (swaps asyncpg for psycopg2)."""
        return self.database_url.replace("+asyncpg", "")


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache so the .env file is only read once.
    The same Settings object is reused everywhere in the app.
    """
    return Settings()
