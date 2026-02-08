"""
SQLAlchemy declarative base.

All ORM models inherit from this Base class. It tells SQLAlchemy
how to map Python classes to database tables.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass
