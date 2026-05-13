"""
SQLAlchemy engine, session factory, and declarative base.

Uses synchronous SQLite — no async ORM.  Call ``init_db()`` once at
startup to create all tables.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def init_db() -> None:
    """Create all tables that don't already exist."""
    from models import dataset, model_artifact  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
