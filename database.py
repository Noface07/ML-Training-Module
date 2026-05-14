"""
SQLAlchemy engine, session factory, and declarative base.

Uses synchronous SQLite — no async ORM.  Call ``init_db()`` once at
startup to create all tables.
"""

from __future__ import annotations

from sqlalchemy import create_engine
import sqlalchemy
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
    """Create tables that don't exist and add any missing columns to existing tables.

    SQLAlchemy's ``create_all`` will only create *new* tables — it won't
    alter existing ones.  This lightweight migration inspects each table
    via ``PRAGMA table_info`` and issues ``ALTER TABLE … ADD COLUMN`` for
    any columns defined in the ORM model but absent from the DB.
    """
    from models import dataset, model_artifact, tag_profile  # noqa: F401 — ensure models are registered

    # 1. Create any brand-new tables
    Base.metadata.create_all(bind=engine)

    # 2. Patch missing columns on existing tables (SQLite only)
    import logging
    _log = logging.getLogger(__name__)

    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {
                row[1]
                for row in conn.execute(
                    sqlalchemy.text(f"PRAGMA table_info('{table.name}')")
                )
            }
            for col in table.columns:
                if col.name not in existing:
                    col_type = col.type.compile(dialect=engine.dialect)
                    stmt = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}"
                    conn.execute(sqlalchemy.text(stmt))
                    _log.info("Migration: added column %s.%s (%s)", table.name, col.name, col_type)
        conn.commit()

def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
