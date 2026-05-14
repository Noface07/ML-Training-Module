"""
ORM model for the ``tag_profiles`` table.

Maps unique deterministic hashes of sorted integer tag collections
to human-readable profile names.
"""

from __future__ import annotations

import datetime
from sqlalchemy import Column, DateTime, Integer, String
from database import Base


class TagProfile(Base):
    """Persisted mapping of tag collection hashes to named profiles."""

    __tablename__ = "tag_profiles"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    profile_name: str = Column(String, nullable=False)
    tag_hash: str = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
