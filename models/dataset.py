"""
ORM model for the ``dataset_records`` table.

Each row represents a single ingested parquet file after validation,
cleaning, and storage.
"""

from __future__ import annotations

import datetime
from sqlalchemy import Column, DateTime, Integer, JSON, String
from database import Base


class DatasetRecord(Base):
    """Persisted metadata about an ingested dataset."""

    __tablename__ = "dataset_records"

    id: str = Column(String, primary_key=True)
    original_filename: str = Column(String, nullable=False)
    stored_path: str = Column(String, nullable=False)
    row_count: int = Column(Integer, nullable=False)
    col_count: int = Column(Integer, nullable=False)
    tags_detected = Column(JSON, nullable=True)
    quality_report = Column(JSON, nullable=True)
    range_metadata = Column(JSON, nullable=True)
    created_at = Column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
