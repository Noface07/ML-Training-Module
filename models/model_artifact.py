"""
ORM model for the ``model_artifacts`` table.

Each row represents a trained (or in-training / failed) ML model artifact
together with its full lineage metadata.
"""

from __future__ import annotations

import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from database import Base


class ModelArtifact(Base):
    """Persisted metadata + lineage for a trained model."""

    __tablename__ = "model_artifacts"

    id: str = Column(String, primary_key=True)
    use_case: str = Column(String, nullable=False)
    model_type: str = Column(String, nullable=False)
    version: int = Column(Integer, nullable=False)
    model_name: str | None = Column(String, nullable=True)
    artifact_path: str = Column(String, nullable=True)
    metadata_path: str = Column(String, nullable=True)
    dataset_id: str = Column(
        String, ForeignKey("dataset_records.id"), nullable=False
    )
    profile_id: int | None = Column(
        Integer, ForeignKey("tag_profiles.id", ondelete="SET NULL"), nullable=True
    )
    feature_schema_id: str = Column(String, nullable=True)
    feature_schema_snapshot = Column(JSON, nullable=True)
    hparams_used = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    range_metadata = Column(JSON, nullable=True)
    tags_used = Column(JSON, nullable=True)
    mandatory_features_used = Column(JSON, nullable=True)
    optional_features_used = Column(JSON, nullable=True)
    cross_tag_features_used = Column(JSON, nullable=True)
    training_duration_seconds: float = Column(Float, nullable=True)
    status: str = Column(String, default="training", nullable=False)
    error_message: str = Column(String, nullable=True)
    created_at = Column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    completed_at = Column(DateTime, nullable=True)
