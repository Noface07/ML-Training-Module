"""
Artifact service — model save/load and registry CRUD.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from config import settings
from models.model_artifact import ModelArtifact
from utils.errors import ErrorCode, raise_error

logger = logging.getLogger(__name__)


def get_model(db: Session, model_id: str) -> ModelArtifact:
    """Fetch a model artifact by ID or raise 404."""
    record = db.query(ModelArtifact).filter(ModelArtifact.id == model_id).first()
    if not record:
        raise_error(ErrorCode.MODEL_NOT_FOUND, f"No model found with id: {model_id}", status_code=404)
    return record


def list_models(
    db: Session,
    use_case: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[ModelArtifact]]:
    """List model artifacts with optional filters."""
    q = db.query(ModelArtifact)
    if use_case:
        q = q.filter(ModelArtifact.use_case == use_case)
    if status:
        q = q.filter(ModelArtifact.status == status)
    total = q.count()
    items = q.order_by(ModelArtifact.created_at.desc()).offset(offset).limit(limit).all()
    return total, items


def delete_model(db: Session, model_id: str) -> None:
    """Delete a model artifact record and its files from disk."""
    record = get_model(db, model_id)
    if record.status == "training":
        raise_error(ErrorCode.MODEL_STILL_TRAINING, "Cannot delete a model that is still training.", status_code=409)

    # Remove artifact files
    for path_str in (record.artifact_path, record.metadata_path):
        if path_str:
            p = Path(path_str)
            if p.exists():
                p.unlink()

    db.delete(record)
    db.commit()


def next_version(db: Session, use_case: str) -> int:
    """Get the next auto-incremented version number for a use-case."""
    from sqlalchemy import func
    max_ver = db.query(func.max(ModelArtifact.version)).filter(
        ModelArtifact.use_case == use_case
    ).scalar()
    return (max_ver or 0) + 1
