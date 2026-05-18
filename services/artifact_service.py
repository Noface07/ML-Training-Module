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
from utils.imp_metrics import enrich_metrics_with_imp, inject_group_ranking

logger = logging.getLogger(__name__)


def _inject_imp(record: ModelArtifact) -> ModelArtifact:
    """Enrich a ModelArtifact record's metrics with the ``imp`` sub-object.

    Works by replacing the in-memory ``metrics`` attribute with an enriched
    copy — the database row is never written back, preserving backward
    compatibility for all existing stored metrics.
    """
    record.metrics = enrich_metrics_with_imp(record.model_type, record.metrics)
    return record


def get_model(db: Session, model_id: str) -> ModelArtifact:
    """Fetch a model artifact by ID or raise 404."""
    record = db.query(ModelArtifact).filter(ModelArtifact.id == model_id).first()
    if not record:
        raise_error(ErrorCode.MODEL_NOT_FOUND, f"No model found with id: {model_id}", status_code=404)
        
    # Inject standalone base imp metrics first
    record = _inject_imp(record)
    
    # If the model is completed, fetch peers and compute ranking
    if record.status == "completed":
        # Fetch only completed peers in the same use_case for ranking computation
        peers = db.query(ModelArtifact).filter(
            ModelArtifact.use_case == record.use_case,
            ModelArtifact.status == "completed",
            ModelArtifact.id != model_id
        ).all()
        # Enrich peers
        for peer in peers:
            _inject_imp(peer)
            
        all_models = [record] + peers
        inject_group_ranking(all_models)
        
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
    
    # Enrich all returned items with standalone imp
    for item in items:
        _inject_imp(item)
        
    # To compute accurate rankings for returned models, we need ALL completed models
    # in their respective use cases.
    if items:
        use_cases = {item.use_case for item in items if item.status == "completed"}
        if use_cases:
            # Fetch all completed models in these use cases
            all_peers = db.query(ModelArtifact).filter(
                ModelArtifact.use_case.in_(use_cases),
                ModelArtifact.status == "completed"
            ).all()
            
            # Create a lookup for peers
            peer_map = {peer.id: peer for peer in all_peers}
            
            # Enrich all peers
            for peer in all_peers:
                _inject_imp(peer)
                
            # Compute rankings globally for these use cases
            inject_group_ranking(all_peers)
            
            # Transfer the computed rankings back to our paginated items
            for item in items:
                if item.status == "completed" and item.id in peer_map:
                    # peer_map[item.id] has the updated ranking
                    item.metrics["imp"]["ranking"] = peer_map[item.id].metrics["imp"]["ranking"]
                    
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
