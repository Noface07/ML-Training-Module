"""
Profile service — native SQLAlchemy CRUD operations for Tag Profiles.
"""

from __future__ import annotations

import logging
from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from models.tag_profile import TagProfile
from models.model_artifact import ModelArtifact
from utils.hashing import compute_tag_hash

logger = logging.getLogger(__name__)


def resolve_or_create_profile(
    db: Session, tag_ids: list[int], profile_name: str | None = None
) -> tuple[TagProfile, bool]:
    """Resolve an existing TagProfile by tag_hash, or create one.

    Returns:
        tuple of (TagProfile instance, created boolean)
    """
    tag_hash = compute_tag_hash(tag_ids)

    # 1. Lookup existing
    existing = db.query(TagProfile).filter(TagProfile.tag_hash == tag_hash).first()
    if existing:
        return existing, False

    # 2. Derive auto-name if omitted
    name = profile_name
    if not name or not name.strip():
        name = f"profile_{tag_hash[:8]}"
    else:
        name = name.strip()

    # 3. Insert new profile with robust race-condition rollback handling
    profile = TagProfile(profile_name=name, tag_hash=tag_hash)
    try:
        db.add(profile)
        db.commit()
        db.refresh(profile)
        logger.info("Created new TagProfile %d (%s)", profile.id, profile.profile_name)
        return profile, True
    except IntegrityError:
        # Concurrent race condition occurred — row was inserted by another session
        db.rollback()
        existing = db.query(TagProfile).filter(TagProfile.tag_hash == tag_hash).first()
        if existing:
            return existing, False
        raise


def update_profile_name(
    db: Session, profile_id: int, new_name: str
) -> TagProfile | None:
    """Update the profile_name for a given profile_id."""
    profile = db.query(TagProfile).filter(TagProfile.id == profile_id).first()
    if not profile:
        return None

    profile.profile_name = new_name.strip()
    db.commit()
    db.refresh(profile)
    return profile


def delete_profile(db: Session, profile_id: int) -> bool:
    """Delete a profile and all natively associated models (cascading files)."""
    profile = db.query(TagProfile).filter(TagProfile.id == profile_id).first()
    if not profile:
        return None

    # Find associated models
    models = db.query(ModelArtifact).filter(ModelArtifact.profile_id == profile_id).all()
    from services.artifact_service import delete_model

    for m in models:
        try:
            delete_model(db, m.id)
        except Exception as exc:
            logger.warning(
                "Could not delete model %s during profile %d deletion: %s",
                m.id,
                profile_id,
                exc,
            )

    db.delete(profile)
    db.commit()
    logger.info("Deleted TagProfile %d and its associated models.", profile_id)
    return True


def get_profile_by_tag_hash(db: Session, tag_ids: list[int]) -> TagProfile | None:
    """Lookup a TagProfile given a raw list of integer tag IDs."""
    tag_hash = compute_tag_hash(tag_ids)
    return db.query(TagProfile).filter(TagProfile.tag_hash == tag_hash).first()


def list_profiles(db: Session) -> list[dict[str, Any]]:
    """List all profiles annotated with their associated model count."""
    profiles = db.query(TagProfile).order_by(TagProfile.created_at.desc()).all()
    results = []
    for p in profiles:
        count = db.query(ModelArtifact).filter(ModelArtifact.profile_id == p.id).count()
        results.append({
            "id": p.id,
            "profile_name": p.profile_name,
            "tag_hash": p.tag_hash,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
            "model_count": count,
        })
    return results
