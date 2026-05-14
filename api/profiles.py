"""
Tag Profile management REST endpoints.

GET    /v1/profiles
GET    /v1/profiles/{profile_id}
PATCH  /v1/profiles/{profile_id}
DELETE /v1/profiles/{profile_id}
GET    /v1/profiles/{profile_id}/models
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models.tag_profile import TagProfile
from models.model_artifact import ModelArtifact
from schemas.profile import (
    ProfileUpdateRequest,
    TagProfileListResponse,
    TagProfileResponse,
)
from schemas.registry import ModelListResponse
from services.profile_service import (
    delete_profile,
    list_profiles,
    update_profile_name,
)

from utils.errors import ErrorCode, raise_error

router = APIRouter()


@router.get("/profiles", response_model=TagProfileListResponse)
def list_profiles_endpoint(db: Session = Depends(get_db)):
    """List all deterministic Tag Profiles annotated with their model counts."""
    profiles = list_profiles(db)
    return TagProfileListResponse(items=profiles)


@router.get("/profiles/{profile_id}", response_model=TagProfileResponse)
def get_profile_endpoint(profile_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a specific Tag Profile."""
    profile = db.query(TagProfile).filter(TagProfile.id == profile_id).first()
    if not profile:
        raise_error(
            ErrorCode.PROFILE_NOT_FOUND,
            f"TagProfile with ID {profile_id} not found.",
            status_code=404,
        )
    return profile


@router.patch("/profiles/{profile_id}", response_model=TagProfileResponse)
def update_profile_endpoint(
    profile_id: int, request: ProfileUpdateRequest, db: Session = Depends(get_db)
):
    """Update the human-readable name of a Tag Profile."""
    profile = update_profile_name(db, profile_id, request.profile_name)
    if not profile:
        raise_error(
            ErrorCode.PROFILE_NOT_FOUND,
            f"TagProfile with ID {profile_id} not found.",
            status_code=404,
        )
    return profile


@router.delete("/profiles/{profile_id}")
def delete_profile_endpoint(profile_id: int, db: Session = Depends(get_db)):
    """Delete a Tag Profile and cascade deletion to all associated models/files."""
    success = delete_profile(db, profile_id)
    if not success:
        raise_error(
            ErrorCode.PROFILE_NOT_FOUND,
            f"TagProfile with ID {profile_id} not found.",
            status_code=404,
        )
    return {"deleted": True, "profile_id": profile_id}


@router.get("/profiles/{profile_id}/models", response_model=ModelListResponse)
def get_profile_models_endpoint(profile_id: int, db: Session = Depends(get_db)):
    """List all trained model artifacts native to a specific Tag Profile."""
    profile = db.query(TagProfile).filter(TagProfile.id == profile_id).first()
    if not profile:
        raise_error(
            ErrorCode.PROFILE_NOT_FOUND,
            f"TagProfile with ID {profile_id} not found.",
            status_code=404,
        )

    models = (
        db.query(ModelArtifact)
        .filter(ModelArtifact.profile_id == profile_id)
        .order_by(ModelArtifact.created_at.desc())
        .all()
    )
    return {"total": len(models), "items": models}
