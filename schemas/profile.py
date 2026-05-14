"""
Pydantic schemas for the Tag Profile API.
"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from schemas.registry import ModelArtifactResponse


class TagProfileResponse(BaseModel):
    """Full TagProfile record."""

    id: int
    profile_name: str
    tag_hash: str
    created_at: datetime
    updated_at: datetime


class TagProfileListItem(TagProfileResponse):
    """TagProfile record annotated with model count."""

    model_count: int


class TagProfileListResponse(BaseModel):
    """List of TagProfiles."""

    items: list[TagProfileListItem]


class ProfileUpdateRequest(BaseModel):
    """Request body for PATCH /v1/profiles/{id}."""

    profile_name: str = Field(..., description="New profile name")
