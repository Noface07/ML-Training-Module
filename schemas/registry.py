"""Pydantic schemas for the model-registry API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelArtifactResponse(BaseModel):
    """Full model-artifact record returned from the registry."""

    id: str
    use_case: str
    model_type: str
    version: int
    model_name: str | None = None
    artifact_path: str | None = None
    metadata_path: str | None = None
    dataset_id: str
    feature_schema_id: str | None = None
    feature_schema_snapshot: Any | None = None
    hparams_used: Any | None = None
    metrics: Any | None = None
    feature_importance: Any | None = None
    range_metadata: Any | None = None
    tags_used: Any | None = None
    optional_features_used: Any | None = None
    training_duration_seconds: float | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ModelListResponse(BaseModel):
    """Paginated list of model artifacts."""

    total: int = Field(description="Total matching records.")
    items: list[ModelArtifactResponse]


class DeleteResponse(BaseModel):
    """Response from DELETE /v1/models/{model_id}."""

    deleted: bool
    model_id: str
