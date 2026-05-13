"""Pydantic schemas for feature-extraction and hparam endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureExtractRequest(BaseModel):
    """Request body for POST /v1/features/extract."""

    dataset_id: str = Field(description="UUID of the ingested dataset.")


class PerTagFeatures(BaseModel):
    """Mandatory vs optional per-tag feature classification."""

    mandatory: list[str] = Field(description="Suffixes required for every tag.")
    optional: list[str] = Field(description="Suffixes that can be toggled on/off.")


class CrossTagFeatures(BaseModel):
    """Cross-tag feature availability."""

    available: list[str] = Field(description="All known cross-tag feature names.")
    present_in_data: list[str] = Field(description="Cross-tag features found in this dataset.")


class FeatureExtractResponse(BaseModel):
    """Response from POST /v1/features/extract."""

    feature_schema_id: str = Field(description="UUID for this schema snapshot.")
    dataset_id: str
    tags: list[str] = Field(description="Tag IDs detected.")
    per_tag_features: PerTagFeatures
    cross_tag_features: CrossTagFeatures
    target_col: str | None = Field(description="Target column name if present.")
    target_present: bool
    total_columns: int
    resolved_columns_sample: list[str] = Field(
        description="Sample of resolved column names."
    )


class HparamDetail(BaseModel):
    """Single hyperparameter definition."""

    value: Any = Field(description="Default value.")
    type: str = Field(description="Python type hint string.")
    min: float | None = Field(default=None, description="Minimum allowed value.")
    max: float | None = Field(default=None, description="Maximum allowed value.")
    description: str = Field(default="", description="Human-readable description.")


class HparamResponse(BaseModel):
    """Response from GET /v1/hparams/{use_case}."""

    use_case: str
    model_type: str
    output_tag: str
    requires_training: bool
    tier1: dict[str, HparamDetail] = Field(default_factory=dict)
    tier2: dict[str, HparamDetail] = Field(default_factory=dict)
    tier3: dict[str, HparamDetail] = Field(default_factory=dict)
