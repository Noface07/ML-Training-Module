"""Pydantic schemas for the training API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request body for POST /v1/train."""

    dataset_id: str = Field(description="UUID of the ingested dataset.")
    feature_schema_id: str = Field(description="UUID returned by /features/extract.")
    use_case: str = Field(description="Use-case key from the registry.")
    model_name: str | None = Field(
        default=None,
        description="Optional custom name for the trained model.",
    )
    tags: list[str] = Field(description="Tag IDs to include in training.")
    mandatory_features: list[str] = Field(
        default_factory=list,
        description=(
            "Mandatory per-tag suffixes to always include. "
            "If empty, auto-detected from the dataset schema "
            "(raw, roll_mean, roll_std, roc_1 + any _MAN suffixed features)."
        ),
    )
    optional_features: list[str] = Field(
        default_factory=list,
        description="Optional per-tag suffixes to include.",
    )
    cross_tag_features: list[str] = Field(
        default_factory=list,
        description="Specific cross-tag features to include.",
    )
    include_cross_tag_features: bool = Field(
        default=False,
        description="Include cross-tag features if available (fallback if cross_tag_features list is empty).",
    )
    target_col: str | None = Field(
        default=None,
        description="Target column name (required for supervised use cases).",
    )
    train_split: float = Field(
        default=0.8,
        ge=0.1,
        le=0.99,
        description="Fraction of data used for training.",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of cross-validation folds.",
    )
    hparams: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional hyperparameter overrides.",
    )
    feature_selection: bool = Field(
        default=True,
        description="If true, try combinations of optional features and pick the best via CV.",
    )


class TrainResponse(BaseModel):
    """Immediate 202 response after a training job is spawned."""

    model_id: str = Field(description="UUID of the model artifact.")
    model_name: str | None = Field(default=None, description="Name of the model artifact.")
    job_id: str = Field(description="Job identifier (same as model_id).")
    status: str = Field(default="training")
    stream_url: str = Field(description="SSE stream URL for live logs.")
    use_case: str
    output_tag: str


class TrainResultEvent(BaseModel):
    """Final SSE event payload emitted when training completes."""

    model_id: str
    model_name: str | None = None
    status: str
    use_case: str
    output_tag: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    training_duration_seconds: float
    artifact_path: str
