"""Pydantic schemas for the ingest API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    """Optional configuration for the ingest endpoint."""

    timestamp_col: str = Field(
        default="timestamp",
        description="Name of the timestamp column in the uploaded file.",
    )
    timestamp_format: str = Field(
        default="auto",
        description="Expected timestamp format. 'auto' tries multiple.",
    )
    forward_fill_limit: int = Field(
        default=5,
        ge=0,
        description="Maximum consecutive NaN rows to forward-fill.",
    )
    backward_fill_first_rows: bool = Field(
        default=True,
        description="Backward-fill leading NaNs after forward fill.",
    )
    drop_duplicate_timestamps: bool = Field(
        default=True,
        description="Drop rows with duplicate timestamp values (keep first).",
    )


class QualityReport(BaseModel):
    """Data-quality summary produced during ingestion."""

    null_rows_dropped: int = Field(description="Fully-null rows removed.")
    duplicate_timestamps_dropped: int = Field(description="Duplicate-ts rows removed.")
    null_pct_by_column: dict[str, float] = Field(description="Null % per column after cleaning.")
    timestamp_gaps: list[dict[str, Any]] = Field(description="Detected timestamp gaps.")
    fill_operations: dict[str, int] = Field(description="Forward/backward fill counts.")


class IngestResponse(BaseModel):
    """Response returned after successful ingestion."""

    dataset_id: str = Field(description="UUID of the newly created dataset record.")
    row_count: int = Field(description="Rows in the cleaned dataset.")
    col_count: int = Field(description="Columns in the cleaned dataset.")
    tags_detected: list[str] = Field(description="Tag IDs found in column names.")
    quality_report: QualityReport
    range_metadata: dict[str, dict[str, float]] = Field(
        description="Per-tag min/max from raw columns."
    )
    profile_id: int | None = Field(default=None, description="ID of the resolved Tag Profile.")
    profile_name: str | None = Field(default=None, description="Name of the resolved Tag Profile.")
    tag_hash: str | None = Field(default=None, description="Deterministic hash of the detected tags.")
    file_hash: str | None = Field(default=None, description="SHA-256 hash of the ingested parquet file.")
    is_duplicate: bool = Field(default=False, description="True if the file was previously ingested and processing was skipped.")
    created_at: datetime


class DatasetPreviewResponse(BaseModel):
    """Response returned when previewing a dataset's rows and schema."""

    dataset_id: str = Field(description="UUID of the dataset.")
    original_filename: str = Field(description="Original filename uploaded.")
    row_count: int = Field(description="Total rows in the dataset.")
    col_count: int = Field(description="Total columns in the dataset.")
    columns: list[str] = Field(description="List of column names.")
    rows: list[dict[str, Any]] = Field(description="Preview rows as dictionary records.")
