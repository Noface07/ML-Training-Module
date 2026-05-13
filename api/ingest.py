"""
POST /v1/ingest — Parquet file upload, validation, cleaning, storage.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
import pandas as pd

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from database import get_db
from models.dataset import DatasetRecord
from schemas.ingest import DatasetPreviewResponse, IngestConfig, IngestResponse
from services.ingest_service import ingest_file
from utils.errors import ErrorCode, raise_error

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest(
    file: UploadFile = File(..., description="Parquet file to ingest"),
    config: Optional[str] = Form(None, description="Optional JSON config string"),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Ingest a parquet file: validate, clean, store, and return metadata."""

    # Parse optional config
    cfg = IngestConfig()
    if config:
        try:
            cfg = IngestConfig(**json.loads(config))
        except Exception as exc:
            raise_error(ErrorCode.INVALID_FILE_FORMAT, f"Invalid config JSON: {exc}", status_code=400)

    # Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise_error(ErrorCode.INVALID_FILE_FORMAT, "Uploaded file is empty.", status_code=400)

    # Run ingest pipeline
    result = ingest_file(
        file_bytes=file_bytes,
        original_filename=file.filename or "unknown.parquet",
        timestamp_col=cfg.timestamp_col,
        timestamp_format=cfg.timestamp_format,
        forward_fill_limit=cfg.forward_fill_limit,
        backward_fill_first_rows=cfg.backward_fill_first_rows,
        drop_duplicate_timestamps=cfg.drop_duplicate_timestamps,
    )

    # Persist to DB
    record = DatasetRecord(
        id=result["dataset_id"],
        original_filename=result["original_filename"],
        stored_path=result["stored_path"],
        row_count=result["row_count"],
        col_count=result["col_count"],
        tags_detected=result["tags_detected"],
        quality_report=result["quality_report"],
        range_metadata=result["range_metadata"],
        created_at=result["created_at"],
    )
    db.add(record)
    db.commit()

    logger.info("Ingested dataset %s (%d rows, %d cols)", result["dataset_id"], result["row_count"], result["col_count"])

    return IngestResponse(**result)


@router.get("/dataset/{dataset_id}/preview", response_model=DatasetPreviewResponse, status_code=200)
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return for preview"),
    db: Session = Depends(get_db),
) -> DatasetPreviewResponse:
    """Preview the schema and first N rows of an ingested dataset."""
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise_error(ErrorCode.DATASET_NOT_FOUND, f"Dataset {dataset_id} not found.", status_code=404)

    if not os.path.exists(record.stored_path):
        raise_error(ErrorCode.INTERNAL_ERROR, f"Stored file for dataset {dataset_id} is missing on disk.", status_code=500)

    try:
        df = pd.read_parquet(record.stored_path)
        preview_df = df.head(limit)
        # Convert NaN/NaT to None for valid JSON serialization
        preview_df = preview_df.replace({pd.NA: None, float("nan"): None})
        rows = json.loads(preview_df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        raise_error(ErrorCode.INTERNAL_ERROR, f"Failed to read dataset parquet file: {exc}", status_code=500)

    return DatasetPreviewResponse(
        dataset_id=record.id,
        original_filename=record.original_filename,
        row_count=record.row_count,
        col_count=record.col_count,
        columns=list(df.columns),
        rows=rows,
    )
