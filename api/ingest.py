"""
POST /v1/ingest — Parquet file upload, validation, cleaning, storage.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from database import get_db
from models.dataset import DatasetRecord
from schemas.ingest import IngestConfig, IngestResponse
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
