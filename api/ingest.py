"""
POST /v1/ingest — Parquet file upload, validation, cleaning, storage.
"""

from __future__ import annotations

import hashlib
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
from services.profile_service import resolve_or_create_profile
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

    # Hash the file for deduplication
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check if this exact file was already ingested
    existing_record = db.query(DatasetRecord).filter(DatasetRecord.file_hash == file_hash).first()
    if existing_record:
        logger.info("File deduplication hit for hash %s -> returning existing dataset_id %s", file_hash, existing_record.id)
        
        # Resolve the profile to return in response
        profile_id, profile_name, tag_hash = None, None, None
        raw_tags: list[str] = existing_record.tags_detected or []
        if raw_tags:
            try:
                tag_ids = [int(t) for t in raw_tags]
                profile, _ = resolve_or_create_profile(db, tag_ids)
                profile_id = profile.id
                profile_name = profile.profile_name
                tag_hash = profile.tag_hash
            except Exception as exc:
                logger.warning("Could not resolve tag profile for existing dataset %s: %s", existing_record.id, exc)
                
        return IngestResponse(
            dataset_id=existing_record.id,
            row_count=existing_record.row_count,
            col_count=existing_record.col_count,
            tags_detected=existing_record.tags_detected or [],
            quality_report=existing_record.quality_report or {},
            range_metadata=existing_record.range_metadata or {},
            profile_id=profile_id,
            profile_name=profile_name,
            tag_hash=tag_hash,
            file_hash=file_hash,
            is_duplicate=True,
            created_at=existing_record.created_at,
        )

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
        file_hash=file_hash,
    )
    db.add(record)
    db.commit()

    logger.info("Ingested dataset %s (%d rows, %d cols)", result["dataset_id"], result["row_count"], result["col_count"])

    # Resolve or create a Tag Profile from the detected tags
    profile_id = None
    profile_name = None
    tag_hash = None
    raw_tags: list[str] = result.get("tags_detected", [])
    if raw_tags:
        try:
            tag_ids = [int(t) for t in raw_tags]
            profile, _ = resolve_or_create_profile(db, tag_ids)
            profile_id = profile.id
            profile_name = profile.profile_name
            tag_hash = profile.tag_hash
            logger.info(
                "Resolved TagProfile %d (%s) for dataset %s",
                profile.id,
                profile.profile_name,
                result["dataset_id"],
            )
        except Exception as exc:
            logger.warning("Could not resolve tag profile for dataset %s: %s", result["dataset_id"], exc)

    return IngestResponse(
        **result,
        profile_id=profile_id,
        profile_name=profile_name,
        tag_hash=tag_hash,
        file_hash=file_hash,
        is_duplicate=False,
    )


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
