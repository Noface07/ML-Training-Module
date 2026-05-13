"""
POST /v1/train         — Spawn training job.
GET  /v1/train/{job_id}/stream — SSE log stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from config import USE_CASE_REGISTRY, settings
from database import get_db
from models.dataset import DatasetRecord
from models.model_artifact import ModelArtifact
from schemas.train import TrainRequest, TrainResponse
from services.artifact_service import next_version
from services.feature_service import get_cached_schema
from services.hparam_service import merge_hparams
from utils.column_parser import (
    MANDATORY_SUFFIXES, build_column_list, build_mandatory_columns,
    build_optional_groups, parse_columns,
)
from utils.errors import ErrorCode, raise_error

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train", status_code=202)
def start_training(
    body: TrainRequest,
    db: Session = Depends(get_db),
) -> TrainResponse:
    """Validate, spawn training subprocess, return 202 immediately."""

    # 1. Validate dataset
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == body.dataset_id).first()
    if not dataset:
        raise_error(ErrorCode.DATASET_NOT_FOUND, f"No dataset found with id: {body.dataset_id}", status_code=404)

    # 2. Validate use_case
    if body.use_case not in USE_CASE_REGISTRY:
        raise_error(ErrorCode.UNKNOWN_USE_CASE, f"Unknown use case: {body.use_case}")

    uc_info = USE_CASE_REGISTRY[body.use_case]

    # 3. Validate target col for supervised use cases
    if uc_info["requires_training"] and uc_info["model_type"] in ("xgboost_clf", "xgboost_reg"):
        if not body.target_col:
            raise_error(ErrorCode.MISSING_TARGET_COL, "target_col is required for this use case.", field="target_col")
        ds_tags = dataset.tags_detected or []
        # Quick col-name check via cached schema or stored parquet
        import pyarrow.parquet as pq
        schema = pq.read_schema(dataset.stored_path)
        col_names = schema.names
        if body.target_col not in col_names:
            raise_error(ErrorCode.MISSING_TARGET_COL, f"Target column '{body.target_col}' not in dataset.", field="target_col")

    # 4. Validate tags exist
    ds_tags_set = set(dataset.tags_detected or [])
    missing_tags = [t for t in body.tags if t not in ds_tags_set]
    if missing_tags:
        raise_error(ErrorCode.TAGS_NOT_IN_DATASET, f"Tags not found in dataset: {missing_tags}", detail=missing_tags)

    # 5. Validate optional features — dynamically discovered
    # Read the schema to find valid suffixes
    import pyarrow.parquet as pq
    schema = pq.read_schema(dataset.stored_path)
    col_names = schema.names
    fs = parse_columns(col_names)
    valid_optional = set(fs.per_tag_features_optional)
    invalid = [f for f in body.optional_features if f not in valid_optional]
    if invalid:
        raise_error(
            ErrorCode.INVALID_OPTIONAL_FEATURE,
            f"Invalid optional features: {invalid}. Valid: {sorted(valid_optional)}",
            detail=invalid,
        )

    # 5b. Validate selective cross-tag features
    valid_cross_tag = set(fs.cross_tag_present)
    invalid_ct = [f for f in body.cross_tag_features if f not in valid_cross_tag]
    if invalid_ct:
        raise_error(
            ErrorCode.INVALID_OPTIONAL_FEATURE,
            f"Invalid cross-tag features: {invalid_ct}. Valid: {sorted(valid_cross_tag)}",
            detail=invalid_ct,
        )

    # Resolve selected cross-tag features list
    if body.cross_tag_features:
        resolved_cross_tag = sorted(body.cross_tag_features)
    elif body.include_cross_tag_features:
        resolved_cross_tag = sorted(fs.cross_tag_present)
    else:
        resolved_cross_tag = []

    # 6. Merge hparams
    hparams_merged = merge_hparams(body.use_case, body.hparams)

    # 7. Resolve column list
    feature_columns = build_column_list(
        tags=body.tags,
        mandatory_suffixes=fs.per_tag_features_mandatory,
        optional_suffixes=body.optional_features,
        include_cross_tag=body.include_cross_tag_features,
        cross_tag_present=fs.cross_tag_present,
        tag_column_map=fs.tag_column_map,
        target_col=None,  # target handled separately
        selected_cross_tag_features=body.cross_tag_features,
    )

    # 8. Create model artifact record
    model_id = str(uuid.uuid4())
    version = next_version(db, body.use_case)

    artifact = ModelArtifact(
        id=model_id,
        use_case=body.use_case,
        model_type=uc_info["model_type"],
        version=version,
        dataset_id=body.dataset_id,
        feature_schema_id=body.feature_schema_id,
        feature_schema_snapshot={"feature_columns": feature_columns, "target_col": body.target_col},
        hparams_used=hparams_merged,
        range_metadata=dataset.range_metadata,
        tags_used=body.tags,
        optional_features_used=body.optional_features,
        cross_tag_features_used=resolved_cross_tag,
        status="training",
    )
    db.add(artifact)
    db.commit()

    # 9. Write job config
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    config_path = os.path.join(settings.LOGS_DIR, f"{model_id}_config.json")

    # Build mandatory / optional groups for feature selection
    mandatory_columns = build_mandatory_columns(
        tags=body.tags,
        mandatory_suffixes=fs.per_tag_features_mandatory,
        tag_column_map=fs.tag_column_map,
    )
    optional_feature_groups = build_optional_groups(
        tags=body.tags,
        optional_suffixes=body.optional_features,
        tag_column_map=fs.tag_column_map,
    )

    job_config = {
        "model_id": model_id,
        "dataset_id": body.dataset_id,
        "dataset_path": dataset.stored_path,
        "use_case": body.use_case,
        "model_type": uc_info["model_type"],
        "feature_columns": feature_columns,
        "mandatory_columns": mandatory_columns,
        "optional_feature_groups": optional_feature_groups,
        "cross_tag_columns": resolved_cross_tag,
        "feature_selection": body.feature_selection,
        "target_col": body.target_col,
        "tags": body.tags,
        "train_split": body.train_split,
        "cv_folds": body.cv_folds,
        "hparams_merged": hparams_merged,
        "range_metadata": dataset.range_metadata,
        "log_path": os.path.join(settings.LOGS_DIR, f"{model_id}.log"),
        "artifact_dir": settings.ARTIFACTS_DIR,
        "db_url": settings.DB_URL,
    }
    with open(config_path, "w") as f:
        json.dump(job_config, f, indent=2, default=str)

    # 10. Spawn training subprocess
    trainer_module = {
        "xgboost_clf": "training.xgboost_clf_trainer",
        "xgboost_reg": "training.xgboost_reg_trainer",
        "isolation_forest": "training.isolation_forest_trainer",
        "statistical": "training.statistical_trainer",
    }[uc_info["model_type"]]

    subprocess.Popen(
        [sys.executable, "-m", trainer_module, "--job-config", config_path],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    logger.info("Training job %s spawned for use_case=%s", model_id, body.use_case)

    return TrainResponse(
        model_id=model_id,
        job_id=model_id,
        status="training",
        stream_url=f"/v1/train/{model_id}/stream",
        use_case=body.use_case,
        output_tag=uc_info["output_tag"],
    )


@router.get("/train/{job_id}/stream")
async def stream_logs(job_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that tails the training log file."""
    record = db.query(ModelArtifact).filter(ModelArtifact.id == job_id).first()
    if not record:
        raise_error(ErrorCode.JOB_NOT_FOUND, f"No training job found with id: {job_id}", status_code=404)

    log_path = os.path.join(settings.LOGS_DIR, f"{job_id}.log")
    result_path = os.path.join(settings.LOGS_DIR, f"{job_id}_result.json")

    async def event_generator() -> AsyncGenerator[dict, None]:
        last_pos = 0
        while True:
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                for line in new_lines:
                    line = line.strip()
                    if line:
                        yield {"data": line}

            # Check if training finished
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    result = json.load(f)
                yield {"event": "status", "data": json.dumps({"status": result.get("status", "completed")})}
                yield {"event": "result", "data": json.dumps(result)}
                break

            await asyncio.sleep(settings.SSE_POLL_INTERVAL_MS / 1000.0)

    return EventSourceResponse(event_generator())
