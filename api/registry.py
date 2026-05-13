"""
Model registry CRUD endpoints.

GET    /v1/models
GET    /v1/models/{model_id}
DELETE /v1/models/{model_id}
GET    /v1/models/{model_id}/download
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from schemas.registry import DeleteResponse, ModelArtifactResponse, ModelListResponse
from services.artifact_service import delete_model, get_model, list_models
from utils.errors import ErrorCode, raise_error

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
def list_models_endpoint(
    use_case: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> ModelListResponse:
    """List all model artifacts with optional filters."""
    total, items = list_models(db, use_case=use_case, status=status, limit=limit, offset=offset)
    return ModelListResponse(
        total=total,
        items=[ModelArtifactResponse.model_validate(i.__dict__) for i in items],
    )


@router.get("/models/{model_id}", response_model=ModelArtifactResponse)
def get_model_endpoint(model_id: str, db: Session = Depends(get_db)) -> ModelArtifactResponse:
    """Return full model artifact record."""
    record = get_model(db, model_id)
    return ModelArtifactResponse.model_validate(record.__dict__)


@router.delete("/models/{model_id}", response_model=DeleteResponse)
def delete_model_endpoint(model_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    """Delete a model artifact and its files."""
    delete_model(db, model_id)
    return DeleteResponse(deleted=True, model_id=model_id)


@router.get("/models/{model_id}/download")
def download_model(model_id: str, db: Session = Depends(get_db)):
    """Download the trained model .pkl file."""
    record = get_model(db, model_id)
    if not record.artifact_path or not Path(record.artifact_path).exists():
        raise_error(ErrorCode.ARTIFACT_FILE_MISSING, "Model artifact file not found on disk.", status_code=404)
    return FileResponse(
        path=record.artifact_path,
        filename=f"{model_id}.pkl",
        media_type="application/octet-stream",
    )
