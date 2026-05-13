"""
POST /v1/features/extract — Feature schema extraction.
GET  /v1/hparams/{use_case} — Default hyperparameters.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models.dataset import DatasetRecord
from schemas.features import FeatureExtractRequest, FeatureExtractResponse, HparamResponse
from services.feature_service import extract_features
from services.hparam_service import get_hparams
from utils.errors import ErrorCode, raise_error

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/features/extract", response_model=FeatureExtractResponse)
def feature_extract(
    body: FeatureExtractRequest,
    db: Session = Depends(get_db),
) -> FeatureExtractResponse:
    """Extract and classify feature columns from a stored dataset."""
    record = db.query(DatasetRecord).filter(DatasetRecord.id == body.dataset_id).first()
    if not record:
        raise_error(ErrorCode.DATASET_NOT_FOUND, f"No dataset found with id: {body.dataset_id}", status_code=404)

    result = extract_features(body.dataset_id, record.stored_path)
    return FeatureExtractResponse(**result)


@router.get("/hparams/{use_case}", response_model=HparamResponse)
def hparams(use_case: str) -> HparamResponse:
    """Return default hyperparameters for a use-case."""
    result = get_hparams(use_case)
    return HparamResponse(**result)
