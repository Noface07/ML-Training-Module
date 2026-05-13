"""
Feature-extraction service.

Reads a stored parquet's column names (zero-cost schema read) and
classifies them using the dynamic column parser.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from config import settings
from utils.column_parser import parse_columns, FeatureSchema

logger = logging.getLogger(__name__)

_schema_cache: dict[str, dict[str, Any]] = {}


def extract_features(dataset_id: str, stored_path: str) -> dict[str, Any]:
    """Read column names from parquet and classify them dynamically."""
    parquet_path = Path(stored_path)
    if not parquet_path.exists():
        parquet_path = Path(settings.DATASETS_DIR) / f"{dataset_id}.parquet"

    schema = pq.read_schema(str(parquet_path))
    columns: list[str] = schema.names
    fs: FeatureSchema = parse_columns(columns)
    feature_schema_id = str(uuid.uuid4())

    result = {
        "feature_schema_id": feature_schema_id,
        "dataset_id": dataset_id,
        "tags": fs.tags,
        "per_tag_features": {
            "mandatory": fs.per_tag_features_mandatory,
            "optional": fs.per_tag_features_optional,
        },
        "cross_tag_features": {
            "available": fs.cross_tag_available,
            "present_in_data": fs.cross_tag_present,
        },
        "target_col": fs.target_col,
        "target_present": fs.target_present,
        "total_columns": fs.total_columns,
        "resolved_columns_sample": columns[:20],
    }

    _schema_cache[feature_schema_id] = {
        "dataset_id": dataset_id,
        "schema": fs,
        "raw": result,
    }
    return result


def get_cached_schema(feature_schema_id: str) -> dict[str, Any] | None:
    """Retrieve a previously generated feature schema from cache."""
    return _schema_cache.get(feature_schema_id)
