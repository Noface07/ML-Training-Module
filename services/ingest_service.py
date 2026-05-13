"""
Ingest service — validation, cleaning, pivot, fill, range computation.

This module contains *all* business logic for the ``POST /v1/ingest``
endpoint.  It has no dependency on FastAPI.
"""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import settings
from utils.column_parser import parse_column
from utils.errors import ErrorCode, raise_error

logger = logging.getLogger(__name__)

# ── Timestamp formats to attempt when ``auto`` is specified ────────────
_TIMESTAMP_FORMATS: list[str] = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y%m%d%H%M%S",
]


def _try_parse_timestamps(
    series: pd.Series,
    fmt: str,
) -> pd.Series | None:
    """Try to parse a Series with a single format; return None on failure."""
    if fmt == "auto":
        # Try explicit formats first, fall back to pd.to_datetime inference
        for f in _TIMESTAMP_FORMATS:
            try:
                return pd.to_datetime(series, format=f)
            except (ValueError, TypeError):
                continue
        # Last resort — pandas inference
        try:
            return pd.to_datetime(series, infer_datetime_format=True)
        except Exception:
            return None
    else:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            return None


def _detect_timestamp_gaps(
    ts_series: pd.Series,
) -> list[dict[str, Any]]:
    """Return a list of gaps larger than the median interval."""
    if len(ts_series) < 3:
        return []
    diffs = ts_series.diff().dropna()
    median_diff = diffs.median()
    threshold = median_diff * 2
    gaps = []
    for idx in diffs[diffs > threshold].index:
        gaps.append(
            {
                "start": str(ts_series.iloc[idx - 1]),
                "end": str(ts_series.iloc[idx]),
                "gap_minutes": round(diffs.loc[idx].total_seconds() / 60, 2),
            }
        )
    return gaps


def ingest_file(
    file_bytes: bytes,
    original_filename: str,
    timestamp_col: str = "timestamp",
    timestamp_format: str = "auto",
    forward_fill_limit: int = 5,
    backward_fill_first_rows: bool = True,
    drop_duplicate_timestamps: bool = True,
) -> dict[str, Any]:
    """Run the full ingest pipeline and return the result dict.

    Args:
        file_bytes: Raw bytes of the uploaded parquet file.
        original_filename: Client-provided filename.
        timestamp_col: Column name to use as timestamp.
        timestamp_format: Expected format or ``"auto"``.
        forward_fill_limit: Max consecutive NaN rows to forward-fill.
        backward_fill_first_rows: Backward-fill leading NaNs.
        drop_duplicate_timestamps: Remove duplicate timestamp rows.

    Returns:
        Dictionary matching the ``IngestResponse`` schema.
    """

    dataset_id = str(uuid.uuid4())

    # ── Step 1 — Load & validate ───────────────────────────────────────
    try:
        df = pd.read_parquet(io.BytesIO(file_bytes))
    except Exception as exc:
        raise_error(
            ErrorCode.INVALID_FILE_FORMAT,
            f"Failed to read parquet file: {exc}",
            status_code=400,
        )

    if timestamp_col not in df.columns:
        raise_error(
            ErrorCode.MISSING_TIMESTAMP_COL,
            f"Timestamp column '{timestamp_col}' not found in dataset. "
            f"Available columns: {list(df.columns[:20])}",
            field="timestamp",
        )

    # Parse timestamps
    parsed = _try_parse_timestamps(df[timestamp_col], timestamp_format)
    if parsed is None:
        sample = df[timestamp_col].head(5).tolist()
        raise_error(
            ErrorCode.TIMESTAMP_PARSE_FAILURE,
            f"Could not parse timestamps. Sample values: {sample}",
            field=timestamp_col,
        )
    df[timestamp_col] = parsed

    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Detect gaps (informational)
    timestamp_gaps = _detect_timestamp_gaps(df[timestamp_col])

    # ── Step 2 — Clean ─────────────────────────────────────────────────
    initial_len = len(df)

    # Drop fully-null rows
    null_mask = df.drop(columns=[timestamp_col]).isnull().all(axis=1)
    null_rows_dropped = int(null_mask.sum())
    df = df[~null_mask].reset_index(drop=True)

    # Drop duplicate timestamps
    dup_ts_dropped = 0
    if drop_duplicate_timestamps:
        before = len(df)
        df = df.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(
            drop=True
        )
        dup_ts_dropped = before - len(df)

    # ── Step 3 — Pivot to wide format if needed ────────────────────────
    if "tag_id" in df.columns and "value" in df.columns:
        logger.info("Long-format detected — pivoting to wide format.")
        df = df.pivot_table(
            index=timestamp_col, columns="tag_id", values="value"
        ).reset_index()
        df.columns = [
            f"tag_{c}_raw" if c != timestamp_col else c for c in df.columns
        ]

    # ── Step 4 — Fill ──────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    forward_filled = 0
    backward_filled = 0

    if forward_fill_limit > 0 and numeric_cols:
        before_nulls = int(df[numeric_cols].isnull().sum().sum())
        df[numeric_cols] = df[numeric_cols].ffill(limit=forward_fill_limit)
        after_nulls = int(df[numeric_cols].isnull().sum().sum())
        forward_filled = before_nulls - after_nulls

    if backward_fill_first_rows and numeric_cols:
        before_nulls = int(df[numeric_cols].isnull().sum().sum())
        df[numeric_cols] = df[numeric_cols].bfill()
        after_nulls = int(df[numeric_cols].isnull().sum().sum())
        backward_filled = before_nulls - after_nulls

    # ── Step 5 — Compute range metadata ────────────────────────────────
    range_metadata: dict[str, dict[str, float]] = {}
    for col in df.columns:
        pc = parse_column(col)
        if pc.is_tag_feature and pc.suffix == "raw":
            tag_id = pc.tag_id
            range_metadata[tag_id] = {
                "min": float(df[col].min()) if not df[col].isnull().all() else 0.0,
                "max": float(df[col].max()) if not df[col].isnull().all() else 0.0,
            }

    # Detect tags
    tags_detected: set[str] = set()
    for col in df.columns:
        pc = parse_column(col)
        if pc.is_tag_feature and pc.tag_id:
            tags_detected.add(pc.tag_id)

    # Null pct by column (after cleaning)
    null_pct: dict[str, float] = {}
    for col in df.columns:
        pct = float(df[col].isnull().mean())
        if pct > 0:
            null_pct[col] = round(pct, 6)

    # ── Step 6 — Store ─────────────────────────────────────────────────
    store_dir = Path(settings.DATASETS_DIR)
    store_dir.mkdir(parents=True, exist_ok=True)
    stored_path = store_dir / f"{dataset_id}.parquet"
    df.to_parquet(str(stored_path), index=False)

    quality_report = {
        "null_rows_dropped": null_rows_dropped,
        "duplicate_timestamps_dropped": dup_ts_dropped,
        "null_pct_by_column": null_pct,
        "timestamp_gaps": timestamp_gaps,
        "fill_operations": {
            "forward_filled": forward_filled,
            "backward_filled": backward_filled,
        },
    }

    return {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
        "stored_path": str(stored_path.resolve()),
        "row_count": len(df),
        "col_count": len(df.columns),
        "tags_detected": sorted(tags_detected),
        "quality_report": quality_report,
        "range_metadata": range_metadata,
        "created_at": datetime.utcnow(),
    }
