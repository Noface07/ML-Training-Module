"""
Custom exception classes and standardised error-response helpers.

Every API error is returned as::

    {
        "error_code": "DATASET_NOT_FOUND",
        "message":    "No dataset found with id: abc-123",
        "field":      null,
        "detail":     null
    }
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    """Canonical error codes used across all API responses."""

    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    MISSING_TIMESTAMP_COL = "MISSING_TIMESTAMP_COL"
    TIMESTAMP_PARSE_FAILURE = "TIMESTAMP_PARSE_FAILURE"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    UNKNOWN_USE_CASE = "UNKNOWN_USE_CASE"
    MISSING_TARGET_COL = "MISSING_TARGET_COL"
    TAGS_NOT_IN_DATASET = "TAGS_NOT_IN_DATASET"
    INVALID_OPTIONAL_FEATURE = "INVALID_OPTIONAL_FEATURE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_STILL_TRAINING = "MODEL_STILL_TRAINING"
    ARTIFACT_FILE_MISSING = "ARTIFACT_FILE_MISSING"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SCHEMA_NOT_FOUND = "SCHEMA_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    PROFILE_NOT_FOUND = "PROFILE_NOT_FOUND"


class MLPlatformError(HTTPException):
    """Structured HTTP exception carrying an ``ErrorCode``."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 422,
        field: str | None = None,
        detail: Any | None = None,
    ) -> None:
        body = {
            "error_code": code.value,
            "message": message,
            "field": field,
            "detail": detail,
        }
        super().__init__(status_code=status_code, detail=body)


def raise_error(
    code: ErrorCode,
    message: str,
    status_code: int = 422,
    field: str | None = None,
    detail: Any | None = None,
) -> None:
    """Convenience helper — raises :class:`MLPlatformError`."""
    raise MLPlatformError(
        code=code,
        message=message,
        status_code=status_code,
        field=field,
        detail=detail,
    )


def error_response(
    code: ErrorCode,
    message: str,
    status_code: int = 422,
    field: str | None = None,
    detail: Any | None = None,
) -> JSONResponse:
    """Return a ``JSONResponse`` (useful in exception handlers)."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": code.value,
            "message": message,
            "field": field,
            "detail": detail,
        },
    )
