"""
Application configuration and use-case registry.

Settings are loaded from environment variables / .env file via pydantic-settings.
The USE_CASE_REGISTRY is the single source of truth for mapping use-case names
to model types, output tags, and training requirements.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the ML Platform."""

    # ── Paths ──────────────────────────────────────────────────────────
    ARTIFACTS_DIR: str = "artifacts"
    DATASETS_DIR: str = "datasets"
    LOGS_DIR: str = "logs"
    DB_URL: str = "sqlite:///./ml_platform.db"

    # ── Ingest defaults ────────────────────────────────────────────────
    DEFAULT_TIMESTAMP_COL: str = "timestamp"
    DEFAULT_FORWARD_FILL_LIMIT: int = 5
    DEFAULT_BACKWARD_FILL: bool = True
    DEFAULT_TRAIN_SPLIT: float = 0.8
    DEFAULT_CV_FOLDS: int = 5

    # ── SSE ─────────────────────────────────────────────────────────────
    SSE_POLL_INTERVAL_MS: int = 200

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


# ── Use-case registry ──────────────────────────────────────────────────
# Maps every supported use-case to its model type, output tag, and
# whether it needs a full training run (as opposed to a statistical /
# rule-based pipeline).

USE_CASE_REGISTRY: dict[str, dict] = {
    "failure_prediction": {
        "model_type": "xgboost_clf",
        "output_tag": "risk_score",
        "requires_training": True,
    },
    "risk_scoring": {
        "model_type": "xgboost_clf",
        "output_tag": "risk_score",
        "requires_training": True,
    },
    "rul": {
        "model_type": "xgboost_reg",
        "output_tag": "rul_hours",
        "requires_training": True,
    },
    "next_interval": {
        "model_type": "xgboost_reg",
        "output_tag": "next_value",
        "requires_training": True,
    },
    "anomaly_multivariate": {
        "model_type": "isolation_forest",
        "output_tag": "anomaly_score",
        "requires_training": True,
    },
    "kpi_prediction": {
        "model_type": "xgboost_reg",
        "output_tag": "kpi_value",
        "requires_training": True,
    },
    "anomaly_univariate": {
        "model_type": "statistical",
        "output_tag": "anomaly_score",
        "requires_training": False,
    },
    "adaptive_threshold": {
        "model_type": "statistical",
        "output_tag": "dynamic_threshold",
        "requires_training": False,
    },
    "early_warning": {
        "model_type": "statistical",
        "output_tag": "warning_score",
        "requires_training": False,
    },
    "health_index": {
        "model_type": "statistical",
        "output_tag": "health_score",
        "requires_training": False,
    },
    "drift_detection": {
        "model_type": "statistical",
        "output_tag": "drift_score",
        "requires_training": False,
    },
    "pattern_detection": {
        "model_type": "statistical",
        "output_tag": "pattern_flag",
        "requires_training": False,
    },
    "data_quality": {
        "model_type": "statistical",
        "output_tag": "data_quality_score",
        "requires_training": False,
    },
}
