"""
Hyperparameter default registry per use-case.
"""

from __future__ import annotations
from typing import Any
from config import USE_CASE_REGISTRY
from utils.errors import ErrorCode, raise_error

HPARAM_REGISTRY: dict[str, dict[str, Any]] = {
    "failure_prediction": {
        "model_type": "xgboost_clf",
        "tier1": {
            "n_estimators":     {"value": 300,  "type": "int",   "min": 50,   "max": 2000, "description": "Number of boosting rounds"},
            "max_depth":        {"value": 6,    "type": "int",   "min": 2,    "max": 12,   "description": "Max tree depth"},
            "learning_rate":    {"value": 0.05, "type": "float", "min": 0.001,"max": 0.3,  "description": "Step size shrinkage"},
            "subsample":        {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Row sampling ratio per tree"},
            "colsample_bytree": {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Feature sampling ratio per tree"},
        },
        "tier2": {
            "min_child_weight": {"value": 5,      "type": "int",    "min": 1, "max": 50,   "description": "Min sum of instance weight in leaf"},
            "scale_pos_weight": {"value": "auto",  "type": "string", "description": "Class imbalance weight. 'auto' = neg/pos ratio"},
            "gamma":            {"value": 0.1,     "type": "float",  "min": 0, "max": 5.0,  "description": "Min loss reduction for split"},
            "reg_alpha":        {"value": 0.0,     "type": "float",  "min": 0, "max": 10.0, "description": "L1 regularization"},
            "reg_lambda":       {"value": 1.0,     "type": "float",  "min": 0, "max": 10.0, "description": "L2 regularization"},
        },
        "tier3": {
            "n_jobs":                {"value": -1,    "type": "int",    "description": "Parallel threads (-1 = all cores)"},
            "random_state":          {"value": 42,    "type": "int",    "description": "Reproducibility seed"},
            "eval_metric":           {"value": "auc", "type": "string", "description": "Evaluation metric for early stopping"},
            "early_stopping_rounds": {"value": 30,    "type": "int",    "description": "Stop if no improvement for N rounds"},
        },
    },
    "risk_scoring": {
        "model_type": "xgboost_clf",
        "tier1": {
            "n_estimators":     {"value": 300,  "type": "int",   "min": 50,   "max": 2000, "description": "Number of boosting rounds"},
            "max_depth":        {"value": 6,    "type": "int",   "min": 2,    "max": 12,   "description": "Max tree depth"},
            "learning_rate":    {"value": 0.05, "type": "float", "min": 0.001,"max": 0.3,  "description": "Step size shrinkage"},
            "subsample":        {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Row sampling ratio per tree"},
            "colsample_bytree": {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Feature sampling ratio per tree"},
        },
        "tier2": {
            "min_child_weight": {"value": 1,     "type": "int",    "min": 1, "max": 50,   "description": "Min sum of instance weight in leaf"},
            "scale_pos_weight": {"value": 1.0,   "type": "float",  "min": 0.1,"max": 100, "description": "Fixed class weight for scoring"},
            "gamma":            {"value": 0.1,   "type": "float",  "min": 0, "max": 5.0,  "description": "Min loss reduction for split"},
            "reg_alpha":        {"value": 0.0,   "type": "float",  "min": 0, "max": 10.0, "description": "L1 regularization"},
            "reg_lambda":       {"value": 1.0,   "type": "float",  "min": 0, "max": 10.0, "description": "L2 regularization"},
        },
        "tier3": {
            "n_jobs":                {"value": -1,    "type": "int",    "description": "Parallel threads"},
            "random_state":          {"value": 42,    "type": "int",    "description": "Seed"},
            "eval_metric":           {"value": "auc", "type": "string", "description": "Eval metric"},
            "early_stopping_rounds": {"value": 30,    "type": "int",    "description": "Early stopping rounds"},
        },
    },
    "rul": {
        "model_type": "xgboost_reg",
        "tier1": {
            "n_estimators":     {"value": 500,  "type": "int",   "min": 50,   "max": 3000, "description": "Number of boosting rounds"},
            "max_depth":        {"value": 5,    "type": "int",   "min": 2,    "max": 12,   "description": "Max tree depth"},
            "learning_rate":    {"value": 0.03, "type": "float", "min": 0.001,"max": 0.3,  "description": "Step size shrinkage"},
            "subsample":        {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Row sampling ratio"},
            "colsample_bytree": {"value": 0.7,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Feature sampling ratio"},
        },
        "tier2": {
            "min_child_weight": {"value": 3,   "type": "int",   "min": 1, "max": 50,   "description": "Min instance weight in leaf"},
            "gamma":            {"value": 0.0, "type": "float", "min": 0, "max": 5.0,  "description": "Min loss reduction"},
            "reg_alpha":        {"value": 0.1, "type": "float", "min": 0, "max": 10.0, "description": "L1 reg"},
            "reg_lambda":       {"value": 1.0, "type": "float", "min": 0, "max": 10.0, "description": "L2 reg"},
        },
        "tier3": {
            "n_jobs":                {"value": -1,     "type": "int",    "description": "Parallel threads"},
            "random_state":          {"value": 42,     "type": "int",    "description": "Seed"},
            "eval_metric":           {"value": "rmse", "type": "string", "description": "Eval metric"},
            "early_stopping_rounds": {"value": 50,     "type": "int",    "description": "Early stopping rounds"},
        },
    },
    "next_interval": {
        "model_type": "xgboost_reg",
        "tier1": {
            "n_estimators":     {"value": 400,  "type": "int",   "min": 50,   "max": 2000, "description": "Boosting rounds"},
            "max_depth":        {"value": 5,    "type": "int",   "min": 2,    "max": 12,   "description": "Max depth"},
            "learning_rate":    {"value": 0.05, "type": "float", "min": 0.001,"max": 0.3,  "description": "Learning rate"},
            "subsample":        {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Row sampling"},
            "colsample_bytree": {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Col sampling"},
        },
        "tier2": {
            "min_child_weight": {"value": 3,   "type": "int",   "min": 1, "max": 50,  "description": "Min child weight"},
            "gamma":            {"value": 0.0, "type": "float", "min": 0, "max": 5.0, "description": "Gamma"},
            "reg_alpha":        {"value": 0.0, "type": "float", "min": 0, "max": 10,  "description": "L1"},
            "reg_lambda":       {"value": 1.0, "type": "float", "min": 0, "max": 10,  "description": "L2"},
        },
        "tier3": {
            "n_jobs":                {"value": -1,     "type": "int",    "description": "Threads"},
            "random_state":          {"value": 42,     "type": "int",    "description": "Seed"},
            "eval_metric":           {"value": "rmse", "type": "string", "description": "Metric"},
            "early_stopping_rounds": {"value": 30,     "type": "int",    "description": "Early stop"},
        },
    },
    "kpi_prediction": {
        "model_type": "xgboost_reg",
        "tier1": {
            "n_estimators":     {"value": 400,  "type": "int",   "min": 50,   "max": 2000, "description": "Boosting rounds"},
            "max_depth":        {"value": 6,    "type": "int",   "min": 2,    "max": 12,   "description": "Max depth"},
            "learning_rate":    {"value": 0.05, "type": "float", "min": 0.001,"max": 0.3,  "description": "Learning rate"},
            "subsample":        {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Row sampling"},
            "colsample_bytree": {"value": 0.8,  "type": "float", "min": 0.5,  "max": 1.0,  "description": "Col sampling"},
        },
        "tier2": {
            "min_child_weight": {"value": 3,   "type": "int",   "min": 1, "max": 50,  "description": "Min child weight"},
            "gamma":            {"value": 0.0, "type": "float", "min": 0, "max": 5.0, "description": "Gamma"},
            "reg_alpha":        {"value": 0.0, "type": "float", "min": 0, "max": 10,  "description": "L1"},
            "reg_lambda":       {"value": 1.0, "type": "float", "min": 0, "max": 10,  "description": "L2"},
        },
        "tier3": {
            "n_jobs":                {"value": -1,     "type": "int",    "description": "Threads"},
            "random_state":          {"value": 42,     "type": "int",    "description": "Seed"},
            "eval_metric":           {"value": "rmse", "type": "string", "description": "Metric"},
            "early_stopping_rounds": {"value": 30,     "type": "int",    "description": "Early stop"},
        },
    },
    "anomaly_multivariate": {
        "model_type": "isolation_forest",
        "tier1": {
            "n_estimators":  {"value": 200,    "type": "int",   "min": 50,   "max": 1000, "description": "Number of isolation trees"},
            "contamination": {"value": 0.05,   "type": "float", "min": 0.01, "max": 0.5,  "description": "Expected anomaly proportion"},
            "max_samples":   {"value": "auto",  "type": "string","description": "Samples per tree. 'auto' = min(256, n)"},
        },
        "tier2": {
            "max_features": {"value": 1.0, "type": "float", "min": 0.1, "max": 1.0, "description": "Features per tree"},
            "random_state": {"value": 42,  "type": "int",   "description": "Seed"},
        },
        "tier3": {},
    },
}


def get_hparams(use_case: str) -> dict[str, Any]:
    """Return default hyperparameters for a use-case.

    Args:
        use_case: Key from the use-case registry.

    Returns:
        Dict with model_type, output_tag, requires_training, and tier1-3.

    Raises:
        MLPlatformError: If the use-case is unknown.
    """
    if use_case not in USE_CASE_REGISTRY:
        raise_error(ErrorCode.UNKNOWN_USE_CASE, f"Unknown use case: {use_case}", status_code=404)

    uc_info = USE_CASE_REGISTRY[use_case]

    if use_case not in HPARAM_REGISTRY:
        # Statistical use-cases have no trainable hparams
        return {
            "use_case": use_case,
            "model_type": uc_info["model_type"],
            "output_tag": uc_info["output_tag"],
            "requires_training": uc_info["requires_training"],
            "tier1": {},
            "tier2": {},
            "tier3": {},
        }

    entry = HPARAM_REGISTRY[use_case]
    return {
        "use_case": use_case,
        "model_type": uc_info["model_type"],
        "output_tag": uc_info["output_tag"],
        "requires_training": uc_info["requires_training"],
        "tier1": entry.get("tier1", {}),
        "tier2": entry.get("tier2", {}),
        "tier3": entry.get("tier3", {}),
    }


def merge_hparams(use_case: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge user overrides onto the defaults for a use-case.

    Args:
        use_case: Key from the use-case registry.
        overrides: User-provided hparam overrides.

    Returns:
        Flat dict of final hyperparameter values.
    """
    defaults = get_hparams(use_case)
    merged: dict[str, Any] = {}
    for tier in ("tier1", "tier2", "tier3"):
        for key, spec in defaults.get(tier, {}).items():
            merged[key] = spec["value"]
    merged.update(overrides)
    return merged
