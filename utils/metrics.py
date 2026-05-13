"""
Evaluation metric helpers for each model family.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    """Compute metrics for a binary classifier.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary of metric name → value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = None
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute metrics for a regressor.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric name → value.
    """
    metrics: dict[str, Any] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    # MAPE — guard against zero division
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        metrics["mape"] = float(
            mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
        )
    else:
        metrics["mape"] = None
    return metrics


def isolation_forest_metrics(scores: np.ndarray, labels: np.ndarray, contamination: float) -> dict[str, Any]:
    """Compute metrics for an Isolation Forest.

    Args:
        scores: Anomaly decision scores (lower = more anomalous).
        labels: Predicted labels (+1 = normal, -1 = anomaly).
        contamination: Contamination parameter used.

    Returns:
        Dictionary of metric name → value.
    """
    anomaly_count = int((labels == -1).sum())
    total = len(labels)
    return {
        "contamination_used": float(contamination),
        "anomaly_count": anomaly_count,
        "anomaly_pct": float(anomaly_count / total) if total else 0.0,
        "score_distribution": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "p5": float(np.percentile(scores, 5)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p95": float(np.percentile(scores, 95)),
        },
    }
