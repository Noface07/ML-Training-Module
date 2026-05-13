"""
Isolation Forest trainer.

Used for: anomaly_multivariate.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from training.base_trainer import BaseTrainer
from utils.metrics import isolation_forest_metrics


class IsolationForestTrainer(BaseTrainer):
    """Isolation Forest anomaly-detection trainer."""

    def _get_if_hparams(self) -> dict[str, Any]:
        hparams = dict(self.job_config.get("hparams_merged", {}))
        for key in ("early_stopping_rounds", "eval_metric", "scale_pos_weight",
                     "n_jobs", "colsample_bytree", "subsample", "learning_rate",
                     "max_depth", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"):
            hparams.pop(key, None)
        return hparams

    # ── Feature selection interface ────────────────────────────────────

    def quick_fit(self, X: pd.DataFrame, y: Any) -> Any:
        model = IsolationForest(**self._get_if_hparams())
        model.fit(X)
        return model

    def quick_score(self, model: Any, X: pd.DataFrame, y: Any) -> dict[str, float]:
        scores = model.decision_function(X)
        labels = model.predict(X)
        anomaly_pct = float((labels == -1).sum() / len(labels)) if len(labels) else 0.0
        return {
            "anomaly_pct": round(anomaly_pct, 6),
            "score_mean": round(float(np.mean(scores)), 6),
            "score_std": round(float(np.std(scores)), 6),
        }

    def primary_metric(self) -> tuple[str, bool]:
        return ("score_std", True)  # higher variance = better separation

    # ── Standard trainer interface ─────────────────────────────────────

    def load_data(self):
        cfg = self.job_config
        df = pd.read_parquet(cfg["dataset_path"])
        feature_cols = cfg["feature_columns"]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available]
        split_idx = int(len(df) * cfg.get("train_split", 0.8))
        return X.iloc[:split_idx], X.iloc[split_idx:], None, None

    def train(self, X_train: pd.DataFrame, y_train: Any) -> Any:
        model = IsolationForest(**self._get_if_hparams())
        model.fit(X_train)
        self.log("INFO", "Isolation Forest fitted", {"n_samples": len(X_train)})
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: Any) -> dict:
        scores = model.decision_function(X_test)
        labels = model.predict(X_test)
        contamination = self.job_config.get("hparams_merged", {}).get("contamination", 0.05)
        return isolation_forest_metrics(scores, labels, contamination)

    def get_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        return {name: 0.0 for name in feature_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args()
    with open(args.job_config) as f:
        config = json.load(f)
    IsolationForestTrainer(config).run()
