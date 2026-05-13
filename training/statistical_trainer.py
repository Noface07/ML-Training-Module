"""
Statistical / rule-based trainer.

Used for: anomaly_univariate, adaptive_threshold, early_warning,
health_index, drift_detection, pattern_detection, data_quality.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from training.base_trainer import BaseTrainer


class StatisticalTrainer(BaseTrainer):
    """Statistical / rule-based trainer (no real model training)."""

    # ── Feature selection interface ────────────────────────────────────

    def quick_fit(self, X: pd.DataFrame, y: Any) -> Any:
        return {"means": X.mean().to_dict(), "stds": X.std().to_dict()}

    def quick_score(self, model: dict, X: pd.DataFrame, y: Any) -> dict[str, float]:
        means, stds = model["means"], model["stds"]
        z_vals = []
        for col in X.columns:
            if col in means and stds.get(col, 0) > 0:
                z_vals.extend(np.abs((X[col] - means[col]) / stds[col]).tolist())
        mean_z = float(np.mean(z_vals)) if z_vals else 0.0
        pct_above_3 = float(np.mean([z > 3 for z in z_vals])) if z_vals else 0.0
        return {"mean_z": round(mean_z, 6), "pct_above_3": round(pct_above_3, 6)}

    def primary_metric(self) -> tuple[str, bool]:
        return ("mean_z", False)  # lower mean-z = better fit

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
        stats = {
            "means": X_train.mean().to_dict(),
            "stds": X_train.std().to_dict(),
            "mins": X_train.min().to_dict(),
            "maxs": X_train.max().to_dict(),
            "medians": X_train.median().to_dict(),
        }
        self.log("INFO", "Statistical baselines computed", {"n_features": len(X_train.columns)})
        return stats

    def evaluate(self, model: dict, X_test: pd.DataFrame, y_test: Any) -> dict:
        means, stds = model["means"], model["stds"]
        z_scores = {}
        for col in X_test.columns:
            if col in means and stds.get(col, 0) > 0:
                z = np.abs((X_test[col] - means[col]) / stds[col])
                z_scores[col] = {
                    "mean_z": round(float(z.mean()), 4),
                    "max_z": round(float(z.max()), 4),
                    "pct_above_2": round(float((z > 2).mean()), 4),
                    "pct_above_3": round(float((z > 3).mean()), 4),
                }
        return {"type": "statistical", "n_features": len(X_test.columns), "z_score_summary": z_scores}

    def get_feature_importance(self, model: dict, feature_names: list[str]) -> dict[str, float]:
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args()
    with open(args.job_config) as f:
        config = json.load(f)
    StatisticalTrainer(config).run()
