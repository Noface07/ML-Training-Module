"""
XGBoost regression trainer.

Used for: rul, next_interval, kpi_prediction.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from training.base_trainer import BaseTrainer
from utils.metrics import regression_metrics


class XGBoostRegTrainer(BaseTrainer):
    """XGBoost regressor trainer."""

    def _get_xgb_hparams(self) -> dict[str, Any]:
        hparams = dict(self.job_config.get("hparams_merged", {}))
        hparams.pop("early_stopping_rounds", None)
        hparams.pop("eval_metric", None)
        hparams.pop("scale_pos_weight", None)
        return hparams

    # ── Feature selection interface ────────────────────────────────────

    def quick_fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        hparams = self._get_xgb_hparams()
        model = XGBRegressor(**hparams, eval_metric="rmse")
        model.fit(X, y, verbose=False)
        return model

    def quick_score(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        y_pred = model.predict(X)
        m = regression_metrics(y.values, y_pred)
        return {k: round(float(v), 6) if v is not None else 0.0 for k, v in m.items()}

    def primary_metric(self) -> tuple[str, bool]:
        return ("rmse", False)  # lower is better

    # ── Standard trainer interface ─────────────────────────────────────

    def load_data(self):
        cfg = self.job_config
        df = pd.read_parquet(cfg["dataset_path"])
        feature_cols = cfg["feature_columns"]
        target_col = cfg["target_col"]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available]
        y = df[target_col]
        split_idx = int(len(df) * cfg.get("train_split", 0.8))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        hparams = self._get_xgb_hparams()
        cv_folds = self.job_config.get("cv_folds", 5)
        model = XGBRegressor(**hparams, eval_metric="rmse")

        if cv_folds > 1:
            kf = KFold(n_splits=cv_folds, shuffle=False)
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                Xf, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                yf, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
                pred = model.predict(Xv)
                fold_m = regression_metrics(yv.values, pred)
                cv_scores.append(fold_m["rmse"])
                self.log("INFO", f"Final CV fold {fold_idx+1}/{cv_folds}", {
                    k: round(float(v), 6) if v is not None else None for k, v in fold_m.items()
                })
            self.job_config["_cv_scores"] = cv_scores

        model.fit(X_train, y_train, verbose=False)
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test.values, y_pred)
        cv_scores = self.job_config.get("_cv_scores", [])
        if cv_scores:
            metrics["cv_rmse_mean"] = float(np.mean(cv_scores))
            metrics["cv_rmse_std"] = float(np.std(cv_scores))
        return metrics

    def get_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: round(float(imp), 6) for name, imp in pairs[:50]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args()
    with open(args.job_config) as f:
        config = json.load(f)
    XGBoostRegTrainer(config).run()
