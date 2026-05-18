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
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
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
            best_iterations = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                Xf, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                yf, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
                
                if hasattr(model, "best_iteration") and model.best_iteration:
                    best_iterations.append(model.best_iteration)
                    
                pred = model.predict(Xv)
                fold_m = regression_metrics(yv.values, pred)
                cv_scores.append(fold_m["rmse"])
                self.log("INFO", f"Final CV fold {fold_idx+1}/{cv_folds}", {
                    k: round(float(v), 6) if v is not None else None for k, v in fold_m.items()
                })
            self.job_config["_cv_scores"] = cv_scores
            
            avg_best = int(np.mean(best_iterations)) if best_iterations else hparams.get("n_estimators", 100)
            model = XGBRegressor(**{**hparams, "n_estimators": avg_best}, eval_metric="rmse")

        model.fit(X_train, y_train, verbose=False)

        # Store training data reference for train-set evaluation in evaluate()
        self._X_train = X_train
        self._y_train = y_train
        self._train_samples = len(X_train)

        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute regression metrics on test set, plus train metrics for overfitting gap."""
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test.values, y_pred)

        # ── CV statistics ─────────────────────────────────────────────
        cv_scores = self.job_config.get("_cv_scores", [])
        if cv_scores:
            metrics["cv_rmse_mean"] = float(np.mean(cv_scores))
            metrics["cv_rmse_std"] = float(np.std(cv_scores))

        # ── Dataset stats ─────────────────────────────────────────────
        metrics["dataset_stats"] = {
            "train_samples": getattr(self, "_train_samples", None),
            "test_samples": len(X_test),
            "target_unique_values": len(np.unique(self._y_train)) if hasattr(self, "_y_train") else None,
        }

        # ── Test metrics snapshot (explicit) ──────────────────────────
        metrics["test_metrics"] = {
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        }

        # ── Train metrics (for overfitting gap) ───────────────────────
        X_tr = getattr(self, "_X_train", None)
        y_tr = getattr(self, "_y_train", None)
        train_metrics: dict[str, float | None] = {"r2": None, "rmse": None, "mae": None}
        if X_tr is not None and y_tr is not None:
            try:
                tr_pred = model.predict(X_tr)
                tr_m = regression_metrics(y_tr.values, tr_pred)
                train_metrics = {
                    "r2":   round(float(tr_m["r2"]),  6) if tr_m.get("r2")  is not None else None,
                    "rmse": round(float(tr_m["rmse"]), 6) if tr_m.get("rmse") is not None else None,
                    "mae":  round(float(tr_m["mae"]),  6) if tr_m.get("mae")  is not None else None,
                }
            except Exception:
                pass
        metrics["train_metrics"] = train_metrics
        
        # ── Generalization metrics ────────────────────────────────────
        test_r2 = metrics["test_metrics"].get("r2")
        train_r2 = train_metrics.get("r2")
        
        gen_metrics = {
            "overfit_gap_r2": None,
            "is_overfit": False,
            "is_underfit": False,
            "has_generalization_failure": False,
        }
        
        if test_r2 is not None and train_r2 is not None:
            gap = round(float(train_r2 - test_r2), 6)
            gen_metrics["overfit_gap_r2"] = gap
            gen_metrics["is_overfit"] = gap > 0.2
            gen_metrics["is_underfit"] = test_r2 < 0.0
            gen_metrics["has_generalization_failure"] = gap > 0.5
            
        metrics["generalization_metrics"] = gen_metrics

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
