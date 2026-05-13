"""
XGBoost binary-classification trainer.

Used for: failure_prediction, risk_scoring.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from training.base_trainer import BaseTrainer
from utils.metrics import classification_metrics


class XGBoostClfTrainer(BaseTrainer):
    """XGBoost classifier trainer."""

    def _get_xgb_hparams(self) -> dict[str, Any]:
        """Extract and clean hparams for XGBClassifier."""
        hparams = dict(self.job_config.get("hparams_merged", {}))
        hparams.pop("early_stopping_rounds", None)
        hparams.pop("eval_metric", None)
        return hparams

    # ── Feature selection interface ────────────────────────────────────

    def quick_fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit XGBoost quickly without CV or logging."""
        hparams = self._get_xgb_hparams()
        if hparams.get("scale_pos_weight") == "auto":
            pos = y.sum()
            hparams["scale_pos_weight"] = float((len(y) - pos) / pos) if pos > 0 else 1.0
        model = XGBClassifier(**hparams, eval_metric="auc", use_label_encoder=False)
        model.fit(X, y, verbose=False)
        return model

    def quick_score(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Score on validation data — returns all classification metrics."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, y_prob)
        except ValueError:
            auc = 0.0
        return {
            "auc": round(float(auc), 6),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 6),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 6),
        }

    def primary_metric(self) -> tuple[str, bool]:
        """AUC — higher is better."""
        return ("auc", True)

    # ── Standard trainer interface ─────────────────────────────────────

    def load_data(self):
        """Load parquet, select columns, time-ordered split (no shuffle)."""
        cfg = self.job_config
        df = pd.read_parquet(cfg["dataset_path"])
        feature_cols = cfg["feature_columns"]
        target_col = cfg["target_col"]
        # Only select columns that exist
        available = [c for c in feature_cols if c in df.columns]
        X = df[available]
        y = df[target_col]
        split_idx = int(len(df) * cfg.get("train_split", 0.8))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train XGBoost classifier with CV + final fit."""
        hparams = self._get_xgb_hparams()
        cv_folds = self.job_config.get("cv_folds", 5)

        if hparams.get("scale_pos_weight") == "auto":
            pos = y_train.sum()
            hparams["scale_pos_weight"] = float((len(y_train) - pos) / pos) if pos > 0 else 1.0

        model = XGBClassifier(**hparams, eval_metric="auc", use_label_encoder=False)

        # Cross-validation with per-fold logging
        if cv_folds > 1:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                Xf, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                yf, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)

                y_pred = model.predict(Xv)
                y_prob = model.predict_proba(Xv)[:, 1]
                try:
                    auc = roc_auc_score(yv, y_prob)
                except ValueError:
                    auc = 0.0

                fold_metrics = {
                    "auc": round(float(auc), 6),
                    "f1": round(float(f1_score(yv, y_pred, zero_division=0)), 6),
                    "precision": round(float(precision_score(yv, y_pred, zero_division=0)), 6),
                    "recall": round(float(recall_score(yv, y_pred, zero_division=0)), 6),
                }
                cv_scores.append(auc)
                self.log("INFO", f"Final CV fold {fold_idx+1}/{cv_folds}", fold_metrics)

            self.job_config["_cv_scores"] = cv_scores

        # Final fit on full training data
        model.fit(X_train, y_train, verbose=False)
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute classification metrics on test set."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test.values, y_pred, y_prob)

        cv_scores = self.job_config.get("_cv_scores", [])
        if cv_scores:
            metrics["cv_auc_mean"] = float(np.mean(cv_scores))
            metrics["cv_auc_std"] = float(np.std(cv_scores))

        metrics["class_distribution"] = {
            "test_pos": int(y_test.sum()),
            "test_neg": int(len(y_test) - y_test.sum()),
        }
        return metrics

    def get_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importances from the XGBoost model."""
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: round(float(imp), 6) for name, imp in pairs[:50]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args()
    with open(args.job_config) as f:
        config = json.load(f)
    XGBoostClfTrainer(config).run()
