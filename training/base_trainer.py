"""
Abstract base class for all trainers.

The ``run()`` orchestrator is shared; subclasses implement the abstract
methods.  When ``feature_selection`` is enabled in the job config, the
runner tries combinations of optional feature suffixes (keeping all
mandatory features fixed), evaluates each combination via k-fold CV
with full per-fold metric logging, and picks the best before doing
the final training run.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Max optional suffixes for exhaustive search (2^N).  Above this we
# fall back to greedy forward selection.
_EXHAUSTIVE_LIMIT = 10


class BaseTrainer(ABC):
    """Lifecycle runner for a single training job."""

    def __init__(self, job_config: dict[str, Any]) -> None:
        self.job_config = job_config
        self.model_id: str = job_config["model_id"]
        self.dataset_id: str = job_config["dataset_id"]
        self.log_path: str = job_config.get("log_path", f"logs/{self.model_id}.log")
        self.artifact_dir: str = job_config.get("artifact_dir", "artifacts")
        self.db_url: str = job_config.get("db_url", "sqlite:///./ml_platform.db")

    # ── Logging ────────────────────────────────────────────────────────
    def log(self, level: str, msg: str, data: dict | None = None) -> None:
        """Append a structured JSON log line to the log file."""
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "msg": msg,
            "data": data or {},
        }
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ── Abstract interface (existing) ──────────────────────────────────
    @abstractmethod
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
        """Return (X_train, X_test, y_train, y_test)."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series | None) -> Any:
        """Return a trained model object (may include internal CV)."""

    @abstractmethod
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series | None) -> dict:
        """Return metrics dict."""

    @abstractmethod
    def get_feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Return {feature_name: importance}."""

    # ── New abstract interface for feature selection ───────────────────
    @abstractmethod
    def quick_fit(self, X: pd.DataFrame, y: pd.Series | None) -> Any:
        """Fit a model quickly (no CV, no logging). Used during feature selection."""

    @abstractmethod
    def quick_score(self, model: Any, X: pd.DataFrame, y: pd.Series | None) -> dict[str, float]:
        """Score a fitted model on validation data. Returns {metric: value}."""

    @abstractmethod
    def primary_metric(self) -> tuple[str, bool]:
        """Return (metric_name, higher_is_better)."""

    # ── Feature selection ──────────────────────────────────────────────
    def _run_feature_selection(
        self,
        df: pd.DataFrame,
        y: pd.Series | None,
    ) -> list[str]:
        """Try combinations of optional feature groups and pick the best.

        Always keeps all mandatory columns.  Varies which optional suffix
        groups are included.  Evaluates each combination via k-fold CV.

        Returns:
            The best column list to use for final training.
        """
        mandatory_cols: list[str] = self.job_config.get("mandatory_columns", [])
        groups: dict[str, list[str]] = self.job_config.get("optional_feature_groups", {})
        cross_tag_cols: list[str] = self.job_config.get("cross_tag_columns", [])
        cv_folds: int = self.job_config.get("cv_folds", 5)

        base_cols = mandatory_cols + cross_tag_cols
        suffixes = sorted(groups.keys())
        n = len(suffixes)

        metric_name, higher_is_better = self.primary_metric()

        self.log("INFO", "Feature selection started", {
            "mandatory_features": len(mandatory_cols),
            "optional_groups": n,
            "suffixes": suffixes,
            "cv_folds": cv_folds,
            "metric": metric_name,
        })

        # ── Generate combinations ──────────────────────────────────────
        if n <= _EXHAUSTIVE_LIMIT:
            # Exhaustive: all 2^N subsets including empty
            all_combos: list[tuple[str, ...]] = [()]
            for r in range(1, n + 1):
                all_combos.extend(itertools.combinations(suffixes, r))
            self.log("INFO", f"Exhaustive search: {len(all_combos)} combinations")
        else:
            # Greedy forward selection
            all_combos = self._greedy_forward_combos(suffixes)
            self.log("INFO", f"Greedy forward selection: {len(all_combos)} steps")

        # ── Evaluate each combination ──────────────────────────────────
        results: list[dict[str, Any]] = []

        for combo_idx, combo in enumerate(all_combos):
            combo_label = list(combo) if combo else ["mandatory_only"]

            # Build column set for this combination
            combo_cols = list(base_cols)
            for suffix in combo:
                combo_cols.extend(groups[suffix])

            # Filter to columns that actually exist in df
            available = [c for c in combo_cols if c in df.columns]
            X_sub = df[available]

            # K-fold CV
            from sklearn.model_selection import KFold, StratifiedKFold
            if y is not None and self._is_classifier():
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=False)
                split_iter = kf.split(X_sub, y)
            else:
                kf = KFold(n_splits=cv_folds, shuffle=False)
                split_iter = kf.split(X_sub)

            fold_metrics: list[dict[str, float]] = []

            for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
                Xf = X_sub.iloc[train_idx]
                Xv = X_sub.iloc[val_idx]
                yf = y.iloc[train_idx] if y is not None else None
                yv = y.iloc[val_idx] if y is not None else None

                model = self.quick_fit(Xf, yf)
                scores = self.quick_score(model, Xv, yv)
                fold_metrics.append(scores)

                self.log("INFO", f"FeatureSelection combo {combo_idx+1}/{len(all_combos)} fold {fold_idx+1}/{cv_folds}", {
                    "combo": combo_label,
                    "n_features": len(available),
                    **{k: round(v, 6) for k, v in scores.items()},
                })

            # Compute mean of primary metric
            mean_val = float(np.mean([m[metric_name] for m in fold_metrics]))
            std_val = float(np.std([m[metric_name] for m in fold_metrics]))

            # Compute means for all metrics
            all_metric_means = {}
            for key in fold_metrics[0]:
                all_metric_means[f"mean_{key}"] = round(float(np.mean([m[key] for m in fold_metrics])), 6)
                all_metric_means[f"std_{key}"] = round(float(np.std([m[key] for m in fold_metrics])), 6)

            self.log("INFO", f"FeatureSelection combo {combo_idx+1}/{len(all_combos)} summary", {
                "combo": combo_label,
                "n_features": len(available),
                f"mean_{metric_name}": round(mean_val, 6),
                f"std_{metric_name}": round(std_val, 6),
                **all_metric_means,
            })

            results.append({
                "combo": combo,
                "combo_label": combo_label,
                "columns": available,
                "mean_primary": mean_val,
                "std_primary": std_val,
                "all_means": all_metric_means,
            })

        # ── Pick the best ──────────────────────────────────────────────
        if higher_is_better:
            best = max(results, key=lambda r: r["mean_primary"])
        else:
            best = min(results, key=lambda r: r["mean_primary"])

        # Rank all combinations
        ranked = sorted(results, key=lambda r: r["mean_primary"], reverse=higher_is_better)
        ranking_table = []
        for rank, r in enumerate(ranked, 1):
            ranking_table.append({
                "rank": rank,
                "combo": r["combo_label"],
                "n_features": len(r["columns"]),
                f"mean_{metric_name}": round(r["mean_primary"], 6),
                f"std_{metric_name}": round(r["std_primary"], 6),
            })

        self.log("INFO", "Feature selection complete", {
            "best_combo": best["combo_label"],
            "best_n_features": len(best["columns"]),
            f"best_mean_{metric_name}": round(best["mean_primary"], 6),
            "total_combos_evaluated": len(results),
            "ranking": ranking_table,
        })

        return best["columns"]

    def _greedy_forward_combos(self, suffixes: list[str]) -> list[tuple[str, ...]]:
        """Generate greedy forward-selection combos for large N."""
        combos: list[tuple[str, ...]] = [()]  # baseline
        for suffix in suffixes:
            combos.append((suffix,))  # each individual
        return combos

    def _is_classifier(self) -> bool:
        """Check if this is a classification use case."""
        model_type = self.job_config.get("model_type", "")
        return model_type in ("xgboost_clf",)

    # ── DB update helper ───────────────────────────────────────────────
    def _update_db(self, updates: dict[str, Any]) -> None:
        """Update the model_artifact row in the database."""
        engine = create_engine(self.db_url, connect_args={"check_same_thread": False})
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            from models.dataset import DatasetRecord  # noqa: F401
            from models.model_artifact import ModelArtifact
            record = session.query(ModelArtifact).filter(ModelArtifact.id == self.model_id).first()
            if record:
                for k, v in updates.items():
                    setattr(record, k, v)
                session.commit()
        finally:
            session.close()

    # ── Orchestrator ───────────────────────────────────────────────────
    def run(self) -> None:
        """Full lifecycle: [feature selection] → load → train → evaluate → save."""
        start = time.time()
        try:
            self.log("INFO", "Training started", {"use_case": self.job_config.get("use_case")})

            # ── Load full data ─────────────────────────────────────────
            X_train, X_test, y_train, y_test = self.load_data()
            self.log("INFO", "Dataset loaded", {
                "train_rows": len(X_train), "test_rows": len(X_test), "cols": X_train.shape[1],
            })

            # ── Feature selection (if enabled + optional groups exist) ─
            groups = self.job_config.get("optional_feature_groups", {})
            do_fs = self.job_config.get("feature_selection", False) and len(groups) > 0

            if do_fs:
                best_cols = self._run_feature_selection(X_train, y_train)
                # Re-select columns for train and test
                X_train = X_train[[c for c in best_cols if c in X_train.columns]]
                X_test = X_test[[c for c in best_cols if c in X_test.columns]]
                self.log("INFO", "Using best feature set from selection", {
                    "n_features": X_train.shape[1],
                })
            else:
                self.log("INFO", "Feature selection disabled or no optional groups, using all selected features")

            # ── Train (with internal CV + final fit) ───────────────────
            model = self.train(X_train, y_train)
            self.log("INFO", "Model training complete")

            # ── Evaluate on held-out test set ──────────────────────────
            metrics = self.evaluate(model, X_test, y_test)
            self.log("INFO", "Evaluation complete", {"metrics": metrics})

            feature_names = list(X_train.columns)
            importance = self.get_feature_importance(model, feature_names)

            # ── Save model ─────────────────────────────────────────────
            base_filename = self.model_id
            if self.job_config.get("model_name"):
                clean_name = "".join(c for c in self.job_config["model_name"] if c not in r'/\:*?"<>|')
                if clean_name.strip():
                    base_filename = clean_name.strip()

            artifact_path = os.path.join(self.artifact_dir, f"{base_filename}.pkl")
            os.makedirs(self.artifact_dir, exist_ok=True)
            joblib.dump(model, artifact_path)

            meta_path = os.path.join(self.artifact_dir, f"{base_filename}_meta.json")
            meta = {
                "model_id": self.model_id,
                "model_name": self.job_config.get("model_name"),
                "use_case": self.job_config.get("use_case"),
                "hparams": self.job_config.get("hparams_merged", {}),
                "metrics": metrics,
                "feature_importance": importance,
                "columns": feature_names,
                "tags_used": self.job_config.get("tags", []),
                "range_metadata": self.job_config.get("range_metadata", {}),
                "feature_selection_used": do_fs,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            self.log("INFO", "Artifact saved", {"path": artifact_path})

            duration = time.time() - start

            result_payload = {
                "model_id": self.model_id,
                "model_name": self.job_config.get("model_name"),
                "status": "completed",
                "use_case": self.job_config.get("use_case", ""),
                "output_tag": self.job_config.get("output_tag", ""),
                "metrics": metrics,
                "feature_importance": importance,
                "training_duration_seconds": round(duration, 2),
                "artifact_path": artifact_path,
            }
            result_path = os.path.join("logs", f"{self.model_id}_result.json")
            with open(result_path, "w") as f:
                json.dump(result_payload, f, default=str)

            self.log("INFO", "Training complete", {"duration_seconds": round(duration, 2)})

            self._update_db({
                "status": "completed",
                "metrics": metrics,
                "feature_importance": importance,
                "artifact_path": artifact_path,
                "metadata_path": meta_path,
                "training_duration_seconds": round(duration, 2),
                "completed_at": datetime.utcnow(),
                "optional_features_used": list(
                    self.job_config.get("_best_optional_suffixes", self.job_config.get("optional_feature_groups", {}).keys())
                ),
            })

        except Exception as exc:
            duration = time.time() - start
            tb = traceback.format_exc()
            self.log("ERROR", f"Training failed: {exc}", {"traceback": tb})

            result_payload = {
                "model_id": self.model_id,
                "model_name": self.job_config.get("model_name"),
                "status": "failed",
                "use_case": self.job_config.get("use_case", ""),
                "output_tag": self.job_config.get("output_tag", ""),
                "error": str(exc),
                "training_duration_seconds": round(duration, 2),
                "artifact_path": "",
            }
            result_path = os.path.join("logs", f"{self.model_id}_result.json")
            with open(result_path, "w") as f:
                json.dump(result_payload, f, default=str)

            self._update_db({
                "status": "failed",
                "error_message": str(exc),
                "training_duration_seconds": round(duration, 2),
                "completed_at": datetime.utcnow(),
            })
            raise
