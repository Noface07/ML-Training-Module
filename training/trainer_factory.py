"""
Trainer factory — routes model_type → trainer class.
"""

from __future__ import annotations

import importlib
from typing import Type

from training.base_trainer import BaseTrainer

TRAINER_MAP: dict[str, str] = {
    "xgboost_clf":      "training.xgboost_clf_trainer.XGBoostClfTrainer",
    "xgboost_reg":      "training.xgboost_reg_trainer.XGBoostRegTrainer",
    "isolation_forest":  "training.isolation_forest_trainer.IsolationForestTrainer",
    "statistical":       "training.statistical_trainer.StatisticalTrainer",
}


def get_trainer_class(model_type: str) -> Type[BaseTrainer]:
    """Dynamically import and return the trainer class for a model type.

    Args:
        model_type: Key from TRAINER_MAP (e.g. ``"xgboost_clf"``).

    Returns:
        The trainer class (not an instance).

    Raises:
        ValueError: If model_type is not in TRAINER_MAP.
    """
    if model_type not in TRAINER_MAP:
        raise ValueError(f"Unknown model_type: {model_type}. Available: {list(TRAINER_MAP)}")

    dotted_path = TRAINER_MAP[model_type]
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
