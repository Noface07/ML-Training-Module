"""
Production-grade important metrics and ranking engine for the /v1/models API.

Injects an ``imp`` sub-object into each model's ``metrics`` dict containing:
  - The 3 most important evaluation metrics (primary / secondary / tertiary)
  - A ``ranking`` sub-object for within-family leaderboards
  - A ``deployment_health`` sub-object for operational risk assessment
  - A ``model_fit_analysis`` for underfit/overfit/generalization states
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

def normalize_positive_metric(value: float | None, lo: float = 0.0, hi: float = 1.0) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    span = hi - lo
    if span <= 0:
        return 0.0
    return clamp_score((value - lo) / span * 100.0)


def normalize_negative_metric(value: float | None, lo: float = 0.0, hi: float = 1.0) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    span = hi - lo
    if span <= 0:
        return 0.0
    return clamp_score((1.0 - (value - lo) / span) * 100.0)


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def apply_exponential_penalty(score: float, penalty_strength: float) -> float:
    """Apply an exponential decay penalty to a score."""
    if penalty_strength <= 0:
        return score
    return score * math.exp(-penalty_strength)


# ---------------------------------------------------------------------------
# Internal safe-get & Ranking Scope
# ---------------------------------------------------------------------------

def _safe(metrics: dict[str, Any], key: str) -> float | None:
    val = metrics.get(key)
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    return fval if math.isfinite(fval) else None


def _round6(v: float | None) -> float | None:
    return None if v is None else round(v, 6)


def resolve_ranking_group(model_type: str) -> str:
    """Determine the ranking group for a given model type."""
    if model_type in ("xgboost_clf", "lightgbm_clf", "catboost_clf", "random_forest_clf"):
        return "classification"
    if model_type in ("xgboost_reg", "linear_reg", "random_forest_reg"):
        return "regression"
    if model_type in ("isolation_forest", "one_class_svm", "autoencoder", "lof"):
        return "anomaly_detection"
    return "unknown"


# ---------------------------------------------------------------------------
# Anomaly separation (derived from score_distribution)
# ---------------------------------------------------------------------------

def compute_anomaly_separation_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    dist = metrics.get("score_distribution")
    if not isinstance(dist, dict):
        return {"p50_p5_gap": None, "score_std": None, "p95_p50_gap": None}

    p5  = _safe(dist, "p5")
    p50 = _safe(dist, "p50")
    p95 = _safe(dist, "p95")
    std = _safe(dist, "std")

    return {
        "p50_p5_gap":  _round6(p50 - p5)   if (p50 is not None and p5  is not None) else None,
        "score_std":   _round6(std),
        "p95_p50_gap": _round6(p95 - p50)  if (p95 is not None and p50 is not None) else None,
    }


# ---------------------------------------------------------------------------
# Overfitting / generalization analysis
# ---------------------------------------------------------------------------

def analyze_model_fit(metrics: dict[str, Any], model_type: str) -> dict[str, bool]:
    """Analyze fit state: underfit, overfit, generalization failure."""
    train = metrics.get("train_metrics") or {}
    test  = metrics.get("test_metrics")  or {}
    
    result = {
        "is_underfit": False,
        "is_overfit": False,
        "has_generalization_failure": False,
    }

    if model_type == "xgboost_clf":
        test_auc = _safe(test, "auc_roc") if test else _safe(metrics, "auc_roc")
        train_auc = _safe(train, "auc_roc")
        
        # Underfit: poor on both
        if test_auc is not None and test_auc < 0.65 and (train_auc is None or train_auc < 0.65):
            result["is_underfit"] = True
            
        gap = None
        if train_auc is not None and test_auc is not None:
            gap = train_auc - test_auc
            
        # Overfit: good on train, poor on test, large gap
        if gap is not None and gap > 0.10 and train_auc is not None and train_auc > 0.8:
            result["is_overfit"] = True
            
        # Gen failure: extreme gap
        if gap is not None and gap > 0.15:
            result["has_generalization_failure"] = True
            
    elif model_type == "xgboost_reg":
        test_r2  = _safe(test, "r2")  if test else _safe(metrics, "r2")
        train_r2 = _safe(train, "r2")
        
        # Underfit: poor on both
        if test_r2 is not None and test_r2 < 0.20 and (train_r2 is None or train_r2 < 0.20):
            result["is_underfit"] = True
            
        gap = None
        if train_r2 is not None and test_r2 is not None:
            gap = train_r2 - test_r2
            
        # Overfit
        if gap is not None and gap > 0.15 and train_r2 is not None and train_r2 > 0.60:
            result["is_overfit"] = True
            
        # Gen failure
        if gap is not None and gap > 0.25:
            result["has_generalization_failure"] = True

    elif model_type == "isolation_forest":
        sep = compute_anomaly_separation_metrics(metrics)
        p50_p5_gap = sep["p50_p5_gap"]
        score_std  = sep["score_std"]
        rob = metrics.get("robustness_metrics") or {}
        drift = _safe(rob, "train_test_score_drift")

        if p50_p5_gap is not None and p50_p5_gap < 0.005 and score_std is not None and score_std < 0.005:
            result["is_underfit"] = True
            
        if drift is not None and drift > 0.10:
            result["is_overfit"] = True
            
        if drift is not None and drift > 0.15:
            result["has_generalization_failure"] = True

    return result

def compute_overfit_metrics(
    metrics: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    """Retained for backward compatibility. Computes gaps."""
    train = metrics.get("train_metrics") or {}
    test  = metrics.get("test_metrics")  or {}
    result: dict[str, Any] = {}

    if model_type == "xgboost_clf":
        test_auc = _safe(test, "auc_roc") if test else _safe(metrics, "auc_roc")
        test_f1  = _safe(test, "f1")      if test else _safe(metrics, "f1")
        train_auc = _safe(train, "auc_roc")
        train_f1  = _safe(train, "f1")

        gap_auc = _round6(train_auc - test_auc) if (train_auc is not None and test_auc is not None) else None
        gap_f1  = _round6(train_f1  - test_f1)  if (train_f1  is not None and test_f1  is not None) else None

        result["overfit_gap_auc"] = gap_auc
        result["overfit_gap_f1"]  = gap_f1

    elif model_type == "xgboost_reg":
        test_r2  = _safe(test, "r2")  if test else _safe(metrics, "r2")
        train_r2 = _safe(train, "r2")

        gap_r2 = _round6(train_r2 - test_r2) if (train_r2 is not None and test_r2 is not None) else None
        result["overfit_gap_r2"] = gap_r2

    fit_state = analyze_model_fit(metrics, model_type)
    result["is_underfit"] = fit_state["is_underfit"]
    
    return result


# ---------------------------------------------------------------------------
# Calibration quality score
# ---------------------------------------------------------------------------

def compute_calibration_score(calibration_metrics: dict[str, Any] | None) -> float:
    if not isinstance(calibration_metrics, dict):
        return 50.0
    ll = _safe(calibration_metrics, "log_loss")
    bs = _safe(calibration_metrics, "brier_score")
    scores = []
    if ll is not None:
        scores.append(normalize_negative_metric(ll, lo=0.0, hi=1.5))
    if bs is not None:
        scores.append(normalize_negative_metric(bs, lo=0.0, hi=0.25))
    if not scores:
        return 50.0
    return clamp_score(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Hard Penalties
# ---------------------------------------------------------------------------

def compute_hard_penalties(base_score: float, metrics: dict[str, Any], model_type: str) -> float:
    """Stack exponential penalties for overfitting, instability, and underfitting."""
    stats = metrics.get("dataset_stats") or {}
    train_samples = _safe(stats, "train_samples") or 1000.0
    
    size_discount = 1.0
    if train_samples < 500:
        size_discount = 0.5
        
    fit = analyze_model_fit(metrics, model_type)
    score = base_score
    
    if fit.get("is_underfit"):
        score = apply_exponential_penalty(score, 0.35)
        
    gen = compute_overfit_metrics(metrics, model_type)
        
    if model_type == "xgboost_clf":
        gap = gen.get("overfit_gap_auc")
        cv_std = _safe(metrics, "cv_auc_std")
        test_auc = _safe(metrics.get("test_metrics", {}), "auc_roc") or _safe(metrics, "auc_roc")
        
        if test_auc is not None and test_auc < 0.60:
            score = apply_exponential_penalty(score, 0.51)
            
        if gap is not None and gap > 0.15:
            penalty = (gap - 0.10) * 5.0
            score = apply_exponential_penalty(score, penalty)
            
        if cv_std is not None and cv_std > 0.05:
            penalty = (cv_std - 0.05) * 5.0 * size_discount
            score = apply_exponential_penalty(score, penalty)
            
    elif model_type == "xgboost_reg":
        gap = gen.get("overfit_gap_r2")
        cv_std = _safe(metrics, "cv_rmse_std")
        cv_mean = _safe(metrics, "cv_rmse_mean")
        test_r2 = _safe(metrics.get("test_metrics", {}), "r2") or _safe(metrics, "r2")
        
        if test_r2 is not None and test_r2 < 0.15:
            score = apply_exponential_penalty(score, 0.60)
            
        if gap is not None and gap > 0.15:
            penalty = (gap - 0.10) * 3.0
            score = apply_exponential_penalty(score, penalty)
            
        if cv_std is not None and cv_mean is not None and cv_mean > 0:
            cv_ratio = cv_std / cv_mean
            if cv_ratio > 0.10:
                penalty = (cv_ratio - 0.10) * 2.0 * size_discount
                score = apply_exponential_penalty(score, penalty)
                
    elif model_type == "isolation_forest":
        rob = metrics.get("robustness_metrics") or {}
        drift = _safe(rob, "train_test_score_drift")
        
        if drift is not None and drift > 0.05:
            penalty = (drift - 0.05) * 20.0
            score = apply_exponential_penalty(score, penalty)
            
    return score


# ---------------------------------------------------------------------------
# Generalization (overfit penalty) score component
# ---------------------------------------------------------------------------

def compute_generalization_score(model_type: str, metrics: dict[str, Any]) -> float:
    gen = compute_overfit_metrics(metrics, model_type)
    if model_type == "xgboost_clf":
        gap = gen.get("overfit_gap_auc")
        if gap is None:
            return 70.0
        return normalize_negative_metric(max(0.0, gap), lo=0.0, hi=0.5)
    elif model_type == "xgboost_reg":
        gap = gen.get("overfit_gap_r2")
        if gap is None:
            return 70.0
        return normalize_negative_metric(max(0.0, gap), lo=0.0, hi=0.5)
    elif model_type == "isolation_forest":
        rob = metrics.get("robustness_metrics") or {}
        drift = _safe(rob, "train_test_score_drift")
        if drift is None:
            return 70.0
        return normalize_negative_metric(drift, lo=0.0, hi=0.10)
    return 70.0


def _contamination_consistency_score(metrics: dict[str, Any]) -> float:
    target = _safe(metrics, "contamination_used")
    actual = _safe(metrics, "anomaly_pct")
    if target is None or actual is None:
        return 50.0
    diff = abs(actual - target)
    return normalize_negative_metric(diff, lo=0.0, hi=0.20)


# ---------------------------------------------------------------------------
# Confidence Level & Deployment Readiness
# ---------------------------------------------------------------------------

def compute_deployment_readiness(score: float) -> str:
    if score >= 85: return "production_ready"
    if score >= 70: return "good"
    if score >= 50: return "experimental"
    if score >= 30: return "weak"
    return "unsafe"


def compute_confidence_level(model_type: str, metrics: dict[str, Any], final_score: float) -> str:
    stats = metrics.get("dataset_stats") or {}
    train_samples = _safe(stats, "train_samples") or 1000.0
    fit = analyze_model_fit(metrics, model_type)

    if fit.get("is_underfit") or final_score < 40:
        return "low"
        
    if train_samples < 200:
        return "low" if final_score < 60 else "medium"

    gen = compute_overfit_metrics(metrics, model_type)
    if model_type == "xgboost_clf":
        cv_std = _safe(metrics, "cv_auc_std")
        gap = gen.get("overfit_gap_auc")
        if cv_std is not None and cv_std > 0.15:
            return "low"
        if gap is not None and gap > 0.20:
            return "low"
        if (cv_std is not None and cv_std < 0.05) and (gap is None or gap < 0.05) and final_score > 75:
            return "high"
            
    elif model_type == "xgboost_reg":
        cv_std = _safe(metrics, "cv_rmse_std")
        gap = gen.get("overfit_gap_r2")
        if cv_std is not None and cv_std > 0.3:
            return "low"
        if gap is not None and gap > 0.30:
            return "low"
        if (cv_std is not None and cv_std < 0.05) and (gap is None or gap < 0.05) and final_score > 75:
            return "high"
            
    elif model_type == "isolation_forest":
        rob = metrics.get("robustness_metrics") or {}
        drift = _safe(rob, "train_test_score_drift")
        sep = compute_anomaly_separation_metrics(metrics)
        std = sep.get("score_std")
        if drift is not None and drift > 0.05:
            return "low"
        if std is not None and std < 0.005:
            return "low"
        if (drift is None or drift < 0.01) and (std is not None and std > 0.02) and final_score > 75:
            return "high"

    return "medium"


# ---------------------------------------------------------------------------
# Per-family rank score calculators
# ---------------------------------------------------------------------------

def _rank_score_classification(metrics: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    test = metrics.get("test_metrics") or {}
    auc  = _safe(test, "auc_roc") if test else _safe(metrics, "auc_roc")
    f1   = _safe(test, "f1") if test else _safe(metrics, "f1")
    cv_mean = _safe(metrics, "cv_auc_mean")
    cv_std  = _safe(metrics, "cv_auc_std")
    calib   = metrics.get("calibration_metrics")

    s_auc    = normalize_positive_metric(auc,      lo=0.5, hi=1.0)
    s_f1     = normalize_positive_metric(f1,       lo=0.0, hi=1.0)
    s_cv     = normalize_positive_metric(cv_mean,  lo=0.5, hi=1.0) if cv_mean is not None else s_auc
    s_std    = normalize_negative_metric(cv_std,   lo=0.0, hi=0.5) if cv_std  is not None else 50.0
    s_gen    = compute_generalization_score("xgboost_clf", metrics)
    s_calib  = compute_calibration_score(calib)

    predictive = 0.25 * s_auc + 0.15 * s_f1
    stability = 0.15 * s_cv + 0.10 * s_std
    generalization = 0.20 * s_gen
    calibration = 0.15 * s_calib
    
    raw_base = predictive + stability + generalization + calibration
    penalized = compute_hard_penalties(raw_base, metrics, "xgboost_clf")
    
    breakdown = {
        "predictive_power": round(predictive, 2),
        "stability": round(stability, 2),
        "generalization": round(generalization, 2),
        "calibration": round(calibration, 2),
        "penalties": round(penalized - raw_base, 2)
    }
    return clamp_score(raw_base), clamp_score(penalized), breakdown


def _rank_score_regression(metrics: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    test = metrics.get("test_metrics") or {}
    r2   = _safe(test, "r2") if test else _safe(metrics, "r2")
    mae  = _safe(test, "mae") if test else _safe(metrics, "mae")
    cv_mean = _safe(metrics, "cv_rmse_mean")
    cv_std  = _safe(metrics, "cv_rmse_std")

    stability_val = 0.0
    if cv_mean is not None and cv_std is not None:
        ratio = cv_std / (abs(cv_mean) + 1e-9)
        stability_val = normalize_negative_metric(ratio, lo=0.0, hi=1.0)

    s_r2      = normalize_positive_metric(r2,       lo=0.0, hi=1.0)
    s_cv_mean = normalize_negative_metric(cv_mean,  lo=0.0, hi=5.0) if cv_mean is not None else 50.0
    s_cv_std  = normalize_negative_metric(cv_std,   lo=0.0, hi=2.0) if cv_std  is not None else 50.0
    s_mae     = normalize_negative_metric(mae,      lo=0.0, hi=5.0) if mae     is not None else 50.0
    s_gen     = compute_generalization_score("xgboost_reg", metrics)

    predictive = 0.25 * s_r2 + 0.15 * s_mae
    stability = 0.20 * s_cv_mean + 0.10 * s_cv_std + 0.10 * stability_val
    generalization = 0.20 * s_gen
    
    raw_base = predictive + stability + generalization
    penalized = compute_hard_penalties(raw_base, metrics, "xgboost_reg")
    
    breakdown = {
        "predictive_power": round(predictive, 2),
        "stability": round(stability, 2),
        "generalization": round(generalization, 2),
        "penalties": round(penalized - raw_base, 2)
    }
    return clamp_score(raw_base), clamp_score(penalized), breakdown


def _rank_score_isolation_forest(metrics: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    sep  = compute_anomaly_separation_metrics(metrics)
    rob  = metrics.get("robustness_metrics") or {}
    drift = _safe(rob, "train_test_score_drift")

    s_gap1  = normalize_positive_metric(sep["p50_p5_gap"],  lo=0.0, hi=0.5)
    s_std   = normalize_positive_metric(sep["score_std"],   lo=0.0, hi=0.3)
    s_gap2  = normalize_positive_metric(sep["p95_p50_gap"], lo=0.0, hi=0.5)
    s_contam = _contamination_consistency_score(metrics)
    s_drift  = normalize_negative_metric(drift, lo=0.0, hi=0.10) if drift is not None else 70.0

    predictive = 0.30 * s_gap1 + 0.15 * s_gap2
    stability = 0.15 * s_std
    robustness = 0.20 * s_contam + 0.20 * s_drift

    raw_base = predictive + stability + robustness
    penalized = compute_hard_penalties(raw_base, metrics, "isolation_forest")
    
    breakdown = {
        "predictive_power": round(predictive, 2),
        "stability": round(stability, 2),
        "robustness": round(robustness, 2),
        "penalties": round(penalized - raw_base, 2)
    }
    return clamp_score(raw_base), clamp_score(penalized), breakdown


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_target_schema(model_type: str, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that the target schema aligns with the model type."""
    warnings = []
    is_valid = True
    stats = metrics.get("dataset_stats", {})
    unique_vals = stats.get("target_unique_values")

    if resolve_ranking_group(model_type) == "regression":
        if unique_vals is not None and unique_vals <= 2:
            warnings.append("regression_target_appears_binary")
            is_valid = False

    return is_valid, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_model_rank_score(model_type: str, metrics: dict[str, Any] | None) -> tuple[float, float, dict[str, float]]:
    """Compute raw quality score [0, 100], deployment health score [0, 100], and component breakdown.
    Returns:
        (raw_quality_score, deployment_health_score, breakdown_dict)
    """
    if not isinstance(metrics, dict):
        return 0.0, 0.0, {}

    if model_type == "xgboost_clf":
        return _rank_score_classification(metrics)
    elif model_type == "xgboost_reg":
        return _rank_score_regression(metrics)
    elif model_type == "isolation_forest":
        return _rank_score_isolation_forest(metrics)
    else:
        return 0.0, 0.0, {}


def build_imp_metrics(model_type: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = {}

    raw_quality_score, health_score_raw, breakdown = compute_model_rank_score(model_type, metrics)
    deployment_health_score = round(health_score_raw, 2)
    
    confidence = compute_confidence_level(model_type, metrics, deployment_health_score)
    readiness = compute_deployment_readiness(deployment_health_score)
    
    fit_analysis = analyze_model_fit(metrics, model_type)
    target_valid, schema_warnings = validate_target_schema(model_type, metrics)

    if model_type == "xgboost_clf":
        test = metrics.get("test_metrics") or {}
        auc  = _round6(_safe(test, "auc_roc") if test else _safe(metrics, "auc_roc"))
        f1   = _round6(_safe(test, "f1")      if test else _safe(metrics, "f1"))
        std  = _round6(_safe(metrics, "cv_auc_std"))

        primary   = {"metric": "auc_roc",    "value": auc,  "direction": "higher_is_better"}
        secondary = {"metric": "f1",         "value": f1,   "direction": "higher_is_better"}
        tertiary  = {"metric": "cv_auc_std", "value": std,  "direction": "lower_is_better"}

    elif model_type == "xgboost_reg":
        test     = metrics.get("test_metrics") or {}
        r2       = _round6(_safe(test, "r2")   if test else _safe(metrics, "r2"))
        mae      = _round6(_safe(test, "mae")  if test else _safe(metrics, "mae"))
        cv_rmse  = _round6(_safe(metrics, "cv_rmse_mean"))

        primary   = {"metric": "r2",           "value": r2,      "direction": "higher_is_better"}
        secondary = {"metric": "cv_rmse_mean", "value": cv_rmse, "direction": "lower_is_better"}
        tertiary  = {"metric": "mae",          "value": mae,     "direction": "lower_is_better"}

    elif model_type == "isolation_forest":
        sep = compute_anomaly_separation_metrics(metrics)
        primary   = {"metric": "p50_p5_gap",  "value": sep["p50_p5_gap"],  "direction": "higher_is_better"}
        secondary = {"metric": "score_std",   "value": sep["score_std"],   "direction": "higher_is_better"}
        tertiary  = {"metric": "p95_p50_gap", "value": sep["p95_p50_gap"], "direction": "higher_is_better"}

    else:
        primary   = {"metric": None, "value": None, "direction": None}
        secondary = {"metric": None, "value": None, "direction": None}
        tertiary  = {"metric": None, "value": None, "direction": None}

    return {
        "_raw_quality_score": round(raw_quality_score, 2), # Internal use for group ranking
        "ranking": {
            "local_rank_score": None, # Computed globally during injection
            "leaderboard_position": None,
            "total_competitors": 1,
            "ranking_available": False,
            "ranking_reason": "insufficient_models",
            "ranking_scope": "within_use_case",
            "ranking_group": resolve_ranking_group(model_type)
        },
        "deployment_health": {
            "deployment_health_score": deployment_health_score,
            "confidence_level": confidence,
            "deployment_readiness": readiness,
            "score_breakdown": breakdown
        },
        "model_fit_analysis": fit_analysis,
        "schema_warnings": schema_warnings,
        "validation_checks": {
            "target_schema_valid": target_valid
        },
        "primary":   primary,
        "secondary": secondary,
        "tertiary":  tertiary,
    }


def enrich_metrics_with_imp(model_type: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    base: dict[str, Any] = dict(metrics) if isinstance(metrics, dict) else {}
    base["generalization_metrics"] = compute_overfit_metrics(base, model_type)
    base["imp"] = build_imp_metrics(model_type, base)
    return base


def normalize_metric_score(
    value: float | None,
    *,
    direction: str,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    if direction == "higher_is_better":
        return normalize_positive_metric(value, lo=lo, hi=hi)
    return normalize_negative_metric(value, lo=lo, hi=hi)

def inject_group_ranking(models: list[Any]) -> list[Any]:
    """
    Computes true peer-relative percentiles for local_rank_score across models.
    Expects models to already have `.metrics["imp"]` populated with `_raw_quality_score`.
    Models are grouped by (use_case, ranking_group).
    """
    from collections import defaultdict
    
    # Group models
    groups = defaultdict(list)
    for m in models:
        if not m.metrics or "imp" not in m.metrics:
            continue
        imp = m.metrics["imp"]
        ranking = imp.get("ranking")
        if not ranking:
            continue
            
        group_key = (m.use_case, ranking["ranking_group"])
        groups[group_key].append(m)
        
    # Rank each group
    for group_key, group_models in groups.items():
        total_competitors = len(group_models)
        
        if total_competitors < 2:
            for m in group_models:
                m.metrics["imp"]["ranking"].update({
                    "local_rank_score": None,
                    "leaderboard_position": None,
                    "total_competitors": total_competitors,
                    "ranking_available": False,
                    "ranking_reason": "insufficient_models"
                })
                # Clean up internal key
                m.metrics["imp"].pop("_raw_quality_score", None)
            continue
            
        # Extract scores
        scores_with_models = []
        for m in group_models:
            raw_score = m.metrics["imp"].get("_raw_quality_score", 0.0)
            scores_with_models.append((raw_score, m))
            
        # Sort by raw score descending
        scores_with_models.sort(key=lambda x: x[0], reverse=True)
        
        # Compute percentiles and positions
        for i, (score, m) in enumerate(scores_with_models):
            position = i + 1
            # Simple percentile rank: (N - position) / (N - 1) * 100
            # If N=2, top gets 100, bottom gets 0.
            percentile = ((total_competitors - position) / (total_competitors - 1)) * 100.0
            
            m.metrics["imp"]["ranking"].update({
                "local_rank_score": round(percentile, 2),
                "leaderboard_position": position,
                "total_competitors": total_competitors,
                "ranking_available": True,
                "ranking_reason": None
            })
            # Clean up internal key
            m.metrics["imp"].pop("_raw_quality_score", None)
            
    return models
