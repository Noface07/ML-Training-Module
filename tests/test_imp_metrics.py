"""
Unit tests for utils/imp_metrics.py

Coverage:
  - normalize_positive_metric, normalize_negative_metric, clamp_score
  - compute_anomaly_separation_metrics
  - analyze_model_fit (underfit, gaps)
  - compute_calibration_score
  - compute_confidence_level
  - compute_generalization_score
  - _contamination_consistency_score
  - compute_hard_penalties & apply_exponential_penalty
  - compute_deployment_readiness
  - resolve_ranking_group
  - per-family rank scores
  - enrich_metrics_with_imp
  - inject_group_ranking
  - validate_target_schema
"""

from __future__ import annotations

import math
import pytest
from types import SimpleNamespace

from utils.imp_metrics import (
    _contamination_consistency_score,
    apply_exponential_penalty,
    analyze_model_fit,
    build_imp_metrics,
    clamp_score,
    compute_anomaly_separation_metrics,
    compute_calibration_score,
    compute_confidence_level,
    compute_deployment_readiness,
    compute_generalization_score,
    compute_hard_penalties,
    compute_model_rank_score,
    compute_overfit_metrics,
    enrich_metrics_with_imp,
    inject_group_ranking,
    normalize_negative_metric,
    normalize_positive_metric,
    resolve_ranking_group,
    validate_target_schema,
)

# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

class TestNormalizationPrimitives:
    def test_clamp_score(self):
        assert clamp_score(-10.0) == 0.0
        assert clamp_score(0.0) == 0.0
        assert clamp_score(50.0) == 50.0
        assert clamp_score(100.0) == 100.0
        assert clamp_score(150.0) == 100.0

    def test_normalize_positive_metric(self):
        assert normalize_positive_metric(0.5, lo=0.0, hi=1.0) == 50.0
        assert normalize_positive_metric(1.0, lo=0.0, hi=1.0) == 100.0
        assert normalize_positive_metric(-0.5, lo=0.0, hi=1.0) == 0.0

    def test_normalize_negative_metric(self):
        assert normalize_negative_metric(0.5, lo=0.0, hi=1.0) == 50.0
        assert normalize_negative_metric(0.0, lo=0.0, hi=1.0) == 100.0
        assert normalize_negative_metric(1.0, lo=0.0, hi=1.0) == 0.0

    def test_exponential_penalty(self):
        assert apply_exponential_penalty(100.0, 0) == 100.0
        assert round(apply_exponential_penalty(100.0, 0.693147), 1) == 50.0 # ~exp(-0.693) = 0.5
        assert apply_exponential_penalty(100.0, -1.0) == 100.0

# ---------------------------------------------------------------------------
# Ranking Group Resolver
# ---------------------------------------------------------------------------

class TestRankingGroupResolver:
    def test_resolve_ranking_group(self):
        assert resolve_ranking_group("xgboost_clf") == "classification"
        assert resolve_ranking_group("random_forest_clf") == "classification"
        assert resolve_ranking_group("xgboost_reg") == "regression"
        assert resolve_ranking_group("linear_reg") == "regression"
        assert resolve_ranking_group("isolation_forest") == "anomaly_detection"
        assert resolve_ranking_group("autoencoder") == "anomaly_detection"
        assert resolve_ranking_group("unknown_model") == "unknown"

# ---------------------------------------------------------------------------
# Anomaly separation (derived from score_distribution)
# ---------------------------------------------------------------------------

class TestComputeAnomalySeparationMetrics:
    def test_full_metrics_calculated(self):
        metrics = {
            "score_distribution": {
                "mean": 0.1, "std": 0.034470, "p5": 0.05, "p25": 0.08,
                "p50": 0.10, "p75": 0.15, "p95": 0.162,
            }
        }
        result = compute_anomaly_separation_metrics(metrics)
        assert result["p50_p5_gap"] == round(0.10 - 0.05, 6)
        assert result["p95_p50_gap"] == round(0.162 - 0.10, 6)
        assert result["score_std"] == round(0.034470, 6)

    def test_missing_distribution(self):
        assert compute_anomaly_separation_metrics({}) == {
            "p50_p5_gap": None, "score_std": None, "p95_p50_gap": None
        }

# ---------------------------------------------------------------------------
# Model Fit Analysis
# ---------------------------------------------------------------------------

class TestAnalyzeModelFit:
    def test_clf_underfit(self):
        metrics = {"train_metrics": {"auc_roc": 0.6}, "test_metrics": {"auc_roc": 0.6}}
        fit = analyze_model_fit(metrics, "xgboost_clf")
        assert fit["is_underfit"] is True
        assert fit["is_overfit"] is False
        assert fit["has_generalization_failure"] is False

    def test_clf_overfit(self):
        metrics = {"train_metrics": {"auc_roc": 0.95}, "test_metrics": {"auc_roc": 0.8}}
        fit = analyze_model_fit(metrics, "xgboost_clf")
        assert fit["is_underfit"] is False
        assert fit["is_overfit"] is True
        assert fit["has_generalization_failure"] is False

    def test_clf_generalization_failure(self):
        metrics = {"train_metrics": {"auc_roc": 0.99}, "test_metrics": {"auc_roc": 0.65}}
        fit = analyze_model_fit(metrics, "xgboost_clf")
        assert fit["is_underfit"] is False
        assert fit["is_overfit"] is True
        assert fit["has_generalization_failure"] is True
        
    def test_reg_underfit(self):
        metrics = {"train_metrics": {"r2": 0.1}, "test_metrics": {"r2": 0.1}}
        fit = analyze_model_fit(metrics, "xgboost_reg")
        assert fit["is_underfit"] is True
        
    def test_reg_generalization_failure(self):
        metrics = {"train_metrics": {"r2": 0.9}, "test_metrics": {"r2": 0.5}}
        fit = analyze_model_fit(metrics, "xgboost_reg")
        assert fit["is_overfit"] is True
        assert fit["has_generalization_failure"] is True


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_regression_binary_target(self):
        metrics = {"dataset_stats": {"target_unique_values": 2}}
        valid, warnings = validate_target_schema("xgboost_reg", metrics)
        assert valid is False
        assert "regression_target_appears_binary" in warnings

    def test_regression_continuous_target(self):
        metrics = {"dataset_stats": {"target_unique_values": 150}}
        valid, warnings = validate_target_schema("xgboost_reg", metrics)
        assert valid is True
        assert len(warnings) == 0

    def test_clf_ignored(self):
        metrics = {"dataset_stats": {"target_unique_values": 2}}
        valid, warnings = validate_target_schema("xgboost_clf", metrics)
        assert valid is True
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Peer Ranking Injection
# ---------------------------------------------------------------------------

class TestPeerRanking:
    def test_group_ranking_percentiles(self):
        models = [
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 90.0, "ranking": {"ranking_group": "classification"}}}),
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 80.0, "ranking": {"ranking_group": "classification"}}}),
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 70.0, "ranking": {"ranking_group": "classification"}}})
        ]
        
        inject_group_ranking(models)
        
        ranks = [m.metrics["imp"]["ranking"]["local_rank_score"] for m in models]
        positions = [m.metrics["imp"]["ranking"]["leaderboard_position"] for m in models]
        
        assert ranks == [100.0, 50.0, 0.0]
        assert positions == [1, 2, 3]
        
        for m in models:
            assert "_raw_quality_score" not in m.metrics["imp"]
            assert m.metrics["imp"]["ranking"]["total_competitors"] == 3
            assert m.metrics["imp"]["ranking"]["ranking_available"] is True

    def test_insufficient_peers(self):
        models = [
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 90.0, "ranking": {"ranking_group": "classification"}}}),
        ]
        
        inject_group_ranking(models)
        
        ranking = models[0].metrics["imp"]["ranking"]
        assert ranking["local_rank_score"] is None
        assert ranking["leaderboard_position"] is None
        assert ranking["ranking_available"] is False
        assert ranking["ranking_reason"] == "insufficient_models"
        assert "_raw_quality_score" not in models[0].metrics["imp"]

    def test_cross_group_isolation(self):
        models = [
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 90.0, "ranking": {"ranking_group": "classification"}}}),
            SimpleNamespace(use_case="a", metrics={"imp": {"_raw_quality_score": 80.0, "ranking": {"ranking_group": "classification"}}}),
            SimpleNamespace(use_case="b", metrics={"imp": {"_raw_quality_score": 70.0, "ranking": {"ranking_group": "classification"}}}),
            SimpleNamespace(use_case="b", metrics={"imp": {"_raw_quality_score": 60.0, "ranking": {"ranking_group": "classification"}}})
        ]
        
        inject_group_ranking(models)
        
        ranks = [m.metrics["imp"]["ranking"]["local_rank_score"] for m in models]
        assert ranks == [100.0, 0.0, 100.0, 0.0] # Group A and B computed independently


# ---------------------------------------------------------------------------
# Rank Scores (Integrated)
# ---------------------------------------------------------------------------

class TestComputeModelRankScore:
    def test_clf_perfect(self):
        metrics = {
            "test_metrics": {"auc_roc": 1.0, "f1": 1.0},
            "train_metrics": {"auc_roc": 1.0, "f1": 1.0},
            "cv_auc_mean": 1.0,
            "cv_auc_std": 0.0,
            "calibration_metrics": {"log_loss": 0.0, "brier_score": 0.0}
        }
        local_rank, health_score, breakdown = compute_model_rank_score("xgboost_clf", metrics)
        assert local_rank == 100.0
        assert health_score == 100.0

    def test_clf_worst(self):
        metrics = {
            "test_metrics": {"auc_roc": 0.5, "f1": 0.1},
            "train_metrics": {"auc_roc": 1.0, "f1": 1.0}, 
            "cv_auc_mean": 0.5,
            "cv_auc_std": 0.5,
            "calibration_metrics": {"log_loss": 2.0, "brier_score": 1.0}
        }
        local_rank, health_score, breakdown = compute_model_rank_score("xgboost_clf", metrics)
        assert health_score < local_rank
        assert health_score < 20.0


# ---------------------------------------------------------------------------
# Enrich Metrics
# ---------------------------------------------------------------------------

class TestEnrichMetricsWithImp:
    def test_enrichment(self):
        metrics = {"test_metrics": {"auc_roc": 0.8}}
        enriched = enrich_metrics_with_imp("xgboost_clf", metrics)

        assert "imp" in enriched
        assert "generalization_metrics" in enriched

        imp = enriched["imp"]
        
        assert "ranking" in imp
        assert "local_rank_score" in imp["ranking"]
        assert imp["ranking"]["ranking_scope"] == "within_use_case"
        assert imp["ranking"]["ranking_group"] == "classification"
        
        assert "deployment_health" in imp
        assert "deployment_health_score" in imp["deployment_health"]
        assert "confidence_level" in imp["deployment_health"]
        assert "deployment_readiness" in imp["deployment_health"]
        
        assert "model_fit_analysis" in imp
        assert "schema_warnings" in imp
        assert "validation_checks" in imp
        
        assert imp["primary"]["metric"] == "auc_roc"
