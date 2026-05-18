# ML Platform API — Step-by-Step Usage Guide

The workflow is always: **Ingest → (Preview Dataset) → Extract Features → (Check Hparams) → Train → Monitor → Browse/Manage → (Profiles)**

---

## Step 1 — Ingest a Parquet File

```
POST http://localhost:8000/v1/ingest
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | Your `.parquet` file |
| `config` | text | ❌ | Optional JSON config string |

**Postman setup:**
- Body → `form-data`
- Key: `file` → type: **File** → select your `.parquet`
- Key: `config` → type: **Text** → paste the JSON below (or omit for defaults)

**Config (optional):**
```json
{
    "timestamp_col": "timestamp",
    "timestamp_format": "auto",
    "forward_fill_limit": 5,
    "backward_fill_first_rows": true,
    "drop_duplicate_timestamps": true
}
```

**Response (200):**
```json
{
    "dataset_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",   ← SAVE THIS
    "row_count": 10420,
    "col_count": 134,
    "tags_detected": ["12446", "12447", "12448"],
    "quality_report": { ... },
    "range_metadata": { "12446": {"min": 0.12, "max": 148.7}, ... },
    "profile_id": 1,                                         ← auto-resolved Tag Profile
    "profile_name": "profile_a3f92c11",                     ← auto-named (renameable)
    "tag_hash": "a3f92c11d4e7...",                          ← deterministic 64-char hash
    "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "is_duplicate": false,                                   ← true if file was previously ingested
    "created_at": "2026-05-12T09:30:00Z"
}
```

> [!IMPORTANT]
> Copy the `dataset_id` from the response — you need it for every subsequent call.
> The `profile_id` is automatically resolved from your tag collection — datasets with the same tags always map to the same profile.
> **Deduplication:** If you upload the exact same file twice, the system skips processing and returns `is_duplicate: true` with the original `dataset_id`.

---

## Step 1b — Preview Dataset (Optional)

Inspect the first N rows and column schema of any ingested dataset — useful for remote Flutter/mobile clients.

```
GET http://localhost:8000/v1/dataset/{dataset_id}/preview
GET http://localhost:8000/v1/dataset/{dataset_id}/preview?limit=50
```

| Query Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `100` | Max rows to return (1 – 1000) |

**Response (200):**
```json
{
    "dataset_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "original_filename": "sensor_data.parquet",
    "row_count": 10420,
    "col_count": 134,
    "columns": ["timestamp", "tag_100_raw", "tag_100_roll_mean", "will_fail", ...],
    "rows": [
        { "timestamp": "2026-01-01T00:00:00", "tag_100_raw": 51.23, "will_fail": 0, ... },
        { "timestamp": "2026-01-01T00:01:00", "tag_100_raw": 51.87, "will_fail": 0, ... }
    ]
}
```

> [!TIP]
> Use `?limit=5` for a fast schema inspection without transferring large payloads.

---

## Step 2 — Extract Features

This tells you what tags and features exist in your dataset.

```
POST http://localhost:8000/v1/features/extract
Content-Type: application/json
```

**Payload:**
```json
{
    "dataset_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

**Response (200):**
```json
{
    "feature_schema_id": "a1b2c3d4-...",   ← SAVE THIS
    "dataset_id": "f47ac10b-...",
    "tags": ["100", "200"],                ← available tag IDs
    "per_tag_features": {
        "mandatory": ["raw", "roc_1", "roll_mean", "roll_std"],
        "optional": [                      ← these are what you can pick from
            "custom_feature_1",
            "custom_feature_2",
            "dist_to_max",
            "my_special_metric",
            "pct_range"
        ]
    },
    "cross_tag_features": {
        "available": ["system_avg_pct", "system_max_pct", "system_tags_high"],
        "present_in_data": ["system_avg_pct"]
    },
    "target_col": "will_fail",
    "target_present": true,
    "total_columns": 17
}
```

> [!TIP]
> **Read this response carefully!** The `tags`, `optional` features, and `target_col` are what you use to build your training payload. Only pick tags and optional features that appear here.

---

## Step 3 (Optional) — Check Default Hyperparameters

```
GET http://localhost:8000/v1/hparams/{use_case}
```

**Available use cases:**

| Use Case | Model Type | Requires Training |
|---|---|---|
| `failure_prediction` | xgboost_clf | ✅ |
| `risk_scoring` | xgboost_clf | ✅ |
| `rul` | xgboost_reg | ✅ |
| `next_interval` | xgboost_reg | ✅ |
| `kpi_prediction` | xgboost_reg | ✅ |
| `anomaly_multivariate` | isolation_forest | ✅ |
| `anomaly_univariate` | statistical | ❌ |
| `adaptive_threshold` | statistical | ❌ |
| `early_warning` | statistical | ❌ |
| `health_index` | statistical | ❌ |
| `drift_detection` | statistical | ❌ |
| `pattern_detection` | statistical | ❌ |
| `data_quality` | statistical | ❌ |

**Example:**
```
GET http://localhost:8000/v1/hparams/failure_prediction
```

**Response** shows tier1/tier2/tier3 defaults with min/max ranges — you only need to override what you want to change.

---

## Step 4 — Train a Model

```
POST http://localhost:8000/v1/train
Content-Type: application/json
```

### Rules for building the payload:
1. **`tags`** — only use tag IDs from Step 2's `tags` array
2. **`mandatory_features`** — optional override; defaults to auto-detected mandatory set (`raw`, `roc_1`, `roll_mean`, `roll_std`)
3. **`optional_features`** — only use suffixes from Step 2's `per_tag_features.optional` array; these are evaluated via combinatorial feature selection (exhaustive if ≤10 groups, greedy forward selection if more)
4. **`feature_selection`** — set to `true` to enable automated cross-validation based feature subset selection; defaults to `false`
5. **`cross_tag_features`** — selectively specify cross-tag features from Step 2's `cross_tag_features.available` array
6. **`target_col`** — required for `xgboost_clf` and `xgboost_reg` use cases; use the value from Step 2
7. **`model_name`** — optional human-readable name for the saved `.pkl` file (e.g. `"My Sensor Model v2"`)
8. **`hparams`** — only include keys you want to override; everything else uses defaults
9. **Not every tag has every optional feature** — the system handles this automatically (skips missing combinations)

---

### Example A — Failure Prediction (XGBoost Classifier)

```json
{
    "model_name": "Failure Model v1",
    "dataset_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "feature_schema_id": "a1b2c3d4-e5f6-...",
    "use_case": "failure_prediction",
    "tags": ["100", "200"],
    "mandatory_features": ["raw", "roc_1", "roll_mean", "roll_std"],
    "optional_features": ["pct_range", "dist_to_max", "custom_feature_1"],
    "feature_selection": true,
    "cross_tag_features": ["system_avg_pct"],
    "include_cross_tag_features": true,
    "target_col": "will_fail",
    "train_split": 0.8,
    "cv_folds": 5,
    "hparams": {
        "n_estimators": 400,
        "max_depth": 7,
        "learning_rate": 0.05
    }
}
```

### Example B — RUL Prediction (XGBoost Regressor)

```json
{
    "dataset_id": "f47ac10b-...",
    "feature_schema_id": "a1b2c3d4-...",
    "use_case": "rul",
    "tags": ["100", "200"],
    "optional_features": ["pct_range"],
    "feature_selection": true,
    "include_cross_tag_features": false,
    "target_col": "rul_hours",
    "train_split": 0.8,
    "cv_folds": 5,
    "hparams": {
        "n_estimators": 500,
        "learning_rate": 0.03
    }
}
```

### Example C — Anomaly Detection (Isolation Forest)

```json
{
    "dataset_id": "f47ac10b-...",
    "feature_schema_id": "a1b2c3d4-...",
    "use_case": "anomaly_multivariate",
    "tags": ["100", "200"],
    "optional_features": [],
    "feature_selection": false,
    "include_cross_tag_features": false,
    "target_col": null,
    "train_split": 0.8,
    "cv_folds": 5,
    "hparams": {
        "contamination": 0.05
    }
}
```

> [!NOTE]
> Anomaly detection does NOT need a `target_col` — set it to `null`.

### Example D — Minimal Payload (all defaults)

```json
{
    "dataset_id": "f47ac10b-...",
    "feature_schema_id": "a1b2c3d4-...",
    "use_case": "failure_prediction",
    "tags": ["100", "200"],
    "target_col": "will_fail"
}
```

Everything else (`model_name`, `optional_features`, `feature_selection`, `include_cross_tag_features`, `train_split`, `cv_folds`, `hparams`) defaults automatically.

---

**Response (202 Accepted):**
```json
{
    "model_id": "e8f9a0b1-...",       ← SAVE THIS
    "model_name": "Failure Model v1",
    "job_id": "e8f9a0b1-...",
    "status": "training",
    "stream_url": "/v1/train/e8f9a0b1-.../stream",
    "use_case": "failure_prediction",
    "output_tag": "risk_score"
}
```

> [!TIP]
> The model is automatically linked to the Tag Profile resolved during ingest. Use `GET /v1/profiles/{profile_id}/models` to browse all versions trained on the same tag set.

---

## Step 5 — Stream Training Logs (SSE)

```
GET http://localhost:8000/v1/train/{model_id}/stream
Accept: text/event-stream
```

> [!WARNING]
> Postman has limited SSE support. Use **curl** for real-time streaming:
> ```bash
> curl -N http://localhost:8000/v1/train/{model_id}/stream
> ```

The stream emits one JSON log line per event as training progresses, then closes with a final `result` event containing the full metrics payload. The stream ends **immediately** after `"Training complete"` is logged — if you see it hanging after that message, check that the server process hasn't stalled.

**Typical log sequence:**
```
Training started
Dataset loaded
Feature selection started          ← only if feature_selection=true and optional groups exist
Exhaustive search: N combinations  ← or "Greedy forward selection" if >10 suffix groups
FeatureSelection combo 1/N fold 1/5
...
Feature selection complete
Model training complete
Evaluation complete
Artifact saved
Training complete                  ← stream ends here
```

---

## Step 6 — Browse & Manage Models

### List all models
```
GET http://localhost:8000/v1/models
GET http://localhost:8000/v1/models?use_case=failure_prediction
GET http://localhost:8000/v1/models?status=completed
GET http://localhost:8000/v1/models?use_case=rul&status=completed&limit=10&offset=0
```

### Get model details
```
GET http://localhost:8000/v1/models/{model_id}
```

### Download model file
```
GET http://localhost:8000/v1/models/{model_id}/download
```
Returns the `.pkl` file with the custom `model_name` as the suggested filename.

### Delete a model
```
DELETE http://localhost:8000/v1/models/{model_id}
```

> [!CAUTION]
> Delete removes both the DB record AND the `.pkl` / `_meta.json` files from disk. This is irreversible.

---

## Model Response — `imp` Object (Important Metrics)

Every **completed** model response from `GET /v1/models` and `GET /v1/models/{model_id}` now includes an `imp` sub-object inside `metrics`. This is computed at read-time and is never stored in the database — it is always current relative to all peer models.

```json
{
    "metrics": {
        "r2": 0.87,
        "rmse": 0.42,
        "mae": 0.31,
        "cv_rmse_mean": 0.45,
        "cv_rmse_std": 0.03,
        "train_metrics": { "r2": 0.94, "rmse": 0.28, "mae": 0.22 },
        "test_metrics":  { "r2": 0.87, "rmse": 0.42, "mae": 0.31 },
        "generalization_metrics": {
            "overfit_gap_r2": 0.07,
            "is_overfit": false,
            "is_underfit": false,
            "has_generalization_failure": false
        },
        "dataset_stats": {
            "train_samples": 8000,
            "test_samples": 2000,
            "target_unique_values": 412
        },
        "imp": {
            "primary":   { "metric": "r2",           "value": 0.87,   "direction": "higher_is_better" },
            "secondary": { "metric": "cv_rmse_mean",  "value": 0.45,   "direction": "lower_is_better" },
            "tertiary":  { "metric": "mae",           "value": 0.31,   "direction": "lower_is_better" },
            "ranking": {
                "local_rank_score": 83.33,
                "leaderboard_position": 1,
                "total_competitors": 4,
                "ranking_available": true,
                "ranking_reason": null,
                "ranking_scope": "within_use_case",
                "ranking_group": "regression"
            },
            "deployment_health": {
                "deployment_health_score": 74.21,
                "confidence_level": "high",
                "deployment_readiness": "good",
                "score_breakdown": {
                    "predictive_power": 32.5,
                    "stability": 18.2,
                    "generalization": 14.1,
                    "penalties": 0.0
                }
            },
            "model_fit_analysis": {
                "is_underfit": false,
                "is_overfit": false,
                "has_generalization_failure": false
            },
            "schema_warnings": [],
            "validation_checks": {
                "target_schema_valid": true
            }
        }
    }
}
```

### `imp` Field Reference

| Field | Description |
|---|---|
| `primary` / `secondary` / `tertiary` | The 3 most important metrics for this model family with their direction |
| `ranking.local_rank_score` | Percentile rank among peers in the same use case (`0`–`100`). `null` if fewer than 2 completed peers exist |
| `ranking.leaderboard_position` | Integer position (1 = best). `null` if ranking unavailable |
| `ranking.total_competitors` | Number of completed models in the same ranking group |
| `ranking.ranking_available` | `false` when fewer than 2 peers exist |
| `ranking.ranking_group` | One of `"classification"`, `"regression"`, `"anomaly_detection"` |
| `deployment_health.deployment_health_score` | Penalized quality score `[0–100]` after applying overfit/stability penalties |
| `deployment_health.confidence_level` | `"low"` / `"medium"` / `"high"` |
| `deployment_health.deployment_readiness` | `"unsafe"` / `"weak"` / `"experimental"` / `"good"` / `"production_ready"` |
| `deployment_health.score_breakdown` | Component contributions to the score |
| `model_fit_analysis.is_underfit` | Model performs poorly on both train and test |
| `model_fit_analysis.is_overfit` | Large train-test performance gap detected |
| `model_fit_analysis.has_generalization_failure` | Extreme train-test divergence — model should not be deployed |

### Primary metrics by model family

| Model Family | Primary | Secondary | Tertiary |
|---|---|---|---|
| `xgboost_clf` | `auc_roc` ↑ | `f1` ↑ | `cv_auc_std` ↓ |
| `xgboost_reg` | `r2` ↑ | `cv_rmse_mean` ↓ | `mae` ↓ |
| `isolation_forest` | `p50_p5_gap` ↑ | `score_std` ↑ | `p95_p50_gap` ↑ |

> [!NOTE]
> `↑` = higher is better, `↓` = lower is better.
> Ranking is always computed relative to **peers in the same use case** at read-time. Adding or removing models from the DB immediately changes `local_rank_score` and `leaderboard_position` on the next GET request.

---

## `generalization_metrics` — Train vs. Test Gap

Every supervised model (`xgboost_clf`, `xgboost_reg`) now reports a `generalization_metrics` block inside `metrics`:

**Regression:**
```json
"generalization_metrics": {
    "overfit_gap_r2": 0.07,
    "is_overfit": false,
    "is_underfit": false,
    "has_generalization_failure": false
}
```

**Classification:**
```json
"generalization_metrics": {
    "overfit_gap_auc": 0.04,
    "is_overfit": false,
    "is_underfit": false,
    "has_generalization_failure": false
}
```

| Flag | Condition |
|---|---|
| `is_overfit` (reg) | `train_r2 - test_r2 > 0.2` |
| `has_generalization_failure` (reg) | `train_r2 - test_r2 > 0.5` |
| `is_underfit` (reg) | `test_r2 < 0.0` |
| `is_overfit` (clf) | `train_auc - test_auc > 0.15` |
| `has_generalization_failure` (clf) | `train_auc - test_auc > 0.3` |

---

## Feature Selection Behaviour

When `feature_selection: true` is set and at least one `optional_features` suffix group is specified, the trainer runs automated CV-based feature subset evaluation before the final fit:

- **≤ 10 suffix groups** → exhaustive search (all 2^N subsets evaluated)
- **> 10 suffix groups** → true **greedy forward selection** — starts with the best single suffix, then at each step adds the suffix that most improves the primary CV metric, stopping when no remaining suffix yields further improvement

The best-performing column set is selected and used for all final training and evaluation. The log stream will show per-combo, per-fold metrics during this phase.

> [!NOTE]
> The internal feature selection metric (`score_std` for Isolation Forest, `rmse` for regression, `auc_roc` for classification) is **independent** from the `imp` ranking metrics. The `imp` block always uses the test-set evaluation for leaderboard ranking regardless of what metric was used internally for feature selection.

When `feature_selection: false` (default), `optional_features_used` in the stored record is always `[]` — not the list of all available optional groups.

---

## Step 7 — Tag Profile Management

Every ingest auto-resolves a **Tag Profile** — a named group that maps a deterministic hash of integer tag IDs to datasets and models. Datasets ingested with the same tag collection always land in the same profile.

### List all profiles
```
GET http://localhost:8000/v1/profiles
```

**Response:**
```json
{
    "items": [
        {
            "id": 1,
            "profile_name": "profile_a3f92c11",
            "tag_hash": "a3f92c11d4e7...",
            "created_at": "...",
            "updated_at": "...",
            "model_count": 3
        }
    ]
}
```

### Get profile details
```
GET http://localhost:8000/v1/profiles/{profile_id}
```

### Rename a profile
```
PATCH http://localhost:8000/v1/profiles/{profile_id}
Content-Type: application/json
```
```json
{
    "profile_name": "Engine Sensors Group A"
}
```

### List all models under a profile
```
GET http://localhost:8000/v1/profiles/{profile_id}/models
```
Returns all model artifacts ever trained on datasets belonging to this profile.

### Delete a profile
```
DELETE http://localhost:8000/v1/profiles/{profile_id}
```

> [!CAUTION]
> Deleting a profile cascades to delete all associated model artifacts and their physical files. This is irreversible.

---

## Quick Reference — Field Descriptions

| Field | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | string | — | From Step 1 response |
| `feature_schema_id` | string | — | From Step 2 response |
| `model_name` | string | `null` | Optional friendly name for `.pkl` file |
| `use_case` | string | — | See table in Step 3 |
| `tags` | string[] | — | Tag IDs from Step 2 `tags` array |
| `mandatory_features` | string[] | auto-detected | Override default mandatory feature set |
| `optional_features` | string[] | `[]` | Suffixes from Step 2 `optional` array |
| `feature_selection` | bool | `false` | Enable CV-based optional feature subset selection |
| `cross_tag_features` | string[] | `[]` | Selectively include specific cross-tag features |
| `include_cross_tag_features` | bool | `false` | Fallback flag to include all available cross-tag features |
| `target_col` | string | `null` | Required for clf/reg, null for anomaly |
| `train_split` | float | `0.8` | Train/test ratio (0.1 – 0.99) |
| `cv_folds` | int | `5` | Cross-validation folds (2 – 20) |
| `hparams` | object | `{}` | Only override what you need |

---

## Quick Reference — All Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/ingest` | Upload & ingest a Parquet file |
| `GET` | `/v1/dataset/{dataset_id}/preview` | Preview dataset rows and schema |
| `POST` | `/v1/features/extract` | Discover tags and feature schema |
| `GET` | `/v1/hparams/{use_case}` | Get default hyperparameters |
| `POST` | `/v1/train` | Spawn a training job |
| `GET` | `/v1/train/{model_id}/stream` | SSE stream training logs |
| `GET` | `/v1/models` | List model artifacts (filterable) |
| `GET` | `/v1/models/{model_id}` | Get model details |
| `GET` | `/v1/models/{model_id}/download` | Download `.pkl` file |
| `DELETE` | `/v1/models/{model_id}` | Delete model + files |
| `GET` | `/v1/profiles` | List all Tag Profiles |
| `GET` | `/v1/profiles/{profile_id}` | Get profile details |
| `PATCH` | `/v1/profiles/{profile_id}` | Rename a profile |
| `GET` | `/v1/profiles/{profile_id}/models` | Models under a profile |
| `DELETE` | `/v1/profiles/{profile_id}` | Delete profile + all its models |
| `GET` | `/health` | Health check |
