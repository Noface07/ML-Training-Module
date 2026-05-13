# ML Platform API — Step-by-Step Usage Guide 

The workflow is always: **Ingest → Extract Features → (Check Hparams) → Train → Monitor/Browse**

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
    "created_at": "2026-05-12T09:30:00Z"
}
```

> [!IMPORTANT]
> Copy the `dataset_id` from the response — you need it for every subsequent call.

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
2. **`optional_features`** — only use suffixes from Step 2's `per_tag_features.optional` array
3. **`cross_tag_features`** — selectively specify cross-tag features from Step 2's `cross_tag_features.available` array
4. **`target_col`** — required for `xgboost_clf` and `xgboost_reg` use cases; use the value from Step 2
5. **`hparams`** — only include keys you want to override; everything else uses defaults
6. **Not every tag has every optional feature** — the system handles this automatically (skips missing combinations)

---

### Example A — Failure Prediction (XGBoost Classifier)

```json
{
    "dataset_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "feature_schema_id": "a1b2c3d4-e5f6-...",
    "use_case": "failure_prediction",
    "tags": ["100", "200"],
    "optional_features": ["pct_range", "dist_to_max", "custom_feature_1"],
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
    "include_cross_tag_features": false,
    "target_col": "will_fail",
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

Everything else (`optional_features`, `include_cross_tag_features`, `train_split`, `cv_folds`, `hparams`) defaults automatically.

---

**Response (202 Accepted):**
```json
{
    "model_id": "e8f9a0b1-...",       ← SAVE THIS
    "job_id": "e8f9a0b1-...",
    "status": "training",
    "stream_url": "/v1/train/e8f9a0b1-.../stream",
    "use_case": "failure_prediction",
    "output_tag": "risk_score"
}
```

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

The stream emits JSON log lines as they appear, then a final `result` event with metrics.

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

### Delete a model
```
DELETE http://localhost:8000/v1/models/{model_id}
```

> [!CAUTION]
> Delete removes both the DB record AND the `.pkl` / `_meta.json` files from disk. This is irreversible.

---

## Quick Reference — Field Descriptions

| Field | Type | Default | Notes |
|---|---|---|---|
| `dataset_id` | string | — | From Step 1 response |
| `feature_schema_id` | string | — | From Step 2 response |
| `use_case` | string | — | See table in Step 3 |
| `tags` | string[] | — | Tag IDs from Step 2 `tags` array |
| `optional_features` | string[] | `[]` | Suffixes from Step 2 `optional` array |
| `cross_tag_features` | string[] | `[]` | Selectively include specific cross-tag features |
| `include_cross_tag_features` | bool | `false` | Fallback flag to include all available cross-tag features |
| `target_col` | string | `null` | Required for clf/reg, null for anomaly |
| `train_split` | float | `0.8` | Train/test ratio (0.1 – 0.99) |
| `cv_folds` | int | `5` | Cross-validation folds (2 – 20) |
| `hparams` | object | `{}` | Only override what you need |
