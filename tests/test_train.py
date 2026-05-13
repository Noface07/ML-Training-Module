"""Tests for the train API."""

from __future__ import annotations


def _setup_dataset(client, sample_parquet_bytes) -> tuple[str, str]:
    """Ingest + extract features, return (dataset_id, schema_id)."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    dataset_id = resp.json()["dataset_id"]

    resp2 = client.post("/v1/features/extract", json={"dataset_id": dataset_id})
    schema_id = resp2.json()["feature_schema_id"]
    return dataset_id, schema_id


def test_train_spawn(client, sample_parquet_bytes):
    """POST /v1/train returns 202 with job info."""
    dataset_id, schema_id = _setup_dataset(client, sample_parquet_bytes)

    resp = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "failure_prediction",
        "tags": ["100", "200"],
        "optional_features": ["pct_range"],
        "include_cross_tag_features": True,
        "target_col": "will_fail",
    })
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "training"
    assert "stream_url" in data
    assert data["use_case"] == "failure_prediction"


def test_train_unknown_use_case(client, sample_parquet_bytes):
    """Unknown use case returns 422."""
    dataset_id, schema_id = _setup_dataset(client, sample_parquet_bytes)

    resp = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "nonexistent",
        "tags": ["100"],
        "target_col": "will_fail",
    })
    assert resp.status_code == 422


def test_train_missing_tags(client, sample_parquet_bytes):
    """Requesting tags not in the dataset returns 422."""
    dataset_id, schema_id = _setup_dataset(client, sample_parquet_bytes)

    resp = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "failure_prediction",
        "tags": ["100", "999"],
        "target_col": "will_fail",
    })
    assert resp.status_code == 422


def test_train_selective_cross_tag(client, sample_parquet_bytes):
    """POST /v1/train supports specific cross_tag_features list."""
    dataset_id, schema_id = _setup_dataset(client, sample_parquet_bytes)

    resp = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "failure_prediction",
        "tags": ["100", "200"],
        "optional_features": ["pct_range"],
        "cross_tag_features": ["factory_temp"],
        "target_col": "will_fail",
    })
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "training"


def test_train_invalid_cross_tag(client, sample_parquet_bytes):
    """Requesting invalid cross-tag features returns 422."""
    dataset_id, schema_id = _setup_dataset(client, sample_parquet_bytes)

    resp = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "failure_prediction",
        "tags": ["100", "200"],
        "cross_tag_features": ["invalid_cross_tag"],
        "target_col": "will_fail",
    })
    assert resp.status_code == 422
