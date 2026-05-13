"""Tests for the model registry API."""

from __future__ import annotations


def test_list_models_empty(client):
    """GET /v1/models on empty DB returns empty list."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_get_model_not_found(client):
    """GET /v1/models/{id} returns 404 for unknown ID."""
    resp = client.get("/v1/models/nonexistent-uuid")
    assert resp.status_code == 404


def test_delete_model_not_found(client):
    """DELETE /v1/models/{id} returns 404 for unknown ID."""
    resp = client.delete("/v1/models/nonexistent-uuid")
    assert resp.status_code == 404


def test_registry_model_name(client, sample_parquet_bytes):
    """Verify model_name is exposed in model registry endpoints."""
    # 1. Ingest
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    dataset_id = resp.json()["dataset_id"]

    # 2. Extract features
    resp2 = client.post("/v1/features/extract", json={"dataset_id": dataset_id})
    schema_id = resp2.json()["feature_schema_id"]

    # 3. Train with custom model_name
    resp3 = client.post("/v1/train", json={
        "dataset_id": dataset_id,
        "feature_schema_id": schema_id,
        "use_case": "failure_prediction",
        "model_name": "Explorer Premium Model",
        "tags": ["100", "200"],
        "target_col": "will_fail",
    })
    assert resp3.status_code == 202
    model_id = resp3.json()["model_id"]

    # 4. Get from registry
    resp4 = client.get(f"/v1/models/{model_id}")
    assert resp4.status_code == 200
    assert resp4.json()["model_name"] == "Explorer Premium Model"

    # 5. List models
    resp5 = client.get("/v1/models")
    assert resp5.status_code == 200
    items = resp5.json()["items"]
    assert any(i["id"] == model_id and i["model_name"] == "Explorer Premium Model" for i in items)
