"""Tests for the features API."""

from __future__ import annotations


def _ingest_first(client, sample_parquet_bytes) -> str:
    """Helper: ingest a file and return the dataset_id."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    return resp.json()["dataset_id"]


def test_extract_features_dynamic(client, sample_parquet_bytes):
    """Feature extraction should discover ALL suffixes dynamically."""
    dataset_id = _ingest_first(client, sample_parquet_bytes)

    resp = client.post("/v1/features/extract", json={"dataset_id": dataset_id})
    assert resp.status_code == 200
    data = resp.json()

    # Mandatory suffixes should be detected
    assert "raw" in data["per_tag_features"]["mandatory"]
    assert "roll_mean" in data["per_tag_features"]["mandatory"]

    # Custom/extra suffixes should appear as optional
    optional = data["per_tag_features"]["optional"]
    assert "custom_feature_1" in optional
    assert "custom_feature_2" in optional
    assert "my_special_metric" in optional
    assert "pct_range" in optional

    # Tags
    assert "100" in data["tags"]
    assert "200" in data["tags"]

    # Target
    assert data["target_present"] is True


def test_extract_features_missing_dataset(client):
    """Requesting features for a non-existent dataset returns 404."""
    resp = client.post("/v1/features/extract", json={"dataset_id": "nonexistent"})
    assert resp.status_code == 404


def test_hparams_known(client):
    """GET /v1/hparams/failure_prediction returns tier1-3."""
    resp = client.get("/v1/hparams/failure_prediction")
    assert resp.status_code == 200
    data = resp.json()
    assert data["use_case"] == "failure_prediction"
    assert "tier1" in data
    assert "n_estimators" in data["tier1"]


def test_hparams_unknown(client):
    """Unknown use case returns 404."""
    resp = client.get("/v1/hparams/nonexistent_case")
    assert resp.status_code == 404


def test_extract_features_man_opt_overrides(client, sample_parquet_bytes_man_opt):
    """Features with _MAN suffix should be mandatory, _OPT should be optional."""
    # Ingest
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes_man_opt, "application/octet-stream")},
    )
    dataset_id = resp.json()["dataset_id"]

    # Extract
    resp = client.post("/v1/features/extract", json={"dataset_id": dataset_id})
    assert resp.status_code == 200
    data = resp.json()

    mandatory = data["per_tag_features"]["mandatory"]
    optional = data["per_tag_features"]["optional"]

    # _MAN overrides should be in mandatory
    assert "custom_feature_1_MAN" in mandatory
    assert "custom_feature_2_MAN" in mandatory
    assert "dist_to_max_MAN" in mandatory
    assert "pct_range_MAN" in mandatory

    # _OPT overrides should be in optional
    assert "pct_range_OPT" in optional

    # Default mandatory (raw, roll_mean, etc.) still mandatory
    assert "raw" in mandatory
    assert "roll_mean" in mandatory

    # Default optional (no suffix override) still optional
    assert "my_special_metric" in optional

    # _MAN features must NOT be in optional
    assert "custom_feature_1_MAN" not in optional
    assert "custom_feature_2_MAN" not in optional
