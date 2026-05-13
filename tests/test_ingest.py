"""Tests for the ingest API."""

from __future__ import annotations


def test_ingest_success(client, sample_parquet_bytes):
    """Upload a valid parquet and verify the response."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "dataset_id" in data
    assert data["row_count"] == 200
    assert set(data["tags_detected"]) == {"100", "200"}
    assert "quality_report" in data
    assert "range_metadata" in data
    # Range should have entries for raw tags
    assert "100" in data["range_metadata"]
    assert "200" in data["range_metadata"]


def test_ingest_empty_file(client):
    """Uploading an empty file returns 400."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("empty.parquet", b"", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_ingest_invalid_parquet(client):
    """Uploading non-parquet data returns 400."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("bad.parquet", b"not a parquet file", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_preview_dataset(client, sample_parquet_bytes):
    """Previewing an ingested dataset returns column list and dictionary rows."""
    # 1. Ingest
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    dataset_id = resp.json()["dataset_id"]

    # 2. Preview
    resp2 = client.get(f"/v1/dataset/{dataset_id}/preview?limit=5")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["dataset_id"] == dataset_id
    assert data["row_count"] == 200
    assert len(data["rows"]) == 5
    assert len(data["columns"]) > 0
