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
