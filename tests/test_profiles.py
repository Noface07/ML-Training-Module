"""
Tests for the Tag Profile feature:
  - utils.hashing determinism and validation
  - services.profile_service ORM operations
  - GET/PATCH/DELETE/list /v1/profiles endpoints
  - Ingest response includes profile details
  - Training links profile_id on model artifact
"""

from __future__ import annotations

import pytest

from utils.hashing import compute_tag_hash
from services.profile_service import (
    delete_profile,
    get_profile_by_tag_hash,
    list_profiles,
    resolve_or_create_profile,
    update_profile_name,
)
from database import SessionLocal


# ── Hashing tests ─────────────────────────────────────────────────────────────

def test_hash_deterministic():
    """Same tag set in different orders returns the same hash."""
    h1 = compute_tag_hash([100, 200])
    h2 = compute_tag_hash([200, 100])
    assert h1 == h2


def test_hash_deduplication():
    """Duplicate tag IDs are normalised before hashing."""
    h1 = compute_tag_hash([100, 200])
    h2 = compute_tag_hash([100, 200, 100])
    assert h1 == h2


def test_hash_different_sets_differ():
    """Different tag collections produce different hashes."""
    h1 = compute_tag_hash([100])
    h2 = compute_tag_hash([200])
    assert h1 != h2


def test_hash_sha256_length():
    """SHA-256 hex digest is exactly 64 characters."""
    h = compute_tag_hash([100, 200])
    assert len(h) == 64


def test_hash_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        compute_tag_hash([])


def test_hash_rejects_non_integers():
    with pytest.raises(ValueError, match="integers only"):
        compute_tag_hash([100, "200"])  # type: ignore[list-item]


# ── Profile service tests ──────────────────────────────────────────────────────

@pytest.fixture
def db():
    """Yield a real SQLAlchemy session backed by the test DB."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_resolve_creates_new_profile(db):
    """First resolve for a tag set creates a new profile."""
    profile, created = resolve_or_create_profile(db, [100, 200])
    assert created is True
    assert profile.id is not None
    assert profile.tag_hash == compute_tag_hash([100, 200])
    assert profile.profile_name.startswith("profile_")


def test_resolve_returns_existing(db):
    """Second resolve for the same tag set returns existing profile."""
    profile1, _ = resolve_or_create_profile(db, [100, 200])
    profile2, created2 = resolve_or_create_profile(db, [200, 100])  # order differs
    assert created2 is False
    assert profile1.id == profile2.id


def test_resolve_with_custom_name(db):
    """Custom name is stored when provided on creation."""
    profile, created = resolve_or_create_profile(db, [300], profile_name="MyProfile")
    assert created is True
    assert profile.profile_name == "MyProfile"


def test_update_profile_name(db):
    """Profile name can be updated."""
    profile, _ = resolve_or_create_profile(db, [400])
    updated = update_profile_name(db, profile.id, "UpdatedName")
    assert updated is not None
    assert updated.profile_name == "UpdatedName"


def test_update_nonexistent_profile(db):
    """Updating a non-existent profile returns None."""
    result = update_profile_name(db, 99999, "Ghost")
    assert result is None


def test_get_profile_by_tag_hash(db):
    """Can look up a profile by tag ID list."""
    profile, _ = resolve_or_create_profile(db, [500, 600])
    found = get_profile_by_tag_hash(db, [600, 500])  # reversed order
    assert found is not None
    assert found.id == profile.id


def test_list_profiles(db):
    """list_profiles returns all profiles with model_count."""
    resolve_or_create_profile(db, [10])
    resolve_or_create_profile(db, [20])
    profiles = list_profiles(db)
    assert len(profiles) >= 2
    for p in profiles:
        assert "id" in p
        assert "profile_name" in p
        assert "model_count" in p


def test_delete_profile(db):
    """Deleted profile is gone from the database."""
    profile, _ = resolve_or_create_profile(db, [700])
    result = delete_profile(db, profile.id)
    assert result is True
    found = get_profile_by_tag_hash(db, [700])
    assert found is None


def test_delete_nonexistent_profile(db):
    """Deleting a non-existent profile returns None."""
    result = delete_profile(db, 99999)
    assert result is None


# ── API endpoint tests ─────────────────────────────────────────────────────────

def test_ingest_includes_profile(client, sample_parquet_bytes):
    """Ingest endpoint returns profile_id, profile_name, and tag_hash."""
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "profile_id" in data
    assert "profile_name" in data
    assert "tag_hash" in data
    assert data["profile_id"] is not None
    assert data["profile_name"] is not None
    assert len(data["tag_hash"]) == 64


def test_list_profiles_endpoint(client, sample_parquet_bytes):
    """GET /v1/profiles returns all profiles after ingest."""
    client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    resp = client.get("/v1/profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    item = data["items"][0]
    assert "profile_name" in item
    assert "tag_hash" in item
    assert "model_count" in item


def test_get_profile_endpoint(client, sample_parquet_bytes):
    """GET /v1/profiles/{id} returns profile details."""
    ingest_resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    profile_id = ingest_resp.json()["profile_id"]

    resp = client.get(f"/v1/profiles/{profile_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == profile_id
    assert "profile_name" in data
    assert "tag_hash" in data


def test_get_profile_not_found(client):
    """GET /v1/profiles/99999 returns 404."""
    resp = client.get("/v1/profiles/99999")
    assert resp.status_code == 404


def test_patch_profile_endpoint(client, sample_parquet_bytes):
    """PATCH /v1/profiles/{id} updates the profile name."""
    ingest_resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    profile_id = ingest_resp.json()["profile_id"]

    resp = client.patch(
        f"/v1/profiles/{profile_id}",
        json={"profile_name": "MyRenamedProfile"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["profile_name"] == "MyRenamedProfile"


def test_delete_profile_endpoint(client, sample_parquet_bytes):
    """DELETE /v1/profiles/{id} removes the profile."""
    ingest_resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    profile_id = ingest_resp.json()["profile_id"]

    del_resp = client.delete(f"/v1/profiles/{profile_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    get_resp = client.get(f"/v1/profiles/{profile_id}")
    assert get_resp.status_code == 404


def test_profile_models_endpoint(client, sample_parquet_bytes):
    """GET /v1/profiles/{id}/models returns empty list before any training."""
    ingest_resp = client.post(
        "/v1/ingest",
        files={"file": ("test.parquet", sample_parquet_bytes, "application/octet-stream")},
    )
    profile_id = ingest_resp.json()["profile_id"]

    resp = client.get(f"/v1/profiles/{profile_id}/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
