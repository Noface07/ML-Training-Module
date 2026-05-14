"""
Deterministic hashing utility for integer tag collections.
"""

from __future__ import annotations

import hashlib
import json
from typing import List


def compute_tag_hash(tag_ids: List[int]) -> str:
    """Deterministically hash a collection of tag IDs.

    Steps:
      1. Validate that tag_ids is a non-empty list of integers.
      2. Deduplicate tag IDs.
      3. Sort ascending.
      4. Serialize to canonical compact JSON array string.
      5. Return SHA-256 hex digest.

    Example:
        compute_tag_hash([200, 100])    -> same hash as compute_tag_hash([100, 200])
        compute_tag_hash([100, 200])    -> "abc123..."
    """
    if not isinstance(tag_ids, list) or not tag_ids:
        raise ValueError("tag_ids must be a non-empty list of integers.")

    for t in tag_ids:
        if not isinstance(t, int):
            raise ValueError(f"tag_ids must contain integers only, got: {type(t)}")

    normalized = sorted(set(tag_ids))
    canonical = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
