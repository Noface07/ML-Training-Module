"""
Shared pytest fixtures.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure we use a temp DB for tests
os.environ["DB_URL"] = "sqlite:///./test_ml_platform.db"

from main import app
from database import init_db, Base, engine


@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_parquet_bytes():
    """Generate a small sample parquet file with tag columns."""
    np.random.seed(42)
    n = 200
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "tag_100_raw": np.random.randn(n).cumsum() + 50,
        "tag_100_roll_mean": np.random.randn(n).cumsum() + 50,
        "tag_100_roll_std": np.abs(np.random.randn(n)) * 2,
        "tag_100_roc_1": np.random.randn(n) * 0.5,
        "tag_100_pct_range": np.random.rand(n),
        "tag_100_dist_to_max": np.random.rand(n) * 10,
        "tag_100_custom_feature_1": np.random.rand(n),
        "tag_100_custom_feature_2": np.random.rand(n),
        "tag_200_raw": np.random.randn(n).cumsum() + 30,
        "tag_200_roll_mean": np.random.randn(n).cumsum() + 30,
        "tag_200_roll_std": np.abs(np.random.randn(n)) * 1.5,
        "tag_200_roc_1": np.random.randn(n) * 0.3,
        "tag_200_pct_range": np.random.rand(n),
        "tag_200_my_special_metric": np.random.rand(n) * 5,
        "system_avg_pct": np.random.rand(n),
        "factory_temp": np.random.rand(n) * 100,
        "will_fail": np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_parquet_bytes_man_opt():
    """Parquet with _MAN and _OPT suffix overrides on custom features."""
    np.random.seed(42)
    n = 200
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min"),
        # standard mandatory
        "tag_100_raw": np.random.randn(n).cumsum() + 50,
        "tag_100_roll_mean": np.random.randn(n).cumsum() + 50,
        "tag_100_roll_std": np.abs(np.random.randn(n)) * 2,
        "tag_100_roc_1": np.random.randn(n) * 0.5,
        # _MAN override → should become mandatory
        "tag_100_custom_feature_1_MAN": np.random.rand(n),
        "tag_100_custom_feature_2_MAN": np.random.rand(n),
        "tag_100_dist_to_max_MAN": np.random.rand(n) * 10,
        # _OPT override → should become optional
        "tag_100_pct_range_OPT": np.random.rand(n),
        # no override → default optional
        "tag_100_my_special_metric": np.random.rand(n),
        # second tag
        "tag_200_raw": np.random.randn(n).cumsum() + 30,
        "tag_200_roll_mean": np.random.randn(n).cumsum() + 30,
        "tag_200_roll_std": np.abs(np.random.randn(n)) * 1.5,
        "tag_200_roc_1": np.random.randn(n) * 0.3,
        "tag_200_pct_range_MAN": np.random.rand(n),
        "tag_200_my_special_metric": np.random.rand(n) * 5,
        "will_fail": np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
