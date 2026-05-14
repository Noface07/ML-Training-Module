"""
ML Platform API — FastAPI application entry point.

Registers all routers and creates required directories on startup.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import settings
from database import init_db
from api import ingest, features, train, registry, profiles
from utils.errors import MLPlatformError

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + create directories."""
    init_db()
    for d in [settings.ARTIFACTS_DIR, settings.DATASETS_DIR, settings.LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.info("ML Platform started — DB initialised, dirs ready.")
    yield
    logger.info("ML Platform shutting down.")


app = FastAPI(
    title="ML Platform API",
    version="1.0.0",
    description="Production-grade Industrial ML Platform for predictive maintenance.",
    lifespan=lifespan,
)

# ── Exception handler for structured errors ────────────────────────────
@app.exception_handler(MLPlatformError)
async def ml_error_handler(request: Request, exc: MLPlatformError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


# ── Register routers ──────────────────────────────────────────────────
app.include_router(ingest.router,    prefix="/v1", tags=["Ingest"])
app.include_router(features.router,  prefix="/v1", tags=["Features"])
app.include_router(train.router,     prefix="/v1", tags=["Train"])
app.include_router(registry.router,  prefix="/v1", tags=["Registry"])
app.include_router(profiles.router,  prefix="/v1", tags=["Profiles"])


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok"}
