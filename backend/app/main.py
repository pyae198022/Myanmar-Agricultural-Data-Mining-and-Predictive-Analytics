"""
FastAPI application — entry point for the backend.

Local development (from the `backend/` directory):
    uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000

Production (Render):
    uvicorn app.main:app --host 0.0.0.0 --port $PORT

`--reload-dir app` keeps auto-reload enabled but scopes the file watcher to the
application source tree only. This prevents spurious reloads when things that
are NOT part of the running app change (e.g. `tests/`, `scripts/`, `models/`,
`data/`, or a `venv/` that sits inside the watched directory). Do NOT run
uvicorn with `--reload` from the repository root: the watcher would then also
cover `venv/*.py`, and any package file change would trigger a reload.

The frontend (Vite dev server) will proxy /api requests to this backend.

Environment variables:
  FRONTEND_ORIGIN  — comma-separated list of allowed frontend origins for CORS.
                     Set on Render to the Vercel production URL, e.g.:
                     https://your-app.vercel.app
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.predict import router as predict_router
from app.services.model_downloader import initialize_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initializes models on startup."""
    logger.info("Starting AgriPredict ML Backend...")
    
    # Initialize/download model artifacts
    initialize_models()
    
    yield
    
    logger.info("Shutting down AgriPredict ML Backend...")


app = FastAPI(
    lifespan=lifespan,
    title="AgriPredict ML Backend",
    description="Agricultural Decision Support & Crop Prediction API",
    version="1.0.0",
)

# ─── CORS — localhost dev + configured production origins ────────────────────

LOCAL_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]


def _parse_frontend_origins() -> list[str]:
    raw = os.environ.get("FRONTEND_ORIGIN", "").strip()
    if not raw:
        return []
    origins = []
    for item in raw.split(","):
        item = item.strip().rstrip("/")
        if item:
            origins.append(item)
    return origins


ALLOWED_ORIGINS = LOCAL_DEV_ORIGINS + _parse_frontend_origins()
logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────────────────────────

app.include_router(predict_router, prefix="/api")
