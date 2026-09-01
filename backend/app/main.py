"""
FastAPI application — entry point for the backend.

Run with:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

The frontend (Vite dev server) will proxy /api requests to this backend.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="AgriPredict ML Backend",
    description="Agricultural Decision Support & Crop Prediction API",
    version="1.0.0",
)

# ─── CORS — allow Vite dev server ───────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────────────────────────

app.include_router(predict_router, prefix="/api")
