"""
API routes for agricultural predictions.

All prediction endpoints return 503 Service Unavailable when model artifacts are
not installed or when a variant's required inputs cannot be supplied. No fake
predictions are ever returned.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.agricultural import (
    ComparisonTask,
    ComparisonResponse,
    CropTypePredictionRequest,
    CropTypePredictionResponse,
    CropYieldPredictionRequest,
    CropYieldPredictionResponse,
    HealthResponse,
    ModelVariant,
    YieldLevelPredictionRequest,
    YieldLevelPredictionResponse,
)
from app.services.model_loader import ArtifactNotFoundError, model_loader
from app.services.prediction import PredictionError, prediction_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Report backend status and per-task/per-variant model availability."""
    status = model_loader.get_loaded_status()
    any_available = any(
        available for per_task in status.values() for available in per_task.values()
    )
    return HealthResponse(
        status="healthy" if any_available else "no_models_loaded",
        version="1.0.0",
        models_loaded=status,
    )


# ─── Crop Type Classification ───────────────────────────────────────────────

@router.post("/predict/crop-type", response_model=CropTypePredictionResponse)
def predict_crop_type(request: CropTypePredictionRequest):
    try:
        result = prediction_service.predict_crop_type(
            input_data=request.model_dump(mode="json"),
            variant=request.model_variant.value,
        )
        return CropTypePredictionResponse(**result)
    except PredictionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ArtifactNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in crop type prediction")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


# ─── Yield Level Classification ─────────────────────────────────────────────

@router.post("/predict/yield-level", response_model=YieldLevelPredictionResponse)
def predict_yield_level(request: YieldLevelPredictionRequest):
    try:
        result = prediction_service.predict_yield_level(
            input_data=request.model_dump(mode="json"),
            variant=request.model_variant.value,
        )
        return YieldLevelPredictionResponse(**result)
    except PredictionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ArtifactNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in yield level prediction")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


# ─── Crop Yield Regression ──────────────────────────────────────────────────

@router.post("/predict/crop-yield", response_model=CropYieldPredictionResponse)
def predict_crop_yield(request: CropYieldPredictionRequest):
    try:
        result = prediction_service.predict_crop_yield(
            input_data=request.model_dump(mode="json"),
            variant=request.model_variant.value,
        )
        return CropYieldPredictionResponse(**result)
    except PredictionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ArtifactNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in crop yield prediction")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


# ─── Model Comparison ────────────────────────────────────────────────────────

@router.get("/compare", response_model=ComparisonResponse)
def compare_models(
    task: ComparisonTask = ComparisonTask.CROP_TYPE,
    region: str = "Mandalay",
    year: int = 2025,
    crop_type: str = "Rice",
    sown_acre: float = 200,
    soil_type: str = "Alluvial",
    avg_temperature: float = 28,
    total_rainfall: float = 150,
    avg_humidity: float = 65,
    water_source: str = "Canal",
    seeding_season: str = "Rainy",
):
    """Compare all real available model variants for a task (metrics from artifacts)."""
    input_data = {
        "region": region,
        "year": year,
        "crop_type": crop_type,
        "sown_acre": sown_acre,
        "soil_type": soil_type,
        "avg_temperature": avg_temperature,
        "total_rainfall": total_rainfall,
        "avg_humidity": avg_humidity,
        "water_source": water_source,
        "seeding_season": seeding_season,
    }

    try:
        result = prediction_service.compare_models(
            input_data=input_data,
            task_id=task.value,
        )
        return ComparisonResponse(**result)
    except PredictionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except ArtifactNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in model comparison")
        raise HTTPException(status_code=500, detail=f"Internal comparison error: {e}")


# ─── Historical Trends ──────────────────────────────────────────────────────

@router.get("/historical")
def get_historical_trends(
    region: str = "Mandalay",
    crop_type: str = "Paddy",
):
    """
    Return REAL historical yield trends for a region and crop from the bundled
    cleaned dataset (aggregated by year). Only real values are returned; no
    fabricated data. Returns 404 when no matching real rows exist.
    """
    from app.preprocessing.reference import historical_trends_for

    trends = historical_trends_for(region=region, crop_type=crop_type)
    if trends is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No real historical data found for region='{region}' "
                f"and crop_type='{crop_type}'."
            ),
        )
    return trends


@router.get("/historical/overview")
def get_historical_overview(
    region: str = "Mandalay",
):
    """
    Return REAL per-crop historical yield series for a region from the bundled
    cleaned dataset (aggregated by year), including the available year range.
    Only real values are returned; no fabricated data. Returns 404 when the
    region has no matching real rows.
    """
    from app.preprocessing.reference import historical_overview

    overview = historical_overview(region=region)
    if overview is None:
        raise HTTPException(
            status_code=404,
            detail=f"No real historical data found for region='{region}'.",
        )
    return overview
