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


# ─── Data Statistics ─────────────────────────────────────────────────────────

@router.get("/stats")
def get_data_statistics():
    """
    Return descriptive statistics computed from the REAL bundled dataset
    (``backend/data/cleaned_data.csv``): dataset overview, numerical statistics,
    categorical distributions, and data-quality details. All values are derived
    at request time via the existing data-loading utility; nothing is hardcoded.
    """
    from app.preprocessing.reference import data_statistics

    return data_statistics()


# ─── Descriptive Mining ──────────────────────────────────────────────────────

@router.get("/descriptive-mining")
def get_descriptive_mining():
    """
    Return the descriptive-mining report computed from the REAL bundled dataset.

    Covers the four Project Book methods where real results can be produced from
    ``backend/data/cleaned_data.csv``:
      - Correlation Analysis (Pearson)
      - Association Rule Mining (Apriori: Yield_Level + Crop_Type)
      - Frequent Pattern Mining (Apriori itemsets: Crop Yield + Crop Type)
      - Sequential Pattern Mining (order-2 / order-3 by Region-Crop_Type)

    Every value is computed at request time from the dataset; none are hardcoded
    or fabricated. No model artifacts are modified.
    """
    from app.preprocessing.descriptive import descriptive_mining_report

    return descriptive_mining_report()


# ─── Clustering Analysis ──────────────────────────────────────────────────────

@router.get("/clustering/overview")
def get_clustering_overview():
    """
    Return the K-Means clustering overview computed from the real bundled dataset.

    Uses the Project Book methodology:
      - exact Chapter 3.1.6 input features
      - log1p on the specified skewed features
      - Min-Max scaling to [0, 1]
      - KMeans with random_state=42 and n_init=10
      - final configured K = 8
    """
    from app.services.clustering import clustering_overview

    return clustering_overview()


@router.get("/clustering/optimal-k")
def get_clustering_optimal_k():
    """
    Return inertia and silhouette results for candidate K values 2 through 10,
    plus the configured final selection K=8.
    """
    from app.services.clustering import clustering_optimal_k

    return clustering_optimal_k()


@router.get("/clustering/profiles")
def get_clustering_profiles():
    """
    Return cluster counts, percentages, numeric means, dominant categorical
    attributes, and generated interpretation rules for each cluster.
    """
    from app.services.clustering import clustering_profiles

    return clustering_profiles()


@router.get("/clustering/evaluation")
def get_clustering_evaluation():
    """
    Return overall and per-cluster silhouette statistics for the final K=8
    clustering result, derived from the normalized feature matrix.
    """
    from app.services.clustering import clustering_evaluation

    return clustering_evaluation()


# ─── Chapter 4 ROC / AUC Evaluation (Crop Type) ──────────────────────────────

@router.get("/evaluation/roc")
def get_roc_evaluation(variant: str = "baseline"):
    """
    Chapter 4 ROC-Curve & AUC evaluation for the Crop Type task.

    Returns per-class AUC and test support read from the REAL Chapter 4 result
    files, plus the real one-vs-rest ROC curve points recomputed from the
    deployed Crop Type model artifacts on the held-out 2022-2023 (990-row) test
    set. Macro / weighted ROC-AUC and accuracy are derived from the supplied
    evaluation data.

    Query params:
      - variant: "baseline" (default) or "feature_engineering"
    """
    from app.preprocessing.roc_evaluation import (
        available_variants,
        roc_evaluation_report,
        roc_evaluation_summary,
    )

    variants = available_variants()
    if variant not in variants:
        raise HTTPException(
            status_code=404,
            detail=f"No Chapter 4 ROC/AUC result available for variant='{variant}'. "
                   f"Available: {variants}",
        )

    report = roc_evaluation_report(variant)
    summary = roc_evaluation_summary()
    return {
        "task": "Crop Type",
        "selected_variant": variant,
        "report": report,
        "summary": summary,
    }


# ─── Crop Yield Evaluation (Actual vs Predicted / CV / Explainability) ──────

@router.get("/evaluation/crop-yield/actual-vs-predicted")
def get_crop_yield_actual_vs_predicted(variant: str = "feature_engineering"):
    """
    Return REAL actual-vs-predicted points for a Crop Yield variant, computed
    from the held-out 2023 test set with the deployed model + exact
    preprocessing pipeline. Metrics (R² / RMSE / MAE) are derived from the
    actuals and predictions. Unavailable variants are reported, not fabricated.
    """
    from app.services.crop_yield_evaluation import actual_vs_predicted

    return actual_vs_predicted(variant=variant)


@router.get("/evaluation/crop-yield/cross-validation")
def get_crop_yield_cross_validation():
    """
    Return the Project Book's documented NN cross-validation reference values
    for Crop Yield, plus a clear note that the deployed Random Forest artifacts
    were not subjected to the same rigorous CV procedure (no invented RF rows).
    """
    from app.services.crop_yield_evaluation import cross_validation

    return cross_validation()


@router.get("/evaluation/crop-yield/feature-importance")
def get_crop_yield_feature_importance(variant: str = "advanced"):
    """
    Return native feature importance from a deployed Crop Yield Random Forest
    model, mapped to readable source features. SHAP is not fabricated here.
    Unavailable variants are reported, not invented.
    """
    from app.services.crop_yield_evaluation import feature_importance

    return feature_importance(variant=variant)


# ─── Crop Yield Level Evaluation (binary ROC/AUC + 5-Fold Stratified CV) ─────

@router.get("/evaluation/yield-level/roc")
def get_yield_level_roc(variant: str = "feature_engineering"):
    """
    Standard BINARY ROC curve + AUC for a Yield Level model, computed from the
    DEPLOYED MLP artifact on the real 2023 (495-row) hold-out test set.
    pos_label='High'. One-vs-Rest is NOT used (Yield Level is binary; OvR is
    only for the 33-class Crop Type task). Baseline is guarded and reported
    unavailable rather than fabricated.
    """
    from app.services.yield_level_evaluation import test_set_roc_auc

    return test_set_roc_auc(variant=variant)


@router.get("/evaluation/yield-level/roc/available")
def get_yield_level_roc_available():
    """Report which Yield Level variants expose real ROC/AUC data."""
    from app.services.yield_level_evaluation import roc_availability

    return {"task": "Crop Yield Level", "models": roc_availability()}


@router.get("/evaluation/yield-level/cross-validation")
def get_yield_level_cross_validation(random_seed: int = 42, n_splits: int = 5):
    """
    Real 5-Fold Stratified Cross-Validation for the Yield Level classification
    task, computed ONLY on the 2012-2022 training rows (2023 hold-out excluded).
    The FE and Advanced MLPs are re-fitted per fold with the documented
    deterministic hyperparameters; association rules are mined from the training
    rows each CV run uses. Reports Accuracy / Precision / Recall / F1 / ROC-AUC
    as Mean +/- Std, plus the Project Book Table 4.9 reference for comparison.
    Baseline is excluded (leakage-guarded). Values are real, not fitted.
    """
    from app.services.yield_level_evaluation import cross_validation

    return cross_validation(random_seed=random_seed, n_splits=n_splits)
