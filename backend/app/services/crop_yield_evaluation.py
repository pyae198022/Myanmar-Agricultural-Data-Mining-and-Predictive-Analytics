"""
Crop Yield evaluation service — Actual vs Predicted, Cross-Validation, and
Feature Importance for the Crop Yield regression task.

All values come from real, available sources; nothing is fabricated:

* ACTUAL VS PREDICTED: computed at request time from the real held-out 2023
  test set (``Crop_Yield`` is the target) using the exact deployed crop_yield
  artifacts and the existing preprocessing pipeline. R² / RMSE / MAE are
  derived from the actual test target and the model's real predictions.
* CROSS-VALIDATION: reported as the Project Book's rigorous CV results, which
  are defined for the NEURAL NETWORK configurations only. The DEPLOYED
  crop_yield artifacts in this application are Random Forest regressors and
  were NOT subjected to the same rigorous CV procedure, so they are reported
  as not-evaluated rather than given invented values.
* FEATURE IMPORTANCE: native ``feature_importances_`` from the deployed Random
  Forest artifacts, mapped back to readable source feature names. SHAP is not
  generated because the deployed model is a Random Forest (not the Project Book
  NN), so native importance is reported and explicitly labelled as such.

No model is retrained and no artifact is modified.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from app.models.config import get_variant_spec
from app.preprocessing import reference
from app.services.model_loader import ArtifactNotFoundError, model_loader
from app.services.preprocessing import PreprocessingService
from app.services.prediction import PredictionService

logger = logging.getLogger(__name__)

TASK_ID = "crop_yield"

# Result caches for the expensive request-time computations. Both depend only on
# the deployed artifacts and the fixed 2023 held-out test set, so the results
# are stable across requests; caching them avoids repeating the per-row
# preprocessing + prediction loop on every page load / page reload.
_ACTUAL_VS_PREDICTED_CACHE: dict[str, dict] = {}
_FEATURE_IMPORTANCE_CACHE: dict[str, dict] = {}

# Variants that are real, deployed Random Forest crop_yield models with native
# feature importances and real held-out predictions.
_REAL_VARIANTS = ["feature_engineering", "advanced"]

# Readable display labels used across the evaluation responses.
VARIANT_LABELS = {
    "baseline": "Baseline Model",
    "feature_engineering": "Feature Engineering",
    "advanced": "Advanced-Association",
}

# ─── Project Book documented reference: NN cross-validation ──────────────────
# The Project Book reports rigorous cross-validation results for the three
# Neural Network configurations. The DEPLOYED crop_yield artifacts are Random
# Forest regressors, so these reference values are presented as documented
# Project Book NN CV results (NOT the deployed models' CV). RF variants are
# reported as not-evaluated to avoid implying they share these results.
_PROJECT_BOOK_NN_CV = [
    {
        "variant": "baseline",
        "label": "Baseline NN",
        "r2": {"mean": 0.9750, "std": 0.0103},
        "rmse": {"mean": 0.6164, "std": 0.1157},
        "mae": {"mean": 0.2807, "std": 0.0573},
    },
    {
        "variant": "feature_engineering",
        "label": "Feature Engineering NN",
        "r2": {"mean": 0.9765, "std": 0.0037},
        "rmse": {"mean": 0.6062, "std": 0.0426},
        "mae": {"mean": 0.1978, "std": 0.0161},
    },
    {
        "variant": "advanced",
        "label": "Association-Enhanced NN",
        "r2": {"mean": 0.9834, "std": 0.0046},
        "rmse": {"mean": 0.5069, "std": 0.0631},
        "mae": {"mean": 0.1830, "std": 0.0308},
    },
]


def _real_test_data() -> pd.DataFrame:
    """The exact held-out Crop Yield test set (2023), sorted chronologically.

    Mirrors the existing evaluation system (495 rows).
    """
    df = reference.load_clean_data()
    df = df.sort_values(["Region", "Crop_Type", "Year"]).reset_index(drop=True)
    return df[df["Year"] == 2023]


def _row_to_input(row: pd.Series) -> dict:
    """Canonical snake_case backend input for a dataset row."""
    return {
        "region": str(row["Region"]),
        "year": int(row["Year"]),
        "crop_type": str(row["Crop_Type"]),
        "sown_acre": float(row["Sown_Acre"]),
        "soil_type": str(row["Soil_Type"]),
        "avg_temperature": float(row["Avg_Temperature"]),
        "total_rainfall": float(row["Total_Rainfall"]),
        "avg_humidity": float(row["Avg_Humidity"]),
        "water_source": str(row["Water_Source"]),
        "seeding_season": str(row["Seeding_Season"]),
    }


def _metrics_from_actuals(y_actual: np.ndarray, y_pred: np.ndarray) -> dict:
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": round(float(r2_score(y_actual, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_actual, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_actual, y_pred)), 4),
        "n_test": int(len(y_actual)),
    }


# ─── Actual vs Predicted ─────────────────────────────────────────────────────

def availability() -> list[dict]:
    """Report which crop_yield variants are real and available for evaluation."""
    rows = []
    for variant in ["baseline", "feature_engineering", "advanced"]:
        real = variant in _REAL_VARIANTS
        rows.append({
            "variant": variant,
            "name": VARIANT_LABELS[variant],
            "available": real and bool(model_loader.get_model(TASK_ID, variant)),
        })
    return rows


def _variant_available(variant: str) -> bool:
    if variant not in _REAL_VARIANTS:
        return False
    try:
        model_loader.get_model(TASK_ID, variant)
        return True
    except ArtifactNotFoundError:
        return False


def actual_vs_predicted(variant: str = "feature_engineering") -> dict:
    """Return REAL actual-vs-predicted points for a variant on the 2023 test set.

    Uses the deployed model + exact preprocessing pipeline, no retraining.
    """
    if not _variant_available(variant):
        return {
            "variant": variant,
            "name": VARIANT_LABELS.get(variant, variant),
            "available": False,
            "reason": "crop_yield/baseline is guarded (requires target-leakage "
                      "inputs); only realistic variants expose evaluation data.",
            "points": [],
            "metrics": None,
            "n_test": 0,
        }

    if variant in _ACTUAL_VS_PREDICTED_CACHE:
        return deepcopy(_ACTUAL_VS_PREDICTED_CACHE[variant])

    test = _real_test_data()
    model = model_loader.get_model(TASK_ID, variant)
    preprocessor = model_loader.get_preprocessor(TASK_ID, variant)
    preproc = PreprocessingService()

    y_actual = test["Crop_Yield"].astype(float).values
    years = test["Year"].astype(int).values
    crops = test["Crop_Type"].astype(str).values
    y_pred = []
    for _, row in test.iterrows():
        X = preproc.transform(_row_to_input(row), TASK_ID, variant, model, preprocessor)
        y_pred.append(float(model.predict(X)[0]))

    y_pred = np.asarray(y_pred, dtype=float)
    metrics = _metrics_from_actuals(y_actual, y_pred)

    # Return all real points (495 rows). Each point also carries the year and
    # crop so the frontend can offer a light, honest visualization.
    points = [
        {
            "actual": round(float(a), 6),
            "predicted": round(float(p), 6),
            "year": int(y),
            "crop_type": str(c),
        }
        for a, p, y, c in zip(y_actual, y_pred, years, crops)
    ]

    result = {
        "variant": variant,
        "name": VARIANT_LABELS[variant],
        "available": True,
        "task": "Crop Yield",
        "test_years": [2023],
        "n_test": int(len(points)),
        "points": points,
        "metrics": metrics,
        "note": (
            "Actual vs Predicted is computed from the real 2023 held-out test "
            "set using the deployed model and the exact preprocessing pipeline. "
            "Predicted values are the model's raw outputs (unrounded); metrics "
            "are derived from these actuals and predictions."
        ),
    }
    _ACTUAL_VS_PREDICTED_CACHE[variant] = deepcopy(result)
    return result


# ─── Cross-Validation ────────────────────────────────────────────────────────

def cross_validation() -> dict:
    """Return the Project Book NN CV reference table plus the RF non-evaluation note.

    The application has no stored CV artifacts, and the deployed crop_yield
    models are Random Forest (not NN), so the rigorous CV results shown are the
    documented Project Book NN reference values. RF variants are explicitly
    reported as not-evaluated rather than given invented numbers.
    """
    models = []
    for ref in _PROJECT_BOOK_NN_CV:
        models.append({
            "variant": ref["variant"],
            "label": ref["label"],
            "r2": ref["r2"],
            "rmse": ref["rmse"],
            "mae": ref["mae"],
            "algorithm": "NN (Project Book)",
            "source": "project_book_reference",
        })

    # The deployed crop_yield artifacts are RandomForestRegressor; they were not
    # subjected to the same rigorous CV procedure, so no RF rows are invented.
    return {
        "task": "Crop Yield",
        "models": models,
        "deployed_algorithms": {
            "feature_engineering": "RandomForestRegressor",
            "advanced": "RandomForestRegressor",
        },
        "note": (
            "Cross-validation results above are the Project Book's documented "
            "reference values for the NEURAL NETWORK configurations. The deployed "
            "crop_yield artifacts are Random Forest regressors, which were not "
            "evaluated with the same rigorous cross-validation procedure, so no "
            "Random Forest cross-validation rows are reported."
        ),
    }


# ─── Feature Importance ──────────────────────────────────────────────────────

def feature_importance(variant: str = "advanced") -> dict:
    """Return native feature importance from the deployed Random Forest model.

    SHAP is not generated because the deployed model is a Random Forest (the
    Project Book's Association-Enhanced NN is not the deployed artifact); native
    RF ``feature_importances_`` are reported and clearly labelled as such.
    """
    if not _variant_available(variant):
        return {
            "variant": variant,
            "name": VARIANT_LABELS.get(variant, variant),
            "available": False,
            "reason": "No real importance data is available for this variant.",
            "features": [],
            "type": None,
        }

    if variant in _FEATURE_IMPORTANCE_CACHE:
        return deepcopy(_FEATURE_IMPORTANCE_CACHE[variant])

    model = model_loader.get_model(TASK_ID, variant)
    spec = get_variant_spec(TASK_ID, variant)

    importances = PredictionService()._extract_feature_importance(model, spec)
    if not importances:
        return {
            "variant": variant,
            "name": VARIANT_LABELS[variant],
            "available": False,
            "reason": "The deployed model exposes no native feature importances.",
            "features": [],
            "type": None,
        }

    result = {
        "variant": variant,
        "name": VARIANT_LABELS[variant],
        "available": True,
        "task": "Crop Yield",
        "type": "Feature Importance",
        "method": "native_random_forest",
        "features": importances,  # [{feature, importance}], sorted desc, sum ~ 1
        "note": (
            "Higher feature importance indicates a stronger contribution to the "
            "model's prediction. This is a measure of association, not causation. "
            "Importance is derived from the deployed Random Forest model's native "
            "feature_importances_ and aggregated to readable source features."
        ),
    }
    _FEATURE_IMPORTANCE_CACHE[variant] = deepcopy(result)
    return result
