"""
Central model configuration and variant registry.

Single source of truth for:
- Model variant naming (backend <-> frontend)
- Model artifact paths per task x variant
- Feature lists per task x variant (matching the REAL trained artifacts)
- Expected categorical/numeric columns per variant
- Which variants are guarded (require inputs not obtainable from a user)

The generated artifacts are the source of truth: the feature lists below match
exactly the ``features`` metadata recorded in ``models/model_registry.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── Model artifact root ─────────────────────────────────────────────────────

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"


# ─── Variant definitions ─────────────────────────────────────────────────────

VARIANT_IDS = ("baseline", "feature_engineering", "advanced")

VARIANT_DISPLAY_NAMES = {
    "baseline": "Baseline Model",
    "feature_engineering": "Feature Engineering",
    "advanced": "Advanced-Association",
}

VARIANT_DESCRIPTIONS = {
    "baseline": "Raw unscaled data with original features",
    "feature_engineering": "Optimized selected features with scaling",
    "advanced": "Apriori rule-based interaction features (Soil x Water, etc.)",
}


# ─── Shared feature blocks ───────────────────────────────────────────────────

# 12 selected features + 3 interactions (used by yield_level & crop_yield FE/advanced)
_EXISTING_FEATURES = [
    "Year",
    "Crop_Type",
    "Sown_Acre",
    "Soil_Type",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Water_Source",
    "Seeding_Season",
    "Crop_x_WaterSource",
    "Crop_x_SeedingSeason",
    "Crop_x_SoilType",
]

_EXISTING_CATEGORICAL = [
    "Crop_Type",
    "Soil_Type",
    "Water_Source",
    "Seeding_Season",
    "Crop_x_WaterSource",
    "Crop_x_SeedingSeason",
    "Crop_x_SoilType",
]

_EXISTING_NUMERIC = [
    "Year",
    "Sown_Acre",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
]

ASSOCIATION_FEATURES = [
    "Assoc_High_Confidence",
    "Assoc_High_Lift",
    "Assoc_Low_Confidence",
    "Assoc_Low_Lift",
    "Assoc_Rule_Count",
]

SEQUENCE_FEATURES = [
    "Prev_Yield_High",
    "Prev_Yield_Low",
    "Prev2_Persistent",
    "Prev3_Persistent",
]


# ─── Task configs ────────────────────────────────────────────────────────────


@dataclass
class VariantSpec:
    variant: str
    features: list[str]
    categorical: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    # True when prediction requires inputs that a user cannot meaningfully
    # supply (e.g. Harvested_Acre/Production_Ton baseline leakage features).
    guarded: bool = False
    guarded_reason: str = ""
    # Feature engineering performed on raw input before feeding the artifact.
    # Each entry maps an output feature name -> a tag used by preprocessing.
    exports_feature_importance: bool = True


@dataclass
class TaskConfig:
    task_id: str
    target_column: str
    model_type: str  # "classification" or "regression"
    artifact_subdir: str
    variants: dict[str, VariantSpec] = field(default_factory=dict)


# Base 8 agronomic inputs supplied by the frontend request (camelCase -> snake)
INPUT_KEYS = [
    "region",
    "year",
    "crop_type",
    "sown_acre",
    "soil_type",
    "avg_temperature",
    "total_rainfall",
    "avg_humidity",
    "water_source",
    "seeding_season",
]

# ─── Crop Type ──────────────────────────────────────────────────────────────

CROP_TYPE_TASK = TaskConfig(
    task_id="crop_type",
    target_column="Crop_Type",
    model_type="classification",
    artifact_subdir="crop_type",
    variants={
        "baseline": VariantSpec(
            variant="baseline",
            features=[
                "Region", "Soil_Type", "Water_Source",
                "Avg_Temperature", "Total_Rainfall", "Avg_Humidity",
                "Year", "Sown_Acre",
            ],
            categorical=["Region", "Soil_Type", "Water_Source"],
            numeric=["Avg_Temperature", "Total_Rainfall", "Avg_Humidity", "Year", "Sown_Acre"],
        ),
        "feature_engineering": VariantSpec(
            variant="feature_engineering",
            features=[
                "Log_Sown_Acre", "Log_Total_Rainfall",
                "Avg_Temperature", "Avg_Humidity", "Year",
                "Region", "Soil_Type", "Water_Source",
            ],
            categorical=["Region", "Soil_Type", "Water_Source"],
            numeric=["Log_Sown_Acre", "Log_Total_Rainfall", "Avg_Temperature", "Avg_Humidity", "Year"],
        ),
        # No advanced crop-type model exists in the project.
        "advanced": VariantSpec(
            variant="advanced",
            features=[],
            guarded=True,
            guarded_reason="No advanced crop-type model exists in the project.",
        ),
    },
)

# ─── Yield Level ────────────────────────────────────────────────────────────

YIELD_LEVEL_TASK = TaskConfig(
    task_id="yield_level",
    target_column="Yield_Level",
    model_type="classification",
    artifact_subdir="yield_level",
    variants={
        "baseline": VariantSpec(
            variant="baseline",
            features=[
                "Region", "Year", "Crop_Type", "Sown_Acre",
                "Harvested_Acre", "Production_Ton",
                "Fertilizer_Import_Value(USD)",
                "Avg_Temperature", "Total_Rainfall", "Myanmar_GDP_USD",
                "Soil_Type", "Seeding_Season", "Water_Source", "Avg_Humidity",
            ],
            # get_dummies expands all categoricals; numeric includes raw cols.
            guarded=True,
            guarded_reason=(
                "Baseline yield-level requires Harvested_Acre and Production_Ton "
                "(target-leakage yield-outputs) which cannot be supplied by a "
                "predicting user; guarded to avoid fabricating inputs."
            ),
        ),
        "feature_engineering": VariantSpec(
            variant="feature_engineering",
            features=_EXISTING_FEATURES,
            categorical=_EXISTING_CATEGORICAL,
            numeric=["Year", "Sown_Acre", "Avg_Temperature", "Total_Rainfall", "Avg_Humidity"],
        ),
        "advanced": VariantSpec(
            variant="advanced",
            features=_EXISTING_FEATURES + ASSOCIATION_FEATURES + SEQUENCE_FEATURES,
            categorical=_EXISTING_CATEGORICAL,
            numeric=["Year", "Sown_Acre", "Avg_Temperature", "Total_Rainfall", "Avg_Humidity"]
            + ASSOCIATION_FEATURES + SEQUENCE_FEATURES,
            exports_feature_importance=False,  # MLP has no native importances
        ),
    },
)

# ─── Crop Yield ─────────────────────────────────────────────────────────────

CROP_YIELD_TASK = TaskConfig(
    task_id="crop_yield",
    target_column="Crop_Yield",
    model_type="regression",
    artifact_subdir="crop_yield",
    variants={
        "baseline": VariantSpec(
            variant="baseline",
            features=[
                "Crop_Type", "Sown_Acre", "Harvested_Acre", "Production_Ton",
                "Fertilizer_Import_Value(USD)", "Avg_Temperature",
                "Total_Rainfall", "Myanmar_GDP_USD", "Soil_Type",
                "Seeding_Season", "Water_Source", "Avg_Humidity",
            ],
            categorical=["Crop_Type", "Soil_Type", "Seeding_Season", "Water_Source"],
            numeric=[
                "Sown_Acre", "Harvested_Acre", "Production_Ton",
                "Fertilizer_Import_Value(USD)", "Avg_Temperature",
                "Total_Rainfall", "Myanmar_GDP_USD", "Avg_Humidity",
            ],
            guarded=True,
            guarded_reason=(
                "Baseline crop-yield requires Harvested_Acre and Production_Ton "
                "(target-leakage features where Crop_Yield = Production_Ton / "
                "Harvested_Acre), which a user predicting yield cannot supply. "
                "Guarded to avoid fabricating inputs."
            ),
        ),
        "feature_engineering": VariantSpec(
            variant="feature_engineering",
            features=_EXISTING_FEATURES,
            categorical=_EXISTING_CATEGORICAL,
            numeric=_EXISTING_NUMERIC,
        ),
        "advanced": VariantSpec(
            variant="advanced",
            features=_EXISTING_FEATURES + ASSOCIATION_FEATURES + SEQUENCE_FEATURES,
            categorical=_EXISTING_CATEGORICAL,
            numeric=["Year", "Sown_Acre", "Avg_Temperature", "Total_Rainfall", "Avg_Humidity"]
            + ASSOCIATION_FEATURES + SEQUENCE_FEATURES,
        ),
    },
)


ALL_TASKS: dict[str, TaskConfig] = {
    CROP_TYPE_TASK.task_id: CROP_TYPE_TASK,
    YIELD_LEVEL_TASK.task_id: YIELD_LEVEL_TASK,
    CROP_YIELD_TASK.task_id: CROP_YIELD_TASK,
}


# ─── Artifact path resolution ───────────────────────────────────────────────

def get_model_path(task_id: str, variant: str) -> Path:
    return MODELS_ROOT / task_id / f"{variant}.joblib"


def get_preprocessor_path(task_id: str, variant: str) -> Path:
    return MODELS_ROOT / task_id / f"{variant}_preprocessor.joblib"


def get_metrics_path(task_id: str, variant: str) -> Path:
    return MODELS_ROOT / task_id / f"{variant}_metrics.json"


def get_variant_spec(task_id: str, variant: str) -> Optional[VariantSpec]:
    """Return the variant spec, or None if the task/variant is unknown."""
    task = ALL_TASKS.get(task_id)
    if task is None:
        return None
    return task.variants.get(variant)
