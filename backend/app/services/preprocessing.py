"""
Preprocessing service for transforming raw user inputs into model-ready features.

Reproduces the EXACT feature pipeline used during training for each real model
artifact. There is no hash-based / random encoding anywhere: every value passed
to a model is either a real user input, a real derived agronomic feature, a real
association rule stat, or a real historical yield-level derived from the bundled
clean dataset.

Two artifact styles are handled:
1. sklearn Pipeline / ColumnTransformer artifacts (crop_type, crop_yield, and
   yield_level advanced): the fitted preprocessor lives inside the pipeline, so
   we build the raw feature frame and let the pipeline transform + predict.
2. Raw MLP + a serialized ``{dummy_columns, minmax}`` bundle (yield_level
   feature_engineering): we reproduce ``pd.get_dummies(dtype=float)``, reindex
   to the saved dummy column order, then apply the fitted MinMaxScaler.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.models.config import get_variant_spec
from app.preprocessing import reference

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when a model cannot be prepared due to missing/unsuppliable inputs."""

    def __init__(self, message: str, status_code: int = 400):
        self.status_code = status_code
        super().__init__(message)


# Input keys accepted from the frontend (snake_case canonical).
_NUMERIC_INPUT_KEYS = {
    "year",
    "sown_acre",
    "avg_temperature",
    "total_rainfall",
    "avg_humidity",
}

_CATEGORICAL_INPUT_KEYS = {
    "region",
    "crop_type",
    "soil_type",
    "water_source",
    "seeding_season",
}


class PreprocessingService:
    """
    Builds the exact feature row expected by each trained artifact and returns
    a model-ready feature representation (array or DataFrame).
    """

    def __init__(self):
        pass

    # ─── Public entrypoint ──────────────────────────────────────────────────

    def transform(
        self,
        input_data: dict,
        task_id: str,
        variant: str,
        model: Any = None,
        preprocessor: Any = None,
    ) -> Any:
        """
        Transform raw input into a model-ready feature representation.

        Returns:
          - a pandas DataFrame of raw features for Pipeline models (so the
            pipeline applies its own fitted preprocessor), or
          - a 2D float ndarray for the raw-MLP + dummy/minmax bundle.

        Raises PreprocessingError for guarded/missing variants.
        """
        spec = get_variant_spec(task_id, variant)
        if spec is None:
            raise PreprocessingError(
                f"Unknown task/variant: {task_id}/{variant}", status_code=400
            )
        if spec.guarded:
            raise PreprocessingError(
                spec.guarded_reason or f"{task_id}/{variant} is not available.",
                status_code=503,
            )

        raw = self._normalize_input(input_data)
        frame = self._build_frame(task_id, variant, spec, raw)

        if task_id == "yield_level" and variant == "feature_engineering":
            # Raw MLP + pd.get_dummies + MinMaxScaler bundle.
            if preprocessor is None or not isinstance(preprocessor, dict):
                raise PreprocessingError(
                    f"yield_level/{variant} requires a serialized preprocessing "
                    "bundle (dummy_columns + minmax).",
                    status_code=503,
                )
            dummy_cols = preprocessor["dummy_columns"]
            scaler = preprocessor["minmax"]
            dummies = pd.get_dummies(frame, dtype=float)
            dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
            return scaler.transform(dummies)

        # Pipeline models: pass the raw feature frame; the fitted preprocessor
        # inside the pipeline handles encoding/scaling.
        return frame

    # ─── Input normalisation ────────────────────────────────────────────────

    @staticmethod
    def _normalize_input(input_data: dict) -> dict:
        """
        Map frontend keys to canonical snake_case field names.

        Accepts both snake_case (Pydantic model_dump) and camelCase (raw frontend
        payload) so the API and future direct callers agree.
        """
        aliases = {
            "sownAcre": "sown_acre",
            "avgTemperature": "avg_temperature",
            "totalRainfall": "total_rainfall",
            "avgHumidity": "avg_humidity",
            "waterSource": "water_source",
            "seedingSeason": "seeding_season",
            "cropType": "crop_type",
            "soilType": "soil_type",
        }
        out = {}
        for k, v in input_data.items():
            out[aliases.get(k, k)] = v
        return out

    # ─── Feature frame building ─────────────────────────────────────────────

    def _build_frame(self, task_id: str, variant: str, spec, raw: dict) -> pd.DataFrame:
        """
        Build a DataFrame whose columns match spec.features in order, applying
        the task/variant-specific engineering.
        """
        # Start with the canonical value lookups.
        raw_feature = self._existing_feature_values(raw)

        row = {}
        for col in spec.features:
            if col in raw_feature:
                row[col] = raw_feature[col]
            elif col == "Log_Sown_Acre":
                row[col] = float(np.log1p(max(float(raw.get("sown_acre", 0.0)), 0.0)))
            elif col == "Log_Total_Rainfall":
                row[col] = float(np.log1p(max(float(raw.get("total_rainfall", 0.0)), 0.0)))
            elif col.startswith("Assoc_"):
                # Association features are computed together below.
                continue
            elif col.startswith("Prev_Yield_") or col.startswith("Prev2_") or col.startswith("Prev3_"):
                continue
            else:
                row[col] = None

        # Add the three interaction features if requested.
        if any(c.startswith("Crop_x_") for c in spec.features):
            crop = str(raw.get("crop_type", ""))
            row["Crop_x_WaterSource"] = f"{crop} | {str(raw.get('water_source', ''))}"
            row["Crop_x_SeedingSeason"] = f"{crop} | {str(raw.get('seeding_season', ''))}"
            row["Crop_x_SoilType"] = f"{crop} | {str(raw.get('soil_type', ''))}"

        # Add association features (needs Region/Crop_Type/Soil_Type/Season/Water).
        if any(c.startswith("Assoc_") for c in spec.features):
            assoc_row = {
                "Region": raw.get("region"),
                "Crop_Type": raw.get("crop_type"),
                "Soil_Type": raw.get("soil_type"),
                "Seeding_Season": raw.get("seeding_season"),
                "Water_Source": raw.get("water_source"),
            }
            assoc = reference.build_association_features(assoc_row, task_id)
            row.update(assoc)

        # Add sequential features from historical data.
        if any(c.startswith("Prev_Yield_") or c.startswith("Prev2_") or c.startswith("Prev3_")
               for c in spec.features):
            seq = reference.build_sequential_features(
                region=raw.get("region"),
                crop_type=raw.get("crop_type"),
                year=int(raw.get("year", 0)),
            )
            row.update(seq)

        # Assemble in spec.features order.
        ordered = {col: row.get(col) for col in spec.features}
        return pd.DataFrame([ordered], columns=spec.features)

    @staticmethod
    def _existing_feature_values(raw: dict) -> dict:
        """Map the 12 existing (non-engineered) feature keys to their values."""
        mapping = {
            "Year": raw.get("year"),
            "Crop_Type": raw.get("crop_type"),
            "Sown_Acre": raw.get("sown_acre"),
            "Soil_Type": raw.get("soil_type"),
            "Avg_Temperature": raw.get("avg_temperature"),
            "Total_Rainfall": raw.get("total_rainfall"),
            "Avg_Humidity": raw.get("avg_humidity"),
            "Water_Source": raw.get("water_source"),
            "Seeding_Season": raw.get("seeding_season"),
            # Baseline-exclusive / macro features (all optional real lookups):
            "Region": raw.get("region"),
            "Harvested_Acre": raw.get("harvested_acre"),
            "Production_Ton": raw.get("production_ton"),
        }
        macro = reference.macro_features_for_year(int(raw.get("year", 0) or 0))
        mapping["Fertilizer_Import_Value(USD)"] = raw.get(
            "fertilizer_import_value_usd", macro.get("Fertilizer_Import_Value(USD)")
        )
        mapping["Myanmar_GDP_USD"] = raw.get(
            "myanmar_gdp_usd", macro.get("Myanmar_GDP_USD")
        )
        return mapping
