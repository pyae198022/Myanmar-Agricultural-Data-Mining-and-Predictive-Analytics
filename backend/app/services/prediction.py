"""
Prediction service — the core orchestration layer.

Responsibilities:
- Receive validated input + model variant
- Build the exact feature row via PreprocessingService (real features only)
- Run the REAL trained model.predict / predict_proba
- Extract real feature importances and package stored real metrics
- Return the frontend response contract (camelCase)

Does NOT:
- Create fake predictions, metrics, confidence, or importances
- Use hash/random encoding
- Modify the trained artifacts
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.models.config import ALL_TASKS, VARIANT_IDS, get_variant_spec
from app.services.model_loader import ArtifactNotFoundError, model_loader
from app.services.preprocessing import PreprocessingError, PreprocessingService

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails (missing artifacts, guarded variant, etc.)."""

    def __init__(self, message: str, status_code: int = 503):
        self.status_code = status_code
        super().__init__(message)


class PredictionService:
    """
    Orchestrates: input -> preprocess -> predict -> format response.
    """

    def __init__(self):
        self.preprocessor = PreprocessingService()

    # ─── Public API methods ─────────────────────────────────────────────────

    def predict_crop_type(self, input_data: dict, variant: str) -> dict:
        task_config = ALL_TASKS["crop_type"]
        return self._predict_classification(input_data, variant, task_config)

    def predict_yield_level(self, input_data: dict, variant: str) -> dict:
        task_config = ALL_TASKS["yield_level"]
        return self._predict_classification(input_data, variant, task_config)

    def predict_crop_yield(self, input_data: dict, variant: str) -> dict:
        task_config = ALL_TASKS["crop_yield"]
        return self._predict_regression(input_data, variant, task_config)

    def compare_models(self, input_data: dict, task_id: str) -> dict:
        """
        Run prediction across all REAL available variants for a task.
        Generalized by the registry (e.g. crop_type has no advanced).

        Returns dict with keys: models, best_model, comparison_metrics.
        """
        if task_id not in ALL_TASKS:
            raise PredictionError(f"Unknown comparison task: {task_id}", status_code=400)

        task_config = ALL_TASKS[task_id]
        results = []
        for variant in VARIANT_IDS:
            spec = get_variant_spec(task_id, variant)
            if spec is None or spec.guarded:
                results.append({
                    "variant": variant,
                    "name": _variant_display_name(variant),
                    "prediction": None,
                    "metrics": None,
                    "available": False,
                })
                continue

            try:
                if task_config.model_type == "classification":
                    if task_id == "yield_level":
                        prediction = self.predict_yield_level(input_data, variant)
                    else:
                        prediction = self.predict_crop_type(input_data, variant)
                    metrics = prediction.get("modelMetrics", {})
                    best_key = "accuracy"
                else:
                    prediction = self.predict_crop_yield(input_data, variant)
                    metrics = prediction.get("modelMetrics", {})
                    best_key = "r2"

                results.append({
                    "variant": variant,
                    "name": _variant_display_name(variant),
                    "prediction": prediction,
                    "metrics": metrics,
                    "available": True,
                })
            except (PredictionError, ArtifactNotFoundError) as e:
                results.append({
                    "variant": variant,
                    "name": _variant_display_name(variant),
                    "prediction": None,
                    "metrics": None,
                    "error": str(e),
                    "available": False,
                })

        valid = [r for r in results if r.get("metrics")]
        best_model = None
        if valid:
            if task_config.model_type == "classification":
                best_model = max(valid, key=lambda r: r["metrics"].get("accuracy", 0))["variant"]
            else:
                best_model = max(valid, key=lambda r: r["metrics"].get("r2", 0))["variant"]

        return {
            "models": results,
            "best_model": best_model,
            "comparison_metrics": self._build_comparison_metrics(results, task_id),
        }

    # ─── Internals ──────────────────────────────────────────────────────────

    def _load_artifacts(self, task_id: str, variant: str):
        """Load model + preprocessor + metrics, raising PredictionError on failures."""
        try:
            model = model_loader.get_model(task_id, variant)
        except ArtifactNotFoundError as e:
            raise PredictionError(
                f"Model artifact not available: {e}", status_code=503
            )
        preprocessor = model_loader.get_preprocessor(task_id, variant)
        metrics = model_loader.get_metrics(task_id, variant)
        return model, preprocessor, metrics

    def _predict_classification(self, input_data: dict, variant: str, task_config) -> dict:
        spec = get_variant_spec(task_config.task_id, variant)
        if spec is not None and spec.guarded:
            raise PredictionError(
                spec.guarded_reason or f"{task_config.task_id}/{variant} is not available.",
                status_code=503,
            )
        model, preprocessor, metrics = self._load_artifacts(task_config.task_id, variant)

        try:
            X = self.preprocessor.transform(
                input_data, task_config.task_id, variant, model, preprocessor
            )
        except PreprocessingError as e:
            raise PredictionError(str(e), status_code=e.status_code)

        predicted_class = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            proba_by_class = dict(zip(classes, proba))
        else:
            proba_by_class = {}
            classes = [predicted_class]
            proba = None

        if task_config.task_id == "crop_type":
            return self._format_crop_type(
                predicted_class, proba_by_class, classes, proba, model, spec, metrics
            )
        # yield_level
        return self._format_yield_level(
            predicted_class, proba_by_class, classes, proba, model, spec, metrics
        )

    def _predict_regression(self, input_data: dict, variant: str, task_config) -> dict:
        spec = get_variant_spec(task_config.task_id, variant)
        if spec is not None and spec.guarded:
            raise PredictionError(
                spec.guarded_reason or f"{task_config.task_id}/{variant} is not available.",
                status_code=503,
            )
        model, preprocessor, metrics = self._load_artifacts(task_config.task_id, variant)

        try:
            X = self.preprocessor.transform(
                input_data, task_config.task_id, variant, model, preprocessor
            )
        except PreprocessingError as e:
            raise PredictionError(str(e), status_code=e.status_code)

        predicted_yield = float(model.predict(X)[0])
        if np.isnan(predicted_yield):
            raise PredictionError("Model produced a NaN prediction.", status_code=503)

        feature_importance = self._extract_feature_importance(model, spec)

        return {
            "predictedYield": round(predicted_yield, 4),
            "featureImportance": feature_importance,
            "modelMetrics": metrics if metrics else {},
        }

    # ─── Formatting helpers ─────────────────────────────────────────────────

    def _format_crop_type(self, predicted_class, proba_by_class, classes, proba,
                          model, spec, metrics) -> dict:
        # Top predictions from real probabilities.
        top = sorted(proba_by_class.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_predictions = [
            {"crop": c, "probability": round(float(float(p)), 4)}
            for c, p in top
        ]
        confidence = round(float(proba_by_class[predicted_class]), 4)
        if np.isnan(confidence):
            confidence = 0.0

        return {
            "predictedCrop": str(predicted_class),
            "confidence": confidence,
            "topPredictions": top_predictions,
            "featureImportance": self._extract_feature_importance(model, spec),
            "modelMetrics": metrics if metrics else {},
        }

    def _format_yield_level(self, predicted_class, proba_by_class, classes, proba,
                            model, spec, metrics) -> dict:
        # Yield level is ONLY Low or High.
        p_low = float(proba_by_class.get("Low", 0.0))
        p_high = float(proba_by_class.get("High", 0.0))

        return {
            "predictedLevel": str(predicted_class),
            "probabilityLow": round(p_low, 4),
            "probabilityHigh": round(p_high, 4),
            "featureImportance": self._extract_feature_importance(model, spec),
            "modelMetrics": metrics if metrics else {},
        }

    # ─── Feature importance mapping ─────────────────────────────────────────

    def _extract_feature_importance(self, model: Any, spec) -> list[dict]:
        """
        Extract real importances from the trained RF and map them back to the
        ORIGINAL feature names (aggregating one-hot expanded columns per source
        feature). Returns [] when the model has no importances (e.g. MLP).
        """
        if spec is None or getattr(spec, "exports_feature_importance", True) is False:
            return []

        importances = None

        # Full pipeline: look for the estimator's feature_importances_
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_importances_"):
                    importances = step.feature_importances_
                    break
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return []

        # Map encoded feature names (if available) back to source features.
        source_names = None
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    try:
                        feat_names = step.get_feature_names_out()
                        source_names = [self._base_feature_name(n) for n in feat_names]
                        break
                    except Exception:
                        continue

        total = float(np.sum(importances))
        if total <= 0:
            return []

        # Aggregate: original feature -> sum of its one-hot/numeric importances.
        grouped: dict[str, float] = {}
        if source_names is not None and len(source_names) == len(importances):
            for name, imp in zip(source_names, importances):
                grouped[name] = grouped.get(name, 0.0) + float(imp)
        else:
            # Fallback to spec.features positional (only valid if same length).
            feats = spec.features
            if len(feats) == len(importances):
                grouped = {f: float(i) for f, i in zip(feats, importances)}
            else:
                return []

        result = [
            {"feature": name, "importance": round(imp / total, 4)}
            for name, imp in grouped.items()
            if name
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    @staticmethod
    def _base_feature_name(encoded: str) -> str:
        """
        Reduce a preprocessor feature name like ``cat__Soil_Type_Alluvial`` or
        ``num__Year`` to the source column (``Soil_Type``, ``Year``).

        Handles interaction columns (``cat__Crop_x_WaterSource_Paddy | Rainfed``
        -> ``Crop_x_WaterSource``) and one-hot values.
        """
        # strip step prefix
        if "__" in encoded:
            encoded = encoded.split("__", 1)[1]
        # Known categorical source columns -> keep everything up to the value.
        for col in (
            "Crop_Type", "Soil_Type", "Water_Source", "Seeding_Season",
        ):
            prefix = col + "_"
            if encoded.startswith(prefix):
                return col
        for col in (
            "Crop_x_WaterSource", "Crop_x_SeedingSeason", "Crop_x_SoilType",
        ):
            if encoded.startswith(col + "_"):
                return col
        # Region encoded as Region_<VALUE>
        if encoded.startswith("Region_"):
            return "Region"
        return encoded

    def _build_comparison_metrics(self, results: list[dict], task_id: str) -> list[dict]:
        task_config = ALL_TASKS[task_id]
        if task_config.model_type == "classification":
            metric_keys = [
                ("Accuracy", "accuracy"),
                ("F1-Score", "f1"),
                ("Precision", "precision"),
                ("Recall", "recall"),
            ]
        else:
            metric_keys = [
                ("R² Score", "r2"),
                ("RMSE", "rmse"),
                ("MAE", "mae"),
            ]

        rows = []
        available = [r for r in results if r.get("metrics")]
        for display_name, key in metric_keys:
            row = {"metric": display_name}
            for r in available:
                m = r["metrics"] or {}
                if key in m:
                    row[r["variant"]] = m[key]
            rows.append(row)
        return rows


def _variant_display_name(variant: str) -> str:
    from app.models.config import VARIANT_DISPLAY_NAMES
    return VARIANT_DISPLAY_NAMES.get(variant, variant)


# ─── Singleton ───────────────────────────────────────────────────────────────

prediction_service = PredictionService()
