"""
Model artifact loading abstraction.

Responsibilities:
- Load trained model .joblib files (the real generated artifacts)
- Load preprocessing .joblib files
- Load stored evaluation metrics .json files
- Report availability from models/model_registry.json (source of truth)
- Cache loaded artifacts and raise clear errors when artifacts are missing

Does NOT:
- Train models
- Create fake artifacts
- Perform predictions (that's PredictionService)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib

from app.models.config import (
    ALL_TASKS,
    VARIANT_IDS,
    get_metrics_path,
    get_model_path,
    get_preprocessor_path,
    get_variant_spec,
)
from app.preprocessing.reference import variant_available

logger = logging.getLogger(__name__)


class ArtifactNotFoundError(Exception):
    """Raised when a required model artifact file does not exist."""

    def __init__(self, artifact_type: str, path: Path):
        self.artifact_type = artifact_type
        self.path = path
        super().__init__(
            f"{artifact_type} artifact not found at {path}. "
            f"Please train the model and place the .joblib file in the correct location."
        )


class ModelLoader:
    """
    Manages loading and caching of trained model artifacts.

    Usage:
        loader = ModelLoader()
        model = loader.get_model("crop_type", "baseline")
        preprocessor = loader.get_preprocessor("crop_type", "baseline")
        metrics = loader.get_metrics("crop_type", "baseline")
    """

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._preprocessors: dict[str, Any] = {}
        self._metrics: dict[str, dict] = {}

    def _cache_key(self, task_id: str, variant: str) -> str:
        return f"{task_id}/{variant}"

    # ─── Availability (from the real registry) ──────────────────────────────

    def is_available(self, task_id: str, variant: str) -> bool:
        """
        Whether the variant is marked AVAILABLE in model_registry.json.

        Guards are also honoured: a variant that is guarded is reported as
        NOT available for real prediction (it cannot be used without the
        unsuppliable inputs).
        """
        spec = get_variant_spec(task_id, variant)
        if spec is None:
            return False
        if spec.guarded:
            return False
        return variant_available(task_id, variant)

    def get_available_variants(self, task_id: str) -> list[str]:
        """List variants that can actually be used for prediction."""
        return [
            v for v in VARIANT_IDS
            if self.is_available(task_id, v)
        ]

    # ─── Model loading ───────────────────────────────────────────────────────

    def get_model(self, task_id: str, variant: str) -> Any:
        """
        Load and cache a trained model artifact. Raises if it does not exist.
        """
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task: {task_id}")
        if variant not in VARIANT_IDS:
            raise ValueError(f"Unknown variant: {variant}")

        key = self._cache_key(task_id, variant)
        if key in self._models:
            return self._models[key]

        path = get_model_path(task_id, variant)
        if not path.exists():
            raise ArtifactNotFoundError("model", path)

        logger.info("Loading model artifact: %s", path)
        model = joblib.load(path)
        self._models[key] = model
        return model

    def get_preprocessor(self, task_id: str, variant: str) -> Optional[Any]:
        """
        Load and cache a preprocessing artifact.

        Returns None if no separate preprocessor file exists (e.g. advanced
        models keep their preprocessor inside the pipeline).
        """
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task: {task_id}")
        if variant not in VARIANT_IDS:
            raise ValueError(f"Unknown variant: {variant}")

        key = self._cache_key(task_id, variant)
        if key in self._preprocessors:
            return self._preprocessors[key]

        path = get_preprocessor_path(task_id, variant)
        if not path.exists():
            return None

        logger.info("Loading preprocessor artifact: %s", path)
        preprocessor = joblib.load(path)
        self._preprocessors[key] = preprocessor
        return preprocessor

    def get_metrics(self, task_id: str, variant: str) -> Optional[dict]:
        """Load and cache stored evaluation metrics from a JSON file."""
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task: {task_id}")
        if variant not in VARIANT_IDS:
            raise ValueError(f"Unknown variant: {variant}")

        key = self._cache_key(task_id, variant)
        if key in self._metrics:
            return self._metrics[key]

        path = get_metrics_path(task_id, variant)
        if not path.exists():
            return None

        with open(path, "r") as f:
            metrics = json.load(f)
        self._metrics[key] = metrics
        return metrics

    # ─── Runtime status ──────────────────────────────────────────────────────

    def get_loaded_status(self) -> dict[str, dict[str, bool]]:
        """
        Per-task per-variant availability, e.g.:

            crop_type:  { baseline: true, feature_engineering: true, advanced: false }
        """
        status = {}
        for task_id in ALL_TASKS:
            status[task_id] = {
                v: self.is_available(task_id, v) for v in VARIANT_IDS
            }
        return status

    def is_model_loaded(self, task_id: str, variant: str) -> bool:
        key = self._cache_key(task_id, variant)
        return key in self._models

    def unload_all(self):
        self._models.clear()
        self._preprocessors.clear()
        self._metrics.clear()
        logger.info("All model artifacts unloaded")


# ─── Singleton instance ──────────────────────────────────────────────────────

model_loader = ModelLoader()
