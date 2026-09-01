"""
Generate model_registry.json describing every deployed model artifact.

This is a metadata-only registry written from the metrics JSONs produced
during training and the known project config.
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

# Algorithm / capability info per (task, variant) from training code.
# (pulled from the metrics json files for feature lists, and static here
#  for algorithm/capability info that is identical across a task's models)
TASK_PROPERTIES = {
    "crop_type": {
        "predict_proba": True,
        "feature_importances": True,
    },
    "yield_level": {
        "predict_proba": True,
        "feature_importances": False,  # MLP has no native feature importance
    },
    "crop_yield": {
        "predict_proba": False,
        "feature_importances": True,
    },
}


def load_metrics(task, variant):
    path = MODELS_ROOT / task / f"{variant}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_registry():
    registry = {}

    for task in ["crop_type", "yield_level", "crop_yield"]:
        registry[task] = {}

        variants_in_order = ["baseline", "feature_engineering", "advanced"]

        for variant in variants_in_order:
            metrics = load_metrics(task, variant)

            if metrics is None:
                if task == "crop_type" and variant == "advanced":
                    registry[task][variant] = {"status": "NOT_AVAILABLE"}
                continue

            alg = metrics.get("algorithm")
            registry[task][variant] = {
                "status": "AVAILABLE",
                "task": task,
                "variant": variant,
                "algorithm": alg,
                "target": metrics.get("target"),
                "features": metrics.get("features", []),
                "categorical_features": metrics.get("categorical_features", []),
                "numeric_features": metrics.get("numeric_features", []),
                "preprocessing": metrics.get("preprocessing"),
                "feature_engineering": metrics.get("feature_engineering", []),
                "training_period": metrics.get("train_years", []),
                "test_period": metrics.get("test_years", []),
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k in (
                        "accuracy", "precision", "recall", "f1",
                        "r2", "rmse", "mae",
                    )
                },
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
                "artifact_filename": f"{variant}.joblib",
                "preprocessor_filename": f"{variant}_preprocessor.joblib",
                "predict_proba": TASK_PROPERTIES[task]["predict_proba"],
                "feature_importances_supported": TASK_PROPERTIES[task]["feature_importances"],
            }
            if "leakage_note" in metrics:
                registry[task][variant]["leakage_note"] = metrics["leakage_note"]

        # Ensure crop_type advanced is always marked NOT_AVAILABLE
        if "advanced" not in registry[task]:
            registry[task]["advanced"] = {"status": "NOT_AVAILABLE"}

    return registry


if __name__ == "__main__":
    registry = build_registry()
    out_path = MODELS_ROOT / "model_registry.json"
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps({t: {v: r.get("status") for v, r in registry[t].items()} for t in registry}, indent=2))
