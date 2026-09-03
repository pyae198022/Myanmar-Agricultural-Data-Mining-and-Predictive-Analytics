"""
Chapter 4 ROC Curve / AUC evaluation for the Crop Type task.

Sources of truth (no hardcoding, no fabrication):

* Per-class ROC-AUC and per-class test support are read from the REAL
  Chapter 4 result files that were produced by the project's Crop Type
  evaluation notebooks (``chapter4_class_auc_results_*.csv``).
* Overall accuracy, Macro ROC-AUC and Weighted ROC-AUC are read from the
  REAL Chapter 4 final-evaluation summary files
  (``chapter4_final_evaluation_summary_*.csv``).
* The raw ROC curve POINTS (per-class FPR/TPR/thresholds) are NOT stored in
  those CSVs, so they are recomputed at request time from the actual deployed
  Crop Type model artifacts against the same real 2022-2023 (990-row) held-out
  test set used everywhere else in the system.
* Because the deployed crop-type artifact is a separate RandomForest run from
  the Project Book model that produced the CSVs, the recomputed curve AUC does
  not exactly equal the stored Chapter 4 AUC. That deviation is measured and
  reported in ``report["verification"]`` rather than hidden or rounded away
  (the supplied CSV AUC/summary is displayed as the reference, and the curve
  points come solely from the real deployed artifacts).

Methodology (matches the project notebooks): multiclass One-vs-Rest using
``label_binarize`` + ``roc_curve`` per class, macro = mean of class AUCs,
weighted = mean weighted by per-class test support.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc as _trapz_auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from . import reference
from app.services.model_loader import model_loader
from app.services.preprocessing import PreprocessingService

# Reusable ROC/AUC dictionaries for both Crop Type model variants.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "data"

# filename -> public model label (order matters for the UI model selector).
VARIANT_FILES = [
    ("baseline", "chapter4_class_auc_results_baseline.csv", "chapter4_final_evaluation_summary_baseline.csv"),
    ("feature_engineering", "chapter4_class_auc_results_fe.csv", "chapter4_final_evaluation_summary_fe.csv"),
]

MODEL_LABELS = {
    "baseline": "Crop Type Baseline",
    "feature_engineering": "Crop Type Feature Engineering (FE)",
}

# ─── Server-side cache ───────────────────────────────────────────────────────
# The per-variant ROC report is expensive to compute (it runs predict_proba on
# the full 990-row test set + one-vs-rest ROC for every class). We cache the
# finished report per variant so subsequent requests return immediately with
# identical results. Entries are only inserted AFTER a successful computation;
# a failed calculation never populates the cache.
_roc_report_cache: dict[str, dict] = {}


def _cached_roc_report(variant: str) -> dict:
    """Return the cached report for ``variant``, computing it only on a miss."""
    if variant in _roc_report_cache:
        return deepcopy(_roc_report_cache[variant])
    fresh = _compute_roc_report(variant)
    # Populate only after a fully successful computation.
    _roc_report_cache[variant] = fresh
    return deepcopy(fresh)


def clear_roc_cache() -> None:
    """Drop cached reports (mainly for tests)."""
    _roc_report_cache.clear()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_class_auc(variant: str) -> pd.DataFrame:
    _, auc_file, _ = next(f for f in VARIANT_FILES if f[0] == variant)
    return _read_csv(DATA_DIR / auc_file)


def _load_summary_row(variant: str) -> pd.Series:
    _, _, summary_file = next(f for f in VARIANT_FILES if f[0] == variant)
    df = _read_csv(DATA_DIR / summary_file)
    return df.iloc[0]


def _per_class_curve_points(class_labels, y_test, proba_matrix):
    """
    Recompute the real one-vs-rest FPR/TPR/thresholds for every class.

    Returns a list of row dicts keyed by the true sorted class list. A class
    with only one distinct label in the test set cannot yield an ROC curve and
    is reported as empty (its AUC still comes from the supplied CSV).

    The authoritative per-class AUC comes from the supplied CSV; this routine
    additionally computes the AUC from the recomputed full curve so we can
    honestly disclose how closely the deployed-artifact curves reproduce the
    stored Chapter 4 values (sklearn ``auc`` trapezoid, same as the notebooks).
    """
    n = len(y_test)
    y_bin = label_binarize(y_test, classes=class_labels)
    points = OrderedDict()
    for i, cls in enumerate(class_labels):
        raw = {"class": str(cls), "recomputed_auc": None, "curve": None}
        col = y_bin[:, i]
        if len(np.unique(col)) < 2:
            points[str(cls)] = raw
            points[str(cls)]["curve"] = {"fpr": [], "tpr": [], "thresholds": []}
            continue
        fpr, tpr, thresholds = roc_curve(col, proba_matrix[:, i])
        raw["recomputed_auc"] = float(_trapz_auc(fpr, tpr))
        # Down-sample long curves for a light chart payload while keeping the
        # full span (keeps 33 classes readable without a broken page).
        pts = _sparse_curve(fpr, tpr, thresholds)
        points[str(cls)] = raw
        points[str(cls)]["curve"] = pts
    return points


def _finite(v):
    """JSON-safe float: sklearn emits inf for the first ROC threshold."""
    f = float(v)
    if not np.isfinite(f):
        return None
    return round(f, 6)


def _sparse_curve(fpr, tpr, thresholds, target: int = 25):
    """Return up to ``target`` evenly sampled (fpr, tpr, threshold) points."""
    n = len(fpr)
    if n <= target:
        return {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [_finite(v) for v in thresholds],
        }
    idx = np.unique(np.linspace(0, n - 1, target).astype(int))
    return {
        "fpr": [float(fpr[i]) for i in idx],
        "tpr": [float(tpr[i]) for i in idx],
        "thresholds": [_finite(thresholds[i]) for i in idx],
    }


def _real_test_data():
    """The exact held-out Crop Type test set (2022-2023), sorted chronologically.

    990 rows, matching the existing evaluation system.
    """
    df = reference.load_clean_data()
    df = df.sort_values(["Region", "Crop_Type", "Year"]).reset_index(drop=True)
    return df[df["Year"] > 2021]


def _recompute_curve_data(variant: str) -> dict:
    """Recompute the real one-vs-rest ROC curve points from the deployed artifact."""
    model = model_loader.get_model("crop_type", variant)
    preprocessor = model_loader.get_preprocessor("crop_type", variant)
    preproc = PreprocessingService()
    test = _real_test_data()

    class_labels = np.array(sorted(test["Crop_Type"].astype(str).unique()))

    proba_rows = []
    for _, row in test.iterrows():
        inp = {
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
        X = preproc.transform(inp, "crop_type", variant, model, preprocessor)
        proba_rows.append(model.predict_proba(X)[0].tolist())

    # Align the artifact's probability columns to the full sorted class list.
    classes = list(getattr(model, "classes_", []))
    proba_matrix = np.zeros((len(test), len(class_labels)))
    raw_rows = np.asarray(proba_rows)
    for i_cls, cls in enumerate(classes):
        if cls in class_labels:
            idx = int(np.where(class_labels == cls)[0][0])
            proba_matrix[:, idx] = raw_rows[:, i_cls]

    y_test = test["Crop_Type"].astype(str).values
    per_class = _per_class_curve_points(class_labels, y_test, proba_matrix)

    # Keep the frontend curve contract flat (fpr/tpr/thresholds) and expose the
    # recomputed per-class AUC separately for honest verification.
    curves = {
        str(cls): per_class[str(cls)]["curve"]
        for cls in class_labels
    }
    recomputed_auc = {
        str(cls): per_class[str(cls)]["recomputed_auc"]
        for cls in class_labels
        if per_class[str(cls)].get("recomputed_auc") is not None
    }
    return {
        "class_names": [str(c) for c in class_labels],
        "n_test": int(len(test)),
        "curves": curves,
        "recomputed_auc": recomputed_auc,
    }


def _model_result(variant: str) -> dict:
    auc_df = _load_class_auc(variant)
    summary = _load_summary_row(variant)

    class_auc = {}
    for _, r in auc_df.iterrows():
        class_auc[str(r["Crop_Type"])] = {
            "auc": float(r["AUC"]),
            "test_support": int(r["Test Support"]),
        }

    # Compute macro / weighted from the actual per-class AUC data so they are
    # always derived rather than entered by hand.
    aucs = np.array([v["auc"] for v in class_auc.values()])
    supports = np.array([v["test_support"] for v in class_auc.values()])
    macro_auc = float(np.mean(aucs))
    weighted_auc = float(np.average(aucs, weights=supports))

    return {
        "model": MODEL_LABELS[variant],
        "variant": variant,
        "task": "Crop Type",
        "classification_type": "multiclass",
        "roc_method": "One-vs-Rest (OvR)",
        "test_sample_count": int(summary.get("n_test", int(supports.sum()))),
        "accuracy": float(summary["Accuracy"]),
        "weighted_precision": float(summary["Weighted Precision"]),
        "weighted_recall": float(summary["Weighted Recall"]),
        "weighted_f1": float(summary["Weighted F1-score"]),
        "macro_roc_auc": round(macro_auc, 6),
        "weighted_roc_auc": round(weighted_auc, 6),
        "reported_macro_roc_auc": float(summary["Macro ROC-AUC"]),
        "reported_weighted_roc_auc": float(summary["Weighted ROC-AUC"]),
        "class_auc": {k: class_auc[k] for k in sorted(class_auc)},
        "class_names": sorted(class_auc.keys()),
    }


def available_variants() -> list[str]:
    """Return the Crop Type ROC variants that have real result files."""
    return [v for v, _, _ in VARIANT_FILES if (DATA_DIR / _auc_name(v)).exists()]


def _auc_name(variant: str) -> str:
    _, auc_file, _ = next(f for f in VARIANT_FILES if f[0] == variant)
    return auc_file


def roc_evaluation_report(variant: str) -> dict:
    """Full ROC/AUC report for one Crop Type model variant (server-side cached)."""
    if variant not in available_variants():
        raise KeyError(f"No Chapter 4 ROC/AUC result files for variant '{variant}'")
    return _cached_roc_report(variant)


def _compute_roc_report(variant: str) -> dict:
    """Compute (un-cached) the full ROC/AUC report for one variant."""
    result = _model_result(variant)
    curve = _recompute_curve_data(variant)
    result["n_test"] = curve["n_test"]
    result["curves"] = curve["curves"]
    result["recomputed_auc"] = curve["recomputed_auc"]
    result["curve_class_names"] = curve["class_names"]

    # Honest verification: how closely do curves recomputed from the DEPLOYED
    # artifact reproduce the supplied Chapter 4 AUC values? Because the deployed
    # crop-type artifacts are a separate RandomForest run from the one that
    # produced the Project Book CSVs, a deviation is expected and REPORTED here
    # rather than hidden.
    support = {c: v["test_support"] for c, v in result["class_auc"].items()}
    csv_auc = {c: v["auc"] for c, v in result["class_auc"].items()}
    rec_auc = curve["recomputed_auc"]
    common = [c for c in rec_auc if c in csv_auc and support.get(c, 0) > 0]
    abs_diffs = []
    for c in common:
        abs_diffs.append(abs(rec_auc[c] - csv_auc[c]))
    recomputed_macro = float(np.mean([rec_auc[c] for c in common])) if common else None
    recomputed_weighted = (
        float(np.average([rec_auc[c] for c in common], weights=[support[c] for c in common]))
        if common else None
    )

    result["verification"] = {
        "max_abs_per_class_diff": round(float(max(abs_diffs)), 6) if abs_diffs else None,
        "mean_abs_per_class_diff": round(float(np.mean(abs_diffs)), 6) if abs_diffs else None,
        "csv_macro_roc_auc": round(float(np.mean([csv_auc[c] for c in common])), 6) if common else None,
        "recomputed_macro_roc_auc": round(recomputed_macro, 6) if recomputed_macro is not None else None,
        "csv_weighted_roc_auc": round(_csv_weighted_auc(variant), 6),
        "recomputed_weighted_roc_auc": round(recomputed_weighted, 6) if recomputed_weighted is not None else None,
        "disclosure": (
            "The ROC curve points are recomputed from the deployed Crop Type model "
            "artifacts on the real 2022-2023 test set (same features/split/hyper-"
            "parameters as the Project Book). The supplied Chapter 4 AUC/summary "
            "values come from the Project Book notebook evaluation. Because the "
            "deployed artifact is a separate RandomForest run, the recomputed "
            "per-class AUC differs from the stored value by up to "
            f"{round(float(max(abs_diffs)), 4) if abs_diffs else 0:.4f}; this deviation "
            "is reported, not suppressed."
        ),
    }
    return result


def _csv_weighted_auc(variant: str) -> float:
    auc_df = _load_class_auc(variant)
    aucs = np.array([float(r["AUC"]) for _, r in auc_df.iterrows()])
    sups = np.array([int(r["Test Support"]) for _, r in auc_df.iterrows()])
    return float(np.average(aucs, weights=sups))


def roc_evaluation_summary() -> dict:
    """Summarize both Crop Type ROC variants for the comparison table."""
    models = []
    for variant, _, _ in VARIANT_FILES:
        if variant not in available_variants():
            continue
        r = roc_evaluation_report(variant)
        v = r.get("verification", {})
        models.append(
            {
                "model": r["model"],
                "variant": r["variant"],
                "accuracy": r["accuracy"],
                "macro_roc_auc": r["macro_roc_auc"],
                "weighted_roc_auc": r["weighted_roc_auc"],
                "recomputed_macro_roc_auc": v.get("recomputed_macro_roc_auc"),
                "recomputed_weighted_roc_auc": v.get("recomputed_weighted_roc_auc"),
                "max_abs_per_class_diff": v.get("max_abs_per_class_diff"),
            }
        )
    return {"task": "Crop Type", "models": models}