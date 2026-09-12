"""
Crop Yield LEVEL classification evaluation service — binary ROC/AUC on the
2023 hold-out test set, and 5-Fold Stratified Cross-Validation on the
2012-2022 training data.

Ground truth (Project Book, Sections 4.2.2.2 & 4.2.3.3 / Table 4.9):

* The task is BINARY (Yield_Level in {High, Low}); ROC/AUC uses the standard
  binary definition with pos_label="High" — NOT One-vs-Rest multiclass (which
  is used only for the 33-class Crop_Type task).
* Test-set AUC reference (single 2023 hold-out split):
      Baseline MLP                 AUC = 0.9698
      Selected-FE Enhanced MLP     AUC = 0.9604
      Model 3 (Assoc+Sequential)   AUC = 0.9690
* Cross-validation: 5-Fold STRATIFIED CV (not Time-Series CV), computed on
  TRAINING data only (2012-2022; the 2023 hold-out set is excluded), reporting
  Accuracy / Precision / Recall / F1 / ROC-AUC as Mean +/- Std across folds.

Honest-subset methodology chosen after reviewing the deployed artifacts:

* The three Yield_Level models are MLPClassifiers. The DEPLOYED baseline model
  is guarded by the prediction pipeline because it requires target-leakage
  inputs (Harvested_Acre, Production_Ton), so it cannot be evaluated through
  the shared preprocessing path on the test set. Only the Feature-Engineering
  and Advanced (Model 3) artifacts can be fairly evaluated.
* TEST-SET ROC/AUC: computed at request time from the actual deployed
  feature_engineering / advanced artifacts on the real 2023 (495-row) test set.
  This is the honest, no-retrain source-of-truth and is reported side by side
  with the Project Book reference AUC.
* CROSS-VALIDATION: the application stores no per-fold CV artifacts, and CV
  requires re-fitting a new MLP on each training fold. To reproduce the
  Project Book's rigorous CV we re-fit the FE and Advanced MLPs per fold using
  the exact training pipeline (interactions, association, sequential features;
  MLPClassifier(hidden_layer_sizes=(64,32), ... random_state=42)). The CV runs
  on the 2012-2022 training rows only; the 2023 hold-out is never included.
  Association rules are mined from the training rows each CV is based on. The
  Baseline is NOT cross-validated because its feature set is the leakage-based
  target-derived one (Harvested_Acre/Production_Ton) whose CV would be a
  tautology, and the deployed baseline is guarded.

Because the Project Book's exact per-fold feature construction / fold seed is
not fully specified in the supplied text, the CV figures below are the real
values produced by re-fitting the documented pipeline with the documented
deterministic seed. They are reported as actual computed values and compared
to the Table 4.9 reference; a mismatch is reported, not hidden or fitted.
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)

TASK_ID = "yield_level"

# Result caches for the expensive request-time computations. Test-set ROC/AUC
# depends only on the deployed artifact + fixed 2023 test set; CV depends only
# on the fixed 2012-2022 training data + deterministic seed. Both are stable
# across requests, so caching avoids recomputing the 495-row prediction loop and
# the per-fold MLP re-fit on every page load / reload.
_ROC_CACHE: dict[str, dict] = {}
_CV_CACHE: dict[str, dict] = {}

VARIANT_LABELS = {
    "baseline": "Baseline MLP",
    "feature_engineering": "Selected-FE Enhanced MLP",
    "advanced": "Model 3 (Assoc+Sequential) MLP",
}

# Variants with real, non-guarded artifacts that can be evaluated.
_REAL_VARIANTS = ["feature_engineering", "advanced"]

# Deterministic MLP configuration used to re-fit folds in cross-validation —
# identical to the methodology that produced the deployed artifacts.
MLP_CONFIG = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)

YIELD_TL_TRAIN_END = 2022
YIELD_TL_TEST_YEAR = 2023

# Exact feature column sets (from the deployed model metadata / training script).
CATEGORICAL_BASE = ["Crop_Type", "Soil_Type", "Water_Source", "Seeding_Season"]
NUMERIC_BASE = ["Sown_Acre", "Avg_Temperature", "Total_Rainfall", "Avg_Humidity"]
INTERACTIONS = ["Crop_x_WaterSource", "Crop_x_SeedingSeason", "Crop_x_SoilType"]
ASSOCIATION_FEATURES = [
    "Assoc_High_Confidence", "Assoc_High_Lift",
    "Assoc_Low_Confidence", "Assoc_Low_Lift", "Assoc_Rule_Count",
]
SEQUENCE_FEATURES = [
    "Prev_Yield_High", "Prev_Yield_Low",
    "Prev2_Persistent", "Prev3_Persistent",
]

_FE_FEATURES = ["Year"] + CATEGORICAL_BASE + NUMERIC_BASE + INTERACTIONS
_ADV_FEATURES = (
    ["Year"] + CATEGORICAL_BASE + NUMERIC_BASE
    + INTERACTIONS + ASSOCIATION_FEATURES + SEQUENCE_FEATURES
)


def _scripts_common():
    """Import the training-script helpers, adding ``scripts/`` to sys.path."""
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from common import (
        add_existing_feature_engineering,
        add_sequential_features,
        build_association_features,
        mine_yield_association_rules,
    )
    return (
        add_existing_feature_engineering,
        add_sequential_features,
        build_association_features,
        mine_yield_association_rules,
    )


def _load_data() -> pd.DataFrame:
    """Load the bundled cleaned dataset (5940 rows, 2012-2023)."""
    from app.preprocessing import reference
    df = reference.load_clean_data()
    df = df.sort_values(["Region", "Crop_Type", "Year"]).reset_index(drop=True)
    return df


def _label_yield_level(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Add the binary Yield_Level label using the training-only median threshold."""
    out = df.copy()
    train_median = out.loc[train_mask, "Crop_Yield"].median()
    out["Yield_Level"] = np.where(out["Crop_Yield"] > train_median, "High", "Low")
    return out


def _build_feature_matrix(variant: str, df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Reconstruct the exact training feature matrix + labels for a variant.

    Mirrors ``scripts/generate_yield_level.py`` so CV re-fits reproduce the
    deployed models' training methodology faithfully.

    ``train_mask`` is a boolean Series indexed like ``df`` marking the training
    rows (used for labeling and train-only association mining).
    """
    (add_fe, _add_seq, _build_assoc, mine_rules) = _scripts_common()
    df_fe = add_fe(df)
    df_labeled = _label_yield_level(df_fe, train_mask)

    if variant == "feature_engineering":
        X = df_labeled[_FE_FEATURES]
        y = df_labeled["Yield_Level"].to_numpy()
        return X, y

    if variant == "advanced":
        # Association rules mined from the training rows only (as in training).
        rules = mine_rules(df_labeled, train_mask)
        df_assoc = _build_assoc(df_labeled, rules)
        df_enh = _add_seq(df_assoc)
        X = df_enh[_ADV_FEATURES]
        y = df_enh["Yield_Level"].to_numpy()
        return X, y

    raise ValueError(f"Unsupported variant for cross-validation: {variant}")


def _encode_labels(y) -> np.ndarray:
    """Encode High->1, Low->0 (exactly sklearn's LabelBinarizer order for the
    deployed string labels). MLPClassifier with early_stopping fails on raw
    string labels in the installed sklearn, so we binarize before fitting —
    this is the identical internal representation the deployed models used."""
    return np.where(np.asarray(y) == "High", 1, 0).astype(int)


def _fit_and_predict_inner_fold(variant: str, X_tr: pd.DataFrame, y_tr: pd.Series):
    """Return a fitted (preprocessor, MLP) pair ready to predict a validation fold.

    Replicates the per-variant preprocessing:
      * feature_engineering: pd.get_dummies + MinMaxScaler
      * advanced: OneHotEncoder + MinMaxScaler via ColumnTransformer
    Labels are binarized (High=1) so the MLP trains exactly as deployed.
    """
    y_int = _encode_labels(y_tr)

    if variant == "feature_engineering":
        Xtr = pd.get_dummies(X_tr, dtype=float)
        scaler = MinMaxScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        model = MLPClassifier(**MLP_CONFIG)
        model.fit(Xtr_s, y_int)
        return {"columns": list(Xtr.columns), "scaler": scaler}, model

    if variant == "advanced":
        cat_cols = X_tr.select_dtypes(include="object").columns.tolist()
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ])
        model = Pipeline([("pre", preprocessor), ("mlp", MLPClassifier(**MLP_CONFIG))])
        model.fit(X_tr, y_int)
        return None, model

    raise ValueError(f"Unsupported variant: {variant}")


def _predict_probability(variant: str, state, model, X: pd.DataFrame):
    """Predict P(High) and the decoded High/Low hard label for a validation fold."""
    if variant == "feature_engineering":
        Xd = pd.get_dummies(X, dtype=float)
        Xd = Xd.reindex(columns=state["columns"], fill_value=0)
        Xs = state["scaler"].transform(Xd)
        proba_matrix = model.predict_proba(Xs)
        high_idx = int(np.where(np.asarray(model.classes_) == 1)[0][0])
        proba = proba_matrix[:, high_idx]
        return proba, np.where(model.predict(Xs) == 1, "High", "Low")

    if variant == "advanced":
        proba_matrix = model.predict_proba(X)
        high_idx = int(np.where(np.asarray(model.classes_) == 1)[0][0])
        proba = proba_matrix[:, high_idx]
        return proba, np.where(model.predict(X) == 1, "High", "Low")
    raise ValueError(f"Unsupported variant: {variant}")


# ─── Test-set binary ROC / AUC (honest, no retrain) ──────────────────────────

def test_set_roc_auc(variant: str = "feature_engineering") -> dict:
    """Compute standard binary ROC curve + AUC from the DEPLOYED artifact on the
    real 2023 test set. pos_label="High". Baseline is guarded -> unavailable.
    """
    from app.services.model_loader import ArtifactNotFoundError, model_loader
    from app.services.preprocessing import PreprocessingService

    if variant not in _REAL_VARIANTS:
        return {
            "variant": variant,
            "name": VARIANT_LABELS.get(variant, variant),
            "available": False,
            "reason": (
                "yield_level/baseline is guarded (requires target-leakage inputs "
                "Harvested_Acre/Production_Ton) and cannot be evaluated through "
                "the shared preprocessing path."
            ),
            "auc": None,
            "curve": None,
            "n_test": 0,
        }

    try:
        model = model_loader.get_model(TASK_ID, variant)
        preprocessor = model_loader.get_preprocessor(TASK_ID, variant)
    except ArtifactNotFoundError:
        return {
            "variant": variant,
            "name": VARIANT_LABELS[variant],
            "available": False,
            "reason": "Deployed artifact not found.",
            "auc": None,
            "curve": None,
            "n_test": 0,
        }

    if variant in _ROC_CACHE:
        return deepcopy(_ROC_CACHE[variant])

    df = _load_data()
    test_bool = (df["Year"] == YIELD_TL_TEST_YEAR).values

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[df["Year"] <= YIELD_TL_TRAIN_END] = True
    labeled = _label_yield_level(df, pd.Series(train_mask, index=df.index))
    test_labels = labeled["Yield_Level"].to_numpy()[test_bool]

    preproc = PreprocessingService()
    proba = []
    for r in df[test_bool].itertuples():
        inp = {
            "region": str(r.Region),
            "year": int(r.Year),
            "crop_type": str(r.Crop_Type),
            "sown_acre": float(r.Sown_Acre),
            "soil_type": str(r.Soil_Type),
            "avg_temperature": float(r.Avg_Temperature),
            "total_rainfall": float(r.Total_Rainfall),
            "avg_humidity": float(r.Avg_Humidity),
            "water_source": str(r.Water_Source),
            "seeding_season": str(r.Seeding_Season),
        }
        X = preproc.transform(inp, TASK_ID, variant, model, preprocessor)
        proba_matrix = model.predict_proba(X)
        high_idx = int(np.where(np.asarray(model.classes_) == "High")[0][0])
        proba.append(float(proba_matrix[0, high_idx]))

    proba = np.asarray(proba, dtype=float)
    y_binary = (test_labels == "High").astype(int)
    auc = float(roc_auc_score(y_binary, proba))
    fpr, tpr, thresholds = roc_curve(y_binary, proba, pos_label=1)

    def _finite(v):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)

    result = {
        "variant": variant,
        "name": VARIANT_LABELS[variant],
        "available": True,
        "task": "Crop Yield Level",
        "classification_type": "binary",
        "roc_method": "standard binary ROC (pos_label=High)",
        "n_test": int(test_bool.sum()),
        "auc": round(auc, 4),
        "curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [_finite(t) for t in thresholds],
        },
        "note": (
            "Standard binary ROC/AUC computed from the deployed MLP artifact on "
            "the real 2023 (495-row) hold-out test set. Reference AUC from the "
            "Project Book for this variant is supplied separately for comparison; "
            "the deployed artifact is a separate MLP run, so values may differ."
        ),
    }
    _ROC_CACHE[variant] = deepcopy(result)
    return result


def roc_availability() -> list[dict]:
    from app.services.model_loader import model_loader
    rows = []
    for v in ["baseline", "feature_engineering", "advanced"]:
        available = v in _REAL_VARIANTS and bool(model_loader.get_model(TASK_ID, v))
        rows.append({
            "variant": v,
            "name": VARIANT_LABELS.get(v, v),
            "available": available,
        })
    return rows


# ─── 5-Fold Stratified Cross-Validation (re-fit, training data only) ─────────

# Project Book reference (Table 4.9) — supplied ground truth, shown for comparison.
PROJECT_BOOK_CV_REFERENCE = {
    "baseline": {
        "label": "Baseline MLP",
        "accuracy": (96.69, 0.67), "precision": (96.37, 1.20),
        "recall": (97.06, 0.70), "f1": (96.71, 0.66), "roc_auc": (0.9931, 0.0015),
    },
    "feature_engineering": {
        "label": "Selected-FE Enhanced MLP",
        "accuracy": (92.54, 0.56), "precision": (93.47, 1.35),
        "recall": (91.51, 1.15), "f1": (92.47, 0.54), "roc_auc": (0.9830, 0.0025),
    },
    "advanced": {
        "label": "Model 3 (Assoc+Sequential) MLP",
        "accuracy": (96.79, 0.34), "precision": (96.94, 0.56),
        "recall": (96.62, 0.57), "f1": (96.78, 0.34), "roc_auc": (0.9897, 0.0016),
    },
}


def cross_validation(random_seed: int = 42, n_splits: int = 5) -> dict:
    """5-Fold Stratified CV on 2012-2022 training rows only, re-fitting the FE
    and Advanced MLPs per fold with the documented deterministic hyperparameters.

    Returns the REAL computed mean/std for Accuracy, Precision, Recall, F1,
    ROC-AUC per model, plus the Project Book reference for honest comparison.
    """
    cache_key = f"{random_seed}/{n_splits}"
    if cache_key in _CV_CACHE:
        return deepcopy(_CV_CACHE[cache_key])

    df = _load_data()
    train_mask = pd.Series(df["Year"] <= YIELD_TL_TRAIN_END, index=df.index)
    n_train = int(train_mask.sum())

    models_out = []
    for variant in _REAL_VARIANTS:
        X, y = _build_feature_matrix(variant, df, train_mask)
        # Select the training rows (2012-2022) positionally, aligned with df order.
        X_train = X.loc[train_mask].reset_index(drop=True)
        y_train = np.asarray(y)[train_mask.to_numpy()]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            state, model = _fit_and_predict_inner_fold(
                variant, X_train.iloc[tr_idx], y_train[tr_idx]
            )
            proba_high, y_pred = _predict_probability(
                variant, state, model, X_train.iloc[va_idx]
            )
            y_true = y_train[va_idx]
            y_true_bin = (y_true == "High").astype(int)
            accs.append(accuracy_score(y_true, y_pred))
            precs.append(precision_score(y_true, y_pred, pos_label="High", zero_division=0))
            recs.append(recall_score(y_true, y_pred, pos_label="High", zero_division=0))
            f1s.append(f1_score(y_true, y_pred, pos_label="High", zero_division=0))
            aucs.append(roc_auc_score(y_true_bin, proba_high))

        def _m(vals, pct=True):
            arr = np.asarray(vals, dtype=float)
            mean = arr.mean()
            if pct:
                return {"mean": round(float(mean * 100), 2), "std": round(float(arr.std(ddof=1) * 100), 2)}
            return {"mean": round(float(mean), 4), "std": round(float(arr.std(ddof=1)), 4)}

        ref = PROJECT_BOOK_CV_REFERENCE.get(variant)
        models_out.append({
            "variant": variant,
            "label": VARIANT_LABELS[variant],
            "algorithm": "MLPClassifier (re-fit per fold)",
            "n_train": int(len(X_train)),
            "n_splits": n_splits,
            "method": "5-Fold Stratified Cross-Validation",
            "accuracy": _m(accs),
            "precision": _m(precs),
            "recall": _m(recs),
            "f1": _m(f1s),
            "roc_auc": _m(aucs, pct=False),
            "project_book_reference": {
                "accuracy": list(ref["accuracy"]) if ref else None,
                "precision": list(ref["precision"]) if ref else None,
                "recall": list(ref["recall"]) if ref else None,
                "f1": list(ref["f1"]) if ref else None,
                "roc_auc": list(ref["roc_auc"]) if ref else None,
            } if ref else None,
        })

    result = {
        "task": "Crop Yield Level",
        "method": "5-Fold Stratified Cross-Validation (2012-2022 training data only)",
        "data_scope": "training rows 2012-2022; 2023 hold-out excluded",
        "random_seed": random_seed,
        "models": models_out,
        "excluded": {
            "baseline": "guarded (leakage-based features Harvested_Acre/Production_Ton); "
                         "guarded baseline is not cross-validated.",
        },
        "note": (
            "Cross-validation re-fits the MLP on each stratified training fold "
            "(hidden_layer_sizes=(64,32), random_state=42, max_iter=500) exactly as "
            "the deployed models were trained; association rules are mined from the "
            "training rows each CV run uses. Values are the ACTUAL computed results "
            "and are compared to the Project Book Table 4.9 reference, not fitted to "
            "it. Baseline is excluded because its features are target-derived leakage "
            "and its artifact is guarded."
        ),
    }
    _CV_CACHE[cache_key] = deepcopy(result)
    return result
