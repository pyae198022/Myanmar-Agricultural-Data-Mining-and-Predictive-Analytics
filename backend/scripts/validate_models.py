"""
Validation harness for generated model artifacts.

Loads every .joblib, verifies integrity, predicts on representative REAL rows
from the project dataset, and checks:
  1. artifact loads
  2. expected input columns
  3. prediction works (classification / regression)
  4. predict_proba works for classifiers
  5. feature_importance works where supported
  6. no NaN predictions
  7. crop_type advanced is NOT loaded
  8. association rules exist for advanced yield models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "Downloads" / "Data Mining"
if not DATA_DIR.exists():
    DATA_DIR = Path("/Users/pyaesone/Downloads/Data Mining")

CLEANED_CSV = DATA_DIR / "cleaned_data.csv"

PASS = 0
FAIL = 0


def check(ok: bool, msg: str):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def load_real_rows(n=3):
    """Return a few representative real rows from the cleaned dataset (2023)."""
    df = pd.read_csv(CLEANED_CSV)
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    df = df[df["Year"] == 2023]
    return df.head(n).to_dict("records")


def transform_yield_level_generic(row_df, bundle):
    """Apply pd.get_dummies + MinMaxScaler (baseline/FE yield level)."""
    dummies = pd.get_dummies(row_df, dtype=float)
    dummies = dummies.reindex(columns=bundle["dummy_columns"], fill_value=0)
    return bundle["minmax"].transform(dummies)


def main():
    real_rows = load_real_rows(3)
    if not real_rows:
        print("No real rows loaded — validation cannot proceed.")
        sys.exit(1)
    print(f"Loaded {len(real_rows)} representative real rows from 2023.\n")

    rows_df = pd.DataFrame(real_rows)

    # Build a row with all needed YIELD features for raw-input models
    yield_rows = rows_df.copy()

    # ---- Crop Type ----
    print("=== CROP TYPE ===")
    ct_rows = rows_df[["Region", "Soil_Type", "Water_Source", "Avg_Temperature",
                       "Total_Rainfall", "Avg_Humidity", "Year", "Sown_Acre"]]
    check_model_classifier(
        joblib.load(MODELS_ROOT / "crop_type" / "baseline.joblib"),
        ct_rows, "crop_type/baseline",
    )

    # feature_engineering needs log-engineered features
    ct_fe = ct_rows.copy()
    ct_fe["Log_Sown_Acre"] = np.log1p(ct_fe["Sown_Acre"].clip(lower=0))
    ct_fe["Log_Total_Rainfall"] = np.log1p(ct_fe["Total_Rainfall"].clip(lower=0))
    ct_fe = ct_fe[["Log_Sown_Acre", "Log_Total_Rainfall", "Avg_Temperature",
                   "Avg_Humidity", "Year", "Region", "Soil_Type", "Water_Source"]]
    check_model_classifier(
        joblib.load(MODELS_ROOT / "crop_type" / "feature_engineering.joblib"),
        ct_fe, "crop_type/feature_engineering",
    )

    # crop_type advanced must NOT exist
    check(
        not (MODELS_ROOT / "crop_type" / "advanced.joblib").exists(),
        "crop_type/advanced artifact does NOT exist (as expected)",
    )
    registry = json.loads((MODELS_ROOT / "model_registry.json").read_text())
    check(
        registry["crop_type"]["advanced"].get("status") == "NOT_AVAILABLE",
        "registry marks crop_type/advanced as NOT_AVAILABLE",
    )
    print()

    # ---- Crop Yield regression ----
    print("=== CROP YIELD (regression) ===")
    cy_fe = yield_rows.copy()
    cy_fe["Crop_x_WaterSource"] = cy_fe["Crop_Type"] + " | " + cy_fe["Water_Source"]
    cy_fe["Crop_x_SeedingSeason"] = cy_fe["Crop_Type"] + " | " + cy_fe["Seeding_Season"]
    cy_fe["Crop_x_SoilType"] = cy_fe["Crop_Type"] + " | " + cy_fe["Soil_Type"]

    # baseline regression inputs: everything except Crop_Yield, Year, Region
    cy_base = yield_rows.drop(columns=["Crop_Yield", "Year", "Region"], errors="ignore")
    if "Yield_Level" in cy_base.columns:
        cy_base = cy_base.drop(columns=["Yield_Level"])
    check_model_regressor(
        joblib.load(MODELS_ROOT / "crop_yield" / "baseline.joblib"),
        cy_base, "crop_yield/baseline",
    )

    # feature engineering regression: 12 selected + interactions
    fe_feats = ["Year", "Crop_Type", "Sown_Acre", "Soil_Type", "Avg_Temperature",
                "Total_Rainfall", "Avg_Humidity", "Water_Source", "Seeding_Season",
                "Crop_x_WaterSource", "Crop_x_SeedingSeason", "Crop_x_SoilType"]
    check_model_regressor(
        joblib.load(MODELS_ROOT / "crop_yield" / "feature_engineering.joblib"),
        cy_fe[fe_feats], "crop_yield/feature_engineering",
    )

    # advanced regression: needs association + sequential features.
    # To validate advanced we load the association/sequential builder from common
    # and reconstruct the 21 features for the real rows.
    import sys as _sys
    here = Path(__file__).resolve().parent
    if str(here) not in _sys.path:
        _sys.path.insert(0, str(here))
    from common import (
        add_existing_feature_engineering,
        add_sequential_features,
        build_association_features,
        load_cleaned_data,
        mine_yield_association_rules,
    )
    df = load_cleaned_data()
    df_fe = add_existing_feature_engineering(df)
    train_mask = df_fe["Year"] <= 2022
    train_median = df_fe.loc[train_mask, "Crop_Yield"].median()
    df_fe["Yield_Level"] = np.where(df_fe["Crop_Yield"] > train_median, "High", "Low")
    rules = mine_yield_association_rules(df_fe, train_mask)
    df_assoc = build_association_features(df_fe, rules)
    df_enh = add_sequential_features(df_assoc)

    # match real rows to enhanced rows by (Region, Crop_Type, Year)
    key_cols = ["Region", "Crop_Type", "Year"]
    adv_merge = df_enh.merge(
        yield_rows[key_cols], on=key_cols, how="inner"
    ).drop_duplicates(subset=key_cols).head(len(yield_rows))

    adv_feats = fe_feats + [
        "Assoc_High_Confidence", "Assoc_High_Lift", "Assoc_Low_Confidence",
        "Assoc_Low_Lift", "Assoc_Rule_Count", "Prev_Yield_High", "Prev_Yield_Low",
        "Prev2_Persistent", "Prev3_Persistent",
    ]
    check(
        len(adv_merge) > 0,
        f"advanced regression: matched {len(adv_merge)} real rows to enhanced data",
    )
    check_model_regressor(
        joblib.load(MODELS_ROOT / "crop_yield" / "advanced.joblib"),
        adv_merge[adv_feats], "crop_yield/advanced",
    )
    check(
        (MODELS_ROOT / "crop_yield" / "association_rules.csv").exists(),
        "crop_yield/association_rules.csv exists (advanced model)",
    )
    print()

    # ---- Yield Level classification ----
    print("=== YIELD LEVEL (classification) ===")
    # baseline: drop target + leakage from yield_rows
    yl_base = yield_rows.drop(
        columns=["Yield_Level", "Crop_Yield", "Predicted_Yield_Level", "Probability_High"],
        errors="ignore",
    )
    yl_bundle = joblib.load(MODELS_ROOT / "yield_level" / "baseline_preprocessor.joblib")
    yl_model = joblib.load(MODELS_ROOT / "yield_level" / "baseline.joblib")
    X_yl_base = transform_yield_level_generic(yl_base, yl_bundle)
    yl_pred = yl_model.predict(X_yl_base)
    check(
        not any(pd.isna(p) for p in yl_pred),
        "yield_level/baseline predicts (no NaN)",
    )
    check(
        set(np.unique(yl_pred)).issubset({"Low", "High"}),
        f"yield_level/baseline predictions in {{Low, High}}: {np.unique(yl_pred).tolist()}",
    )
    probs = yl_model.predict_proba(X_yl_base)
    check(
        probs.shape == (len(yield_rows), 2) and np.all(np.isfinite(probs)),
        f"yield_level/baseline predict_proba works (shape {probs.shape})",
    )

    # feature_engineering: 12 selected + interactions
    yl_fe = yield_rows[
        ["Year", "Crop_Type", "Sown_Acre", "Soil_Type", "Avg_Temperature",
         "Total_Rainfall", "Avg_Humidity", "Water_Source", "Seeding_Season"]
    ].copy()
    yl_fe["Crop_x_WaterSource"] = yl_fe["Crop_Type"] + " | " + yl_fe["Water_Source"]
    yl_fe["Crop_x_SeedingSeason"] = yl_fe["Crop_Type"] + " | " + yl_fe["Seeding_Season"]
    yl_fe["Crop_x_SoilType"] = yl_fe["Crop_Type"] + " | " + yl_fe["Soil_Type"]
    yl_fe_bundle = joblib.load(MODELS_ROOT / "yield_level" / "feature_engineering_preprocessor.joblib")
    yl_fe_model = joblib.load(MODELS_ROOT / "yield_level" / "feature_engineering.joblib")
    X_yl_fe = transform_yield_level_generic(yl_fe, yl_fe_bundle)
    yl_fe_pred = yl_fe_model.predict(X_yl_fe)
    check(
        set(np.unique(yl_fe_pred)).issubset({"Low", "High"}),
        f"yield_level/feature_engineering predicts in {{Low, High}}: {np.unique(yl_fe_pred).tolist()}",
    )

    # advanced: 21 features via pipeline
    adv_rows = adv_merge[adv_feats].copy()
    adv_yl_model = joblib.load(MODELS_ROOT / "yield_level" / "advanced.joblib")
    adv_yl_pred = adv_yl_model.predict(adv_rows)
    check(
        set(np.unique(adv_yl_pred)).issubset({"Low", "High"}),
        f"yield_level/advanced predicts in {{Low, High}}: {np.unique(adv_yl_pred).tolist()}",
    )
    adv_probs = adv_yl_model.predict_proba(adv_rows)
    check(
        adv_probs.shape == (len(adv_rows), 2) and np.all(np.isfinite(adv_probs)),
        f"yield_level/advanced predict_proba works (shape {adv_probs.shape})",
    )
    check(
        (MODELS_ROOT / "yield_level" / "association_rules.csv").exists(),
        "yield_level/association_rules.csv exists (advanced model)",
    )
    print()

    print(f"\n===== VALIDATION SUMMARY: {PASS} passed, {FAIL} failed =====")
    sys.exit(1 if FAIL else 0)


def check_model_classifier(model, X, name):
    """Check a classification pipeline that accepts raw input."""
    preds = model.predict(X)
    check(preds is not None and len(preds) == len(X), f"{name} predicts {len(preds)} rows")
    check(
        not any(pd.isna(p) for p in preds),
        f"{name} predictions contain no NaN",
    )
    try:
        probs = model.predict_proba(X)
        check(
            probs.shape[0] == len(X) and np.all(np.isfinite(probs)),
            f"{name} predict_proba works (shape {probs.shape})",
        )
    except Exception as e:
        check(False, f"{name} predict_proba raised: {e}")

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        check(np.all(np.isfinite(fi)) and len(fi) > 0, f"{name} feature_importances_ present")
    elif hasattr(model, "named_steps"):
        imp_found = any(
            hasattr(step, "feature_importances_") for step in model.named_steps.values()
        )
        check(imp_found, f"{name} (pipeline) exposes feature_importances_")


def check_model_regressor(model, X, name):
    preds = model.predict(X)
    check(preds is not None and len(preds) == len(X), f"{name} predicts {len(preds)} rows")
    check(
        not np.isnan(np.array(preds, dtype=float)).any(),
        f"{name} predictions contain no NaN",
    )
    if hasattr(model, "feature_importances_"):
        check(np.all(np.isfinite(model.feature_importances_)), f"{name} feature_importances_ finite")
    elif hasattr(model, "named_steps"):
        imp_found = any(
            hasattr(step, "feature_importances_") for step in model.named_steps.values()
        )
        check(imp_found, f"{name} (pipeline) exposes feature_importances_")


if __name__ == "__main__":
    main()
