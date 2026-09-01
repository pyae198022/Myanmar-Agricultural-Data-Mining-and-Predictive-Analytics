"""
Generate Crop Yield regression model artifacts (baseline, feature_engineering, advanced).

Reproduces the exact Data Mining project methodology using RandomForestRegressor as
the production model (the project reports RF results for all three variants).

Split: Train 2012-2022, Test 2023.

- Baseline: RandomForestRegressor on original features, dropping [Crop_Yield, Year, Region].
  ⚠ PRESERVES Harvested_Acre and Production_Ton per the original project code
    (target leakage is documented, not silently removed).
- Feature Engineering: RandomForestRegressor on the 12 selected features + 3 Crop_x_* interactions.
- Advanced: RandomForestRegressor on 12 selected + 3 interactions + 5 association + 4 sequential,
  using the Model3 association/sequential methodology (TRAIN ONLY).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (
    ASSOC_ENHANCED_FEATURES,
    EXISTING_FE_CATEGORICAL,
    EXISTING_FE_INTERACTIONS,
    EXISTING_FE_NUMERIC,
    SEQUENCE_FEATURES,
    YIELD_TL_TRAIN_END,
    YIELD_TL_TEST_YEAR,
    add_existing_feature_engineering,
    add_sequential_features,
    build_association_features,
    load_cleaned_data,
    load_selected_features,
    mine_yield_association_rules,
    serialize_rules,
)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
CROP_YIELD_DIR = MODELS_ROOT / "crop_yield"

TARGET = "Crop_Yield"

BASELINE_EXCLUDED = {TARGET, "Year", "Region"}


def build_baseline():
    df = load_cleaned_data()

    train_mask = df["Year"] <= YIELD_TL_TRAIN_END
    test_mask = df["Year"] == YIELD_TL_TEST_YEAR

    X = df.drop(columns=BASELINE_EXCLUDED)
    y = df[TARGET]

    X_train, X_test = X[train_mask.values], X[test_mask.values]
    y_train, y_test = y[train_mask.values], y[test_mask.values]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        [
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_years": [int(y) for y in sorted(df.loc[train_mask, "Year"].unique())],
        "test_years": [int(y) for y in sorted(df.loc[test_mask, "Year"].unique())],
    }
    print(f"[crop_yield/baseline] train={len(X_train)} test={len(X_test)} "
          f"R2={metrics['r2']:.6f} RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f}")
    print(f"  FEATURES (preserves leakage Harvested_Acre & Production_Ton): {list(X.columns)}")

    save_model_and_metrics(
        model, "baseline", metrics,
        list(X.columns), cat_cols, num_cols,
        feature_engineering=[],
        leakage_note=(
            "PRESERVES Harvested_Acre and Production_Ton as input features, matching the "
            "original project code. NOTE: Crop_Yield = Production_Ton / Harvested_Acre, "
            "so these are target-leakage features. Kept deliberately to reproduce the "
            "project-reported baseline methodology (R2=0.9873)."
        ),
    )


def _build_fe_selected_df():
    """Build dataframe with the 12 selected features + interactions from raw data.

    The project's FE file contains the selected features (including interactions).
    We reconstruct them from the raw cleaned data to keep raw human-readable inputs.
    """
    df = load_cleaned_data()
    df_fe = add_existing_feature_engineering(df)
    keep = [
        "Year", "Crop_Type", "Sown_Acre", "Soil_Type", "Avg_Temperature",
        "Total_Rainfall", "Avg_Humidity", "Water_Source", "Seeding_Season",
    ] + EXISTING_FE_INTERACTIONS + [TARGET]
    return df_fe[keep].copy(), df


def build_feature_engineering():
    # Use the actual project selected-features file to stay faithful to the pipeline.
    fe_df = load_selected_features()
    # Ensure interactions present (non-normalized file already contains them).
    if "Crop_x_WaterSource" not in fe_df.columns:
        fe_df = add_existing_feature_engineering(fe_df)

    train_mask = fe_df["Year"] <= YIELD_TL_TRAIN_END
    test_mask = fe_df["Year"] == YIELD_TL_TEST_YEAR

    X = fe_df.drop(columns=[TARGET])
    y = fe_df[TARGET]

    X_train, X_test = X[train_mask.values], X[test_mask.values]
    y_train, y_test = y[train_mask.values], y[test_mask.values]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        [
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_years": [int(y) for y in sorted(fe_df.loc[train_mask, "Year"].unique())],
        "test_years": [int(y) for y in sorted(fe_df.loc[test_mask, "Year"].unique())],
    }
    print(f"[crop_yield/feature_engineering] train={len(X_train)} test={len(X_test)} "
          f"R2={metrics['r2']:.6f} RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f}")
    print(f"  FEATURES: {list(X.columns)}")

    save_model_and_metrics(
        model, "feature_engineering", metrics,
        list(X.columns), cat_cols, num_cols,
        feature_engineering=EXISTING_FE_INTERACTIONS,
    )


def build_advanced():
    # Reproduce the association/sequential methodology to build the enhanced features.
    df = load_cleaned_data()
    df_fe = add_existing_feature_engineering(df)

    train_mask = df_fe["Year"] <= YIELD_TL_TRAIN_END

    # Yield_Level labeling (train-only median) for mining & sequential features.
    train_median = df_fe.loc[train_mask, TARGET].median()
    df_fe["Yield_Level"] = np.where(df_fe[TARGET] > train_median, "High", "Low")

    # Association rules (train only)
    yield_rules = mine_yield_association_rules(df_fe, train_mask)
    print(f"  [advanced] mined {len(yield_rules)} yield association rules (train only)")

    # Association + sequential features
    df_assoc = build_association_features(df_fe, yield_rules)
    df_enhanced = add_sequential_features(df_assoc)

    # Drop target/leakage columns (exactly as the project's Model3 code)
    df_enhanced = df_enhanced.drop(
        columns=[
            "Yield_Level",
            "Prev_Yield_Level",
            "Prev2_Yield_Level",
            "Prev3_Yield_Level",
            "Harvested_Acre",
            "Production_Ton",
            "Fertilizer_Import_Value(USD)",
            "Myanmar_GDP_USD",
        ]
    )

    feature_cols = (
        ["Year", "Crop_Type", "Sown_Acre", "Soil_Type", "Avg_Temperature",
         "Total_Rainfall", "Avg_Humidity", "Water_Source", "Seeding_Season"]
        + EXISTING_FE_INTERACTIONS
        + ["Assoc_High_Confidence", "Assoc_High_Lift", "Assoc_Low_Confidence",
           "Assoc_Low_Lift", "Assoc_Rule_Count"]
        + SEQUENCE_FEATURES
    )

    train_df = df_enhanced[df_enhanced["Year"] <= YIELD_TL_TRAIN_END]
    test_df = df_enhanced[df_enhanced["Year"] == YIELD_TL_TEST_YEAR]

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        [
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_years": [int(y) for y in sorted(train_df["Year"].unique())],
        "test_years": [int(y) for y in sorted(test_df["Year"].unique())],
    }
    print(f"[crop_yield/advanced] train={len(X_train)} test={len(X_test)} "
          f"R2={metrics['r2']:.6f} RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f}")

    save_model_and_metrics(
        model, "advanced", metrics,
        feature_cols, cat_cols, num_cols,
        feature_engineering=EXISTING_FE_INTERACTIONS + [
            "association: Assoc_High_Confidence, Assoc_High_Lift, Assoc_Low_Confidence, Assoc_Low_Lift, Assoc_Rule_Count",
            "sequential: Prev_Yield_High, Prev_Yield_Low, Prev2_Persistent, Prev3_Persistent",
        ],
    )

    # Save association rules next to the model
    serialize_rules(yield_rules, CROP_YIELD_DIR / "association_rules.csv")
    print(f"  saved: {CROP_YIELD_DIR / 'association_rules.csv'}")


def save_model_and_metrics(model, variant, metrics, features, categorical, numeric,
                           feature_engineering=None, leakage_note=None):
    CROP_YIELD_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CROP_YIELD_DIR / f"{variant}.joblib"
    preproc_path = CROP_YIELD_DIR / f"{variant}_preprocessor.joblib"
    metrics_path = CROP_YIELD_DIR / f"{variant}_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(model.named_steps["pre"], preproc_path)

    metrics["task"] = "crop_yield"
    metrics["variant"] = variant
    metrics["algorithm"] = "RandomForestRegressor"
    metrics["target"] = "Crop_Yield"
    metrics["features"] = features
    metrics["categorical_features"] = categorical
    metrics["numeric_features"] = numeric
    metrics["preprocessing"] = "OneHotEncoder(handle_unknown='ignore') + numeric passthrough"
    metrics["feature_engineering"] = feature_engineering or []
    metrics["predict_proba"] = False
    metrics["feature_importances"] = True
    if leakage_note:
        metrics["leakage_note"] = leakage_note

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  saved: {model_path}")
    print(f"  saved: {preproc_path}")
    print(f"  saved: {metrics_path}")


if __name__ == "__main__":
    print("===== Crop Yield regression artifact generation =====")
    build_baseline()
    build_feature_engineering()
    build_advanced()
    print("===== Done =====")
