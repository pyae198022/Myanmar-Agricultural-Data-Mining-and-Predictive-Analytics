"""
Generate Crop Type model artifacts (baseline & feature_engineering).

Reproduces the exact Data Mining project methodology:
- Baseline: RandomForestClassifier, 8 baseline features, OneHotEncoder + passthrough,
  chronological split Train 2012-2021 / Test 2022-2023.
- Feature Engineering: RandomForestClassifier, log1p(Sown_Acre) + log1p(Total_Rainfall)
  engineered features, OneHotEncoder + SimpleImputer, same chronological split.

No Advanced crop type model exists in the project — it is NOT generated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (
    CROP_TYPE_TRAIN_END_INCLUSIVE,
    load_cleaned_data,
)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
CROP_TYPE_DIR = MODELS_ROOT / "crop_type"


def build_baseline():
    df = load_cleaned_data()

    TARGET = "Crop_Type"
    BASELINE_FEATURES = [
        "Region",
        "Soil_Type",
        "Water_Source",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Year",
        "Sown_Acre",
    ]
    categorical = ["Region", "Soil_Type", "Water_Source"]
    numeric = ["Avg_Temperature", "Total_Rainfall", "Avg_Humidity", "Year", "Sown_Acre"]

    train_df = df[df["Year"] <= CROP_TYPE_TRAIN_END_INCLUSIVE]
    test_df = df[df["Year"] > CROP_TYPE_TRAIN_END_INCLUSIVE]

    X_train, y_train = train_df[BASELINE_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[BASELINE_FEATURES], test_df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"[crop_type/baseline] train={len(X_train)} rows "
          f"({sorted(train_df.Year.unique())}) test={len(X_test)} rows "
          f"({sorted(test_df.Year.unique())})")
    print(f"  accuracy={accuracy:.6f} precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}")

    save_model_and_metrics(
        model, "baseline",
        {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_years": [int(y) for y in sorted(train_df.Year.unique())],
            "test_years": [int(y) for y in sorted(test_df.Year.unique())],
        },
        BASELINE_FEATURES,
        categorical,
        numeric,
    )


def build_feature_engineering():
    df = load_cleaned_data()

    TARGET = "Crop_Type"
    CATEGORICAL = ["Region", "Soil_Type", "Water_Source"]
    ENGINEERED_NUMERIC = [
        "Log_Sown_Acre",
        "Log_Total_Rainfall",
        "Avg_Temperature",
        "Avg_Humidity",
        "Year",
    ]
    MODEL_FEATURES = ENGINEERED_NUMERIC + CATEGORICAL

    train_df = df[df["Year"] <= CROP_TYPE_TRAIN_END_INCLUSIVE]
    test_df = df[df["Year"] > CROP_TYPE_TRAIN_END_INCLUSIVE]

    def engineer(d):
        out = d.copy()
        out["Log_Sown_Acre"] = np.log1p(out["Sown_Acre"].clip(lower=0))
        out["Log_Total_Rainfall"] = np.log1p(out["Total_Rainfall"].clip(lower=0))
        return out

    X_train = engineer(train_df)[MODEL_FEATURES]
    y_train = train_df[TARGET]
    X_test = engineer(test_df)[MODEL_FEATURES]
    y_test = test_df[TARGET]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ENGINEERED_NUMERIC),
            ("cat", categorical_transformer, CATEGORICAL),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"[crop_type/feature_engineering] train={len(X_train)} rows "
          f"({sorted(train_df.Year.unique())}) test={len(X_test)} rows "
          f"({sorted(test_df.Year.unique())})")
    print(f"  accuracy={accuracy:.6f} precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}")

    save_model_and_metrics(
        model, "feature_engineering",
        {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_years": [int(y) for y in sorted(train_df.Year.unique())],
            "test_years": [int(y) for y in sorted(test_df.Year.unique())],
        },
        MODEL_FEATURES,
        CATEGORICAL,
        ENGINEERED_NUMERIC,
    )


def save_model_and_metrics(model, variant, metrics, features, categorical, numeric):
    CROP_TYPE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CROP_TYPE_DIR / f"{variant}.joblib"
    preproc_path = CROP_TYPE_DIR / f"{variant}_preprocessor.joblib"
    metrics_path = CROP_TYPE_DIR / f"{variant}_metrics.json"

    # Save the complete pipeline as the model artifact; also save the fitted
    # preprocessing step separately for introspection.
    joblib.dump(model, model_path)
    joblib.dump(model.named_steps["preprocessor"], preproc_path)

    metrics["task"] = "crop_type"
    metrics["variant"] = variant
    metrics["algorithm"] = "RandomForestClassifier"
    metrics["target"] = "Crop_Type"
    metrics["features"] = features
    metrics["categorical_features"] = categorical
    metrics["numeric_features"] = numeric
    metrics["preprocessing"] = "OneHotEncoder(handle_unknown='ignore') + numeric passthrough" \
        if variant == "baseline" \
        else "SimpleImputer + OneHotEncoder(handle_unknown='ignore') for cat; SimpleImputer(median) for engineered num"
    metrics["feature_engineering"] = [] if variant == "baseline" \
        else ["Log_Sown_Acre=log1p(Sown_Acre.clip(lower=0))", "Log_Total_Rainfall=log1p(Total_Rainfall.clip(lower=0))"]
    metrics["predict_proba"] = True
    metrics["feature_importances"] = True

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  saved: {model_path}")
    print(f"  saved: {preproc_path}")
    print(f"  saved: {metrics_path}")


if __name__ == "__main__":
    print("===== Crop Type artifact generation =====")
    build_baseline()
    build_feature_engineering()
    print("===== Done =====")
