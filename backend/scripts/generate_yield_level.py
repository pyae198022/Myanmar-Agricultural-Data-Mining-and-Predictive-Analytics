"""
Generate Yield Level classification model artifacts (baseline, feature_engineering, advanced).

Reproduces the exact Data Mining project methodology (all MLPClassifier):

- Baseline: MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
    max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42) on the
    14 original features; pd.get_dummies encoding + MinMaxScaler.
- Feature Engineering: same MLP on the 12 selected features + 3 Crop_x_* interactions;
    pd.get_dummies + MinMaxScaler. (This reproduces the 02 notebook which trained WITHOUT
    association/sequential features.)
- Advanced: same MLP on 12 selected + 3 interactions + 5 association + 4 sequential
    (Model3 methodology, TRAIN-ONLY association mining). OneHotEncoder + MinMaxScaler.

Target: Yield_Level (Low/High, binary). Labeling uses training-only median Crop_Yield.
Split: Train 2012-2022, Test 2023.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from common import (
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
    mine_yield_association_rules,
    serialize_rules,
)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
YIELD_LEVEL_DIR = MODELS_ROOT / "yield_level"

TARGET = "Yield_Level"

MLP_CONFIG = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)


def build_baseline():
    train, test = load_train_test_labeled()

    X_train, y_train = train.drop(columns=[TARGET, "Crop_Yield", "Predicted_Yield_Level", "Probability_High"], errors="ignore"), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET, "Crop_Yield", "Predicted_Yield_Level", "Probability_High"], errors="ignore"), test[TARGET]

    # The baseline uses pd.get_dummies + MinMaxScaler as in the project notebook.
    Xtr = pd.get_dummies(X_train, dtype=float)
    Xte = pd.get_dummies(X_test, dtype=float)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler = MinMaxScaler()
    # NOTE: get_dummies produces only numeric columns; scaling all is faithful since
    # the project scaled the full dummy matrix with MinMaxScaler.
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = MLPClassifier(**MLP_CONFIG)
    model.fit(Xtr_s, y_train)
    y_pred = model.predict(Xte_s)

    metrics = _classification_metrics(y_test, y_pred, len(X_train), len(X_test), train, test)
    print(f"[yield_level/baseline] train={len(X_train)} test={len(X_test)} "
          f"acc={metrics['accuracy']:.6f} f1={metrics['f1']:.6f}")

    # Save model + a serialized preprocessor bundle (dummy columns + scaler).
    save_mlp_artifacts(
        model, "baseline",
        {"dummy_columns": list(Xtr.columns), "minmax": scaler},
        metrics,
        features=list(X_train.columns),
        categorical=[c for c in X_train.columns if X_train[c].dtype == "object"],
        numeric=[c for c in X_train.columns if X_train[c].dtype != "object"],
        encoding="pd.get_dummies(dtype=float) + MinMaxScaler",
    )


def build_feature_engineering():
    # Reconstruct the 12 selected + 3 interaction features from raw data.
    df = load_cleaned_data()
    df_fe = add_existing_feature_engineering(df)

    # Labeling (train-only median) exactly as project
    train_mask = df_fe["Year"] <= YIELD_TL_TRAIN_END
    train_median = df_fe.loc[train_mask, "Crop_Yield"].median()
    df_fe["Yield_Level"] = np.where(df_fe["Crop_Yield"] > train_median, "High", "Low")

    keep = [
        "Year", "Crop_Type", "Sown_Acre", "Soil_Type", "Avg_Temperature",
        "Total_Rainfall", "Avg_Humidity", "Water_Source", "Seeding_Season",
    ] + EXISTING_FE_INTERACTIONS + ["Yield_Level"]

    df_fe = df_fe[keep]

    train_df = df_fe[df_fe["Year"] <= YIELD_TL_TRAIN_END]
    test_df = df_fe[(df_fe["Year"] == YIELD_TL_TEST_YEAR)]

    X_train, y_train = train_df.drop(columns=["Yield_Level"]), train_df["Yield_Level"]
    X_test, y_test = test_df.drop(columns=["Yield_Level"]), test_df["Yield_Level"]

    Xtr = pd.get_dummies(X_train, dtype=float)
    Xte = pd.get_dummies(X_test, dtype=float)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = MLPClassifier(**MLP_CONFIG)
    model.fit(Xtr_s, y_train)
    y_pred = model.predict(Xte_s)

    metrics = _classification_metrics(y_test, y_pred, len(X_train), len(X_test), train_df, test_df)
    print(f"[yield_level/feature_engineering] train={len(X_train)} test={len(X_test)} "
          f"acc={metrics['accuracy']:.6f} f1={metrics['f1']:.6f}")

    save_mlp_artifacts(
        model, "feature_engineering",
        {"dummy_columns": list(Xtr.columns), "minmax": scaler},
        metrics,
        features=list(X_train.columns),
        categorical=EXISTING_FE_CATEGORICAL + EXISTING_FE_INTERACTIONS,
        numeric=EXISTING_FE_NUMERIC + ["Year"],
        encoding="pd.get_dummies(dtype=float) + MinMaxScaler",
        feature_engineering=EXISTING_FE_INTERACTIONS,
    )


def build_advanced():
    df = load_cleaned_data()
    df_fe = add_existing_feature_engineering(df)

    train_mask = df_fe["Year"] <= YIELD_TL_TRAIN_END

    train_median = df_fe.loc[train_mask, "Crop_Yield"].median()
    df_fe["Yield_Level"] = np.where(df_fe["Crop_Yield"] > train_median, "High", "Low")

    yield_rules = mine_yield_association_rules(df_fe, train_mask)
    print(f"  [advanced] mined {len(yield_rules)} yield association rules (train only)")

    df_assoc = build_association_features(df_fe, yield_rules)
    df_enhanced = add_sequential_features(df_assoc)

    # Feature set: 12 selected + 3 interactions + 5 association + 4 sequential
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

    X_train, y_train = train_df[feature_cols], train_df["Yield_Level"]
    X_test, y_test = test_df[feature_cols], test_df["Yield_Level"]

    # Advanced uses sklearn OneHotEncoder + MinMaxScaler via ColumnTransformer (Model3).
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ]
    )

    model = Pipeline([("pre", preprocessor), ("mlp", MLPClassifier(**MLP_CONFIG))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _classification_metrics(y_test, y_pred, len(X_train), len(X_test), train_df, test_df)
    print(f"[yield_level/advanced] train={len(X_train)} test={len(X_test)} "
          f"acc={metrics['accuracy']:.6f} f1={metrics['f1']:.6f}")

    save_mlp_artifacts(
        model, "advanced",
        None,  # preprocessor is inside the pipeline
        metrics,
        features=feature_cols,
        categorical=cat_cols,
        numeric=num_cols,
        encoding="OneHotEncoder(handle_unknown='ignore') + MinMaxScaler (ColumnTransformer)",
        feature_engineering=EXISTING_FE_INTERACTIONS + [
            "association: 5 features",
            "sequential: 4 features",
        ],
    )

    serialize_rules(yield_rules, YIELD_LEVEL_DIR / "association_rules.csv")
    print(f"  saved: {YIELD_LEVEL_DIR / 'association_rules.csv'}")


def _classification_metrics(y_test, y_pred, n_train, n_test, train_df, test_df):
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary", pos_label="High", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary", pos_label="High", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="binary", pos_label="High", zero_division=0)),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_years": [int(y) for y in sorted(train_df["Year"].unique())],
        "test_years": [int(y) for y in sorted(test_df["Year"].unique())],
    }


def save_mlp_artifacts(model, variant, preproc_bundle, metrics, features, categorical,
                       numeric, encoding, feature_engineering=None):
    YIELD_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = YIELD_LEVEL_DIR / f"{variant}.joblib"
    preproc_path = YIELD_LEVEL_DIR / f"{variant}_preprocessor.joblib"
    metrics_path = YIELD_LEVEL_DIR / f"{variant}_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(preproc_bundle, preproc_path)

    metrics["task"] = "yield_level"
    metrics["variant"] = variant
    metrics["algorithm"] = "MLPClassifier"
    metrics["target"] = "Yield_Level"
    metrics["classes"] = ["Low", "High"]
    metrics["features"] = features
    metrics["categorical_features"] = categorical
    metrics["numeric_features"] = numeric
    metrics["preprocessing"] = encoding
    metrics["feature_engineering"] = feature_engineering or []
    metrics["predict_proba"] = True
    metrics["feature_importances"] = False

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  saved: {model_path}")
    print(f"  saved: {preproc_path}")
    print(f"  saved: {metrics_path}")


def load_train_test_labeled():
    from common import TRAIN_LABELED, TEST_LABELED_2023
    train = pd.read_excel(TRAIN_LABELED)
    test = pd.read_excel(TEST_LABELED_2023)
    return train, test


if __name__ == "__main__":
    print("===== Yield Level classification artifact generation =====")
    build_baseline()
    build_feature_engineering()
    build_advanced()
    print("===== Done =====")
