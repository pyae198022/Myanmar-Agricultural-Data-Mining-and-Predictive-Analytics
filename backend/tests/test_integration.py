"""
Integration tests verifying the FastAPI backend correctly loads and uses the
REAL trained model artifacts.

Validates:
1. All 8 model artifacts load (crop_type: baseline/FE; yield_level: baseline/FE/advanced;
   crop_yield: baseline/FE/advanced).
2. Predictions contain no NaN.
3. Crop type prediction is one of the actual trained classes.
4. Yield level is ONLY Low or High (no Medium).
5. predict_proba works for classifiers.
6. Regression prediction is numeric.
7. Metrics match the corresponding *_metrics.json artifact.
8. crop_type advanced is correctly unavailable (503).
9. No hash-based / manual fake encoding remains in preprocessing.
10. No random prediction generation remains in the backend.
11. No setTimeout/mock inference remains in the backend.
12. Association rules load for advanced models.
13. Sequential feature generation does not use future information.
14. Baseline yield_level / crop_yield are guarded (503) per design.
15. API prediction matches a direct preprocess -> model.predict() reconstruction
    on real rows from the bundled dataset.
"""

from __future__ import annotations

import pathlib
import re

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

app_dir = pathlib.Path(__file__).resolve().parent.parent
MODELS = app_dir / "models"
DATA = app_dir / "data" / "cleaned_data.csv"

from app.main import app  # noqa: E402
from app.services.model_loader import model_loader  # noqa: E402
from app.preprocessing import reference  # noqa: E402

client = TestClient(app)

# All 8 real variants and their expected artifact directories.
ALL_REAL_VARIANTS = {
    "crop_type": ["baseline", "feature_engineering"],
    "yield_level": ["baseline", "feature_engineering", "advanced"],
    "crop_yield": ["baseline", "feature_engineering", "advanced"],
}

USABLE = {
    "crop_type": ["baseline", "feature_engineering"],
    "yield_level": ["feature_engineering", "advanced"],
    "crop_yield": ["feature_engineering", "advanced"],
}

# Real input mapping frontend keys -> Pydantic snake_case.
def make_payload(row):
    return {
        "region": row["Region"],
        "year": int(row["Year"]),
        "crop_type": row["Crop_Type"],
        "sown_acre": float(row["Sown_Acre"]),
        "soil_type": row["Soil_Type"],
        "avg_temperature": float(row["Avg_Temperature"]),
        "total_rainfall": float(row["Total_Rainfall"]),
        "avg_humidity": float(row["Avg_Humidity"]),
        "water_source": row["Water_Source"],
        "seeding_season": row["Seeding_Season"],
    }


def load_real_rows(n=3):
    df = pd.read_csv(DATA)
    rows = df[df["Year"] == 2023].drop_duplicates(["Region", "Crop_Type"]).head(n)
    return [make_payload(r) for _, r in rows.iterrows()]


REAL_ROWS = load_real_rows(3)


# ── 1. All 8 variants load ─────────────────────────────────────────────────

def test_all_8_artifacts_load():
    for task, variants in ALL_REAL_VARIANTS.items():
        for variant in variants:
            m = model_loader.get_model(task, variant)
            assert m is not None, f"{task}/{variant} failed to load"


# ── Registry / availability ────────────────────────────────────────────────

def test_health_reports_availability():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    ml = body["models_loaded"]
    assert ml["crop_type"]["advanced"] is False, "crop_type/advanced must be unavailable"
    assert ml["crop_type"]["baseline"] is True
    assert ml["crop_type"]["feature_engineering"] is True
    assert ml["yield_level"]["feature_engineering"] is True
    assert ml["yield_level"]["advanced"] is True
    assert ml["crop_yield"]["feature_engineering"] is True
    assert ml["crop_yield"]["advanced"] is True


# ── 2,3,5. Crop type predictions ───────────────────────────────────────────

@pytest.mark.parametrize("variant", ["baseline", "feature_engineering"])
def test_crop_type_prediction(variant):
    for payload in REAL_ROWS:
        resp = client.post("/api/predict/crop-type",
                           json={**payload, "model_variant": variant})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predictedCrop"]  # non-empty
        assert not np.isnan(body["confidence"])
        assert len(body["topPredictions"]) > 0
        assert all(not np.isnan(p["probability"]) for p in body["topPredictions"])
        assert "modelMetrics" in body and body["modelMetrics"].get("accuracy") is not None


# ── 8. crop_type advanced unavailable ──────────────────────────────────────

def test_crop_type_advanced_unavailable():
    resp = client.post("/api/predict/crop-type",
                       json={**REAL_ROWS[0], "model_variant": "advanced"})
    assert resp.status_code == 503


# ── 4,5. Yield level predictions (Low/High only) ───────────────────────────

@pytest.mark.parametrize("variant", ["feature_engineering", "advanced"])
def test_yield_level_prediction_low_high_only(variant):
    for payload in REAL_ROWS:
        resp = client.post("/api/predict/yield-level",
                           json={**payload, "model_variant": variant})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predictedLevel"] in {"Low", "High"}, body["predictedLevel"]
        assert body["probabilityLow"] >= 0.0 and body["probabilityHigh"] >= 0.0
        # probabilities should sum ~ 1
        assert abs((body["probabilityLow"] + body["probabilityHigh"]) - 1.0) < 1e-3
        assert not np.isnan(body["probabilityLow"])
        assert not np.isnan(body["probabilityHigh"])


# ── 6,7. Crop yield regression predictions ─────────────────────────────────

@pytest.mark.parametrize("variant", ["feature_engineering", "advanced"])
def test_crop_yield_regression(variant):
    for payload in REAL_ROWS:
        resp = client.post("/api/predict/crop-yield",
                           json={**payload, "model_variant": variant})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body["predictedYield"], (int, float))
        assert not np.isnan(body["predictedYield"])
        assert body["modelMetrics"].get("r2") is not None


# ── 14. Baseline guarded (503) ─────────────────────────────────────────────

@pytest.mark.parametrize("task,path", [
    ("yield-level", "/api/predict/yield-level"),
    ("crop-yield", "/api/predict/crop-yield"),
])
def test_baseline_guarded_503(task, path):
    resp = client.post(path, json={**REAL_ROWS[0], "model_variant": "baseline"})
    assert resp.status_code == 503
    assert "Harvested_Acre" in resp.json()["detail"] or "Production_Ton" in resp.json()["detail"]


# ── 7. Metrics match artifacts ─────────────────────────────────────────────

def test_metrics_match_artifacts():
    import json as _json
    for task, variants in USABLE.items():
        for variant in variants:
            stored = _json.loads(
                (MODELS / task / f"{variant}_metrics.json").read_text()
            )
            # Obtain metrics through a prediction response
            if task == "crop_type":
                resp = client.post("/api/predict/crop-type",
                                   json={**REAL_ROWS[0], "model_variant": variant})
            elif task == "yield_level":
                resp = client.post("/api/predict/yield-level",
                                   json={**REAL_ROWS[0], "model_variant": variant})
            else:
                resp = client.post("/api/predict/crop-yield",
                                   json={**REAL_ROWS[0], "model_variant": variant})
            returned = resp.json()["modelMetrics"]
            for key in ("accuracy", "f1", "precision", "recall", "r2", "rmse", "mae"):
                if key in stored:
                    assert returned.get(key) == stored[key], f"{task}/{variant} {key} mismatch"


# ── 12. Association rules load ─────────────────────────────────────────────

def test_association_rules_load():
    for task in ("yield_level", "crop_yield"):
        rules = reference.load_association_rules(task)
        assert len(rules) > 0, f"{task}/association_rules.csv empty"
        assert {"Antecedent", "Consequent", "Confidence", "Lift"} <= set(rules.columns)


# ── 13. Sequential features don't use future info ──────────────────────────

def test_sequential_features_no_future():
    # For year 2023, features must come only from years < 2023.
    seq = reference.build_sequential_features(region="Sagaing", crop_type="Paddy", year=2023)
    history = reference._yield_level_history().get(("Sagaing", "Paddy"), {})
    assert all(field in ("Prev_Yield_High", "Prev_Yield_Low",
                         "Prev2_Persistent", "Prev3_Persistent") for field in seq)
    # Non-fabricated: values are either 0/1 and derive from history only.
    assert all(v in (0, 1) for v in seq.values())
    # Manually recompute using ONLY years < 2023 and compare.
    prior_levels = [history[y] for y in sorted(history) if y < 2023]
    assert len(prior_levels) >= 3, "expected >=3 prior years for Sagaing/Paddy"
    manual = {
        "Prev_Yield_High": 1 if prior_levels[-1] == "High" else 0,
        "Prev_Yield_Low": 1 if prior_levels[-1] == "Low" else 0,
        "Prev2_Persistent": 1 if prior_levels[-1] == prior_levels[-2] else 0,
        "Prev3_Persistent": 1 if (prior_levels[-1] == prior_levels[-2] == prior_levels[-3]) else 0,
    }
    assert seq == manual, f"sequential mismatch: {seq} != {manual}"


# ── 15. API prediction == direct preprocess->predict on real rows ──────────

def test_api_matches_direct_advanced():
    from app.services.prediction import prediction_service
    # For crop_yield/advanced: compare API predictedYield to direct model.predict
    model = model_loader.get_model("crop_yield", "advanced")
    pre = prediction_service.preprocessor
    for payload in REAL_ROWS:
        X = pre.transform(payload, "crop_yield", "advanced", model, None)
        direct = float(model.predict(X)[0])
        api = prediction_service.predict_crop_yield(payload, "advanced")["predictedYield"]
        # API intentionally rounds to 4 decimal places.
        assert abs(direct - api) < 1e-3, f"API {api} != direct {direct}"


# ── Compare endpoint ───────────────────────────────────────────────────────

def test_compare_endpoint():
    resp = client.get("/api/compare", params={"task": "crop_type", **dict(REAL_ROWS[0])})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    variants = [m["variant"] for m in body["models"] if m.get("available")]
    assert "baseline" in variants
    assert "feature_engineering" in variants
    assert "advanced" not in variants, "crop_type must not list advanced as real"
    assert body["best_model"] in variants


def test_historical_returns_real_data():
    # Real bundled dataset values for Mandalay/Paddy must be returned, never fabricated.
    resp = client.get("/api/historical", params={"region": "Mandalay", "crop_type": "Paddy"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("years"), list) and len(body["years"]) > 0
    for key in ("yields", "rainfall", "temperatures", "areas"):
        assert key in body, f"missing historical key: {key}"
        assert isinstance(body[key], list)
        assert len(body[key]) == len(body["years"]), f"{key} length mismatch"

    # Cross-check against the raw dataset directly.
    df = pd.read_csv(DATA)
    mask = (df["Region"] == "Mandalay") & (df["Crop_Type"] == "Paddy")
    years = sorted(df.loc[mask, "Year"].unique().tolist())
    assert body["years"] == years
    expected_yield = df.loc[mask].groupby("Year")["Crop_Yield"].mean()
    assert abs(body["yields"][0] - expected_yield.loc[body["years"][0]]) < 1e-3


def test_historical_missing_data_returns_404():
    resp = client.get("/api/historical", params={"region": "Mandalay", "crop_type": "DoesNotExistCrop"})
    assert resp.status_code == 404, resp.text


# ── Clustering endpoints ─────────────────────────────────────────────────────

def test_clustering_overview_endpoint():
    resp = client.get("/api/clustering/overview")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["total_records"] == 5940
    assert body["n_clusters"] == 8
    assert body["selected_k"] == 8
    # Primary silhouette should be Chapter 4.3 evaluation (0.053031)
    assert abs(body["overall_silhouette"] - 0.053031) < 1e-4
    assert body["hopkins_statistic"] > 0.5

    distribution = body["cluster_distribution"]
    assert len(distribution) == 8
    assert sum(row["count"] for row in distribution) == 5940
    expected_counts = [490, 1312, 844, 917, 789, 853, 341, 394]
    assert [row["count"] for row in distribution] == expected_counts


def test_clustering_optimal_k_endpoint():
    resp = client.get("/api/clustering/optimal-k")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["candidate_k"] == list(range(2, 11))
    assert body["selected_k"] == 8
    assert len(body["results"]) == 9

    selected = next(row for row in body["results"] if row["k"] == 8)
    assert selected["is_selected"] is True
    assert abs(selected["silhouette_score"] - 0.307068) < 1e-4


def test_clustering_profiles_endpoint():
    resp = client.get("/api/clustering/profiles")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["total_records"] == 5940
    assert body["selected_k"] == 8
    assert len(body["profiles"]) == 8

    cluster0 = next(row for row in body["profiles"] if row["cluster_id"] == 0)
    assert cluster0["count"] == 490
    assert cluster0["dominant_region"] == "Mandalay"
    assert cluster0["dominant_crop_type"] == "Paddy"
    assert cluster0["dominant_soil_type"] == "Luvisols / Cambisols / Savanna"
    assert cluster0["dominant_water_source"] == "Rainfed"
    assert "Cluster 0" in cluster0["interpretation_rule"]
    assert set(cluster0["numeric_means"].keys()) == {
        "Sown_Acre",
        "Harvested_Acre",
        "Production_Ton",
        "Fertilizer_Import_Value(USD)",
        "Avg_Temperature",
        "Total_Rainfall",
        "Myanmar_GDP_USD",
        "Avg_Humidity",
    }


def test_clustering_evaluation_endpoint():
    resp = client.get("/api/clustering/evaluation")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Ground truth from Project Book Chapter 4.3 evaluation: 0.053031
    # Using StandardScaler with evaluation features (Crop_Yield included, Myanmar_GDP_USD excluded)
    assert abs(body["overall_silhouette"] - 0.053031) < 1e-4
    assert body["overall_quality"] == "Weak"
    assert len(body["per_cluster"]) == 8

    strongest = body["strongest_cluster"]
    assert strongest["cluster_id"] == 7  # Cluster 7 has highest silhouette (0.4484)
    assert strongest["quality"] == "Fair"
    assert abs(strongest["mean_silhouette"] - 0.4484) < 1e-4


# ── 9,10,11. No fake/random/hash/mock inference in backend ─────────────────

def test_no_hash_based_encoding():
    src = (app_dir / "app" / "services" / "preprocessing.py").read_text()
    assert "hash(" not in src, "hash-based encoding must be removed"
    assert "import random" not in src
    assert "random.seed" not in src
    assert "seeded" not in src

def test_no_random_prediction_generation():
    for path in (app_dir / "app").rglob("*.py"):
        text = path.read_text()
        assert "seededRandom" not in text
        assert "random.random" not in text
        assert "random.uniform" not in text

def test_no_mock_inference_in_backend():
    for path in (app_dir / "app").rglob("*.py"):
        text = path.read_text()
        assert "setTimeout" not in text, f"setTimeout found in {path}"
        assert "mock" not in text.lower(), f"mock found in {path}"


# ── Registry: crop_type advanced NOT_AVAILABLE ────────────────────────────

def test_registry_crop_type_advanced_not_available():
    reg = reference.load_registry()
    assert reg["crop_type"]["advanced"]["status"] == "NOT_AVAILABLE"


# ── Crop Yield evaluation: Actual vs Predicted ───────────────────────────────

def test_crop_yield_actual_vs_predicted_fe():
    resp = client.get("/api/evaluation/crop-yield/actual-vs-predicted?variant=feature_engineering")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_test"] == 495  # real 2023 held-out crop_yield test set
    assert len(body["points"]) == 495
    assert abs(body["metrics"]["r2"] - 0.9867314439759981) < 1e-3
    for pt in body["points"][:5]:
        assert "actual" in pt and "predicted" in pt and "year" in pt and "crop_type" in pt
        assert np.isfinite(pt["actual"]) and np.isfinite(pt["predicted"])


def test_crop_yield_actual_vs_predicted_advanced():
    resp = client.get("/api/evaluation/crop-yield/actual-vs-predicted?variant=advanced")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_test"] == 495
    assert abs(body["metrics"]["r2"] - 0.9870587873710347) < 1e-3


def test_crop_yield_actual_vs_predicted_baseline_unavailable():
    resp = client.get("/api/evaluation/crop-yield/actual-vs-predicted?variant=baseline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["points"] == []
    assert body["n_test"] == 0


# ── Crop Yield evaluation: Cross-Validation ──────────────────────────────────

def test_crop_yield_cross_validation_reference():
    resp = client.get("/api/evaluation/crop-yield/cross-validation")
    assert resp.status_code == 200
    body = resp.json()
    assert [m["label"] for m in body["models"]] == [
        "Baseline NN", "Feature Engineering NN", "Association-Enhanced NN",
    ]
    # Project Book documented values
    fe = body["models"][1]
    assert abs(fe["r2"]["mean"] - 0.9765) < 1e-4
    assert abs(fe["rmse"]["mean"] - 0.6062) < 1e-4
    assert abs(fe["mae"]["mean"] - 0.1978) < 1e-4
    # Random Forest must not be implied to have CV results
    assert body["deployed_algorithms"]["feature_engineering"] == "RandomForestRegressor"
    assert "Random Forest" in body["note"]


# ── Crop Yield evaluation: Feature Importance ────────────────────────────────

def test_crop_yield_feature_importance_advanced():
    resp = client.get("/api/evaluation/crop-yield/feature-importance?variant=advanced")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["type"] == "Feature Importance"
    assert body["method"] == "native_random_forest"
    feats = body["features"]
    assert len(feats) > 0
    # Sorted descending by importance
    imps = [f["importance"] for f in feats]
    assert imps == sorted(imps, reverse=True)
    assert abs(sum(imps) - 1.0) < 0.02


def test_crop_yield_feature_importance_fe():
    resp = client.get("/api/evaluation/crop-yield/feature-importance?variant=feature_engineering")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert len(body["features"]) == 12


def test_crop_yield_feature_importance_baseline_unavailable():
    resp = client.get("/api/evaluation/crop-yield/feature-importance?variant=baseline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["features"] == []


# ── Crop Yield Level evaluation: binary ROC / AUC ────────────────────────────

def test_yield_level_roc_fe():
    resp = client.get("/api/evaluation/yield-level/roc?variant=feature_engineering")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["classification_type"] == "binary"
    assert body["roc_method"] == "standard binary ROC (pos_label=High)"
    assert body["n_test"] == 495
    # Standard binary AUC (not One-vs-Rest). Computed live from the deployed MLP.
    assert 0.0 <= body["auc"] <= 1.0
    assert len(body["curve"]["fpr"]) == len(body["curve"]["tpr"])
    # AUC must be well above random (0.5) for a real trained model.
    assert body["auc"] > 0.9


def test_yield_level_roc_advanced():
    resp = client.get("/api/evaluation/yield-level/roc?variant=advanced")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_test"] == 495
    assert body["auc"] > 0.9


def test_yield_level_roc_baseline_unavailable():
    resp = client.get("/api/evaluation/yield-level/roc?variant=baseline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["auc"] is None
    assert body["curve"] is None
    assert "guarded" in body["reason"]


def test_yield_level_roc_available():
    resp = client.get("/api/evaluation/yield-level/roc/available")
    assert resp.status_code == 200
    body = resp.json()
    variants = {m["variant"]: m["available"] for m in body["models"]}
    assert variants["feature_engineering"] is True
    assert variants["advanced"] is True
    assert variants["baseline"] is False


# ── Crop Yield Level evaluation: 5-Fold Stratified CV ────────────────────────

def test_yield_level_cross_validation_values():
    resp = client.get("/api/evaluation/yield-level/cross-validation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"].startswith("5-Fold Stratified")
    assert "2012-2022" in body["data_scope"]

    by_var = {m["variant"]: m for m in body["models"]}
    assert "feature_engineering" in by_var and "advanced" in by_var
    assert "baseline" not in by_var  # excluded (leakage-guarded)

    fe = by_var["feature_engineering"]
    assert fe["n_train"] == 5445
    assert fe["n_splits"] == 5
    # Real computed values (reproducible: fixed StratifiedKFold random_state=42).
    assert abs(fe["accuracy"]["mean"] - 91.68) < 0.5
    assert abs(fe["precision"]["mean"] - 91.71) < 1.0
    assert abs(fe["recall"]["mean"] - 91.74) < 1.0
    assert abs(fe["f1"]["mean"] - 91.67) < 0.5
    assert abs(fe["roc_auc"]["mean"] - 0.9809) < 0.005

    adv = by_var["advanced"]
    assert adv["n_train"] == 5445
    assert abs(adv["accuracy"]["mean"] - 96.91) < 0.5
    assert abs(adv["precision"]["mean"] - 97.05) < 0.5
    assert abs(adv["recall"]["mean"] - 96.77) < 0.5
    assert abs(adv["f1"]["mean"] - 96.91) < 0.5
    assert abs(adv["roc_auc"]["mean"] - 0.9896) < 0.005

    assert "not fitted" in body["note"]


def test_yield_level_cross_validation_project_book_reference_present():
    resp = client.get("/api/evaluation/yield-level/cross-validation")
    assert resp.status_code == 200
    body = resp.json()
    by_var = {m["variant"]: m for m in body["models"]}
    # Project Book Table 4.9 reference values must be present exactly.
    fe_ref = by_var["feature_engineering"]["project_book_reference"]
    assert abs(fe_ref["accuracy"][0] - 92.54) < 1e-9
    assert abs(fe_ref["roc_auc"][0] - 0.9830) < 1e-9

    adv_ref = by_var["advanced"]["project_book_reference"]
    assert abs(adv_ref["accuracy"][0] - 96.79) < 1e-9
    assert abs(adv_ref["roc_auc"][0] - 0.9897) < 1e-9
    assert adv_ref["accuracy"] == [96.79, 0.34]
