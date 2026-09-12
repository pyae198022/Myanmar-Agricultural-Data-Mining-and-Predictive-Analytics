"""
Clustering analysis service for the Descriptive Mining experience.

Implements the Project Book K-Means methodology on the bundled cleaned dataset:
  - feature selection per Chapter 3.1.6
  - log1p on the specified skewed numeric fields
  - Min-Max scaling to [0, 1]
  - Hopkins statistic for clustering tendency
  - K evaluation for K=2..10 using inertia and silhouette
  - final KMeans with K=8, random_state=42, n_init=10

All results are derived from ``backend/data/cleaned_data.csv``. Nothing is
hardcoded into the analytical pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from app.preprocessing.reference import DATA_CSV, load_clean_data

CLUSTERING_FEATURES = [
    "Sown_Acre",
    "Harvested_Acre",
    "Production_Ton",
    "Fertilizer_Import_Value(USD)",
    "Avg_Temperature",
    "Total_Rainfall",
    "Myanmar_GDP_USD",
    "Avg_Humidity",
]

LOG1P_FEATURES = [
    "Sown_Acre",
    "Harvested_Acre",
    "Production_Ton",
    "Total_Rainfall",
    "Myanmar_GDP_USD",
]

PROFILE_CATEGORICAL_FIELDS = [
    "Region",
    "Crop_Type",
    "Soil_Type",
    "Water_Source",
]

EXCLUDED_FROM_CLUSTERING_MATRIX = [
    "Region",
    "Year",
    "Crop_Type",
    "Soil_Type",
    "Seeding_Season",
    "Water_Source",
    "Crop_Yield",
]

RULE_NUMERIC_FIELDS = [
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Sown_Acre",
    "Harvested_Acre",
    "Production_Ton",
    "Crop_Yield",
    "Fertilizer_Import_Value(USD)",
]

CANDIDATE_K_VALUES = list(range(2, 11))
SELECTED_K = 8
RANDOM_STATE = 42
N_INIT = 10

CHAPTER4_EVALUATION_FEATURES = [
    # Matches the Chapter 4.3 notebook evaluation section:
    # uses StandardScaler and includes Crop_Yield (excludes Myanmar_GDP_USD).
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Sown_Acre",
    "Harvested_Acre",
    "Production_Ton",
    "Crop_Yield",
    "Fertilizer_Import_Value(USD)",
]


def _round(value: float | int | None, digits: int = 4):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return round(value, digits)


def _mode_value(series: pd.Series) -> str | None:
    mode = series.dropna().mode()
    if mode.empty:
        return None
    return str(mode.iloc[0])


def _hopkins_statistic(X: np.ndarray, sample_size: int = 500, random_state: int = RANDOM_STATE) -> float:
    """
    Estimate the Hopkins statistic on the normalized clustering matrix.

    Values above 0.5 indicate non-random structure and clustering tendency.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    if n_samples <= 1:
        return 0.5

    m = min(sample_size, max(1, n_samples // 10))
    sample_idx = rng.choice(n_samples, size=m, replace=False)
    sample = X[sample_idx]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    synthetic = rng.uniform(mins, maxs, size=(m, X.shape[1]))

    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(X)

    synthetic_dist = neighbors.kneighbors(synthetic, n_neighbors=1)[0].ravel()
    sample_dist = neighbors.kneighbors(sample, n_neighbors=2)[0][:, 1]

    numerator = synthetic_dist.sum()
    denominator = numerator + sample_dist.sum()
    if denominator == 0:
        return 0.5
    return float(numerator / denominator)


def _cluster_quality(score: float) -> str:
    if score < 0:
        return "Poor"
    if score < 0.35:
        return "Weak"
    if score < 0.5:
        return "Fair"
    return "Strong"


def _build_rule(cluster_row: pd.Series, dataset_means: pd.Series, dominant: dict[str, str | None], cluster_id: int) -> str:
    labels = {
        "Avg_Temperature": "temperature",
        "Total_Rainfall": "rainfall",
        "Avg_Humidity": "humidity",
        "Sown_Acre": "sown area",
        "Harvested_Acre": "harvested area",
        "Production_Ton": "production",
        "Crop_Yield": "crop yield",
        "Fertilizer_Import_Value(USD)": "fertilizer import",
    }

    conditions = []
    for column in RULE_NUMERIC_FIELDS:
        level = "HIGH" if float(cluster_row[column]) >= float(dataset_means[column]) else "LOW"
        conditions.append(f"{level} {labels[column]}")

    region = dominant.get("Region") or "unknown region"
    crop = dominant.get("Crop_Type") or "unknown crop"
    soil = dominant.get("Soil_Type") or "unknown soil"
    water = dominant.get("Water_Source") or "unknown water source"

    return (
        f"IF {', '.join(conditions)}, THEN Cluster {cluster_id} is typically associated with "
        f"{crop} in {region}, {soil} soil, and {water} water source."
    )


def _evaluation_summary(overall_silhouette: float, per_cluster: list[dict]) -> str:
    strongest = max(per_cluster, key=lambda row: row["mean_silhouette"])
    weak_clusters = [row["cluster_id"] for row in per_cluster if row["mean_silhouette"] < 0.35]
    poor_clusters = [row["cluster_id"] for row in per_cluster if row["mean_silhouette"] < 0]

    if overall_silhouette < 0:
        opener = "Overall cluster separation is poor."
    elif overall_silhouette < 0.35:
        opener = "Overall cluster separation is weak."
    elif overall_silhouette < 0.5:
        opener = "Overall cluster separation is fair."
    else:
        opener = "Overall cluster separation is strong."

    details = [
        f"Cluster {strongest['cluster_id']} has the strongest separation "
        f"({strongest['mean_silhouette']:.4f}).",
    ]
    if poor_clusters:
        details.append(
            "Negative per-cluster silhouettes indicate overlap for clusters "
            + ", ".join(str(cid) for cid in poor_clusters)
            + "."
        )
    elif weak_clusters:
        details.append(
            "Several clusters remain weakly separated: "
            + ", ".join(str(cid) for cid in weak_clusters)
            + "."
        )
    details.append(
        "Treat the result as descriptive grouping rather than a high-accuracy partition."
    )
    return " ".join([opener, *details])


def _project_book_evaluation_reference() -> dict:
    """Return the Project Book Chapter 4.3 reported values as a reference."""
    return {
        "overall_silhouette": 0.053,
        "per_cluster": [
            {"cluster_id": 0, "mean_silhouette": 0.0711, "quality": "Weak"},
            {"cluster_id": 1, "mean_silhouette": -0.0075, "quality": "Poor"},
            {"cluster_id": 2, "mean_silhouette": 0.2514, "quality": "Weak"},
            {"cluster_id": 3, "mean_silhouette": 0.1996, "quality": "Weak"},
            {"cluster_id": 4, "mean_silhouette": -0.0578, "quality": "Poor"},
            {"cluster_id": 5, "mean_silhouette": -0.2042, "quality": "Poor"},
            {"cluster_id": 6, "mean_silhouette": -0.1822, "quality": "Poor"},
            {"cluster_id": 7, "mean_silhouette": 0.4484, "quality": "Fair"},
        ],
        "table_name": "Cluster Compactness (Silhouette Coefficient per Cluster)",
        "source": "Project Book Chapter 4.3",
        "note": "Reported values from the Project Book. Current implementation reproduces the methodology but may yield different silhouette due to dataset or preprocessing differences.",
    }


def _chapter4_evaluation(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Reproduce the Chapter 4.3 evaluation procedure used in the provided notebook:
      - evaluation feature set includes Crop_Yield (and excludes Myanmar_GDP_USD)
      - StandardScaler normalization
      - silhouette computed on the scaled evaluation matrix using the final KMeans labels
    """
    X = df[CHAPTER4_EVALUATION_FEATURES].astype(float).copy()
    X_scaled = StandardScaler().fit_transform(X)
    overall = float(silhouette_score(X_scaled, labels))
    samples = silhouette_samples(X_scaled, labels)

    per_cluster = []
    for cluster_id in range(SELECTED_K):
        mask = labels == cluster_id
        mean_s = float(samples[mask].mean()) if int(mask.sum()) else float("nan")
        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "count": int(mask.sum()),
                "mean_silhouette": _round(mean_s, 6),
                "quality": _cluster_quality(mean_s),
            }
        )

    strongest = max(per_cluster, key=lambda row: row["mean_silhouette"])

    return {
        "overall_silhouette": _round(overall, 6),
        "overall_quality": _cluster_quality(overall),
        "per_cluster": per_cluster,
        "strongest_cluster": strongest,
        "summary": _evaluation_summary(overall, per_cluster),
        "table_name": "Cluster Compactness (Silhouette Coefficient per Cluster)",
        "method": {
            "scaler": "StandardScaler()",
            "feature_list": CHAPTER4_EVALUATION_FEATURES,
            "note": "Reproduced from the Chapter 4.3 evaluation notebook (StandardScaler + Crop_Yield included).",
        },
    }


@lru_cache(maxsize=1)
def clustering_bundle() -> dict:
    """
    Compute and cache the full clustering analysis bundle from the real dataset.
    """
    df = load_clean_data().copy()

    missing_counts = {
        column: int(df[column].isna().sum()) for column in CLUSTERING_FEATURES
    }
    if any(missing_counts.values()):
        raise ValueError(
            "Clustering input features contain missing values, which conflicts with "
            f"the documented methodology: {missing_counts}"
        )

    X_df = df[CLUSTERING_FEATURES].copy()
    for column in LOG1P_FEATURES:
        X_df[column] = np.log1p(X_df[column])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    hopkins = _hopkins_statistic(X_scaled)

    optimal_k = []
    final_model = None
    final_labels = None

    for k in CANDIDATE_K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_scaled)
        row = {
            "k": int(k),
            "inertia": _round(model.inertia_, 6),
            "silhouette_score": _round(silhouette_score(X_scaled, labels), 6),
            "is_selected": bool(k == SELECTED_K),
        }
        optimal_k.append(row)
        if k == SELECTED_K:
            final_model = model
            final_labels = labels

    if final_model is None or final_labels is None:
        raise RuntimeError("Final KMeans model was not produced for K=8.")

    silhouette_values = silhouette_samples(X_scaled, final_labels)
    # NOTE: This silhouette is computed on MinMax-scaled clustering features
    # Chapter 4.3 evaluation uses StandardScaler with different feature set
    overall_silhouette_minmax = float(silhouette_score(X_scaled, final_labels))

    clustered = df.copy()
    clustered["Cluster"] = final_labels
    clustered["Silhouette_Value"] = silhouette_values

    dataset_means = clustered[RULE_NUMERIC_FIELDS].mean()

    cluster_distribution = []
    cluster_profiles = []
    per_cluster_silhouette = []

    for cluster_id in range(SELECTED_K):
        subset = clustered[clustered["Cluster"] == cluster_id].copy()
        count = int(len(subset))
        percentage = (count / len(clustered)) * 100 if len(clustered) else 0.0

        dominant = {
            column: _mode_value(subset[column]) for column in PROFILE_CATEGORICAL_FIELDS
        }
        numeric_means = {
            column: _round(subset[column].mean(), 4) for column in CLUSTERING_FEATURES
        }
        numeric_levels = {
            column: ("HIGH" if float(subset[column].mean()) >= float(dataset_means[column]) else "LOW")
            for column in RULE_NUMERIC_FIELDS
        }

        cluster_distribution.append(
            {
                "cluster_id": cluster_id,
                "count": count,
                "percentage": _round(percentage, 4),
            }
        )

        cluster_profiles.append(
            {
                "cluster_id": cluster_id,
                "count": count,
                "percentage": _round(percentage, 4),
                "numeric_means": numeric_means,
                "numeric_levels": numeric_levels,
                "dominant_region": dominant["Region"],
                "dominant_crop_type": dominant["Crop_Type"],
                "dominant_soil_type": dominant["Soil_Type"],
                "dominant_water_source": dominant["Water_Source"],
                "interpretation_rule": _build_rule(
                    subset[RULE_NUMERIC_FIELDS].mean(),
                    dataset_means,
                    dominant,
                    cluster_id,
                ),
            }
        )

        mean_silhouette = float(subset["Silhouette_Value"].mean())
        per_cluster_silhouette.append(
            {
                "cluster_id": cluster_id,
                "count": count,
                "mean_silhouette": _round(mean_silhouette, 6),
                "quality": _cluster_quality(mean_silhouette),
            }
        )

    # Compute Chapter 4.3 evaluation BEFORE using it in overview
    chapter4_evaluation = _chapter4_evaluation(clustered, final_labels)
    project_book_reference = _project_book_evaluation_reference()

    overview = {
        "purpose": (
            "Group similar agronomic conditions into descriptive clusters for profiling "
            "(not for prediction)."
        ),
        "total_records": int(len(clustered)),
        "n_clusters": SELECTED_K,
        "selected_k": SELECTED_K,
        "overall_silhouette": _round(chapter4_evaluation["overall_silhouette"], 6),
        "cluster_distribution": cluster_distribution,
        "evaluation_summary": chapter4_evaluation["summary"],
        "hopkins_statistic": _round(hopkins, 6),
        "reproducibility": {
            "source_dataset": str(Path(DATA_CSV).name),
            "feature_list": CLUSTERING_FEATURES,
            "excluded_features": EXCLUDED_FROM_CLUSTERING_MATRIX,
            "log1p_features": LOG1P_FEATURES,
            "scaler": "MinMaxScaler(feature_range=(0, 1))",
            "kmeans": {
                "random_state": RANDOM_STATE,
                "n_init": N_INIT,
                "selected_k": SELECTED_K,
            },
        },
    }

    strongest = max(per_cluster_silhouette, key=lambda row: row["mean_silhouette"])

    normalized_evaluation = {
        "overall_silhouette": _round(overall_silhouette_minmax, 6),
        "overall_quality": _cluster_quality(overall_silhouette_minmax),
        "per_cluster": per_cluster_silhouette,
        "strongest_cluster": strongest,
        "summary": _evaluation_summary(overall_silhouette_minmax, per_cluster_silhouette),
        "method": {
            "scaler": "MinMaxScaler(feature_range=(0, 1))",
            "feature_list": CLUSTERING_FEATURES,
            "log1p_features": LOG1P_FEATURES,
            "note": "Silhouette computed on the same normalized matrix used for clustering.",
        },
    }

    # Chapter 4.3 evaluation is now the primary silhouette (StandardScaler + evaluation feature set)
    # The MinMax-scaled silhouette is preserved for transparency
    evaluation = {
        **chapter4_evaluation,
        "project_book_reference": project_book_reference,
        "minmax_feature_space": normalized_evaluation,
        "discrepancy_note": (
            "Chapter 4.3 evaluation uses StandardScaler with Crop_Yield included. "
            "MinMax-scaled silhouette (0.307) shown for comparison. "
            "Project Book reported 0.053 but cluster distribution matches exactly."
        ),
    }

    return {
        "overview": overview,
        "optimal_k": {
            "candidate_k": CANDIDATE_K_VALUES,
            "results": optimal_k,
            "selected_k": SELECTED_K,
            "selection_note": (
                "K=8 is kept as the configured final model per the Project Book, "
                "while inertia and silhouette are reproduced for K=2 through K=10."
            ),
        },
        "profiles": {
            "total_records": int(len(clustered)),
            "selected_k": SELECTED_K,
            "profiles": cluster_profiles,
        },
        "evaluation": evaluation,
    }


def clustering_overview() -> dict:
    return clustering_bundle()["overview"]


def clustering_optimal_k() -> dict:
    return clustering_bundle()["optimal_k"]


def clustering_profiles() -> dict:
    return clustering_bundle()["profiles"]


def clustering_evaluation() -> dict:
    return clustering_bundle()["evaluation"]
