"""
Reference data used at inference time.

The real model artifacts for the advanced variants rely on two kinds of
metadata that cannot come from a single user request:

* Association rules (antecedent -> Yield_Level consequence), mined from the
  TRAINING data only and stored in ``models/{task}/association_rules.csv``.
* Historical Yield_Level per (Region, Crop_Type, Year), used to build the
  sequential features ``Prev_Yield_*``. Only strictly previous years are used
  so that no future/current information leaks into prediction.

Both are derived from the bundled cleaned dataset (``backend/data/cleaned_data.csv``)
using the exact same methodology as training:

* Yield_Level is labeled ``High`` when Crop_Yield > training median (0.332187),
  otherwise ``Low``. The threshold is the project's constant.
* Sequential features look back at previous years only; when no history is
  available the feature values default to 0 (identical to the training-stage
  ``add_sequential_features`` behaviour where NaN shifts become 0).

Nothing here is fabricated: GDP / fertilizer-import values are looked up from
the real historical rows by year, and yield levels come from real Crop_Yield.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ──────────────────────────────────────────────────────────────────

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = BACKEND_ROOT / "data" / "cleaned_data.csv"
MODELS_ROOT = BACKEND_ROOT / "models"
REGISTRY_JSON = MODELS_ROOT / "model_registry.json"

# Project constant: training-only median Crop_Yield used to label Yield_Level.
YIELD_LEVEL_THRESHOLD = 0.332187

# Association mining features (project Model3 methodology) — the antecedent
# columns that a rule may reference. Used to match a row against a rule.
_ASSOC_MINING_COLUMNS = [
    "Region",
    "Crop_Type",
    "Soil_Type",
    "Seeding_Season",
    "Water_Source",
]

# Association / sequential feature names (must match the trained artifacts).
ASSOCIATION_FEATURES = [
    "Assoc_High_Confidence",
    "Assoc_High_Lift",
    "Assoc_Low_Confidence",
    "Assoc_Low_Lift",
    "Assoc_Rule_Count",
]

SEQUENCE_FEATURES = [
    "Prev_Yield_High",
    "Prev_Yield_Low",
    "Prev2_Persistent",
    "Prev3_Persistent",
]

# Sequential feature subsets used to look back before the query year.
_SEQ_LOOKBACK_COLUMNS = ["Year", "Yield_Level"]


@lru_cache(maxsize=1)
def load_clean_data() -> pd.DataFrame:
    """Load the bundled cleaned dataset (drops stray unnamed columns)."""
    df = pd.read_csv(DATA_CSV)
    return df[[c for c in df.columns if not str(c).startswith("Unnamed")]]


@lru_cache(maxsize=1)
def load_registry() -> dict:
    """Load the model registry metadata."""
    if not REGISTRY_JSON.exists():
        return {}
    with open(REGISTRY_JSON, "r") as f:
        return json.load(f)


def variant_available(task_id: str, variant: str) -> bool:
    """Whether a task/variant is marked AVAILABLE in the registry."""
    entry = (load_registry() or {}).get(task_id, {}).get(variant, {})
    return entry.get("status") == "AVAILABLE"


@lru_cache(maxsize=8)
def load_association_rules(task_id: str) -> pd.DataFrame:
    """
    Load the stored association rules for a task.

    The CSV columns are Antecedent, Consequent, Support, Confidence, Lift.
    Antecedent/Consequent are stored as ``Item1|Item2`` strings.
    """
    rules_csv = MODELS_ROOT / task_id / "association_rules.csv"
    if not rules_csv.exists():
        return pd.DataFrame(
            columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
        )
    rules = pd.read_csv(rules_csv)
    for col in ("Antecedent", "Consequent"):
        if col in rules.columns:
            rules[col] = rules[col].fillna("").apply(
                lambda s: set(x for x in str(s).split("|") if x)
            )
    return rules


def _rule_matches(antecedent: set, row: dict) -> bool:
    """Check whether a rule's antecedent items match a feature row."""
    for item in antecedent:
        if "=" not in item:
            return False
        column, value = item.split("=", 1)
        if str(row.get(column)) != value:
            return False
    return True


def build_association_features(feature_row: dict, task_id: str) -> dict:
    """
    Compute the 5 association features for a single row.

    Mirrors training's ``build_association_features``: for each rule whose
    antecedent matches the row, take the best (highest Lift) High and Low rule
    to populate the confidence/lift features, and the rule count.

    ``feature_row`` must contain the antecedent columns (Region, Crop_Type,
    Soil_Type, Seeding_Season, Water_Source).
    """
    rules = load_association_rules(task_id)
    high_rules = [
        r for r in rules.itertuples()
        if "Yield_Level=High" in r.Consequent
    ]
    low_rules = [
        r for r in rules.itertuples()
        if "Yield_Level=Low" in r.Consequent
    ]

    matched_high = [r for r in high_rules if _rule_matches(r.Antecedent, feature_row)]
    matched_low = [r for r in low_rules if _rule_matches(r.Antecedent, feature_row)]

    best_high = max(matched_high, key=lambda r: (r.Lift, r.Confidence)) if matched_high else None
    best_low = max(matched_low, key=lambda r: (r.Lift, r.Confidence)) if matched_low else None

    return {
        "Assoc_High_Confidence": float(best_high.Confidence) if best_high is not None else 0.0,
        "Assoc_High_Lift": float(best_high.Lift) if best_high is not None else 0.0,
        "Assoc_Low_Confidence": float(best_low.Confidence) if best_low is not None else 0.0,
        "Assoc_Low_Lift": float(best_low.Lift) if best_low is not None else 0.0,
        "Assoc_Rule_Count": float(len(matched_high) + len(matched_low)),
    }


@lru_cache(maxsize=32)
def historical_overview(region: str):
    """
    Return REAL per-crop historical yield series for a region, aggregated by
    year, from the bundled cleaned dataset. Only real values are returned.

    Returns ``None`` when the region has no matching real rows, otherwise a dict::

        {
          "region": "Mandalay",
          "year_min": 2012,
          "year_max": 2023,
          "crops": { "Paddy": {"years": [...], "yields": [...]}, ... }
        }
    """
    df = load_clean_data()
    region_name = str(region).strip()
    sub = df[df["Region"].astype(str).str.strip() == region_name]
    if sub.empty:
        return None

    crops: dict[str, dict] = {}
    for crop, grp in sub.groupby("Crop_Type"):
        agg = grp.groupby("Year")["Crop_Yield"].mean().sort_index()
        crops[str(crop)] = {
            "years": [int(y) for y in agg.index.tolist()],
            "yields": [round(float(v), 4) for v in agg.tolist()],
        }

    return {
        "region": region_name,
        "year_min": int(sub["Year"].min()),
        "year_max": int(sub["Year"].max()),
        "crops": crops,
    }


@lru_cache(maxsize=1)
def _yield_level_history() -> dict[tuple, dict[int, str]]:
    """
    Build { (Region, Crop_Type): {Year: Yield_Level} } from the bundled data.

    Yield_Level is derived from real Crop_Yield using the training threshold:
    High if Crop_Yield > 0.332187 else Low.
    """
    df = load_clean_data()
    df = df[["Region", "Crop_Type", "Year", "Crop_Yield"]].copy()
    df["Yield_Level"] = np.where(df["Crop_Yield"] > YIELD_LEVEL_THRESHOLD, "High", "Low")

    history: dict[tuple, dict[int, str]] = {}
    for region, crop, year, level in df[["Region", "Crop_Type", "Year", "Yield_Level"]].itertuples(index=False, name=None):
        history.setdefault((region, crop), {})[int(year)] = level
    return history


def build_sequential_features(region: str, crop_type: str, year: int) -> dict:
    """
    Compute the 4 sequential features for a query (region, crop, year).

    Uses ONLY years strictly before ``year``:
      Prev_Yield_High = 1 if previous-year level is High
      Prev_Yield_Low  = 1 if previous-year level is Low
      Prev2_Persistent = 1 if previous two available years were equal
      Prev3_Persistent = 1 if previous three available years were equal

    If insufficient history exists the value defaults to 0 (matching training's
    NaN-shift -> 0 behaviour). No current/future information is used.
    """
    history = _yield_level_history().get((region, crop_type), {})

    prev_years = sorted((y for y in history if y < year), reverse=True)
    levels = [history[y] for y in prev_years]

    prev = levels[0] if len(levels) >= 1 else None
    prev2 = levels[1] if len(levels) >= 2 else None
    prev3 = levels[2] if len(levels) >= 3 else None

    return {
        "Prev_Yield_High": 1 if prev == "High" else 0,
        "Prev_Yield_Low": 1 if prev == "Low" else 0,
        "Prev2_Persistent": 1 if (prev is not None and prev2 is not None and prev == prev2) else 0,
        "Prev3_Persistent": (
            1
            if (prev is not None and prev2 is not None and prev3 is not None and prev == prev2 == prev3)
            else 0
        ),
    }


@lru_cache(maxsize=1)
def _macro_by_year() -> dict[int, dict[str, float]]:
    """Lookup of country-level macro features by year from real data."""
    df = load_clean_data()
    stats = {}
    for year, grp in df.groupby("Year"):
        fert = grp["Fertilizer_Import_Value(USD)"].dropna()
        gdp = grp["Myanmar_GDP_USD"].dropna()
        stats[int(year)] = {
            "Fertilizer_Import_Value(USD)": float(fert.mean()) if len(fert) else None,
            "Myanmar_GDP_USD": float(gdp.mean()) if len(gdp) else None,
        }
    return stats


def macro_features_for_year(year: int) -> dict:
    """Return real GDP / fertilizer-import values for a given year (may be empty)."""
    return _macro_by_year().get(int(year), {})


def historical_trends_for(region: str, crop_type: str):
    """
    Return REAL historical trend series for a (region, crop_type) from the bundled
    cleaned dataset, aggregated by year.

    Returns a dict with lists ``years, yields, rainfall, temperatures, areas`` or
    ``None`` when no matching real rows exist. No fabricated values are produced.
    """
    df = load_clean_data()
    mask = (df["Region"].astype(str).str.strip() == str(region).strip()) & (
        df["Crop_Type"].astype(str).str.strip() == str(crop_type).strip()
    )
    sub = df.loc[mask]
    if sub.empty:
        return None

    agg = sub.groupby("Year").agg(
        yield_=("Crop_Yield", "mean"),
        rainfall=("Total_Rainfall", "mean"),
        temperature=("Avg_Temperature", "mean"),
        area=("Sown_Acre", "mean"),
    ).sort_index()

    return {
        "years": [int(y) for y in agg.index.tolist()],
        "yields": [round(float(v), 4) for v in agg["yield_"].tolist()],
        "rainfall": [round(float(v), 2) for v in agg["rainfall"].tolist()],
        "temperatures": [round(float(v), 2) for v in agg["temperature"].tolist()],
        "areas": [round(float(v), 2) for v in agg["area"].tolist()],
    }

