"""
Shared utilities for reproducing the Data Mining project's exact methodology
and generating deployable model artifacts.

This module centralizes:
- Data source paths
- Chronological train/test splits
- Labeling (training-only median Yield_Level threshold)
- Feature engineering (interactions, association rules, sequential features)

IMPORTANT: This reproduces the project methodology faithfully. Any known
methodology quirks (e.g. leakage in the baseline regression) are preserved
and documented in the artifact metadata rather than silently changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ─── Data source paths ──────────────────────────────────────────────────────

DATA_DIR = Path(
    os.environ.get(
        "AGRIPREDICT_DATA_DIR",
        "/Users/pyaesone/Downloads/Data Mining",
    )
)

CLEANED_DATA_CSV = DATA_DIR / "cleaned_data.csv"

CLASSIFICATION_DIR = DATA_DIR / "Yield_Ton" / "Classification"
TRAIN_LABELED = CLASSIFICATION_DIR / "Train_Labeled_Crop_Yield.xlsx"
TEST_LABELED_2023 = CLASSIFICATION_DIR / "Test_Labeled_Crop_Yield_2023.xlsx"

FINAL_SELECTED_NONNORM = (
    DATA_DIR / "Yield_Ton" / "Final_Selected_Features_NonNormalized (2).xlsx"
)

ASSOC_ENHANCED_FEATURES = (
    DATA_DIR / "Yield_Ton" / "Model 3" / "Association_Enhanced_Features (1) (1).xlsx"
)

# ─── Target/split constants (from project) ──────────────────────────────────

YIELD_LEVEL_THRESHOLD = 0.332187        # training-only median Crop_Yield
YIELD_TL_TRAIN_END = 2022               # Yield level / regression train end
YIELD_TL_TEST_YEAR = 2023
CROP_TYPE_TRAIN_END = 2021              # Crop type train end
CROP_TYPE_TRAIN_END_INCLUSIVE = 2021

# ─── Association mining constants (from project) ────────────────────────────

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MAX_ITEMSET_SIZE = 3

# ─── Selected features / interactions (from project) ────────────────────────

EXISTING_FE_CATEGORICAL = [
    "Crop_Type",
    "Soil_Type",
    "Water_Source",
    "Seeding_Season",
]

EXISTING_FE_NUMERIC = [
    "Sown_Acre",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
]

EXISTING_FE_INTERACTIONS = [
    "Crop_x_WaterSource",
    "Crop_x_SeedingSeason",
    "Crop_x_SoilType",
]

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


# ─── Data loading ───────────────────────────────────────────────────────────

def load_cleaned_data() -> pd.DataFrame:
    """Load the raw cleaned dataset (15 columns, 2012-2023, 5940 rows)."""
    df = pd.read_csv(CLEANED_DATA_CSV)
    return df[[c for c in df.columns if not c.startswith("Unnamed")]]


def load_train_test_labeled() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pre-labeled train (2012-2022) and test (2023) yield files."""
    train = pd.read_excel(TRAIN_LABELED)
    test = pd.read_excel(TEST_LABELED_2023)
    return train, test


def load_selected_features() -> pd.DataFrame:
    """Load Final Selected Features Non-Normalized (Year..Crop_Yield, 13 cols)."""
    return pd.read_excel(FINAL_SELECTED_NONNORM)


# ─── Interaction features ───────────────────────────────────────────────────

def add_existing_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 Crop_x_* interaction features exactly as the project does."""
    out = data.copy()
    out["Crop_x_WaterSource"] = (
        out["Crop_Type"].astype(str) + " | " + out["Water_Source"].astype(str)
    )
    out["Crop_x_SeedingSeason"] = (
        out["Crop_Type"].astype(str) + " | " + out["Seeding_Season"].astype(str)
    )
    out["Crop_x_SoilType"] = (
        out["Crop_Type"].astype(str) + " | " + out["Soil_Type"].astype(str)
    )
    return out


# ─── Manual Apriori (exact project algorithm) ───────────────────────────────

def apriori_manual(transactions, min_support=MIN_SUPPORT, max_k=MAX_ITEMSET_SIZE):
    """Manual Apriori from the project's Association_Enhanced_NN notebook."""
    from collections import Counter

    n = len(transactions)
    all_items = set()
    for transaction in transactions:
        all_items |= transaction

    counter_1 = Counter()
    for transaction in transactions:
        for item in transaction:
            counter_1[frozenset([item])] += 1

    frequent_1 = {
        itemset: count
        for itemset, count in counter_1.items()
        if count / n >= min_support
    }

    all_frequent = dict(frequent_1)
    current_level = frequent_1
    k = 2

    while current_level and k <= max_k:
        previous = list(current_level.keys())
        candidates = set()
        for i in range(len(previous)):
            for j in range(i + 1, len(previous)):
                union = previous[i] | previous[j]
                if len(union) == k:
                    candidates.add(union)

        counter_k = Counter()
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counter_k[candidate] += 1

        frequent_k = {
            itemset: count
            for itemset, count in counter_k.items()
            if count / n >= min_support
        }

        all_frequent.update(frequent_k)
        current_level = frequent_k
        k += 1

    return all_frequent


def generate_rules(freq_itemsets, n, min_confidence=MIN_CONFIDENCE):
    """Generate association rules exactly as the project does."""
    import itertools

    rules = []
    for itemset, count in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        itemset_support = count / n
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_count = freq_itemsets.get(antecedent, 0)
                if antecedent_count == 0:
                    continue
                antecedent_support = antecedent_count / n
                confidence = itemset_support / antecedent_support
                if confidence < min_confidence:
                    continue
                consequent_count = freq_itemsets.get(consequent, 0)
                consequent_support = consequent_count / n
                if consequent_support == 0:
                    continue
                lift = confidence / consequent_support
                rules.append(
                    {
                        "Antecedent": set(antecedent),
                        "Consequent": set(consequent),
                        "Support": itemset_support,
                        "Confidence": confidence,
                        "Lift": lift,
                    }
                )

    if not rules:
        return pd.DataFrame(
            columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
        )

    return (
        pd.DataFrame(rules)
        .sort_values(["Lift", "Confidence"], ascending=False)
        .reset_index(drop=True)
    )


def mine_yield_association_rules(df: pd.DataFrame, train_mask) -> pd.DataFrame:
    """
    Mine Yield_Level association rules from TRAINING data only.

    Exactly reproduces the project's Model3 methodology:
    - mining_columns: Region, Crop_Type, Soil_Type, Seeding_Season, Water_Source, Yield_Level
    - min_support=0.05, max_k=3, min_confidence=0.60
    - keep consequents containing 'Yield_Level=' and Lift > 1
    """
    mining_columns = [
        "Region",
        "Crop_Type",
        "Soil_Type",
        "Seeding_Season",
        "Water_Source",
        "Yield_Level",
    ]

    train = df.loc[train_mask]
    transactions = train[mining_columns].astype(str).apply(
        lambda row: frozenset(
            f"{col}={value}" for col, value in zip(mining_columns, row)
        ),
        axis=1,
    ).tolist()

    freq = apriori_manual(transactions)
    rules = generate_rules(freq, len(transactions))

    yield_rules = rules[
        rules["Consequent"].apply(
            lambda s: any(item.startswith("Yield_Level=") for item in s)
        )
    ].copy()
    yield_rules = yield_rules[yield_rules["Lift"] > 1].copy()
    return yield_rules.reset_index(drop=True)


def rule_matches_row(antecedent, row):
    for item in antecedent:
        if "=" not in item:
            return False
        column, value = item.split("=", 1)
        if str(row[column]) != value:
            return False
    return True


def build_association_features(data: pd.DataFrame, yield_rules: pd.DataFrame) -> pd.DataFrame:
    """Attach the 5 association features to each row exactly as the project does."""
    high_rules = []
    low_rules = []
    for _, rule in yield_rules.iterrows():
        consequent = list(rule["Consequent"])
        if "Yield_Level=High" in consequent:
            high_rules.append(rule)
        if "Yield_Level=Low" in consequent:
            low_rules.append(rule)

    out = data.copy()
    high_confidence, high_lift, low_confidence, low_lift, rule_count = [], [], [], [], []

    for _, row in out.iterrows():
        matched_high = [r for r in high_rules if rule_matches_row(r["Antecedent"], row)]
        matched_low = [r for r in low_rules if rule_matches_row(r["Antecedent"], row)]

        best_high = max(matched_high, key=lambda r: (r["Lift"], r["Confidence"])) if matched_high else None
        best_low = max(matched_low, key=lambda r: (r["Lift"], r["Confidence"])) if matched_low else None

        high_confidence.append(best_high["Confidence"] if best_high is not None else 0.0)
        high_lift.append(best_high["Lift"] if best_high is not None else 0.0)
        low_confidence.append(best_low["Confidence"] if best_low is not None else 0.0)
        low_lift.append(best_low["Lift"] if best_low is not None else 0.0)
        rule_count.append(len(matched_high) + len(matched_low))

    out["Assoc_High_Confidence"] = high_confidence
    out["Assoc_High_Lift"] = high_lift
    out["Assoc_Low_Confidence"] = low_confidence
    out["Assoc_Low_Lift"] = low_lift
    out["Assoc_Rule_Count"] = rule_count

    return out


def add_sequential_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 sequential features grouped by (Region, Crop_Type), previous years only."""
    out = data.copy()
    out = out.sort_values(["Region", "Crop_Type", "Year"])
    grouped = out.groupby(["Region", "Crop_Type"], sort=False)

    out["Prev_Yield_Level"] = grouped["Yield_Level"].shift(1)
    out["Prev2_Yield_Level"] = grouped["Yield_Level"].shift(2)
    out["Prev3_Yield_Level"] = grouped["Yield_Level"].shift(3)

    out["Prev_Yield_High"] = out["Prev_Yield_Level"].eq("High").astype(int)
    out["Prev_Yield_Low"] = out["Prev_Yield_Level"].eq("Low").astype(int)
    out["Prev2_Persistent"] = (
        out["Prev_Yield_Level"].notna()
        & out["Prev2_Yield_Level"].notna()
        & out["Prev_Yield_Level"].eq(out["Prev2_Yield_Level"])
    ).astype(int)
    out["Prev3_Persistent"] = (
        out["Prev_Yield_Level"].notna()
        & out["Prev2_Yield_Level"].notna()
        & out["Prev3_Yield_Level"].notna()
        & out["Prev_Yield_Level"].eq(out["Prev2_Yield_Level"])
        & out["Prev2_Yield_Level"].eq(out["Prev3_Yield_Level"])
    ).astype(int)

    return out


# ─── Artifact helpers ───────────────────────────────────────────────────────

def save_artifact(pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def serialize_rules(rules_df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert sets to sorted lists for clean JSON/CSV serialization.
    rules_df = rules_df.copy()
    rules_df["Antecedent"] = rules_df["Antecedent"].apply(lambda s: "|".join(sorted(s)))
    rules_df["Consequent"] = rules_df["Consequent"].apply(lambda s: "|".join(sorted(s)))
    rules_df.to_csv(path, index=False)


def load_rules(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ─── Metric comparison helper ───────────────────────────────────────────────

def compare_metrics(reported: dict, reproduced: dict):
    print("\n── METRIC COMPARISON (project-reported vs reproduced) ──")
    for key in reported:
        if key in reproduced:
            r = reported[key]
            m = reproduced[key]
            diff = m - r
            print(f"  {key:12s} reported={r:<.6f} reproduced={m:<.6f} diff={diff:+.6f}")
        else:
            print(f"  {key:12s} reported={r:.6f} (not reproduced)")
